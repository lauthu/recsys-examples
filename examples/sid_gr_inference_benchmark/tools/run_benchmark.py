#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the SID-GR / SGLang HTTP beam-search benchmark in Docker.

Each clean run and the profiled run is a fresh container and server process.
OUTPUT receives report.md, sunburst.svg and profile.nsys-rep; intermediate
files are kept in OUTPUT/.work only when a run fails.
"""

import argparse
import hashlib
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from bench import catalog, report, sunburst, trace
from bench.common import ENGINES, SIDTRIE_FILE, is_sglang, read_json, write_json

TOOLS = Path(__file__).resolve().parent
LABEL = "sid-gr-inference-benchmark"
CACHE_ENV = """XDG_CACHE_HOME HF_HOME TORCH_HOME TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR
    TORCHINDUCTOR_CACHE_DIR CUDA_CACHE_PATH FLASHINFER_WORKSPACE_BASE TVM_FFI_CACHE_DIR
    CUTE_DSL_CACHE_DIR QUACK_CACHE_DIR SGLANG_CACHE_DIR SGLANG_JIT_CACHE_DIR
    NUMBA_CACHE_DIR MPLCONFIGDIR""".split()
# Set by the runner or the client from the workload, catalog and container layout.
MANAGED_ENV = set(CACHE_ENV) | set(
    """BENCH_PROFILE BENCH_ENGINE PYTHONPATH PYTHON_BIN HOME CUDA_VISIBLE_DEVICES
    MODEL_DIR GR_MODEL_DIR GR_DECODE_ATTEN_ROOT GR_HTTP_HOST GR_HTTP_PORT GR_DEVICE
    GR_CONTEXT_LEN GR_BEAM_WIDTH GR_DECODE_STEPS GR_MAX_BATCH_SIZE GR_DECODE_BACKEND
    GR_BEAM_KV_POOL_CAPACITY GR_CONTEXT_KV_POOL_CAPACITY GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS
    GR_SUPPRESS_SPECIAL_TOKENS_ON_IGNORE_EOS GR_CATALOG_JSONL GR_ALLOW_CATALOG_RELOAD
    SGLANG_PROFILE_V2 SGLANG_BEAM_TRIE_TOPK_LOGPROB""".split()
)
SHARED = "SGLANG_BEAM_SHARED_CONTEXT_ATTENTION"


def fail(message):
    raise ValueError(message)


def require(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        fail(f"{name} must be an integer >= {minimum}")


def require_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        fail(f"{name} needs exactly the keys {sorted(keys)}")


def load_config(path):
    config = read_json(path)
    engine = config.get("engine")
    if engine not in ENGINES:
        fail(f"engine must be one of {', '.join(ENGINES)}")
    keys = ["engine", "workload", "runs", "cpus", "gpu", "env"]
    keys += ["server_settings"] if is_sglang(engine) else []
    keys += ["trie_config"] if engine == "sglang-sidtrie" else []
    require_keys(config, keys, "the config")
    workload, runs, cpus, env = (config[k] for k in ("workload", "runs", "cpus", "env"))
    minimums = {"context_length": 1, "beam_width": 1, "output_tokens": 2, "seed": 0}
    require_keys(workload, minimums, "workload")
    for key, minimum in minimums.items():
        require(workload[key], f"workload.{key}", minimum)
    minimums = {"clean": 1, "requests": 1, "warmups": 0, "profile_requests": 1}
    require_keys(runs, minimums, "runs")
    for key, minimum in minimums.items():
        require(runs[key], f"runs.{key}", minimum)
    require_keys(cpus, ("server", "client"), "cpus")
    for key, value in cpus.items():
        if not isinstance(value, list) or not value or len(set(value)) != len(value):
            fail(f"cpus.{key} must be a nonempty list of distinct CPU numbers")
        for cpu in value:
            require(cpu, f"cpus.{key} entry", 0)
    if set(cpus["server"]) & set(cpus["client"]):
        fail("cpus.server and cpus.client must not overlap")
    gpu = config["gpu"]
    if type(gpu) is not str or not re.fullmatch(r"\d+|GPU-[\w-]+", gpu):
        fail('gpu must be one device index such as "0" or a GPU UUID')
    if not isinstance(env, dict) or any(type(v) is not str for v in env.values()):
        fail('env values must be strings, for example "1" rather than true')
    if MANAGED_ENV & env.keys():
        fail(f"env sets runner-managed variables {sorted(MANAGED_ENV & env.keys())}")
    if is_sglang(engine):
        check_server_settings(engine, config["server_settings"])
    if engine == "sglang-sidtrie":
        if workload["output_tokens"] != 3:
            fail("sglang-sidtrie needs workload.output_tokens 3 (three SID codebooks)")
        if env.get(SHARED, "false") not in ("true", "false"):
            fail(f'{SHARED} must be "true" or "false"')
        trie = config["trie_config"]
        require_keys(trie, ("token_start", "codebook_size"), "trie_config")
        require(trie["token_start"], "trie_config.token_start", 0)
        require(trie["codebook_size"], "trie_config.codebook_size")
    return config


def check_server_settings(engine, settings):
    managed = {"model_path", "host", "port", "beam_search_constraint_dict"}
    if managed & settings.keys():
        fail(f"server_settings must not set {sorted(managed & settings.keys())}")
    if settings.get("tp_size", 1) != 1 or not settings.get("disable_overlap_schedule"):
        fail("server_settings need tp_size 1 and disable_overlap_schedule true")
    legacy = {"enable_beam_search", "enable_beam_cascade_attention"} & settings.keys()
    if engine == "sglang-beam-search-new" and not settings.get("enable_beam_search"):
        fail("sglang-beam-search-new needs server_settings.enable_beam_search true")
    if engine == "sglang-sidtrie" and legacy:
        fail(f"sglang-sidtrie uses the native beam_width API; remove {sorted(legacy)}")


def load_model(model, workload):
    config = read_json(model / "config.json")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        fail("the model checkpoint must be bfloat16")
    names = [p.name for p in model.iterdir() if p.suffix == ".safetensors"]
    if not set(names) - {SIDTRIE_FILE}:
        fail(f"{model} has no safetensors weights")
    length = workload["context_length"] + workload["output_tokens"]
    if length > config.get("max_position_embeddings", length):
        fail("context_length + output_tokens exceeds max_position_embeddings")
    if workload["beam_width"] > config["vocab_size"]:
        fail("beam_width exceeds the vocabulary size")
    return config


def make_workload(config, model_config):
    """Seeded random token IDs without special tokens; identical for every engine."""
    workload, runs = config["workload"], config["runs"]
    special = set()
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = model_config.get(key)
        special.update(value if isinstance(value, list) else [value])
    allowed = [t for t in range(model_config["vocab_size"]) if t not in special]
    rng, length = random.Random(workload["seed"]), workload["context_length"]
    count = runs["warmups"] + max(runs["requests"], runs["profile_requests"])
    rows = [rng.choices(allowed, k=length) for _ in range(count)]
    return [
        {"request_id": f"request-{i}", "input_ids": row} for i, row in enumerate(rows)
    ]


def cpu_list(cpus):
    """[0, 1, 2, 5] -> "0-2,5"."""
    spans = []
    for cpu in sorted(cpus):
        if spans and cpu == spans[-1][1] + 1:
            spans[-1][1] = cpu
        else:
            spans.append([cpu, cpu])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


def git(path, *args):
    command = ["git", "-C", str(path), *args]
    return subprocess.run(command, capture_output=True, text=True).stdout


def checkout(path, *pathspec, untracked="no"):
    """Report label and a fingerprint that changes with the checked-out files."""
    commit = git(path, "rev-parse", "HEAD").strip()
    if not commit:
        return "(not a git checkout)", None
    status = git(path, "status", "--porcelain", f"-u{untracked}", "--", *pathspec)
    diff = git(path, "diff", "HEAD", "--", *pathspec)
    label = commit[:12] + (" with uncommitted changes" if status else "")
    return label, hashlib.sha256((commit + status + diff).encode()).hexdigest()


def input_state(args, engine):
    """Source, tools and model identity; compared before and after the runs."""
    paths = ["examples/sid-gr-inference", "corelib/gr_decode_atten"]
    tools = [p for p in sorted(TOOLS.rglob("*")) if p.is_file()]
    tools = [p.read_bytes() for p in tools if "__pycache__" not in p.parts]
    model = [
        (p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in args.model.iterdir()
    ]
    return {
        "source": checkout(args.source, *(["python"] if is_sglang(engine) else paths)),
        "tools": checkout(TOOLS, ".", untracked="normal")[0],
        "tools_hash": hashlib.sha256(b"".join(tools)).hexdigest(),
        "model": sorted(model),
    }


def docker_command(args, config, work, cache, image, name, run):
    sidtrie = config["engine"] == "sglang-sidtrie"
    source = "/sglang" if is_sglang(config["engine"]) else "/source"
    model = "/model-source" if sidtrie else "/model"
    mounts = [(TOOLS, "/benchmark-tools", True), (work, "/work", False)]
    mounts += [(cache, "/cache", False), (args.source, source, True)]
    mounts += [(args.model, model, True)]
    if sidtrie:  # links to /model-source plus the generated trie head
        mounts.append((work / "model", "/model", True))
    env = {key: f"/cache/{key.lower()}" for key in CACHE_ENV}
    env.update(HOME="/cache/home", USER="benchmark", LOGNAME="benchmark")
    env.update(PYTHONPATH="/benchmark-tools", PYTHONDONTWRITEBYTECODE="1")
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cpus = cpu_list(config["cpus"]["server"] + config["cpus"]["client"])
    argv = ["docker", "run", "--rm", "--init", "--name", name, "--label", LABEL]
    argv += ["--network", "none", "--gpus", f"device={config['gpu']}"]
    argv += ["--cpuset-cpus", cpus, "--user", f"{os.getuid()}:{os.getgid()}"]
    argv += ["--shm-size", "4g", "--cap-add", "SYS_PTRACE"]
    argv += ["--security-opt", "seccomp=unconfined"]
    for key, value in sorted(env.items()):
        argv += ["--env", f"{key}={value}"]
    for path, target, readonly in mounts:
        option = f"type=bind,src={path},dst={target}"
        argv += ["--mount", option + (",readonly" if readonly else "")]
    argv += [image, "python3", "-m", "bench.client", "--plan", "/work/plan.json"]
    return argv + ["--run", run] + (["--profile"] if run == "profile" else [])


def run_container(argv, name, log):
    try:
        with log.open("w") as stream:
            code = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
    finally:
        # Never leave a container measuring after the runner stops for any reason.
        subprocess.run(["docker", "kill", name], capture_output=True)
    if code.returncode:
        server_log = log.with_suffix("") / "server.log"
        fail(f"container exited with {code.returncode}; see {log} and {server_log}")


def prepare(args, config, model_config, work):
    """Write plan.json, the workload and the engine's catalog files."""
    engine, workload, runs = config["engine"], config["workload"], config["runs"]
    vocab, width = model_config["vocab_size"], workload["beam_width"]
    length = workload["output_tokens"]
    plan_catalog, info = None, None
    if args.catalog:
        data = args.catalog.read_bytes()
        paths, info = catalog.parse(
            data, output_tokens=length, beam_width=width, vocab_size=vocab
        )
        info["name"] = args.catalog.name
        (work / "catalog.jsonl").write_bytes(data)
        source = "/work/catalog.jsonl"
        if engine == "sglang-beam-search-new":
            source = "/work/catalog.pkl"
            (work / "catalog.pkl").write_bytes(catalog.sglang_trie(paths))
        elif engine == "sglang-sidtrie":
            source, view = f"/model/{SIDTRIE_FILE}", work / "model"
            view.mkdir()
            for item in args.model.iterdir():
                if item.name != SIDTRIE_FILE:
                    (view / item.name).symlink_to(f"/model-source/{item.name}")
            trie = catalog.sidtrie(paths, **config["trie_config"], vocab_size=vocab)
            (view / SIDTRIE_FILE).write_bytes(trie)
        jsonl = "/work/catalog.jsonl"
        plan_catalog = dict(path=jsonl, server_path=jsonl, server_source=source)
        plan_catalog["items"] = info["items"]
    elif engine == "sglang-sidtrie":
        fail("sglang-sidtrie needs --catalog")
    write_json(work / "workload.json", make_workload(config, model_config))
    plan = {"engine": engine, "workload": workload, "vocab_size": vocab}
    plan.update({key: runs[key] for key in ("warmups", "requests", "profile_requests")})
    plan.update(env=config["env"], server_settings=config.get("server_settings"))
    cpus = config["cpus"]
    plan.update(server_cpus=cpus["server"], client_cpus=cpus["client"])
    plan.update(workload_path="/work/workload.json", catalog=plan_catalog)
    write_json(work / "plan.json", plan)
    return info


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TOOLS / "benchmark.json")
    parser.add_argument(
        "--source", type=Path, required=True, help="recsys-examples or SGLang checkout"
    )
    parser.add_argument("--model", type=Path, required=True, help="bf16 checkpoint")
    parser.add_argument("--image", required=True, help="image with the engine and nsys")
    parser.add_argument("--catalog", type=Path, help="JSONL items that constrain beams")
    parser.add_argument("--output", type=Path, required=True, help="new or empty dir")
    parser.add_argument(
        "--cache-dir", type=Path, help="JIT cache (default: OUTPUT/../.cache/IMAGE)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print docker commands")
    args = parser.parse_args(argv)
    for name in ("source", "model", "catalog", "output", "cache_dir"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    model_config = load_model(args.model, config["workload"])
    entry = "examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
    if is_sglang(config["engine"]):
        entry = "python/sglang/srt/entrypoints/http_server.py"
    if not (args.source / entry).is_file():
        fail(f"--source has no {entry}")
    if args.output.exists() and any(args.output.iterdir()):
        fail(f"output directory {args.output} is not empty")
    runs = [f"clean_{i}" for i in range(1, config["runs"]["clean"] + 1)] + ["profile"]
    work = args.output / ".work"
    if args.dry_run:
        cache = args.cache_dir or args.output.parent / ".cache"
        for run in runs:
            argv = docker_command(args, config, work, cache, args.image, "bench", run)
            print(shlex.join(argv))
        return 0

    inspect = ["docker", "image", "inspect", "--format", "{{.Id}}", args.image]
    image = subprocess.run(inspect, capture_output=True, text=True).stdout.strip()
    if not image:
        fail(f"no local Docker image {args.image}; the runner never pulls images")
    running = ["docker", "ps", "-q", "--filter", f"label={LABEL}"]
    busy = subprocess.run(running, capture_output=True, text=True, check=True).stdout
    if busy.split():
        fail(f"benchmark containers {busy.split()} are running; they would interfere")
    cache = args.cache_dir or args.output.parent / ".cache" / image.split(":")[-1][:12]
    (cache / "home").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True)
    catalog_info = prepare(args, config, model_config, work)
    before = input_state(args, config["engine"])
    for run in runs:
        print(f"== {run}", flush=True)
        name = f"bench-{uuid.uuid4().hex[:12]}"
        argv = docker_command(args, config, work, cache, image, name, run)
        run_container(argv, name, work / f"{run}.log")
    if input_state(args, config["engine"]) != before:
        fail("source, tools or model files changed during the runs; results are mixed")
    write_outputs(args, config, work, runs, catalog_info, before, image)
    shutil.rmtree(work)
    print(f"Wrote report.md, sunburst.svg and profile.nsys-rep to {args.output}")
    return 0


def write_outputs(args, config, work, runs, catalog_info, before, image):
    summary = trace.analyze(work / "profile/trace.sqlite")
    title = f"{report.ENGINE_NAMES[config['engine']]} · {args.model.name}"
    subtitle = f"GPU kernel time per request, mean of {summary['requests']} profiled"
    subtitle += " requests: phase › section › kernel family"
    svg = sunburst.render(summary["work"], summary["requests"], title, subtitle)
    (args.output / "sunburst.svg").write_text(svg)
    shutil.move(work / "profile/trace.nsys-rep", args.output / "profile.nsys-rep")
    context = {
        "config": config,
        "clean": [read_json(work / run / "result.json") for run in runs[:-1]],
        "profile": read_json(work / "profile/result.json"),
        "trace": summary,
        "catalog": catalog_info,
        "model": args.model.name,
        "source": f"{args.source.name} {before['source'][0]}",
        "tools": before["tools"],
        "image": image,
        "cpus": {key: cpu_list(value) for key, value in config["cpus"].items()},
        "command": shlex.join(["python3", *sys.argv]),
    }
    (args.output / "report.md").write_text(report.render(context))


if __name__ == "__main__":
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda signum, frame: sys.exit(128 + signum))
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"benchmark failed: {exc}")
