#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark HTTP beam-search solutions on one scenario, each in Docker.

A scenario (configs/scenarios/*.yaml) fixes the model, workload and runs; a
solution (configs/solutions/*.yaml) fixes how one engine is launched. Every clean run and the profiled run is a fresh container and server process.
Each solution's directory receives report.md, sunburst.svg and profile.nsys-rep;
with several solutions, OUTPUT also gets comparison.md. Intermediate files are
kept in .work only when a run fails.
"""

import argparse
import hashlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from bench import catalog, compare, provenance, report, sunburst, trace
from bench.common import read_json, write_json
from bench.config import (
    Scenario,
    Solution,
    load_model,
    load_scenario,
    load_solution,
    make_workload,
)
from bench.engines import fail

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "bench"
LABEL = "sid-gr-inference-benchmark"
CACHE_ENV = """XDG_CACHE_HOME HF_HOME TORCH_HOME TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR
    TORCHINDUCTOR_CACHE_DIR CUDA_CACHE_PATH FLASHINFER_WORKSPACE_BASE TVM_FFI_CACHE_DIR
    CUTE_DSL_CACHE_DIR QUACK_CACHE_DIR SGLANG_CACHE_DIR SGLANG_JIT_CACHE_DIR
    NUMBA_CACHE_DIR MPLCONFIGDIR""".split()


@dataclass
class Run:
    """One solution to measure: its configs, source checkout, image and output."""

    name: str  # the solution file name without .yaml
    engine: object
    scenario: Scenario
    solution: Solution
    source: Path
    image: str
    output: Path
    solution_path: Path | None = None
    manifest: dict | None = None  # source and image identity, set before the runs

    @property
    def work(self):
        return self.output / ".work"


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


def input_state(args, run):
    """Source, benchmark code and model identity; compared before and after the runs."""
    files = [p for p in sorted(BENCH.rglob("*")) if p.is_file()]
    files = [p.read_bytes() for p in files if "__pycache__" not in p.parts]
    model = [
        (p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in args.model.iterdir()
    ]
    return {
        "source": checkout(run.source, *run.engine.fingerprint),
        "tools": checkout(ROOT, "bench", untracked="normal")[0],
        "tools_hash": hashlib.sha256(b"".join(files)).hexdigest(),
        "model": sorted(model),
    }


def docker_command(args, run, cache, image, name, phase):
    engine, scenario = run.engine, run.scenario
    model = "/model-source" if engine.model_view else "/model"
    mounts = [(BENCH, "/benchmark-tools/bench", True), (run.work, "/work", False)]
    mounts += [(cache, "/cache", False), (run.source, engine.mount, True)]
    mounts += [(args.model, model, True)]
    if engine.model_view:  # links to /model-source plus the generated trie head
        mounts.append((run.work / "model", "/model", True))
    env = {key: f"/cache/{key.lower()}" for key in CACHE_ENV}
    env.update(HOME="/cache/home", USER="benchmark", LOGNAME="benchmark")
    env.update(PYTHONPATH="/benchmark-tools", PYTHONDONTWRITEBYTECODE="1")
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cpus = cpu_list(scenario.cpus.server + scenario.cpus.client)
    argv = ["docker", "run", "--rm", "--init", "--name", name, "--label", LABEL]
    argv += ["--network", "none", "--gpus", f"device={scenario.gpu}"]
    argv += ["--cpuset-cpus", cpus, "--user", f"{os.getuid()}:{os.getgid()}"]
    argv += ["--shm-size", "4g", "--cap-add", "SYS_PTRACE"]
    argv += ["--security-opt", "seccomp=unconfined"]
    for key, value in sorted(env.items()):
        argv += ["--env", f"{key}={value}"]
    for path, target, readonly in mounts:
        option = f"type=bind,src={path},dst={target}"
        argv += ["--mount", option + (",readonly" if readonly else "")]
    argv += [image, "python3", "-m", "bench.client", "--plan", "/work/plan.json"]
    return argv + ["--run", phase] + (["--profile"] if phase == "profile" else [])


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


def prepare(args, run, model_config, workload_rows):
    """Write plan.json, the workload and the engine's catalog files."""
    engine, solution, work = run.engine, run.solution, run.work
    workload, runs = run.scenario.workload, run.scenario.runs
    vocab = model_config["vocab_size"]
    plan_catalog, info = None, None
    if args.catalog:
        data = args.catalog.read_bytes()
        paths, info = catalog.parse(
            data,
            output_tokens=workload.output_tokens,
            beam_width=workload.beam_width,
            vocab_size=vocab,
        )
        info["name"] = args.catalog.name
        (work / "catalog.jsonl").write_bytes(data)
        source = "/work/catalog.jsonl"
        if engine.model_view:
            source, view = f"/model/{engine.catalog_file}", work / "model"
            view.mkdir()
            for item in args.model.iterdir():
                if item.name != engine.catalog_file:
                    (view / item.name).symlink_to(f"/model-source/{item.name}")
            native = engine.convert_catalog(paths, solution, vocab)
            (view / engine.catalog_file).write_bytes(native)
        elif engine.catalog_file:
            source = f"/work/{engine.catalog_file}"
            native = engine.convert_catalog(paths, solution, vocab)
            (work / engine.catalog_file).write_bytes(native)
        jsonl = "/work/catalog.jsonl"
        plan_catalog = dict(path=jsonl, server_path=jsonl, server_source=source)
        plan_catalog["items"] = info["items"]
    elif engine.catalog_required:
        fail(f"{engine.name} needs --catalog")
    write_json(work / "workload.json", workload_rows)
    # The container reads this generated JSON; it never sees the YAML files.
    plan = {"engine": engine.name, "workload": asdict(workload), "vocab_size": vocab}
    plan.update(warmups=runs.warmups, requests=runs.requests)
    plan.update(profile_requests=runs.profile_requests)
    plan.update(env=solution.env, server_settings=solution.server_settings)
    cpus = run.scenario.cpus
    plan.update(server_cpus=cpus.server, client_cpus=cpus.client)
    plan.update(workload_path="/work/workload.json", catalog=plan_catalog)
    write_json(work / "plan.json", plan)
    return info


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution",
        nargs=3,
        action="append",
        required=True,
        metavar=("CONFIG", "SOURCE", "IMAGE"),
        help="solution YAML, its engine source checkout and a local image with its "
        "dependencies and nsys; repeat to compare solutions (see --reference)",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=ROOT / "configs/scenarios/qwen3_0_6b.yaml",
        help="scenario YAML: model pin, workload, runs, CPUs and GPU",
    )
    parser.add_argument("--model", type=Path, required=True, help="bf16 checkpoint")
    parser.add_argument("--catalog", type=Path, help="JSONL items that constrain beams")
    parser.add_argument("--output", type=Path, required=True, help="new or empty dir")
    parser.add_argument(
        "--cache-dir", type=Path, help="JIT cache (default: OUTPUT/../.cache/IMAGE)"
    )
    parser.add_argument(
        "--reference",
        help="solution name (file name without .yaml) the comparison measures "
        "against (default: sglang if given, else the first solution)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run although a source differs from its pinned commit; the report "
        "marks the result as not reproducible",
    )
    parser.add_argument("--dry-run", action="store_true", help="print docker commands")
    args = parser.parse_args(argv)
    for name in ("scenario", "model", "catalog", "output", "cache_dir"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def build_runs(args, scenario):
    runs = []
    for solution_path, source, image in args.solution:
        solution_path = Path(solution_path).expanduser().resolve()
        engine, solution = load_solution(solution_path, scenario.workload)
        source = Path(source).expanduser().resolve()
        if not (source / engine.entry).is_file():
            fail(f"{source} is not {engine.source_label}: no {engine.entry}")
        if args.catalog and not engine.catalog_supported:
            fail(f"{engine.title} cannot constrain beams to a catalog; drop --catalog")
        name = solution_path.stem
        output = args.output if len(args.solution) == 1 else args.output / name
        runs.append(
            Run(name, engine, scenario, solution, source, image, output, solution_path)
        )
    if len({run.name for run in runs}) != len(runs):
        fail("--solution configs need distinct file names")
    if args.reference and args.reference not in {run.name for run in runs}:
        fail(f"--reference {args.reference} is not one of the --solution configs")
    return runs


def main(argv=None):
    args = parse_args(argv)
    scenario = load_scenario(args.scenario)
    model_config = load_model(args.model, scenario.workload)
    runs = build_runs(args, scenario)
    if args.output.exists() and any(args.output.iterdir()):
        fail(f"output directory {args.output} is not empty")
    phases = [f"clean_{i}" for i in range(1, scenario.runs.clean + 1)]
    phases.append("profile")
    if args.dry_run:
        for run in runs:
            cache = args.cache_dir or args.output.parent / ".cache"
            for phase in phases:
                print(
                    shlex.join(
                        docker_command(args, run, cache, run.image, "bench", phase)
                    )
                )
        return 0

    for run in runs:
        run.manifest = {
            "source": provenance.verify_source(
                run.source,
                run.solution.source,
                run.engine.fingerprint,
                args.allow_dirty,
            ),
            "image": provenance.image_info(run.image),
        }
        run.image = run.manifest["image"]["id"]
    model = provenance.verify_model(args.model, scenario.model)
    running = ["docker", "ps", "-q", "--filter", f"label={LABEL}"]
    busy = subprocess.run(running, capture_output=True, text=True, check=True).stdout
    if busy.split():
        fail(f"benchmark containers {busy.split()} are running; they would interfere")
    rows = make_workload(scenario, model_config)
    finished = []
    for run in runs:
        print(f"=== {run.name}", flush=True)
        finished.append(measure(args, run, phases, model_config, rows, model))
    if len(runs) > 1:
        write_comparison(args, runs, finished)
    print(f"Wrote the reports to {args.output}")
    return 0


def measure(args, run, phases, model_config, rows, model):
    """Run every phase for one solution and write its three output files."""
    image = run.image
    cache = args.cache_dir or args.output.parent / ".cache" / image.split(":")[-1][:12]
    (cache / "home").mkdir(parents=True, exist_ok=True)
    run.work.mkdir(parents=True)
    catalog_info = prepare(args, run, model_config, rows)
    before = input_state(args, run)
    for phase in phases:
        print(f"== {phase}", flush=True)
        name = f"bench-{uuid.uuid4().hex[:12]}"
        argv = docker_command(args, run, cache, image, name, phase)
        run_container(argv, name, run.work / f"{phase}.log")
    if input_state(args, run) != before:
        fail("source, benchmark code or model files changed during the runs")
    summary = write_outputs(args, run, phases, catalog_info, before, model)
    shutil.rmtree(run.work)
    return summary


def manifest(args, run, before, model, profile):
    """Everything needed to reproduce the run, printed at the end of the report."""
    return {
        **run.manifest,
        "model": model,
        "benchmark_code": before["tools"],
        "dockerfile": run.engine.dockerfile,
        "configs": {
            "scenario": provenance.sha256_file(args.scenario),
            "solution": provenance.sha256_file(run.solution_path),
        },
        "nsys": profile.get("nsys"),
    }


def write_outputs(args, run, phases, catalog_info, before, model):
    work, cpus = run.work, run.scenario.cpus
    summary = trace.analyze(work / "profile/trace.sqlite")
    title = f"{run.engine.title} · {model['id'].split('/')[-1]}"
    subtitle = f"GPU kernel time per request, mean of {summary['requests']} profiled"
    subtitle += " requests: phase › section › kernel family"
    svg = sunburst.render(summary["work"], summary["requests"], title, subtitle)
    (run.output / "sunburst.svg").write_text(svg)
    shutil.move(work / "profile/trace.nsys-rep", run.output / "profile.nsys-rep")
    clean = [read_json(work / phase / "result.json") for phase in phases[:-1]]
    context = {
        "scenario": run.scenario,
        "solution": run.solution,
        "clean": clean,
        "profile": read_json(work / "profile/result.json"),
        "trace": summary,
        "catalog": catalog_info,
        "model": model["id"].split("/")[-1],
        "manifest": manifest(
            args, run, before, model, read_json(work / "profile/result.json")
        ),
        "cpus": {"server": cpu_list(cpus.server), "client": cpu_list(cpus.client)},
        "command": shlex.join(["python3", *sys.argv]),
        # The first clean run's beams per request, for the cross-engine comparison.
        "outputs": read_json(work / phases[0] / "outputs.json"),
    }
    (run.output / "report.md").write_text(report.render(context))
    return context


def write_comparison(args, runs, finished):
    names = [run.name for run in runs]
    reference = args.reference or ("sglang" if "sglang" in names else names[0])
    text = compare.render(finished, names, reference)
    (args.output / "comparison.md").write_text(text)


if __name__ == "__main__":
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda signum, frame: sys.exit(128 + signum))
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"benchmark failed: {exc}")
