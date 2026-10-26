#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark HTTP beam-search solutions on one scenario, each in Docker.

A scenario (configs/scenarios/*.yaml) fixes what is measured: the model, the
workload, the runs and an optional catalog. A solution (configs/solutions/*.yaml)
fixes how one engine is configured. The command line gives the paths, CPUs and
GPU of this invocation. Every clean run and the profiled run is a fresh
container and server process. Each solution's directory receives report.md,
sunburst.svg and profile.nsys-rep; with several solutions, OUTPUT also gets
comparison.md. Intermediate files are kept in .work only when a run fails.
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

import yaml
from bench import catalog, compare, provenance, report, sunburst, trace
from bench.common import CACHE_ENV, fail, read_json, write_json
from bench.config import (
    Resources,
    Scenario,
    Solution,
    load_model,
    load_scenario,
    load_solution,
    make_workload,
)

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "bench"
LABEL = "sid-gr-inference-benchmark"
PROFILED = ("profile", "load_profile")  # server processes run under Nsight Systems


@dataclass
class Run:
    """One solution to measure: its configs, source checkout, image and output."""

    name: str  # the solution file name without .yaml
    adapter: object  # bench.adapters.Adapter of the solution's engine
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


def cpu_numbers(text):
    """ "0-2,5" -> [0, 1, 2, 5]; the inverse of cpu_list."""
    cpus = []
    for part in text.split(","):
        first, dash, last = part.partition("-")
        if not first.isdigit() or (dash and not last.isdigit()):
            raise argparse.ArgumentTypeError(f"{text!r} is not a CPU list like 0-2,5")
        cpus += range(int(first), int(last or first) + 1)
    return cpus


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
        "source": checkout(run.source, *run.adapter.fingerprint),
        "tools": checkout(ROOT, "bench", untracked="normal")[0],
        "tools_hash": hashlib.sha256(b"".join(files)).hexdigest(),
        "model": sorted(model),
    }


def docker_command(args, run, cache, image, name, phase):
    adapter, resources = run.adapter, args.resources
    view = adapter.model_view and run.scenario.catalog is not None
    model = "/model-source" if view else "/model"
    mounts = [(BENCH, "/benchmark-tools/bench", True), (run.work, "/work", False)]
    mounts += [(cache, "/cache", False), (run.source, adapter.mount, True)]
    mounts += [(args.model, model, True)]
    if view:  # links to /model-source plus the generated trie head
        mounts.append((run.work / "model", "/model", True))
    env = {key: f"/cache/{key.lower()}" for key in CACHE_ENV}
    env.update(HOME="/cache/home", USER="benchmark", LOGNAME="benchmark")
    env.update(PYTHONPATH="/benchmark-tools", PYTHONDONTWRITEBYTECODE="1")
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cpus = cpu_list(resources.server_cpus + resources.client_cpus)
    argv = ["docker", "run", "--rm", "--init", "--name", name, "--label", LABEL]
    argv += ["--network", "none", "--gpus", f"device={resources.gpu}"]
    argv += ["--cpuset-cpus", cpus, "--user", f"{os.getuid()}:{os.getgid()}"]
    argv += ["--shm-size", "4g", "--cap-add", "SYS_PTRACE"]
    argv += ["--security-opt", "seccomp=unconfined"]
    for key, value in sorted(env.items()):
        argv += ["--env", f"{key}={value}"]
    for path, target, readonly in mounts:
        option = f"type=bind,src={path},dst={target}"
        argv += ["--mount", option + (",readonly" if readonly else "")]
    argv += [image, "python3", "-m", "bench.client", "--plan", "/work/plan.json"]
    return argv + ["--run", phase] + (["--profile"] if phase in PROFILED else [])


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


def parse_catalog(args, scenario, vocab):
    """(bytes, token paths, identity) of the --catalog file, checked against the
    workload."""
    workload = scenario.workload
    data = args.catalog.read_bytes()
    paths, info = catalog.parse(
        data,
        output_tokens=workload.output_tokens,
        beam_width=workload.beam_width,
        vocab_size=vocab,
    )
    return data, paths, {**info, "name": args.catalog.name}


def server_source(adapter):
    """Where the server reads the catalog in the container."""
    if adapter.model_view:
        return f"/model/{adapter.catalog_file}"
    return f"/work/{adapter.catalog_file or 'catalog.jsonl'}"


def make_plan(args, run, vocab, catalog_items):
    """plan.json: the container reads this, never the YAML files."""
    scenario, solution, resources = run.scenario, run.solution, args.resources
    plan = {"engine": run.adapter.name, "workload": asdict(scenario.workload)}
    plan.update(levels=scenario.workload.levels)
    plan.update(vocab_size=vocab, dtype=scenario.model.dtype)
    plan.update(warmups=scenario.runs.warmups, requests=scenario.runs.requests)
    plan.update(profile_requests=scenario.runs.profile_requests)
    plan.update(env=solution.env, server_settings=solution.server_settings)
    plan.update(server_cpus=resources.server_cpus, client_cpus=resources.client_cpus)
    plan.update(workload_path="/work/workload.json", catalog=None)
    if catalog_items is not None:
        jsonl = "/work/catalog.jsonl"
        source = server_source(run.adapter)
        plan["catalog"] = dict(path=jsonl, server_path=jsonl, server_source=source)
        plan["catalog"]["items"] = catalog_items
    return plan


def prepare(args, run, model_config, workload_rows):
    """Write plan.json, the workload and the engine's catalog files."""
    adapter, work = run.adapter, run.work
    vocab = model_config["vocab_size"]
    info = None
    if args.catalog:
        data, paths, info = parse_catalog(args, run.scenario, vocab)
        (work / "catalog.jsonl").write_bytes(data)
        if adapter.catalog_file:
            native = adapter.convert_catalog(paths, run.scenario.model, vocab)
            directory = work
            if adapter.model_view:  # links to /model-source plus the native file
                directory = work / "model"
                directory.mkdir()
                for item in args.model.iterdir():
                    if item.name != adapter.catalog_file:
                        (directory / item.name).symlink_to(f"/model-source/{item.name}")
            (directory / adapter.catalog_file).write_bytes(native)
    write_json(work / "workload.json", workload_rows)
    plan = make_plan(args, run, vocab, info["items"] if info else None)
    write_json(work / "plan.json", plan)
    return info


def added_settings(run, plan):
    """What the benchmark adds to the solution's env and server_settings for this
    scenario and machine; PORT stands for the port, which differs per run."""
    env = run.adapter.server(plan, "PORT", profile=False)[1]
    added = {
        "env": {k: v for k, v in sorted(env.items()) if run.solution.env.get(k) != v}
    }
    settings = run.adapter.server_args(plan, "PORT")
    if settings is not None:
        added["server_settings"] = settings
    return added


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
        required=True,
        help="scenario YAML: model pin and dtype, workload, runs, optional catalog pin",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="checkpoint directory matching the scenario's model pin",
    )
    parser.add_argument(
        "--server-cpus",
        type=cpu_numbers,
        required=True,
        help="CPUs for the server, such as 0-9 or 4-6,20-22",
    )
    parser.add_argument(
        "--client-cpus",
        type=cpu_numbers,
        required=True,
        help="CPUs for the client, apart from the server CPUs",
    )
    parser.add_argument(
        "--gpu", required=True, help="one GPU: a device index such as 0, or a UUID"
    )
    parser.add_argument(
        "--catalog", type=Path, help="the catalog JSONL that the scenario pins"
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what the benchmark adds to each solution's settings, and the "
        "docker commands",
    )
    args = parser.parse_args(argv)
    for name in ("scenario", "model", "catalog", "output", "cache_dir"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).expanduser().resolve())
    args.resources = Resources(args.server_cpus, args.client_cpus, args.gpu)
    return args


def check_catalog(args, scenario):
    """The scenario says whether beams follow a catalog; --catalog says where it is."""
    if scenario.catalog is None:
        if args.catalog:
            fail("--catalog needs a scenario that pins it in catalog.sha256")
        return
    if not args.catalog:
        fail("the scenario pins a catalog; give its file with --catalog")
    if provenance.sha256_file(args.catalog) != scenario.catalog.sha256:
        fail(f"{args.catalog} differs from the scenario's catalog.sha256")


def build_runs(args, scenario):
    runs = []
    for solution_path, source, image in args.solution:
        solution_path = Path(solution_path).expanduser().resolve()
        adapter, solution = load_solution(solution_path, scenario)
        source = Path(source).expanduser().resolve()
        if not (source / adapter.entry).is_file():
            fail(f"{source} is not {adapter.source_label}: no {adapter.entry}")
        if scenario.catalog and not adapter.catalog_supported:
            fail(
                f"{adapter.title} cannot constrain beams to a catalog; "
                "use a scenario without one"
            )
        name = solution_path.stem
        output = args.output if len(args.solution) == 1 else args.output / name
        runs.append(
            Run(name, adapter, scenario, solution, source, image, output, solution_path)
        )
    if len({run.name for run in runs}) != len(runs):
        fail("--solution configs need distinct file names")
    if args.reference and args.reference not in {run.name for run in runs}:
        fail(f"--reference {args.reference} is not one of the --solution configs")
    return runs


def main(argv=None):
    args = parse_args(argv)
    scenario = load_scenario(args.scenario)
    check_catalog(args, scenario)
    model_config = load_model(args.model, scenario.workload)
    runs = build_runs(args, scenario)
    if args.output.exists() and any(args.output.iterdir()):
        fail(f"output directory {args.output} is not empty")
    if args.dry_run:
        vocab, items = model_config["vocab_size"], None
        if args.catalog:
            items = parse_catalog(args, scenario, vocab)[2]["items"]
        for run in runs:
            added = added_settings(run, make_plan(args, run, vocab, items))
            print(f"# {run.name}: added to the solution's settings (PORT is per run)")
            print(yaml.safe_dump(added, sort_keys=False).rstrip())
            cache = args.cache_dir or args.output.parent / ".cache"
            for phase in run_phases(scenario, run):
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
                run.adapter.fingerprint,
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
        phases = run_phases(scenario, run)
        finished.append(measure(args, run, phases, model_config, rows, model))
    if len(runs) > 1:
        write_comparison(args, runs, finished)
    print(f"Wrote the reports to {args.output}")
    return 0


def run_phases(scenario, run):
    """The server processes of one solution, in order: the clean runs, the profile
    one request at a time and, with several levels, the profile under load at
    the largest, unless the engine overlaps steps (no step can be told apart)."""
    phases = [f"clean_{i}" for i in range(1, scenario.runs.clean + 1)]
    phases.append("profile")
    if len(scenario.workload.levels) > 1 and not run.adapter.overlaps_steps(
        run.solution
    ):
        phases.append("load_profile")
    return phases


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
    # Newer configs name it "dtype"; Hugging Face reads a config without either as float32.
    dtype = model_config.get("torch_dtype") or model_config.get("dtype") or "float32"
    lengths = prompt_lengths(run.scenario, rows)
    summary = write_outputs(
        args, run, phases, catalog_info, before, model, dtype, lengths
    )
    shutil.rmtree(run.work)
    return summary


def manifest(args, run, before, model, profile):
    """Everything needed to reproduce the run, printed at the end of the report."""
    return {
        **run.manifest,
        "model": model,
        "benchmark_code": before["tools"],
        "recipe": run.adapter.recipe,
        "configs": {
            "scenario": provenance.sha256_file(args.scenario),
            "solution": provenance.sha256_file(run.solution_path),
        },
        "nsys": profile.get("nsys"),
    }


def prompt_lengths(scenario, rows):
    """Prompt tokens of the measured requests, in the order every level sends them."""
    runs = scenario.runs
    measured = rows[
        runs.warmups : runs.warmups + max(runs.requests, runs.profile_requests)
    ]
    return [len(row["input_ids"]) for row in measured]


def write_outputs(
    args, run, phases, catalog_info, before, model, checkpoint_dtype, lengths
):
    work, resources = run.work, args.resources
    summary = trace.analyze(work / "profile/trace.sqlite")
    title = f"{run.adapter.title} · {model['id'].split('/')[-1]}"
    subtitle = f"Server time per request, mean of {summary['windows']} profiled"
    subtitle += " requests: phase › section › kernel family or non-kernel time"
    svg = sunburst.render(
        summary["work"], summary["gaps"], summary["windows"], title, subtitle
    )
    (run.output / "sunburst.svg").write_text(svg)
    shutil.move(work / "profile/trace.nsys-rep", run.output / "profile.nsys-rep")
    clean = [
        read_json(work / p / "result.json") for p in phases if p.startswith("clean")
    ]
    load = None
    if "load_profile" in phases:
        load = {
            "trace": trace.analyze(work / "load_profile/trace.sqlite", window="step"),
            "result": read_json(work / "load_profile/result.json"),
        }
        target = run.output / "load_profile.nsys-rep"
        shutil.move(work / "load_profile/trace.nsys-rep", target)
    context = {
        "scenario": run.scenario,
        "solution": run.solution,
        "clean": clean,
        "profile": read_json(work / "profile/result.json"),
        "trace": summary,
        "load": load,
        "catalog": catalog_info,
        "model": model["id"].split("/")[-1],
        "checkpoint_dtype": checkpoint_dtype,
        "manifest": manifest(
            args, run, before, model, read_json(work / "profile/result.json")
        ),
        "cpus": {
            "server": cpu_list(resources.server_cpus),
            "client": cpu_list(resources.client_cpus),
        },
        "added": added_settings(run, read_json(work / "plan.json")),
        "command": shlex.join(["python3", *sys.argv]),
        "lengths": lengths,
        # The first clean run's beams per level and request, for the comparison.
        "outputs": read_json(work / phases[0] / "outputs.json"),
    }
    (run.output / "report.md").write_text(report.render(context))
    return context


def write_comparison(args, runs, finished):
    names = [run.name for run in runs]
    reference = args.reference or ("sglang" if "sglang" in names else names[0])
    text = compare.render(finished, names, reference)
    (args.output / "comparison.md").write_text(text)
    svg = compare.chart_svg(finished, names)
    if svg:
        (args.output / compare.CHART).write_text(svg)


if __name__ == "__main__":
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda signum, frame: sys.exit(128 + signum))
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"benchmark failed: {exc}")
