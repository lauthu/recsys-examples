# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark one server process inside the container; standard library only.

    python3 -m bench.client --config /work/run_config.json --run round_1 [--profile]

Starts the server and, for each concurrency in turn (level_order), sends warmup
and then measured requests from a closed loop of that many clients (a new
loopback connection per request); checks every returned beam, splits each
request's latency into phases from the server's handler and step times
(timeline) and writes /work/<run>/result.json, with the server's full settings
and environment read once after the last concurrency's measured requests. With
--profile the server runs under Nsight Systems, the capture covers only the
measured requests and the report is exported to SQLite after the server exits:
the run "profile" sends requests one at a time and the server marks each
request; the run "load_profile" keeps runs.profile_concurrency's requests in
flight (by default the largest concurrency's) and the server marks each
engine step.
"""

import argparse
import http.client
import math
import os
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench import catalog, engines
from bench.common import PHASES, call, read_json, write_json
from bench.config import RunConfig, block_layout, load_concurrency
from bench.engines.adapter import AMBIENT

# What the server marks in each profiled run: BENCH_PROFILE for its launcher.
PROFILES = {"profile": "requests", "load_profile": "steps"}


def nsys(name):
    """Capture only between cudaProfilerStart/Stop; trace CUDA graph replays per node."""
    return (
        "nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none "
        "--capture-range=cudaProfilerApi --capture-range-end=stop "
        f"--cuda-graph-trace=node --force-overwrite=true --output /work/{name}/trace"
    ).split()


def export(name):
    return (
        "nsys export --type=sqlite --force-overwrite=true "
        f"--output /work/{name}/trace.sqlite /work/{name}/trace.nsys-rep"
    ).split()


def launch(adapter, config, port, profile, name="profile"):
    """Return (argv, env) for the server, pinned to the server CPUs. The image's
    engine settings (AMBIENT) are dropped: the server sees only the RunConfig's.
    profile is what the server marks ("requests" or "steps"), or empty."""
    argv, env = adapter.server(config, port, profile)
    inherited = {k: v for k, v in os.environ.items() if not k.startswith(AMBIENT)}
    cpus = ",".join(map(str, config.resources.server_cpus))
    argv = ["taskset", "--cpu-list", cpus, *argv]
    return (nsys(name) + argv if profile else argv), {**inherited, **env}


def wait_ready(process, port, path, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with {process.returncode}; see server.log"
            )
        try:
            if call(port, "GET", path, timeout=5).get("ready") is True:
                return
        except (OSError, RuntimeError, ValueError, http.client.HTTPException):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server not ready after {timeout} s; see server.log")


def send(port, body, timeout=120):
    """POST /generate; (response, sent, replied) in perf_counter_ns, the clock the
    server records with: from JSON encoding through the parsed response."""
    start = time.perf_counter_ns()
    response = call(port, "POST", "/generate", body, timeout)
    return response, start, time.perf_counter_ns()


def send_all(port, adapter, rows, workload, concurrency):
    """[(response, sent, replied)] in row order; each client sends its next row on
    a reply."""
    with ThreadPoolExecutor(concurrency) as clients:
        return list(
            clients.map(lambda row: send(port, adapter.payload(row, workload)), rows)
        )


def timeline(measured, timed, runtime, output_tokens):
    """Where each measured request's time went, and the level's engine steps.

    The client's send and reply times and the server's handler and step times
    share one clock, so a request's latency splits exactly into: outside the
    server's handler (client and HTTP), queue (handler start to the start of its
    first prefill step), prefill, decode (prefill end to its last step's end) and
    return (last step's end to the handler's return). A step that is still doing
    its bookkeeping after the handler returned ends the request at the return.
    Returns ({phase: [ms per request]}, [[kind, requests, rows, tokens, schedule
    ms, step ms]]).
    """
    handled, steps = runtime["requests"], runtime["steps"]
    by_request = {}
    for step in steps:
        for rid in step["requests"]:
            by_request.setdefault(rid, []).append(step)
    phases = {name: [] for name in PHASES}
    for row, (_, sent, replied) in zip(measured, timed):
        rid = row["request_id"]
        mine = by_request.get(rid, [])
        prefill = [step for step in mine if step["kind"] == "prefill"]
        decode = len(mine) - len(prefill)
        if rid not in handled or not prefill or decode != output_tokens - 1:
            raise RuntimeError(
                f"{rid}: the server recorded {len(prefill)} prefill and {decode} "
                f"decode steps (expected at least 1 and {output_tokens - 1}) and "
                f"{'a' if rid in handled else 'no'} handler time"
            )
        started, returned = handled[rid]
        first = min(step["scheduled_ns"] for step in prefill)
        # A step can send its results before its own bookkeeping ends; the
        # request's phases end no later than its handler's return.
        prefilled = min(max(step["finished_ns"] for step in prefill), returned)
        finished = min(max(step["finished_ns"] for step in mine), returned)
        spans = {
            "outside": (replied - sent) - (returned - started),
            "queue": first - started,
            "prefill": prefilled - first,
            "decode": finished - prefilled,
            "return": returned - finished,
        }
        if min(spans.values()) < 0:
            raise RuntimeError(f"{rid}: the client and server times disagree: {spans}")
        for name, ns in spans.items():
            phases[name].append(ns / 1e6)
    level_steps = [
        [
            step["kind"],
            len(step["requests"]),
            step["rows"],
            step["tokens"],
            (step["launched_ns"] - step["scheduled_ns"]) / 1e6,
            (step["finished_ns"] - step["scheduled_ns"]) / 1e6,
        ]
        for step in steps
    ]
    return phases, level_steps


def check(result, workload, vocab_size):
    if len(result) != workload.beam_width:
        raise ValueError(f"{len(result)} beams, expected {workload.beam_width}")
    for tokens, score in result:
        if len(tokens) != workload.output_tokens or any(
            type(t) is not int or not 0 <= t < vocab_size for t in tokens
        ):
            raise ValueError(f"invalid beam tokens {tokens}")
        if not math.isfinite(score):
            raise ValueError("non-finite beam score")
    if len({tuple(tokens) for tokens, _ in result}) != len(result):
        raise ValueError("duplicate beams")
    if any(a[1] < b[1] for a, b in zip(result, result[1:])):
        raise ValueError("beams are not sorted by descending score")


def stop(process, pid, profile):
    """Stop the server; under Nsight, stop only the app so nsys can write its report."""
    if process.poll() is None and profile and pid:
        try:
            os.kill(pid, signal.SIGTERM)
            process.wait(timeout=300)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def machine():
    query = "nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader"
    gpu = subprocess.run(query.split(), capture_output=True, text=True, check=True)
    name, uuid, driver = (
        part.strip() for part in gpu.stdout.splitlines()[0].split(",")
    )
    with open("/proc/cpuinfo") as stream:
        cpus = [line.split(":")[1].strip() for line in stream if "model name" in line]
    cpu = cpus[0] if cpus else "unknown"
    return {"gpu": name, "gpu_uuid": uuid, "driver": driver, "cpu": cpu}


def validate(adapter, config, row, response, paths):
    """Check one response; with a catalog, every beam must be a catalog item."""
    result = adapter.beams(response, row["request_id"])
    check(result, config.workload, config.vocab_size)
    if paths is not None and catalog.outside(result, paths):
        raise ValueError(f"{row['request_id']}: returned beams outside the catalog")
    return result


def check_server(adapter, config, port, runtime):
    """The server must run the mounted source, hold its weights in the scenario's
    dtype and hold the managed catalog."""
    adapter.check_source(runtime)
    dtype = config.scenario.model.dtype
    if runtime["dtype"] != dtype:
        raise RuntimeError(
            f"the server holds {runtime['dtype']} weights, not the scenario's {dtype}"
        )
    if config.catalog_items is not None:
        status = adapter.catalog_status(port, runtime)
        catalog.check_status(
            status, source=adapter.catalog_source, items=config.catalog_items
        )


def level_blocks(rows, config):
    """{concurrency: its requests}: workload.json holds one block per
    concurrency, in order (bench.config.make_workload)."""
    size, levels = block_layout(config.scenario)[1], config.workload.levels
    return {k: rows[i * size : (i + 1) * size] for i, k in enumerate(levels)}


def level_order(levels, name, load=None):
    """The concurrency values in the order run `name` measures them. round_i
    rotates them by i - 1, so none always runs first. The run "profile" sends one
    request at a time, because its trace attributes GPU time per request;
    "load_profile" measures concurrency `load` (by default the largest),
    attributing GPU time per step."""
    if name == "profile":
        return [1]
    if name == "load_profile":
        return [load or max(levels)]
    shift = (int(name.rsplit("_", 1)[1]) - 1) % len(levels)
    return levels[shift:] + levels[:shift]


def run(config, name, profile, work=Path("/work")):
    directory = work / name
    directory.mkdir(parents=True, exist_ok=True)
    adapter, workload = engines.get(config.solution.engine), config.workload
    runs = config.scenario.runs
    start_row = block_layout(config.scenario)[0]  # each block's first measured row
    blocks = level_blocks(read_json(work / "workload.json"), config)
    mode = PROFILES[name] if profile else ""
    count = runs.profile_requests if profile else runs.requests
    load = load_concurrency(config.scenario)
    if mode == "steps":  # each client sends profile_requests, as far as the block goes
        count = min(load * count, max(runs.requests, runs.profile_requests))
    paths = None
    if config.catalog_items is not None:
        data = (work / "catalog.jsonl").read_bytes()
        shape = dict(
            output_tokens=workload.output_tokens, beam_width=workload.beam_width
        )
        paths, _ = catalog.parse(data, **shape, vocab_size=config.vocab_size)
    order = level_order(workload.levels, name, load)
    result = {"run": name, "profile": profile, "order": order, "levels": []}
    result.update(machine())
    if profile:
        result["nsys"] = subprocess.run(
            ["nsys", "--version"], capture_output=True, text=True
        ).stdout.strip()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    argv, env = launch(adapter, config, port, mode, name)
    pid = None
    with (directory / "server.log").open("w") as log:
        options = {"stdout": log, "stderr": subprocess.STDOUT, "env": env}
        process = subprocess.Popen(argv, start_new_session=True, **options)
    try:
        wait_ready(process, port, adapter.ready)
        os.sched_setaffinity(0, config.resources.client_cpus)  # threads inherit it
        counters, outputs = {}, {}
        for concurrency in order:
            # Each concurrency sends its own requests; the run "profile", the first's.
            first_block = blocks[workload.levels[0]]
            rows = first_block if mode == "requests" else blocks[concurrency]
            warmups = runs.warmups_at(concurrency)
            measured = rows[start_row : start_row + count]
            warm = send_all(port, adapter, rows[:warmups], workload, concurrency)
            for row, (response, *_) in zip(rows[:warmups], warm):
                validate(adapter, config, row, response, paths)
            # Also takes the warmups' request and step times, which are dropped.
            before = call(port, "GET", "/pilot/runtime")
            pid = before["pid"]
            check_server(adapter, config, port, before)
            if profile:
                adapter.profile(port, start=True)
            first = time.perf_counter()
            timed = send_all(port, adapter, measured, workload, concurrency)
            duration = time.perf_counter() - first
            if profile:
                adapter.profile(port, start=False)
            after = call(port, "GET", "/pilot/runtime")
            # Counters cover only measured requests: warmups fall between snapshots.
            for key, value in after["counters"].items():
                if value is not None:
                    delta = value - before["counters"].get(key, 0)
                    counters[key] = counters.get(key, 0) + delta
            # Output checks run after timing so they never delay the next request.
            outputs[str(concurrency)] = {
                row["request_id"]: validate(adapter, config, row, response, paths)
                for row, (response, *_) in zip(measured, timed)
            }
            latency = [(replied - sent) / 1e6 for _, sent, replied in timed]
            phases, steps = timeline(measured, timed, after, workload.output_tokens)
            del after["requests"], after["steps"]  # kept per level, not in runtime
            result["levels"].append(
                {
                    "concurrency": concurrency,
                    "warmups": warmups,
                    "latency_ms": latency,
                    "duration_s": duration,
                    "phases_ms": phases,
                    "steps": steps,
                }
            )
        # Once, after timing: the server's full settings and its environment.
        server_config = call(port, "GET", "/pilot/config")
        adapter.check_counters(config, counters, count * len(order), after)
        result.update(runtime=after, counters=counters, catalog_beams=None)
        result.update(config=server_config, server_command=argv)
        if paths is not None:
            beams = count * len(order) * workload.beam_width
            result["catalog_beams"] = beams
    finally:
        stop(process, pid, profile)
    if profile:
        subprocess.run(export(name), check=True)
    write_json(directory / "result.json", result)
    if not profile:  # the summary reads these; compact because they are large
        write_json(directory / "outputs.json", outputs, indent=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    run(RunConfig.read(args.config), args.run, args.profile)


if __name__ == "__main__":
    main()
