# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark one server process inside the container; standard library only.

    python3 -m bench.client --plan /work/plan.json --run clean_1 [--profile]

Starts the server and, for each concurrency level in turn (level_order), sends
warmup and then measured requests from a closed loop of that many clients (a new
loopback connection per request); checks every returned beam and writes
/work/<run>/result.json, with the server's full settings and environment read
once after the last level's measured requests. With --profile the
server runs under Nsight Systems, requests go one at a time and the capture
covers only the measured ones; the report is exported to SQLite after the
server exits.
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

from bench import adapters, catalog
from bench.common import call, read_json, write_json

# Capture only between cudaProfilerStart/Stop; trace CUDA graph replays per node.
NSYS = (
    "nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none "
    "--capture-range=cudaProfilerApi --capture-range-end=stop "
    "--cuda-graph-trace=node --force-overwrite=true --output /work/profile/trace"
).split()
EXPORT = (
    "nsys export --type=sqlite --force-overwrite=true "
    "--output /work/profile/trace.sqlite /work/profile/trace.nsys-rep"
).split()


def launch(adapter, plan, port, profile):
    """Return (argv, env) for the server, pinned to the server CPUs. The image's
    engine settings (AMBIENT) are dropped: the server sees only the plan's."""
    argv, env = adapter.server(plan, port, profile)
    inherited = {
        k: v for k, v in os.environ.items() if not k.startswith(adapters.AMBIENT)
    }
    argv = ["taskset", "--cpu-list", ",".join(map(str, plan["server_cpus"])), *argv]
    return (NSYS + argv if profile else argv), {**inherited, **env}


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
    """POST /generate; latency covers JSON encoding through the parsed response."""
    start = time.perf_counter_ns()
    response = call(port, "POST", "/generate", body, timeout)
    return response, (time.perf_counter_ns() - start) / 1e6


def send_all(port, adapter, rows, workload, concurrency):
    """[(response, ms)] in row order; each client sends its next row on a reply."""
    with ThreadPoolExecutor(concurrency) as clients:
        return list(
            clients.map(lambda row: send(port, adapter.payload(row, workload)), rows)
        )


def check(result, workload, vocab_size):
    if len(result) != workload["beam_width"]:
        raise ValueError(f"{len(result)} beams, expected {workload['beam_width']}")
    for tokens, score in result:
        if len(tokens) != workload["output_tokens"] or any(
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


def validate(adapter, plan, row, response, paths):
    """Check one response; with a catalog, every beam must be a catalog item."""
    result = adapter.beams(response, row["request_id"])
    check(result, plan["workload"], plan["vocab_size"])
    if paths is not None and catalog.outside(result, paths):
        raise ValueError(f"{row['request_id']}: returned beams outside the catalog")
    return result


def check_server(adapter, plan, port, runtime):
    """The server must run the mounted source, hold its weights in the scenario's
    dtype and hold the managed catalog."""
    adapter.check_source(runtime)
    if runtime["dtype"] != plan["dtype"]:
        raise RuntimeError(
            f"the server holds {runtime['dtype']} weights, not the scenario's "
            f"{plan['dtype']}"
        )
    if plan["catalog"]:
        status = adapter.catalog_status(port, runtime)
        expected = plan["catalog"]
        catalog.check_status(
            status, source=expected["server_source"], items=expected["items"]
        )


def level_blocks(rows, plan):
    """{concurrency: its requests}: workload.json holds one block per level, in
    level order (bench.config.make_workload)."""
    size = plan["warmups"] + max(plan["requests"], plan["profile_requests"])
    return {k: rows[i * size : (i + 1) * size] for i, k in enumerate(plan["levels"])}


def level_order(levels, name):
    """Concurrency levels in the order run `name` measures them. clean_i rotates
    them by i - 1, so no level always runs first; the profiled run sends one
    request at a time, because the trace attributes GPU time per request."""
    if name == "profile":
        return [1]
    shift = (int(name.rsplit("_", 1)[1]) - 1) % len(levels)
    return levels[shift:] + levels[:shift]


def run(plan, name, profile, work=Path("/work")):
    directory = work / name
    directory.mkdir(parents=True, exist_ok=True)
    adapter, workload = adapters.get(plan["engine"]), plan["workload"]
    warmups = plan["warmups"]
    blocks = level_blocks(read_json(plan["workload_path"]), plan)
    count = plan["profile_requests"] if profile else plan["requests"]
    paths = None
    if plan["catalog"]:
        data = Path(plan["catalog"]["path"]).read_bytes()
        shape = {key: workload[key] for key in ("output_tokens", "beam_width")}
        paths, _ = catalog.parse(data, **shape, vocab_size=plan["vocab_size"])
    order = level_order(plan["levels"], name)
    result = {"run": name, "profile": profile, "order": order, "levels": []}
    result.update(machine())
    if profile:
        result["nsys"] = subprocess.run(
            ["nsys", "--version"], capture_output=True, text=True
        ).stdout.strip()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    argv, env = launch(adapter, plan, port, profile)
    pid = None
    with (directory / "server.log").open("w") as log:
        options = {"stdout": log, "stderr": subprocess.STDOUT, "env": env}
        process = subprocess.Popen(argv, start_new_session=True, **options)
    try:
        wait_ready(process, port, adapter.ready)
        os.sched_setaffinity(0, plan["client_cpus"])  # client threads inherit it
        counters, outputs = {}, {}
        for concurrency in order:
            # Each level sends its own requests; the profiled run, the first level's.
            rows = blocks[plan["levels"][0]] if profile else blocks[concurrency]
            measured = rows[warmups : warmups + count]
            warm = send_all(port, adapter, rows[:warmups], workload, concurrency)
            for row, (response, _) in zip(rows[:warmups], warm):
                validate(adapter, plan, row, response, paths)
            before = call(port, "GET", "/pilot/runtime")
            pid = before["pid"]
            check_server(adapter, plan, port, before)
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
                row["request_id"]: validate(adapter, plan, row, response, paths)
                for row, (response, _) in zip(measured, timed)
            }
            latency = [ms for _, ms in timed]
            result["levels"].append(
                {
                    "concurrency": concurrency,
                    "latency_ms": latency,
                    "duration_s": duration,
                }
            )
        # Once, after timing: the server's full settings and its environment.
        config = call(port, "GET", "/pilot/config")
        adapter.check_counters(plan, counters, count * len(order), after)
        result.update(runtime=after, counters=counters, catalog_beams=None)
        result.update(config=config, server_command=argv)
        if paths is not None:
            beams = count * len(order) * workload["beam_width"]
            result["catalog_beams"] = beams
    finally:
        stop(process, pid, profile)
    if profile:
        subprocess.run(EXPORT, check=True)
    write_json(directory / "result.json", result)
    if not profile:  # the comparison reads these; compact because they are large
        write_json(directory / "outputs.json", outputs, indent=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    run(read_json(args.plan), args.run, args.profile)


if __name__ == "__main__":
    main()
