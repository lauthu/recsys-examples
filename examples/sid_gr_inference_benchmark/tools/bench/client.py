# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark one server process inside the container; standard library only.

    python3 -m bench.client --plan /work/plan.json --run clean_1 [--profile]

Starts the server, sends warmup and then measured requests one at a time
(closed loop, concurrency 1, a new loopback connection per request), checks
every returned beam and writes /work/<run>/result.json. With --profile the
server runs under Nsight Systems and the capture covers only the measured
requests; the report is exported to SQLite after the server exits.
"""

import argparse
import http.client
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from bench import catalog
from bench.common import is_sglang, read_json, write_json

SID_GR_SCRIPT = "/source/examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
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


def call(port, method, path, body=None, timeout=60, text=False):
    """HTTP request to the server; returns parsed JSON, or the raw body if text."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        connection.request(method, path, data, headers)
        response = connection.getresponse()
        payload = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"{method} {path}: HTTP {response.status} {payload[:300]!r}")
    return payload if text else json.loads(payload)


def profile_control(engine, port, start):
    """Start or stop the cudaProfilerApi capture range in the server process."""
    if is_sglang(engine):  # native endpoints answer with plain text
        body = {"activities": ["CUDA_PROFILER"]} if start else None
        route = "/start_profile" if start else "/stop_profile"
        call(port, "POST", route, body, text=True)
    else:
        call(port, "POST", f"/pilot/profile/{'start' if start else 'stop'}")


def server_command(plan, port, profile):
    """Return (argv, env, ready path) for the configured engine."""
    engine, workload = plan["engine"], plan["workload"]
    # Benchmark settings come only from the plan, never from the image's environment.
    ambient = ("GR_", "SGLANG_", "MODEL_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(ambient)}
    env.update(plan["env"], BENCH_PROFILE="1" if profile else "0")
    if is_sglang(engine):
        env.update(PYTHONPATH="/sglang/python:/benchmark-tools", BENCH_ENGINE=engine)
        env["SGLANG_PROFILE_V2"] = "0"  # the cudaProfilerApi control path
        if engine == "sglang-sidtrie":
            # Keep exact full-vocabulary normalization; the approximation changes scores.
            env["SGLANG_BEAM_TRIE_TOPK_LOGPROB"] = "false"
        argv = [
            sys.executable,
            "-m",
            "bench.server_sglang",
            "--plan",
            "/work/plan.json",
        ]
        argv += ["--port", str(port)]
        ready = "/pilot/ready"
    else:
        env.update(
            {
                "PYTHON_BIN": "/benchmark-tools/bench/server_sid_gr.py",
                "MODEL_DIR": "/model",
                "GR_DECODE_ATTEN_ROOT": "/source/corelib/gr_decode_atten",
                "GR_HTTP_HOST": "127.0.0.1",
                "GR_HTTP_PORT": str(port),
                "GR_CONTEXT_LEN": str(workload["context_length"]),
                "GR_BEAM_WIDTH": str(workload["beam_width"]),
                "GR_DECODE_STEPS": str(workload["output_tokens"] - 1),
                "GR_MAX_BATCH_SIZE": "1",
                "GR_BEAM_KV_POOL_CAPACITY": "1",
                "GR_CONTEXT_KV_POOL_CAPACITY": "1",
                "GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS": "1",
                "GR_DEVICE": "cuda",
                "GR_DECODE_BACKEND": "real",
                "GR_SUPPRESS_SPECIAL_TOKENS_ON_IGNORE_EOS": "0",
            }
        )
        if plan["catalog"]:
            env["GR_CATALOG_JSONL"] = plan["catalog"]["server_path"]
        argv = ["bash", SID_GR_SCRIPT, "--no-executor-sync-timing"]
        ready = "/ready"
    argv = ["taskset", "--cpu-list", ",".join(map(str, plan["server_cpus"])), *argv]
    return (NSYS + argv if profile else argv), env, ready


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


def payload(engine, row, workload):
    params = {
        "n": workload["beam_width"],
        "max_new_tokens": workload["output_tokens"],
        "ignore_eos": True,
    }
    if engine == "sglang-beam-search-new":
        params["use_beam_search"] = True
    elif engine == "sglang-sidtrie":
        params["beam_width"] = workload["beam_width"]
    return {
        "input_ids": row["input_ids"],
        "rid": row["request_id"],
        "stream": False,
        "sampling_params": params,
    }


def send(port, body, timeout=120):
    """POST /generate; latency covers JSON encoding through the parsed response."""
    start = time.perf_counter_ns()
    response = call(port, "POST", "/generate", body, timeout)
    return response, (time.perf_counter_ns() - start) / 1e6


def beams(engine, response, rid):
    """Return [(token_ids, cumulative logprob)] in the server's order."""
    meta = response["meta_info"]
    if meta.get("request_id" if engine == "sid-gr" else "id") != rid:
        raise ValueError(f"response does not belong to {rid}")
    result = []
    for beam in meta["beam_results"]:
        tokens, score = beam["output_ids"], float(beam["meta_info"]["sequence_score"])
        # SGLang reports mean logprob per generated token; SID-GR reports the sum.
        result.append((tokens, score * len(tokens) if is_sglang(engine) else score))
    return result


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


def catalog_status(engine, port, runtime):
    if is_sglang(engine):
        return runtime.get("catalog")
    status = call(port, "GET", "/catalog/status")["catalog"] or {}
    return {
        "active": bool(status),
        "source": status.get("source"),
        "items": status.get("item_count"),
    }


def check_sidtrie(plan, counters, requests, length_penalty):
    """Every measured request must take the native SIDTrie path that was requested."""
    decode = counters["forward.decode.graph"] + counters["forward.decode.eager"]
    shared = plan["env"].get("SGLANG_BEAM_SHARED_CONTEXT_ATTENTION", "false") == "true"
    problems = []
    if length_penalty != 1.0:
        problems.append(
            f"length penalty {length_penalty} breaks mean-to-sum score conversion"
        )
    if decode <= 0 or counters["sidtrie.compact_decode"] != decode:
        problems.append("not every decode forward used the compact trie output head")
    if counters["sidtrie.exact_normalizer"] != decode:
        problems.append("not every decode forward kept the full-vocabulary normalizer")
    if counters["sidtrie.device_history"] != requests:
        problems.append("not every request finalized beams with device history")
    if shared and (
        counters["forward.decode.graph"]
        or counters["sidtrie.shared_attention"] < counters["forward.decode.eager"]
    ):
        problems.append("shared-context attention did not run in every decode forward")
    if not shared and counters["sidtrie.shared_attention"]:
        problems.append("shared-context attention ran although it is disabled")
    if problems:
        raise RuntimeError("SIDTrie check failed: " + "; ".join(problems))


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


def validate(plan, row, response, paths):
    """Check one response; with a catalog, every beam must be a catalog item."""
    result = beams(plan["engine"], response, row["request_id"])
    check(result, plan["workload"], plan["vocab_size"])
    if paths is not None and catalog.outside(result, paths):
        raise ValueError(f"{row['request_id']}: returned beams outside the catalog")


def check_server(plan, port, runtime):
    """The server must run the mounted source and hold the managed catalog."""
    if is_sglang(plan["engine"]) and not runtime["sglang"].startswith("/sglang/"):
        raise RuntimeError(f"imported {runtime['sglang']}, not the mounted SGLang")
    if plan["catalog"]:
        status = catalog_status(plan["engine"], port, runtime)
        expected = plan["catalog"]
        catalog.check_status(
            status, source=expected["server_source"], items=expected["items"]
        )


def run(plan, name, profile, work=Path("/work")):
    directory = work / name
    directory.mkdir(parents=True, exist_ok=True)
    engine, workload, warmups = plan["engine"], plan["workload"], plan["warmups"]
    rows = read_json(plan["workload_path"])
    count = plan["profile_requests"] if profile else plan["requests"]
    measured = rows[warmups : warmups + count]
    paths = None
    if plan["catalog"]:
        data = Path(plan["catalog"]["path"]).read_bytes()
        shape = {key: workload[key] for key in ("output_tokens", "beam_width")}
        paths, _ = catalog.parse(data, **shape, vocab_size=plan["vocab_size"])
    result = {"run": name, "profile": profile, **machine()}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    argv, env, ready_path = server_command(plan, port, profile)
    pid = None
    with (directory / "server.log").open("w") as log:
        options = {"stdout": log, "stderr": subprocess.STDOUT, "env": env}
        process = subprocess.Popen(argv, start_new_session=True, **options)
    try:
        wait_ready(process, port, ready_path)
        os.sched_setaffinity(0, plan["client_cpus"])
        for row in rows[:warmups]:
            validate(plan, row, send(port, payload(engine, row, workload))[0], paths)
        before = call(port, "GET", "/pilot/runtime")
        pid = before["pid"]
        check_server(plan, port, before)
        if profile:
            profile_control(engine, port, start=True)
        first = time.perf_counter()
        timed = [send(port, payload(engine, row, workload)) for row in measured]
        duration = time.perf_counter() - first
        if profile:
            profile_control(engine, port, start=False)
        after = call(port, "GET", "/pilot/runtime")
        # Output checks run after timing so they never delay the next request.
        for row, (response, _) in zip(measured, timed):
            validate(plan, row, response, paths)
        counters = {
            key: value - before["counters"].get(key, 0)
            for key, value in after["counters"].items()
            if value is not None
        }
        if engine == "sglang-sidtrie":
            check_sidtrie(plan, counters, len(measured), after["length_penalty"])
        result.update(latency_ms=[ms for _, ms in timed], duration_s=duration)
        result.update(runtime=after, counters=counters, catalog_beams=None)
        if paths is not None:
            result["catalog_beams"] = len(measured) * workload["beam_width"]
    finally:
        stop(process, pid, profile)
    if profile:
        subprocess.run(EXPORT, check=True)
    write_json(directory / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    run(read_json(args.plan), args.run, args.profile)


if __name__ == "__main__":
    main()
