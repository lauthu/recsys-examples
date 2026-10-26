# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import socket
import subprocess
import sys

import pytest
from bench import adapters, client
from bench.common import read_json, write_json

PATHS = [[1000 + a, 2000, 3000] for a in range(4)]
WORKLOAD = {
    "context_length": 4,
    "beam_width": 4,
    "output_tokens": 3,
    "concurrency": 1,
    "seed": 0,
}
# Answers like a SID-GR server with the benchmark wrapper; SGLang's native
# profile endpoints answer with plain text.
FAKE_SERVER = """
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port, mode, paths = int(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
served, in_flight, lock = [0], [0, 0], threading.Lock()  # in flight: [now, max]
seen = set()  # a request ID sent twice fails, as a server could answer it from memory
records = {"requests": {}, "steps": []}  # handler and step times until /pilot/runtime


def step(kind, rid, rows):
    now = time.perf_counter_ns()
    if mode == "slow" and kind == "decode":
        time.sleep(0.01)
    return {"kind": kind, "requests": [rid], "rows": rows, "tokens": 4,
            "scheduled_ns": now, "launched_ns": now, "finished_ns": time.perf_counter_ns()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value):
        body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        counters = {"prefill.graph_replays": served[0], "decode.graph_replays": 2 * served[0],
                    "decode.eager_fallbacks": None,
                    "prefill.graph_captures": served[0] if mode == "captures" else 0}
        catalog = {"source": "/work/catalog.jsonl", "item_count": len(paths)}
        runtime = {"pid": os.getpid(), "counters": counters, "max_in_flight": in_flight[1],
                   "dtype": "bfloat16"}
        if self.path == "/pilot/runtime":
            with lock:
                runtime.update(requests=dict(records["requests"]), steps=list(records["steps"]))
                records["requests"].clear()
                records["steps"].clear()
        config = {"source": "build_info", "settings": {"beam_width": 4},
                  "environment": {"GR_X": "1"}}
        self.reply({"/ready": {"ready": True}, "/catalog/status": {"catalog": catalog},
                    "/pilot/runtime": runtime, "/pilot/config": config}[self.path])

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"null")
        if self.path in ("/start_profile", "/stop_profile"):
            return self.reply("Start profiling.\\n")
        if self.path.startswith("/pilot/profile/"):
            return self.reply({"capturing": self.path.endswith("start")})
        started = time.perf_counter_ns()
        if request["rid"] in seen:
            self.send_response(409)
            self.end_headers()
            return
        seen.add(request["rid"])
        with lock:
            served[0] += 1
            in_flight[0] += 1
            in_flight[1] = max(in_flight)
        if mode == "slow":
            time.sleep(0.2)
        decode = request["sampling_params"]["max_new_tokens"] - 1
        steps = [step("prefill", request["rid"], 1)]
        steps += [step("decode", request["rid"], 4) for _ in range(decode)]
        with lock:
            in_flight[0] -= 1
            if mode != "no_steps":
                records["steps"].extend(steps)
            records["requests"][request["rid"]] = [started, time.perf_counter_ns()]
        beams = [{"output_ids": p, "meta_info": {"sequence_score": -float(i)}}
                 for i, p in enumerate(paths[: request["sampling_params"]["n"]])]
        if mode == "duplicate":
            beams[1] = beams[0]
        self.reply({"meta_info": {"request_id": request["rid"], "beam_results": beams}})


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def fake_server(tmp_path, port, mode="valid"):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    return [sys.executable, str(script), str(port), mode, json.dumps(PATHS)]


def run_fake(
    tmp_path,
    monkeypatch,
    mode="valid",
    concurrency=1,
    dtype="bfloat16",
    name="clean_1",
    profile=False,
):
    rows = [{"item_id": i, "token_ids": p} for i, p in enumerate(PATHS)]
    (tmp_path / "catalog.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    blocks = len(concurrency) if isinstance(concurrency, list) else 1
    write_json(  # one block of 2 warmups + 3 measured requests per level
        tmp_path / "workload.json",
        [
            {"request_id": f"request-{i}", "input_ids": [5, 6, 7, 8]}
            for i in range(5 * blocks)
        ],
    )
    catalog = {
        "path": str(tmp_path / "catalog.jsonl"),
        "server_path": "/work/catalog.jsonl",
        "server_source": "/work/catalog.jsonl",
        "items": len(PATHS),
    }
    plan = {
        "engine": "sid-gr",
        "workload": {**WORKLOAD, "concurrency": concurrency},
        "levels": sorted(
            concurrency if isinstance(concurrency, list) else [concurrency]
        ),
        "warmups": 2,
        "requests": 3,
        "profile_requests": 1,
        "vocab_size": 5000,
        "dtype": dtype,
        "env": {},
        "server_settings": None,
        "server_cpus": sorted(os.sched_getaffinity(0))[:1],  # a CPU this job has
        "client_cpus": sorted(os.sched_getaffinity(0)),
        "workload_path": str(tmp_path / "workload.json"),
        "catalog": catalog,
    }
    monkeypatch.setattr(
        adapters.get("sid-gr"),
        "server",
        lambda plan, port, profile: (
            fake_server(tmp_path, port, mode),
            dict(os.environ),
        ),
    )
    monkeypatch.setattr(
        client, "machine", lambda: {"gpu": "G", "driver": "D", "cpu": "C"}
    )
    client.run(plan, name, profile, work=tmp_path)
    return read_json(tmp_path / name / "result.json")


def test_clean_run_measures_checks_and_counts(tmp_path, monkeypatch):
    result = run_fake(tmp_path, monkeypatch)
    (level,) = result["levels"]
    assert level["concurrency"] == 1 and result["order"] == [1]
    assert len(level["latency_ms"]) == 3 and all(ms > 0 for ms in level["latency_ms"])
    # Counters cover only the measured requests; a disabled runner (None) has no delta.
    assert result["counters"] == {
        "prefill.graph_replays": 3,
        "decode.graph_replays": 6,
        "prefill.graph_captures": 0,
    }
    assert result["catalog_beams"] == 12 and result["gpu"] == "G"
    # The server's own settings and environment, and how it was started.
    assert result["config"]["settings"] == {"beam_width": 4}
    assert result["config"]["environment"] == {"GR_X": "1"}
    assert result["server_command"][:2] == ["taskset", "--cpu-list"]
    # Every measured request's validated beams are kept for the comparison.
    outputs = read_json(tmp_path / "clean_1/outputs.json")["1"]
    assert sorted(outputs) == ["request-2", "request-3", "request-4"]
    assert outputs["request-2"][1] == [PATHS[1], -1.0]
    assert "outputs" not in result


def test_clients_keep_concurrency_requests_in_flight(tmp_path, monkeypatch):
    result = run_fake(tmp_path, monkeypatch, mode="slow", concurrency=3)
    assert result["order"] == [3] and result["runtime"]["max_in_flight"] == 3
    assert len(result["levels"][0]["latency_ms"]) == 3
    # Responses stay with their requests whatever order they finish in.
    outputs = read_json(tmp_path / "clean_1/outputs.json")["3"]
    assert sorted(outputs) == ["request-2", "request-3", "request-4"]


def test_each_level_is_measured_in_turn_without_warmups(tmp_path, monkeypatch):
    levels = [1, 2, 3]
    result = run_fake(tmp_path, monkeypatch, concurrency=levels, name="clean_2")
    # clean_2 starts at the second level, so no level always runs first.
    assert result["order"] == [2, 3, 1]
    assert [level["concurrency"] for level in result["levels"]] == [2, 3, 1]
    assert all(len(level["latency_ms"]) == 3 for level in result["levels"])
    # Each level's warmups fall between snapshots: 3 levels x 3 measured requests.
    assert result["counters"]["prefill.graph_replays"] == 9
    assert result["catalog_beams"] == 3 * 3 * 4
    # Each level sends its own block of requests; the fake server rejects repeats.
    outputs = read_json(tmp_path / "clean_2/outputs.json")
    assert sorted(outputs["1"]) == ["request-2", "request-3", "request-4"]
    assert sorted(outputs["2"]) == ["request-7", "request-8", "request-9"]
    assert sorted(outputs["3"]) == ["request-12", "request-13", "request-14"]


def test_each_latency_splits_into_phases_from_the_server_times(tmp_path, monkeypatch):
    result = run_fake(tmp_path, monkeypatch, mode="slow")
    (level,) = result["levels"]
    phases = level["phases_ms"]
    assert list(phases) == ["outside", "queue", "prefill", "decode", "return"]
    for i, latency in enumerate(level["latency_ms"]):
        assert sum(phases[name][i] for name in phases) == pytest.approx(latency)
        assert phases["decode"][i] >= 20  # two decode steps of 10 ms each
    # The measured requests' steps only: 3 requests x (1 prefill + 2 decode).
    kinds = [step[0] for step in level["steps"]]
    assert kinds == ["prefill", "decode", "decode"] * 3
    assert level["steps"][1][:4] == ["decode", 1, 4, 4]  # kind, requests, rows, tokens
    assert all(ms >= 10 for *_, ms in level["steps"][1:3])
    # The records stay out of the runtime snapshot.
    assert "steps" not in result["runtime"] and "requests" not in result["runtime"]


def test_missing_step_times_fail_the_run(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="recorded 0 prefill and 0 decode steps"):
        run_fake(tmp_path, monkeypatch, mode="no_steps")


def test_client_and_server_times_must_agree():
    step = {"requests": ["r"], "rows": 1, "tokens": 4}
    runtime = {
        "requests": {"r": [100, 900]},
        "steps": [
            {
                **step,
                "kind": "prefill",
                "scheduled_ns": 200,
                "launched_ns": 250,
                "finished_ns": 400,
            },
            {
                **step,
                "kind": "decode",
                "scheduled_ns": 450,
                "launched_ns": 460,
                "finished_ns": 800,
            },
        ],
    }
    measured = [{"request_id": "r"}]
    phases, steps = client.timeline(measured, [(None, 50, 1000)], runtime, 2)
    expected = {
        "outside": 1.5e-4,
        "queue": 1e-4,
        "prefill": 2e-4,
        "decode": 4e-4,
        "return": 1e-4,
    }
    assert phases == {name: [pytest.approx(ms)] for name, ms in expected.items()}
    assert steps[0] == ["prefill", 1, 1, 4, 5e-5, 2e-4]
    # A reply before the handler returned cannot happen on one clock.
    with pytest.raises(RuntimeError, match="times disagree"):
        client.timeline(measured, [(None, 50, 800)], runtime, 2)


def test_level_order_rotates_between_clean_runs():
    levels = [1, 2, 4]
    assert client.level_order(levels, "clean_1") == [1, 2, 4]
    assert client.level_order(levels, "clean_2") == [2, 4, 1]
    assert client.level_order(levels, "clean_3") == [4, 1, 2]
    assert client.level_order(levels, "clean_4") == [1, 2, 4]
    assert client.level_order(levels, "profile") == [1]


def test_sid_gr_fails_when_it_captures_prefill_graphs_while_measured(
    tmp_path, monkeypatch
):
    with pytest.raises(RuntimeError, match="captured 3 prefill CUDA graphs"):
        run_fake(tmp_path, monkeypatch, mode="captures")


def test_invalid_beams_fail_the_run(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="duplicate beams"):
        run_fake(tmp_path, monkeypatch, mode="duplicate")


def test_weights_in_another_dtype_than_the_scenario_fail_the_run(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="holds bfloat16 weights"):
        run_fake(tmp_path, monkeypatch, dtype="float16")


def test_the_server_sees_the_plan_settings_not_the_image_ones(monkeypatch):
    monkeypatch.setenv("SGLANG_FROM_IMAGE", "1")
    monkeypatch.setenv("BENCH_KEPT", "1")
    plan = {"env": {"SGLANG_X": "2"}, "server_cpus": [4, 5], "workload": WORKLOAD}
    argv, env = client.launch(adapters.get("sglang"), plan, 8000, profile=False)
    assert argv[:3] == ["taskset", "--cpu-list", "4,5"]
    assert "SGLANG_FROM_IMAGE" not in env and env["BENCH_KEPT"] == "1"
    assert env["SGLANG_X"] == "2" and env["OMP_NUM_THREADS"] == "2"


def test_profile_control_accepts_sglang_plain_text(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(fake_server(tmp_path, port))
    try:
        client.wait_ready(server, port, "/ready", timeout=30)
        for engine in ("sid-gr", "vedantjh2-sglang"):
            for start in (True, False):
                adapters.get(engine).profile(port, start)
    finally:
        server.kill()
        server.wait()


def test_sglang_scores_are_converted_to_cumulative_logprob():
    beam = {"output_ids": [1, 2, 3], "meta_info": {"sequence_score": -0.5}}
    response = {"meta_info": {"id": "r", "beam_results": [beam]}}
    adapter = adapters.get("vedantjh2-sglang")
    assert adapter.beams(response, "r") == [([1, 2, 3], -1.5)]
    with pytest.raises(ValueError, match="does not belong"):
        adapter.beams(response, "other")


VEDANTJH2_CASES = [
    ("false", {}, None),
    (
        "true",
        {
            "forward.decode.graph": 0,
            "forward.decode.eager": 8,
            "vedantjh2.shared_attention": 224,
        },
        None,
    ),
    ("true", {"vedantjh2.shared_attention": 224}, "shared-context"),
    ("false", {"vedantjh2.shared_attention": 1}, "disabled"),
    ("false", {"vedantjh2.exact_normalizer": 7}, "normalizer"),
    ("false", {"vedantjh2.device_history": 3}, "device history"),
]


@pytest.mark.parametrize("shared, changes, message", VEDANTJH2_CASES)
def test_vedantjh2_checks(shared, changes, message):
    counters = {
        "forward.decode.graph": 8,
        "forward.decode.eager": 0,
        "vedantjh2.compact_decode": 8,
        "vedantjh2.exact_normalizer": 8,
        "vedantjh2.device_history": 4,
        "vedantjh2.shared_attention": 0,
        **changes,
    }
    env = {"SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": shared}
    plan = {"env": env, "catalog": {"items": 256}}
    adapter = adapters.get("vedantjh2-sglang")
    if message is None:
        adapter.check_counters(plan, counters, 4, {"length_penalty": 1.0})
    else:
        with pytest.raises(RuntimeError, match=message):
            adapter.check_counters(plan, counters, 4, {"length_penalty": 1.0})
    with pytest.raises(RuntimeError, match="length penalty"):
        adapter.check_counters(plan, counters, 4, {"length_penalty": 0.5})


@pytest.mark.parametrize("normalizer, message", [(0, None), (8, "ran although")])
def test_vedantjh2_codebook_normalization_skips_the_full_normalizer(
    normalizer, message
):
    counters = {
        "forward.decode.graph": 8,
        "forward.decode.eager": 0,
        "vedantjh2.compact_decode": 8,
        "vedantjh2.exact_normalizer": normalizer,
        "vedantjh2.device_history": 4,
        "vedantjh2.shared_attention": 0,
    }
    env = {"SGLANG_BEAM_TRIE_TOPK_LOGPROB": "true"}
    plan = {"env": env, "catalog": {"items": 256}}
    check = adapters.get("vedantjh2-sglang").check_counters
    if message is None:
        check(plan, counters, 4, {"length_penalty": 1.0})
    else:
        with pytest.raises(RuntimeError, match=message):
            check(plan, counters, 4, {"length_penalty": 1.0})


@pytest.mark.parametrize(
    "name", ["vedantjh2.compact_decode", "vedantjh2.exact_normalizer", None]
)
def test_vedantjh2_without_a_catalog_must_not_run_the_trie_head(name):
    counters = {
        "forward.decode.graph": 8,
        "forward.decode.eager": 0,
        "vedantjh2.compact_decode": 0,
        "vedantjh2.exact_normalizer": 0,
        "vedantjh2.device_history": 0,
        "vedantjh2.shared_attention": 0,
    }
    plan = {"env": {"SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "false"}, "catalog": None}
    adapter = adapters.get("vedantjh2-sglang")
    if name is None:
        adapter.check_counters(plan, counters, 4, {"length_penalty": 1.0})
        return
    with pytest.raises(RuntimeError, match="trie output head ran without a catalog"):
        adapter.check_counters(plan, {**counters, name: 8}, 4, {"length_penalty": 1.0})


def test_public_sglang_uses_native_beam_width_and_checks_the_length_penalty():
    adapter = adapters.get("sglang")
    row = {"request_id": "r", "input_ids": [1]}
    params = adapter.payload(row, WORKLOAD)["sampling_params"]
    assert params["beam_width"] == 4 and params["n"] == 4
    assert "use_beam_search" not in params
    adapter.check_counters({"env": {}}, {}, 1, {"length_penalty": 1.0})
    with pytest.raises(RuntimeError, match="length penalty"):
        adapter.check_counters({"env": {}}, {}, 1, {"length_penalty": 0.5})


def test_the_load_profile_keeps_the_largest_level_in_flight(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "nsys", lambda name: [])  # no Nsight here
    monkeypatch.setattr(client, "export", lambda name: ["true"])
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "nsys").write_text("#!/bin/sh\necho NVIDIA Nsight Systems version 0\n")
    (fake / "nsys").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    result = run_fake(
        tmp_path,
        monkeypatch,
        mode="slow",
        concurrency=[1, 3],
        name="load_profile",
        profile=True,
    )
    # The largest level's own requests: each of 3 clients sends profile_requests (1).
    assert result["order"] == [3] and result["runtime"]["max_in_flight"] == 3
    (level,) = result["levels"]
    assert len(level["latency_ms"]) == 3 and len(level["steps"]) == 9
    outputs = read_json(tmp_path / "load_profile/result.json")
    assert outputs["profile"] is True


def test_profiled_runs_tell_the_server_what_to_mark(monkeypatch):
    plan = {"env": {}, "server_cpus": [4, 5], "workload": WORKLOAD}
    for name, marks in (("profile", "requests"), ("load_profile", "steps")):
        argv, env = client.launch(adapters.get("sglang"), plan, 8000, marks, name)
        assert argv[0] == "nsys" and f"/work/{name}/trace" in argv
        assert env["BENCH_PROFILE"] == marks
    argv, env = client.launch(adapters.get("sglang"), plan, 8000, "")
    assert argv[0] == "taskset" and env["BENCH_PROFILE"] == "0"
    assert client.level_order([1, 2, 4], "load_profile") == [4]


def test_a_step_may_end_after_the_handler_returned():
    """A step can send its results and then do bookkeeping; the request's decode
    ends at its handler's return, not at the step's end."""
    step = {"requests": ["r"], "rows": 1, "tokens": 4}
    runtime = {
        "requests": {"r": [100, 900]},
        "steps": [
            {
                **step,
                "kind": "prefill",
                "scheduled_ns": 200,
                "launched_ns": 250,
                "finished_ns": 400,
            },
            {
                **step,
                "kind": "decode",
                "scheduled_ns": 450,
                "launched_ns": 460,
                "finished_ns": 950,
            },
        ],
    }
    phases, steps = client.timeline(
        [{"request_id": "r"}], [(None, 50, 1000)], runtime, 2
    )
    assert phases["decode"] == [pytest.approx(5e-4)] and phases["return"] == [0.0]
    assert sum(ms for (ms,) in phases.values()) == pytest.approx(9.5e-4)
    assert steps[1][5] == pytest.approx(5e-4)  # the step keeps its own full time
