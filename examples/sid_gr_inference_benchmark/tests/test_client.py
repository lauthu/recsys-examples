# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import socket
import subprocess
import sys

import pytest
from bench import client, engines
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
                    "decode.eager_fallbacks": None}
        catalog = {"source": "/work/catalog.jsonl", "item_count": len(paths)}
        runtime = {"pid": os.getpid(), "counters": counters, "max_in_flight": in_flight[1]}
        self.reply({"/ready": {"ready": True}, "/catalog/status": {"catalog": catalog},
                    "/pilot/runtime": runtime}[self.path])

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"null")
        if self.path in ("/start_profile", "/stop_profile"):
            return self.reply("Start profiling.\\n")
        if self.path.startswith("/pilot/profile/"):
            return self.reply({"capturing": self.path.endswith("start")})
        with lock:
            served[0] += 1
            in_flight[0] += 1
            in_flight[1] = max(in_flight)
        if mode == "slow":
            time.sleep(0.2)
        with lock:
            in_flight[0] -= 1
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


def run_fake(tmp_path, monkeypatch, mode="valid", concurrency=1):
    rows = [{"item_id": i, "token_ids": p} for i, p in enumerate(PATHS)]
    (tmp_path / "catalog.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    write_json(
        tmp_path / "workload.json",
        [{"request_id": f"request-{i}", "input_ids": [5, 6, 7, 8]} for i in range(5)],
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
        "warmups": 2,
        "requests": 3,
        "profile_requests": 1,
        "vocab_size": 5000,
        "env": {},
        "server_settings": None,
        "server_cpus": sorted(os.sched_getaffinity(0))[:1],  # a CPU this job has
        "client_cpus": sorted(os.sched_getaffinity(0)),
        "workload_path": str(tmp_path / "workload.json"),
        "catalog": catalog,
    }
    monkeypatch.setattr(
        engines.get("sid-gr"),
        "server",
        lambda plan, port, profile: (
            fake_server(tmp_path, port, mode),
            dict(os.environ),
        ),
    )
    monkeypatch.setattr(
        client, "machine", lambda: {"gpu": "G", "driver": "D", "cpu": "C"}
    )
    client.run(plan, "clean_1", False, work=tmp_path)
    return read_json(tmp_path / "clean_1/result.json")


def test_clean_run_measures_checks_and_counts(tmp_path, monkeypatch):
    result = run_fake(tmp_path, monkeypatch)
    assert len(result["latency_ms"]) == 3 and all(ms > 0 for ms in result["latency_ms"])
    # Counters cover only the measured requests; a disabled runner (None) has no delta.
    assert result["counters"] == {"prefill.graph_replays": 3, "decode.graph_replays": 6}
    assert result["catalog_beams"] == 12 and result["gpu"] == "G"
    # Every measured request's validated beams are kept for the comparison.
    outputs = read_json(tmp_path / "clean_1/outputs.json")
    assert sorted(outputs) == ["request-2", "request-3", "request-4"]
    assert outputs["request-2"][1] == [PATHS[1], -1.0]
    assert "outputs" not in result


def test_clients_keep_concurrency_requests_in_flight(tmp_path, monkeypatch):
    result = run_fake(tmp_path, monkeypatch, mode="slow", concurrency=3)
    assert result["concurrency"] == 3 and result["runtime"]["max_in_flight"] == 3
    assert len(result["latency_ms"]) == 3
    # Responses stay with their requests whatever order they finish in.
    outputs = read_json(tmp_path / "clean_1/outputs.json")
    assert sorted(outputs) == ["request-2", "request-3", "request-4"]


def test_invalid_beams_fail_the_run(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="duplicate beams"):
        run_fake(tmp_path, monkeypatch, mode="duplicate")


def test_profile_control_accepts_sglang_plain_text(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(fake_server(tmp_path, port))
    try:
        client.wait_ready(server, port, "/ready", timeout=30)
        for engine in ("sid-gr", "vedantjh2-sglang"):
            for start in (True, False):
                engines.get(engine).profile(port, start)
    finally:
        server.kill()
        server.wait()


def test_sglang_scores_are_converted_to_cumulative_logprob():
    beam = {"output_ids": [1, 2, 3], "meta_info": {"sequence_score": -0.5}}
    response = {"meta_info": {"id": "r", "beam_results": [beam]}}
    engine = engines.get("vedantjh2-sglang")
    assert engine.beams(response, "r") == [([1, 2, 3], -1.5)]
    with pytest.raises(ValueError, match="does not belong"):
        engine.beams(response, "other")


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
    engine = engines.get("vedantjh2-sglang")
    if message is None:
        engine.check_counters(plan, counters, 4, {"length_penalty": 1.0})
    else:
        with pytest.raises(RuntimeError, match=message):
            engine.check_counters(plan, counters, 4, {"length_penalty": 1.0})
    with pytest.raises(RuntimeError, match="length penalty"):
        engine.check_counters(plan, counters, 4, {"length_penalty": 0.5})


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
    engine = engines.get("vedantjh2-sglang")
    if name is None:
        engine.check_counters(plan, counters, 4, {"length_penalty": 1.0})
        return
    with pytest.raises(RuntimeError, match="trie output head ran without a catalog"):
        engine.check_counters(plan, {**counters, name: 8}, 4, {"length_penalty": 1.0})


def test_public_sglang_uses_native_beam_width_and_checks_the_length_penalty():
    engine = engines.get("sglang")
    row = {"request_id": "r", "input_ids": [1]}
    params = engine.payload(row, WORKLOAD)["sampling_params"]
    assert params["beam_width"] == 4 and params["n"] == 4
    assert "use_beam_search" not in params
    engine.check_counters({"env": {}}, {}, 1, {"length_penalty": 1.0})
    with pytest.raises(RuntimeError, match="length penalty"):
        engine.check_counters({"env": {}}, {}, 1, {"length_penalty": 0.5})
