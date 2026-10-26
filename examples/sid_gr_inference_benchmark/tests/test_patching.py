# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import enum
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from bench import engines
from bench.engines import sglang_server, sid_gr_server
from bench.engines.patching import (
    Patch,
    apply,
    configuration,
    plain,
    redacted,
    selected,
)

NATIVE = ("sglang", "vedantjh2-sglang")  # both use the native beam_width API


@pytest.fixture
def fake_engine(monkeypatch):
    """A module `fake_engine` with one class whose method we patch."""
    module = types.ModuleType("fake_engine")

    class Server:
        def handle(self, x, *, scale=1):
            return x * scale

    module.Server = Server
    monkeypatch.setitem(sys.modules, "fake_engine", module)
    return module


def counting(calls):
    def hook(original):
        def wrapped(*args, **kwargs):
            calls.append(args[1:])
            return original(*args, **kwargs)

        return wrapped

    return hook


def test_apply_keeps_arguments_and_results_and_logs(fake_engine, capsys):
    calls = []
    patch = Patch("fake_engine:Server.handle", counting(calls), "counts calls")
    assert apply([patch], frozenset(), profile=False) == [patch]
    assert fake_engine.Server().handle(3, scale=2) == 6  # the original's result
    assert calls == [(3,)] and fake_engine.Server.handle.__name__ == "handle"
    out = capsys.readouterr().out
    assert "[bench] patched fake_engine:Server.handle (always): counts calls" in out


def test_selection_by_feature_and_run_kind():
    always = Patch("m:f", None, "")
    profiling = Patch("m:g", None, "", profile_only=True)
    trie = Patch("m:h", None, "", needs="trie_head")
    table = [always, profiling, trie]
    public, vedantjh2 = (engines.get(name).features for name in NATIVE)
    assert selected(table, public, profile=False) == [always]
    assert selected(table, vedantjh2, profile=True) == table


def test_a_missing_target_stops_the_server(fake_engine):
    with pytest.raises(RuntimeError, match="fake_engine:Server.gone is missing"):
        apply([Patch("fake_engine:Server.gone", lambda f: f, "")], frozenset(), False)


@pytest.mark.parametrize("launcher", [sglang_server, sid_gr_server])
def test_launcher_tables_are_well_formed(launcher):
    for patch in launcher.patches():
        module, _, attribute = patch.target.partition(":")
        assert module and attribute and callable(patch.hook) and patch.adds
        known = set().union(*(engines.get(n).features for n in engines.ENGINES))
        assert patch.needs is None or patch.needs in known


def test_latency_runs_patch_only_counters_step_times_and_routes():
    def always(launcher, engine):
        features = engines.get(engine).features
        return [
            p.target.rsplit(":", 1)[1]
            for p in selected(launcher.patches(), features, False)
        ]

    steps = [
        "Scheduler.get_internal_state",
        "ModelRunner.forward",
        "Scheduler.get_next_batch_to_run",
        "Scheduler.run_batch",
        "Scheduler.process_batch_result",
        "ScheduleBatch.copy",
    ]
    assert always(sglang_server, "sglang-beam-search-new") == steps
    assert always(sglang_server, "sglang") == [*steps, "BeamGroup.finalize"]
    assert always(sglang_server, "vedantjh2-sglang") == [
        *steps,
        "BeamGroup.finalize",
        "shared_context_attention",
    ]
    assert always(sid_gr_server, "sid-gr") == [
        "GRHTTPServingAdapter._dispatch",
        "GRHTTPServingAdapter._handle_sglang_generate",
        "GRContinuousServingExecutor._run_prefill",
        "GRContinuousServingExecutor._run_decode_batch",
    ]


def test_the_sid_gr_launcher_loads_nothing_else_from_bench():
    code = "import sys, bench.engines.sid_gr_server; print(sorted(m for m in sys.modules if m.startswith('bench')))"
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.split() == [
        "['bench',",
        "'bench.engines',",
        "'bench.engines.patching',",
        "'bench.engines.sid_gr_server']",
    ]


class Mode(enum.Enum):
    FAST = "fast"


class Record:
    def __init__(self):
        self.name, self.sizes, self._hidden = "x", (1, 2), 3
        self.ratio, self.mode, self.tags = float("inf"), Mode.FAST, {"b", "a"}
        self.nested = {"inner": {"deep": {"deeper": {"deepest": [1]}}}}


def test_configuration_values_become_json_safe():
    found = plain({"record": Record(), 1: None})
    assert found == {
        "record": {
            "name": "x",
            "sizes": [1, 2],
            "ratio": "inf",
            "mode": "fast",
            "tags": ["a", "b"],
            "nested": {"inner": {"deep": "{'deeper': {'deepest': [1]}}"}},  # depth 4
        },
        "1": None,
    }
    json.dumps(found, allow_nan=False)


def test_only_secret_words_are_redacted():
    found = redacted(
        {
            "HF_TOKEN": "t",
            "api_key": "k",
            "ssl_keyfile_password": "p",
            "UNSET_SECRET": "",
            "bos_token_id": 1,
            "num_key_value_heads": 8,
            "max_prefill_tokens": 8,
            "skip_tokenizer_init": True,
            "ssl_keyfile": "/k.pem",
            "nested": {"admin_api_key": "a"},
        }
    )
    assert found == {
        "HF_TOKEN": "redacted",
        "api_key": "redacted",
        "ssl_keyfile_password": "redacted",
        "UNSET_SECRET": "",
        "bos_token_id": 1,
        "num_key_value_heads": 8,
        "max_prefill_tokens": 8,
        "skip_tokenizer_init": True,
        "ssl_keyfile": "/k.pem",
        "nested": {"admin_api_key": "redacted"},
    }


def test_config_snapshot_names_its_source_and_sorts_the_environment():
    found = configuration({"page_size": 1}, "build_info", {"B": "1", "GH_TOKEN": "x"})
    assert found == {
        "source": "build_info",
        "settings": {"page_size": 1},
        "environment": {"B": "1", "GH_TOKEN": "redacted"},
    }
    assert list(configuration({}, "s", {"B": "1", "A": "2"})["environment"]) == [
        "A",
        "B",
    ]


def test_overlap_scheduler_copies_keep_their_batch_phase(monkeypatch):
    """Overlap launches the next batch, then processes a copy of the previous one."""
    pushed = []

    def nvtx(kind, rid, label=None):
        pushed.append((kind, label))
        return contextlib.nullcontext()

    monkeypatch.setattr(sglang_server, "nvtx", nvtx)
    monkeypatch.setattr(sglang_server, "_batch_phase", {})
    monkeypatch.setattr(sglang_server, "_decode_steps", {})

    class Batch:
        def __init__(self, decode):
            self.reqs = [types.SimpleNamespace(rid="r1")]
            self.forward_mode = types.SimpleNamespace(is_decode=lambda: decode)

        copy = sglang_server.keep_step(
            lambda self: Batch(self.forward_mode.is_decode())
        )

    class Scheduler:
        run_batch = sglang_server.batch_scope("run", "scheduler_forward")(
            lambda self, batch: None
        )
        process_batch_result = sglang_server.batch_scope("result", "result")(
            lambda self, batch, result: None
        )

    scheduler, prefill, running = Scheduler(), Batch(False), Batch(True)
    queue = []
    for batch in (prefill, running, running):  # the running batch is reused
        scheduler.run_batch(batch)
        queue.append(batch.copy())
        if len(queue) > 1:
            scheduler.process_batch_result(queue.pop(0), None)
    scheduler.process_batch_result(queue.pop(0), None)
    phases = [label for kind, label in pushed if kind == "phase"]
    assert phases == [
        "prefill",
        "decode_1",
        "prefill",
        "decode_2",
        "decode_1",
        "decode_2",
    ]


def test_overlap_scheduler_copies_keep_their_step_times(monkeypatch):
    """Each step is recorded once, with its kind, requests and rows, whether its
    results are processed right away or, with overlap, from a copy one step late."""
    for name, value in (("_steps", []), ("_open_steps", {}), ("_running", None)):
        monkeypatch.setattr(sglang_server, name, value)

    class Batch:
        def __init__(self, decode, rids):
            self.reqs = [types.SimpleNamespace(rid=rid) for rid in rids]
            self.forward_mode = types.SimpleNamespace(is_decode=lambda: decode)
            self.extend_num_tokens = 8

        copy = sglang_server.keep_step(
            lambda self: Batch(
                self.forward_mode.is_decode(), [r.rid for r in self.reqs]
            )
        )

    def forward(self, batch):  # the forward notes its rows on the running step
        sglang_server._running["rows"] = 4 * len(batch.reqs)

    class Scheduler:
        get_next_batch_to_run = sglang_server.note_schedule(lambda self: None)
        run_batch = sglang_server.open_step(forward)
        process_batch_result = sglang_server.close_step(lambda self, b, r: None)

    scheduler = Scheduler()
    prefill, running = Batch(False, ["r1", "r1"]), Batch(True, ["r1", "r2"])
    queue = []
    for batch in (prefill, running, running):  # overlap: process the previous copy
        scheduler.get_next_batch_to_run()
        scheduler.run_batch(batch)
        queue.append(batch.copy())
        if len(queue) > 1:
            scheduler.process_batch_result(queue.pop(0), None)
    scheduler.process_batch_result(queue.pop(0), None)
    scheduler.get_next_batch_to_run()
    scheduler.run_batch(prefill)  # without overlap: the batch itself
    scheduler.process_batch_result(prefill, None)
    steps = sglang_server.taken(sglang_server._steps)
    assert [(s["kind"], s["requests"], s["rows"], s["tokens"]) for s in steps] == [
        ("prefill", ["r1"], 8, 8),
        ("decode", ["r1", "r2"], 8, None),
        ("decode", ["r1", "r2"], 8, None),
        ("prefill", ["r1"], 8, 8),
    ]
    for s in steps:
        assert s["scheduled_ns"] <= s["launched_ns"] <= s["finished_ns"]
    assert sglang_server._steps == []  # taken records are forgotten


def test_marking_steps_wraps_each_step_in_one_window(monkeypatch):
    """Under load the window opens when the scheduler starts choosing a batch, holds
    the batch's phase and section ranges, and ends after its results; it ends at
    once when no batch was chosen. It is a start/end range, so an engine's own
    push/pop around get_next_batch_to_run (beam-search-new's) cannot close it.
    Phases are the step kind."""
    marks, started = [], []

    def range_start(text):
        started.append(text)
        marks.append(f"start {text}")
        return len(started)

    nvtx = types.SimpleNamespace(
        range_push=lambda text: marks.append(text),
        range_pop=lambda: marks.append("pop"),
        range_start=range_start,
        range_end=lambda handle: marks.append(f"end {started[handle - 1]}"),
    )
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(nvtx=nvtx))
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name, value in (("MARKS", "steps"), ("_capturing", True), ("_window", [])):
        monkeypatch.setattr(sglang_server, name, value)
    monkeypatch.setattr(sglang_server, "_batch_phase", {})
    monkeypatch.setattr(sglang_server, "_windows", iter(range(1, 100)))

    class Batch:
        def __init__(self, decode):
            self.reqs = [types.SimpleNamespace(rid=r) for r in ("r1", "r2")]
            self.forward_mode = types.SimpleNamespace(is_decode=lambda: decode)

    class Scheduler:  # stacked as the profiled process installs them
        get_next_batch_to_run = sglang_server.open_window(lambda self, chosen: chosen)
        run_batch = sglang_server.batch_scope("run", "scheduler_forward")(
            lambda self, batch: None
        )
        process_batch_result = sglang_server.close_window(
            sglang_server.batch_scope("result", "result")(lambda self, b, r: None)
        )

    scheduler = Scheduler()

    def choose(batch):  # the engine's own range around choosing, as in beam-search-new
        nvtx.range_push("engine")
        scheduler.get_next_batch_to_run(types.SimpleNamespace(batch_to_run=batch))
        nvtx.range_pop()

    choose(None)
    for batch in (Batch(False), Batch(True)):
        choose(batch)
        scheduler.run_batch(batch)
        scheduler.process_batch_result(batch, None)

    def step(n, kind):
        return [
            "engine",
            f"start bench/step:{n}",
            "pop",
            f"bench/phase:{n}:{kind}",
            f"bench/section:{n}:scheduler_forward",
            "pop",
            "pop",
            f"bench/phase:{n}:{kind}",
            f"bench/section:{n}:result",
            "pop",
            "pop",
            f"end bench/step:{n}",
        ]

    idle = ["engine", "start bench/step:1", "end bench/step:1", "pop"]
    assert marks == [*idle, *step(2, "prefill"), *step(3, "decode")]
    assert sglang_server._window == []
