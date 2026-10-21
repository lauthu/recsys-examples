# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import sys
import types

import pytest
from bench import engines, server_sglang, server_sid_gr
from bench.patching import Patch, apply, selected


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
    assert apply([patch], "any", profile=False) == [patch]
    assert fake_engine.Server().handle(3, scale=2) == 6  # the original's result
    assert calls == [(3,)] and fake_engine.Server.handle.__name__ == "handle"
    out = capsys.readouterr().out
    assert "[bench] patched fake_engine:Server.handle (always): counts calls" in out


def test_selection_by_engine_and_run_kind():
    always = Patch("m:f", None, "")
    profiling = Patch("m:g", None, "", profile_only=True)
    vedantjh2 = Patch("m:h", None, "", engines=("vedantjh2-sglang",))
    table = [always, profiling, vedantjh2]
    assert selected(table, "sglang", profile=False) == [always]
    assert selected(table, "vedantjh2-sglang", profile=True) == table


def test_a_missing_target_stops_the_server(fake_engine):
    with pytest.raises(RuntimeError, match="fake_engine:Server.gone is missing"):
        apply([Patch("fake_engine:Server.gone", lambda f: f, "")], "x", False)


@pytest.mark.parametrize("launcher", [server_sglang, server_sid_gr])
def test_launcher_tables_are_well_formed(launcher):
    for patch in launcher.patches():
        module, _, attribute = patch.target.partition(":")
        assert module and attribute and callable(patch.hook) and patch.adds
        assert patch.engines is None or set(patch.engines) <= set(engines.ENGINES)


def test_latency_runs_patch_only_counters_and_routes():
    def always(launcher, engine):
        return [
            p.target.rsplit(":", 1)[1]
            for p in selected(launcher.patches(), engine, False)
        ]

    assert always(server_sglang, "sglang-beam-search-new") == [
        "Scheduler.get_internal_state",
        "ModelRunner.forward",
    ]
    assert always(server_sglang, "vedantjh2-sglang") == [
        "Scheduler.get_internal_state",
        "ModelRunner.forward",
        "BeamGroup.finalize",
        "shared_context_attention",
    ]
    assert always(server_sid_gr, "sid-gr") == ["GRHTTPServingAdapter._dispatch"]


def test_overlap_scheduler_copies_keep_their_batch_phase(monkeypatch):
    """Overlap launches the next batch, then processes a copy of the previous one."""
    pushed = []

    def nvtx(kind, rid, label=None):
        pushed.append((kind, label))
        return contextlib.nullcontext()

    monkeypatch.setattr(server_sglang, "nvtx", nvtx)
    monkeypatch.setattr(server_sglang, "_batch_phase", {})
    monkeypatch.setattr(server_sglang, "_decode_steps", {})

    class Batch:
        def __init__(self, decode):
            self.reqs = [types.SimpleNamespace(rid="r1")]
            self.forward_mode = types.SimpleNamespace(is_decode=lambda: decode)

        copy = server_sglang.keep_phase(
            lambda self: Batch(self.forward_mode.is_decode())
        )

    class Scheduler:
        run_batch = server_sglang.batch_scope("run", "scheduler_forward")(
            lambda self, batch: None
        )
        process_batch_result = server_sglang.batch_scope("result", "result")(
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
