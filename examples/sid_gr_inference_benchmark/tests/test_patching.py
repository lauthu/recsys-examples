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
from bench import adapters, server_sglang, server_sid_gr
from bench.patching import Patch, apply, configuration, plain, redacted, selected

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
    public, vedantjh2 = (adapters.get(name).features for name in NATIVE)
    assert selected(table, public, profile=False) == [always]
    assert selected(table, vedantjh2, profile=True) == table


def test_a_missing_target_stops_the_server(fake_engine):
    with pytest.raises(RuntimeError, match="fake_engine:Server.gone is missing"):
        apply([Patch("fake_engine:Server.gone", lambda f: f, "")], frozenset(), False)


@pytest.mark.parametrize("launcher", [server_sglang, server_sid_gr])
def test_launcher_tables_are_well_formed(launcher):
    for patch in launcher.patches():
        module, _, attribute = patch.target.partition(":")
        assert module and attribute and callable(patch.hook) and patch.adds
        known = set().union(*(a.features for a in adapters.ADAPTERS.values()))
        assert patch.needs is None or patch.needs in known


def test_latency_runs_patch_only_counters_and_routes():
    def always(launcher, engine):
        features = adapters.get(engine).features
        return [
            p.target.rsplit(":", 1)[1]
            for p in selected(launcher.patches(), features, False)
        ]

    assert always(server_sglang, "sglang-beam-search-new") == [
        "Scheduler.get_internal_state",
        "ModelRunner.forward",
    ]
    assert always(server_sglang, "sglang") == [
        "Scheduler.get_internal_state",
        "ModelRunner.forward",
        "BeamGroup.finalize",
    ]
    assert always(server_sglang, "vedantjh2-sglang") == [
        "Scheduler.get_internal_state",
        "ModelRunner.forward",
        "BeamGroup.finalize",
        "shared_context_attention",
    ]
    assert always(server_sid_gr, "sid-gr") == ["GRHTTPServingAdapter._dispatch"]


def test_the_sid_gr_launcher_loads_nothing_else_from_bench():
    code = "import sys, bench.server_sid_gr; print(sorted(m for m in sys.modules if m.startswith('bench')))"
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
        "'bench.patching',",
        "'bench.server_sid_gr']",
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
