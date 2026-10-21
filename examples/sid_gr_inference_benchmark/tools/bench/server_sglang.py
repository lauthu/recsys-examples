# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SGLang HTTP server plus benchmark endpoints.

    BENCH_ENGINE=<engine> python3 -m bench.server_sglang --plan /work/plan.json --port PORT

GET /pilot/ready    200 once the server status is Up
GET /pilot/runtime  packages, attention backends, catalog and counters from the scheduler

Counters start at zero: model forwards per phase in CUDA-graph or eager mode
and, for the SIDTrie fork, compact output head, exact normalizer, device-side
beam history and shared-context attention calls. ModelRunner.forward runs in
Python for graph replays too, so forward counters see every forward; kernel
wrappers such as shared-context attention only run in eager mode.
With BENCH_PROFILE=1, NVTX ranges bench/request|phase|section are pushed only
while a native /start_profile capture (activities=["CUDA_PROFILER"]) is active.
"""

import argparse
import contextlib
import functools
import importlib.metadata
import json
import os
import threading
from pathlib import Path

PROFILE = os.environ.get("BENCH_PROFILE") == "1"
SIDTRIE = os.environ.get("BENCH_ENGINE") == "sglang-sidtrie"
PACKAGES = (
    "sglang-kernel torch triton flashinfer-python nvidia-cutlass-dsl transformers"
)
COUNTERS = """forward.prefill.graph forward.prefill.eager forward.decode.graph
    forward.decode.eager sidtrie.compact_decode sidtrie.exact_normalizer
    sidtrie.device_history sidtrie.shared_attention"""
_counts = dict.fromkeys(COUNTERS.split(), 0)
_observed: dict[str, object] = {}
_capturing = False
_context = threading.local()  # request ID of the batch running on this thread
_decode_steps: dict[str, int] = {}
_batch_phase: dict[int, tuple[str, str]] = {}


def _class(value):
    return type(value).__name__ if value is not None else None


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@contextlib.contextmanager
def nvtx(kind, rid, label=None):
    if not (_capturing and rid is not None):
        yield
        return
    import torch

    torch.cuda.nvtx.range_push(f"bench/{kind}:{rid}" + (f":{label}" if label else ""))
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def wrap(owner, name, make):
    original = getattr(owner, name, None)
    if not callable(original):
        raise RuntimeError(
            f"unsupported SGLang revision: {owner.__name__}.{name} is missing"
        )
    setattr(owner, name, functools.wraps(original)(make(original)))


def section(label, counter=None):
    """NVTX section around a call; optionally count calls (in every run)."""

    def make(original):
        def wrapped(*args, **kwargs):
            if counter:
                _counts[counter] += 1
            if not _capturing:
                return original(*args, **kwargs)
            with nvtx("section", getattr(_context, "rid", None), label):
                return original(*args, **kwargs)

        return wrapped

    return make


def catalog_status(scheduler):
    if SIDTRIE:
        trie = getattr(
            getattr(scheduler, "beam_coordinator", None), "trie_constraint", None
        )
        return trie and {
            "active": True,
            "source": trie.config.tensor_path,
            "items": list(trie.level_cardinalities)[-1],
        }
    trie = getattr(getattr(scheduler, "beam_search_constraint", None), "trie", None)
    return trie and {
        "active": True,
        "items": trie.stats()["num_sequences"],
        "source": scheduler.server_args.beam_search_constraint_dict,
    }


def runtime(scheduler):
    import sglang

    runner = scheduler.tp_worker.model_runner
    backend = runner.attn_backend

    def resolved(phase):
        child = getattr(runner, f"{phase}_attn_backend", None)
        child = child or getattr(backend, f"{phase}_backend", None)
        return _class(backend if child is None or isinstance(child, str) else child)

    return {
        "sglang": sglang.__file__,
        "packages": {name: _version(name) for name in PACKAGES.split()},
        "attention": {"prefill": resolved("prefill"), "decode": resolved("decode")},
        "catalog": catalog_status(scheduler),
        "length_penalty": _observed.get("length_penalty"),
        "counters": dict(_counts),
    }


def batch_phase(batch, advance=False, preparation=None):
    """(request ID, "prefill" | "decode_<k>") for a batch holding one request."""
    ids = {str(req.rid) for req in getattr(batch, "reqs", ())}
    if len(ids) != 1:
        return None, None
    rid = ids.pop()
    decode = preparation == "decode" or (
        preparation is None and batch.forward_mode.is_decode()
    )
    step = _decode_steps.get(rid, 0) + 1 if decode else 0
    label = f"decode_{step}" if decode else "prefill"
    if advance:
        _decode_steps[rid] = step
        _batch_phase[id(batch)] = (rid, label)
    return rid, label


def install_sidtrie_hooks():
    from sglang.srt.beam_search import shared_context_attention_kernel as kernel
    from sglang.srt.beam_search.beam_group import BeamGroup
    from sglang.srt.beam_search.coordinator import BeamCoordinator
    from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead

    def finalize(original):
        traced = section("beam_history")(original)

        def wrapped(self, *args, **kwargs):
            result = traced(self, *args, **kwargs)
            _counts["sidtrie.device_history"] += bool(self.device_history)
            _observed["length_penalty"] = self.length_penalty
            return result

        return wrapped

    attention = section("shared_context_attention", counter="sidtrie.shared_attention")
    wrap(kernel, "shared_context_attention", attention)
    wrap(BeamGroup, "finalize", finalize)
    if PROFILE:
        wrap(BeamTrieOutputHead, "project_mixed", section("trie_output_head"))
        for method, label in (
            ("select_leader_prefill", "beam_selection"),
            ("select_and_relay_decode", "beam_selection"),
            ("commit_prefill", "beam_commit"),
            ("commit_decode", "beam_commit"),
            ("_spawn_member_rows", "beam_kv"),
            ("_reclaim_orphans", "beam_kv"),
        ):
            wrap(BeamCoordinator, method, section(label))


def install_scheduler_hooks():
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.model_runner import ModelRunner

    def internal_state(original):
        def wrapped(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            result.internal_state["bench"] = runtime(self)
            return result

        return wrapped

    def forward(original):
        traced = section("model")(original)

        def wrapped(self, batch, *args, **kwargs):
            result = traced(self, batch, *args, **kwargs)
            kind = "decode" if batch.forward_mode.is_decode() else "prefill"
            mode = "graph" if result.can_run_graph else "eager"
            _counts[f"forward.{kind}.{mode}"] += 1
            if SIDTRIE and kind == "decode":
                output = getattr(result, "logits_output", None)
                trie = getattr(batch, "beam_trie_levels", None)
                normalizer = getattr(output, "beam_normalizer", None)
                _counts["sidtrie.compact_decode"] += trie is not None
                _counts["sidtrie.exact_normalizer"] += normalizer is not None
            return result

        return wrapped

    wrap(Scheduler, "get_internal_state", internal_state)
    wrap(ModelRunner, "forward", forward)
    if SIDTRIE:
        install_sidtrie_hooks()
    if PROFILE:
        install_profile_hooks()


def install_profile_hooks():
    """NVTX phases and sections; ranges are pushed only during a capture."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.profiler_manager import (
        SchedulerProfilerManager,
    )

    def start(original):
        def wrapped(self, *args, **kwargs):
            global _capturing
            result = original(self, *args, **kwargs)
            _capturing = bool(self.profile_in_progress)
            return result

        return wrapped

    def stop(original):
        def wrapped(self, *args, **kwargs):
            global _capturing
            try:
                return original(self, *args, **kwargs)
            finally:
                _capturing = False

        return wrapped

    def batch_scope(mode, label):
        def make(original):
            def wrapped(self, *args, **kwargs):
                # ScheduleBatch methods run on the batch; Scheduler methods take it.
                on_batch = mode in ("prefill", "decode")
                batch = self if on_batch else args[0] if args else kwargs["batch"]
                if mode == "result":
                    fallback = (batch_phase(batch)[0], "result")
                    rid, phase = _batch_phase.get(id(batch), fallback)
                else:
                    preparation = None if mode == "run" else mode
                    rid, phase = batch_phase(batch, mode == "run", preparation)
                previous, _context.rid = getattr(_context, "rid", None), rid
                try:
                    with nvtx("phase", rid, phase), nvtx("section", rid, label):
                        return original(self, *args, **kwargs)
                finally:
                    _context.rid = previous

            return wrapped

        return make

    wrap(SchedulerProfilerManager, "_start_profile", start)
    wrap(SchedulerProfilerManager, "_stop_profile", stop)
    for owner, method, mode, label in (
        (ScheduleBatch, "prepare_for_extend", "prefill", "input_preparation"),
        (ScheduleBatch, "prepare_for_decode", "decode", "input_preparation"),
        (Scheduler, "run_batch", "run", "scheduler_forward"),
        (Scheduler, "process_batch_result", "result", "result_processing"),
    ):
        wrap(owner, method, batch_scope(mode, label))
    if not SIDTRIE:
        for method, label in (
            ("process_beam_search_prefill_result", "beam_selection"),
            ("process_beam_search_decode_result", "beam_selection"),
            ("_handle_beam_kv_cache", "beam_kv"),
        ):
            wrap(Scheduler, method, section(label))


def run_scheduler_process(*args, **kwargs):
    """Spawn target: patch the scheduler process before it builds the Scheduler."""
    install_scheduler_hooks()
    from sglang.srt.managers.scheduler import run_scheduler_process as native

    return native(*args, **kwargs)


def install_http_hooks(http_server):
    from fastapi.responses import JSONResponse

    async def ready():
        manager = http_server._global_state.tokenizer_manager
        up = (
            str(manager.server_status) == "ServerStatus.Up"
            and not manager.gracefully_exit
        )
        return JSONResponse({"ready": up}, status_code=200 if up else 503)

    async def runtime_route():
        states = await http_server._global_state.tokenizer_manager.get_internal_state()
        return JSONResponse({"pid": os.getpid(), **states[0]["bench"]})

    http_server.app.add_api_route("/pilot/ready", ready, methods=["GET"])
    http_server.app.add_api_route("/pilot/runtime", runtime_route, methods=["GET"])
    if not PROFILE:
        return

    def request(original):
        async def wrapped(*args, **kwargs):
            obj = kwargs.get("obj", args[0] if args else None)
            with nvtx("request", getattr(obj, "rid", None)):
                return await original(*args, **kwargs)

        return wrapped

    def capture(active):
        def make(original):
            async def wrapped(*args, **kwargs):
                global _capturing
                try:
                    return await original(*args, **kwargs)
                finally:
                    _capturing = active

            return wrapped

        return make

    hooks = {
        "/generate": request,
        "/start_profile": capture(True),
        "/stop_profile": capture(False),
    }
    for route in http_server.app.routes:
        make = hooks.pop(getattr(route, "path", None), None)
        if make:
            wrapped = functools.wraps(route.endpoint)(make(route.endpoint))
            dependency = route.dependant  # codespell:ignore dependant
            route.endpoint = dependency.call = wrapped
    if hooks:
        raise RuntimeError(
            f"unsupported SGLang revision: routes {sorted(hooks)} are missing"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())

    from sglang.srt.entrypoints import http_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    settings = dict(plan["server_settings"], model_path="/model", host="127.0.0.1")
    settings["port"] = args.port
    if plan["catalog"] and plan["engine"] == "sglang-beam-search-new":
        settings["beam_search_constraint_dict"] = plan["catalog"]["server_source"]
    server_args = ServerArgs(**settings)
    if server_args.tp_size != 1 or not server_args.disable_overlap_schedule:
        raise ValueError("NVTX phases need tp_size=1 and disable_overlap_schedule=true")
    install_http_hooks(http_server)
    try:
        http_server.launch_server(
            server_args, run_scheduler_process_func=run_scheduler_process
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
