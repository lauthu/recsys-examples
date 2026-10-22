# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start SGLang's own HTTP server with the benchmark's declared patches.

    BENCH_ENGINE=<engine> python3 -m bench.server_sglang --plan /work/plan.json --port PORT

Compared with the image's native server (`python3 -m sglang.launch_server`):

Same
  - The server: SGLang's http_server.launch_server, run from the mounted source.
    Requests go to SGLang's own /generate; there is no proxy or extra hop.
  - ServerArgs: `server_settings` from the solution file verbatim, plus what
    the adapter sets (Adapter.server_args): model_path, host, port, and from
    the scenario dtype, random_seed, the CUDA graph sizes and request slots
    and, for beam-search-new with a catalog, beam_search_constraint_dict.

Different, and nothing else
  - ROUTES: two added GET routes (/pilot/ready, /pilot/runtime) and, when
    profiling, wrappers on /generate, /start_profile and /stop_profile.
  - PATCHES: the scheduler-process methods listed there. Hooks follow the rules
    in bench/patching.py: original arguments and results, no GPU sync.
  - Latency runs keep only the "always" entries, which count forwards and read
    state (about 0.6 us per forward). "profile" entries exist only in the
    profiled process and push NVTX ranges only while a capture is active.

server.log lists every applied patch at startup.
"""

import argparse
import contextlib
import functools
import importlib.metadata
import json
import os
import threading
from pathlib import Path

from bench import adapters
from bench.patching import Patch, apply

ENGINE = os.environ.get("BENCH_ENGINE")
PROFILE = os.environ.get("BENCH_PROFILE") == "1"
# The engine's code paths (Adapter.features) decide which patches apply.
FEATURES = adapters.get(ENGINE).features if ENGINE else frozenset()
TRIE_HEAD = "trie_head" in FEATURES
PACKAGES = (
    "sglang-kernel torch triton flashinfer-python nvidia-cutlass-dsl transformers"
)
COUNTERS = """forward.prefill.graph forward.prefill.eager forward.decode.graph
    forward.decode.eager forward.decode.rows vedantjh2.compact_decode
    vedantjh2.exact_normalizer vedantjh2.device_history vedantjh2.shared_attention"""


# ---- every difference from the native server ------------------------------
def patches():
    """Scheduler-process patches, installed before SGLang builds the Scheduler."""
    scheduler = "sglang.srt.managers.scheduler:Scheduler"
    batch = "sglang.srt.managers.schedule_batch:ScheduleBatch"
    profiler = (
        "sglang.srt.managers.scheduler_components.profiler_manager"
        ":SchedulerProfilerManager"
    )
    coordinator = "sglang.srt.beam_search.coordinator:BeamCoordinator"
    return [
        # Always: counters and state for /pilot/runtime.
        Patch(
            f"{scheduler}.get_internal_state",
            add_runtime,
            "adds the runtime snapshot that /pilot/runtime returns",
        ),
        Patch(
            "sglang.srt.model_executor.model_runner:ModelRunner.forward",
            count_forward,
            "counts forwards by phase and graph/eager mode, decode rows (and the "
            "vedantjh2 fork's trie-head decode path); NVTX section `model` when "
            "profiling",
        ),
        Patch(
            "sglang.srt.beam_search.beam_group:BeamGroup.finalize",
            record_finalize,
            "records length_penalty and device beam history; NVTX section "
            "`beam_history` when profiling",
            needs="native_beam",
        ),
        Patch(
            "sglang.srt.beam_search.shared_context_attention_kernel"
            ":shared_context_attention",
            section("shared_context_attention", counter="vedantjh2.shared_attention"),
            "counts shared-context attention calls; NVTX section when profiling",
            needs="shared_context",
        ),
        # Profile only: capture state, then NVTX phases and sections.
        Patch(
            f"{profiler}._start_profile",
            capture_started,
            "notes that a capture started",
            profile_only=True,
        ),
        Patch(
            f"{profiler}._stop_profile",
            capture_stopped,
            "notes that the capture stopped",
            profile_only=True,
        ),
        *(
            Patch(
                target,
                batch_scope(mode, label),
                f"NVTX phase range and section `{label}`",
                profile_only=True,
            )
            for target, mode, label in (
                (f"{batch}.prepare_for_extend", "prefill", "input_preparation"),
                (f"{batch}.prepare_for_decode", "decode", "input_preparation"),
                (f"{scheduler}.run_batch", "run", "scheduler_forward"),
                (f"{scheduler}.process_batch_result", "result", "result_processing"),
            )
        ),
        Patch(
            f"{batch}.copy",
            keep_phase,
            "gives the copy the overlap scheduler processes its batch's phase",
            profile_only=True,
        ),
        *(
            Patch(
                f"{coordinator}.{method}",
                section(label),
                f"NVTX section `{label}`",
                profile_only=True,
                needs="native_beam",
            )
            for method, label in (
                ("select_leader_prefill", "beam_selection"),
                ("select_and_relay_decode", "beam_selection"),
                ("commit_prefill", "beam_commit"),
                ("commit_decode", "beam_commit"),
                ("_spawn_member_rows", "beam_kv"),
                ("_reclaim_orphans", "beam_kv"),
            )
        ),
        Patch(
            "sglang.srt.beam_search.trie_output_head:BeamTrieOutputHead.project_mixed",
            section("trie_output_head"),
            "NVTX section `trie_output_head`",
            profile_only=True,
            needs="trie_head",
        ),
        *(
            Patch(
                f"{scheduler}.{method}",
                section(label),
                f"NVTX section `{label}`",
                profile_only=True,
                needs="legacy_beam",
            )
            for method, label in (
                ("process_beam_search_prefill_result", "beam_selection"),
                ("process_beam_search_decode_result", "beam_selection"),
                ("_handle_beam_kv_cache", "beam_kv"),
            )
        ),
    ]


def routes(http_server):
    """HTTP-process changes: (path, hook or None to add, profile_only, adds)."""
    return [
        ("/pilot/ready", ready_route(http_server), False, "added: 200 once Up"),
        ("/pilot/runtime", runtime_route(http_server), False, "added: snapshot"),
        ("/generate", request_range, True, "NVTX request range"),
        ("/start_profile", capture_route(True), True, "notes the capture start"),
        ("/stop_profile", capture_route(False), True, "notes the capture stop"),
    ]


# ---- state the hooks share --------------------------------------------------
_counts = dict.fromkeys(COUNTERS.split(), 0)
_observed: dict[str, object] = {}
_capturing = False
_context = threading.local()  # request ID of the batch running on this thread
_decode_steps: dict[str, int] = {}
_batch_phase: dict[int, tuple[str, str]] = {}


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


# ---- always-on hooks ---------------------------------------------------------
def section(label, counter=None):
    """Count calls (if counter is given); NVTX section while capturing."""

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


def add_runtime(original):
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.internal_state["bench"] = runtime(self)
        return result

    return wrapped


def count_forward(original):
    traced = section("model")(original)

    def wrapped(self, batch, *args, **kwargs):
        result = traced(self, batch, *args, **kwargs)
        kind = "decode" if batch.forward_mode.is_decode() else "prefill"
        mode = "graph" if result.can_run_graph else "eager"
        _counts[f"forward.{kind}.{mode}"] += 1
        if kind == "decode":
            _counts["forward.decode.rows"] += batch.batch_size
        if TRIE_HEAD and kind == "decode":
            output = getattr(result, "logits_output", None)
            trie = getattr(batch, "beam_trie_levels", None)
            normalizer = getattr(output, "beam_normalizer", None)
            _counts["vedantjh2.compact_decode"] += trie is not None
            _counts["vedantjh2.exact_normalizer"] += normalizer is not None
        return result

    return wrapped


def record_finalize(original):
    traced = section("beam_history")(original)

    def wrapped(self, *args, **kwargs):
        result = traced(self, *args, **kwargs)
        _counts["vedantjh2.device_history"] += bool(
            getattr(self, "device_history", False)
        )
        _observed["length_penalty"] = self.length_penalty
        return result

    return wrapped


# ---- profile-only hooks ------------------------------------------------------
def capture_started(original):
    def wrapped(self, *args, **kwargs):
        global _capturing
        result = original(self, *args, **kwargs)
        _capturing = bool(self.profile_in_progress)
        return result

    return wrapped


def capture_stopped(original):
    def wrapped(self, *args, **kwargs):
        global _capturing
        try:
            return original(self, *args, **kwargs)
        finally:
            _capturing = False

    return wrapped


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


def keep_phase(original):
    # The overlap scheduler processes a copy of each batch after launching the next.
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if id(self) in _batch_phase:
            _batch_phase[id(result)] = _batch_phase[id(self)]
        return result

    return wrapped


def batch_scope(mode, label):
    """NVTX phase and section around a batch step; mode says how to find it."""

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


# ---- HTTP routes -------------------------------------------------------------
def ready_route(http_server):
    from fastapi.responses import JSONResponse

    async def ready():
        manager = http_server._global_state.tokenizer_manager
        up = (
            str(manager.server_status) == "ServerStatus.Up"
            and not manager.gracefully_exit
        )
        return JSONResponse({"ready": up}, status_code=200 if up else 503)

    return ready


def runtime_route(http_server):
    from fastapi.responses import JSONResponse

    async def runtime_snapshot():
        states = await http_server._global_state.tokenizer_manager.get_internal_state()
        return JSONResponse({"pid": os.getpid(), **states[0]["bench"]})

    return runtime_snapshot


def request_range(original):
    async def wrapped(*args, **kwargs):
        obj = kwargs.get("obj", args[0] if args else None)
        with nvtx("request", getattr(obj, "rid", None)):
            return await original(*args, **kwargs)

    return wrapped


def capture_route(active):
    def make(original):
        async def wrapped(*args, **kwargs):
            global _capturing
            try:
                return await original(*args, **kwargs)
            finally:
                _capturing = active

        return wrapped

    return make


def install_routes(http_server):
    wrappers = {}
    for path, hook, profile_only, adds in routes(http_server):
        if profile_only and not PROFILE:
            continue
        if path.startswith("/pilot/"):
            http_server.app.add_api_route(path, hook, methods=["GET"])
        else:
            wrappers[path] = hook
        kind = "profile" if profile_only else "always"
        print(f"[bench] route {path} ({kind}): {adds}", flush=True)
    for route in http_server.app.routes:
        make = wrappers.pop(getattr(route, "path", None), None)
        if make:
            wrapped = functools.wraps(route.endpoint)(make(route.endpoint))
            dependency = route.dependant  # codespell:ignore dependant
            route.endpoint = dependency.call = wrapped
    if wrappers:
        raise RuntimeError(
            f"unsupported SGLang revision: routes {sorted(wrappers)} are missing"
        )


# ---- runtime snapshot (/pilot/runtime) ----------------------------------------
def _class(value):
    return type(value).__name__ if value is not None else None


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def catalog_status(scheduler):
    if TRIE_HEAD:
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
    """Facts read from the running scheduler; the client checks and reports them."""
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
        "dtype": str(runner.dtype).removeprefix("torch."),
        "overlap": scheduler.enable_overlap,
        "catalog": catalog_status(scheduler),
        "length_penalty": _observed.get("length_penalty"),
        "counters": dict(_counts),
        "patches": [patch.target for patch in applied],
    }


# ---- launch --------------------------------------------------------------------
applied: list[Patch] = []  # in the scheduler process, where runtime() runs


def run_scheduler_process(*args, **kwargs):
    """Spawn target: patch the scheduler process, then run SGLang's own."""
    applied.extend(apply(patches(), FEATURES, PROFILE))
    from sglang.srt.managers.scheduler import run_scheduler_process as native

    return native(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())

    from sglang.srt.entrypoints import http_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    adapter = adapters.get(plan["engine"])
    settings = {**plan["server_settings"], **adapter.server_args(plan, args.port)}
    server_args = ServerArgs(**settings)
    if server_args.tp_size != 1:
        raise ValueError("NVTX phases need tp_size=1")
    install_routes(http_server)
    try:
        http_server.launch_server(
            server_args, run_scheduler_process_func=run_scheduler_process
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
