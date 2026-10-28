#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run SID-GR's own HTTP server with the benchmark's declared patches.

scripts/serve_qwen3_gr_http.sh execs $PYTHON_BIN; the runner points it here:

    sid_gr_server.py .../tools/serve_qwen3_gr_http.py [args]

gr-inference-internal, which SID-GR was imported from, keeps the classes and
methods patched here and runs under this launcher too. Its server is ASGI and
awaits _handle_sglang_generate_async for each request (the "asgi" entry), and
its script runs $PYTHON_BIN to probe and install packages (`python - <<EOF`,
`python -m pip`): every command other than the server runs unchanged in this
interpreter.

Compared with the native server (serve_qwen3_gr_http.sh with plain python3):

Same
  - The server: serve_qwen3_gr_http.py from the mounted source, run unchanged in
    this process with runpy, with the arguments and environment the script
    builds. Requests go to SID-GR's own handlers; there is no proxy or extra hop.

Different, and nothing else
  - PATCHES: the methods listed there. Hooks follow the rules in
    bench/engines/patching.py: original arguments and results, no GPU sync.
  - "always" entries add four routes and leave every other route to the
    original dispatcher (about 85 ns per request), and note when each request's
    handler starts and returns and when each prefill and decode step starts and
    ends (time.perf_counter_ns, a few microseconds per step):
        GET  /pilot/runtime        packages, the imported engine, attention backends,
                                   CUDA graph counters, gr-inference-internal's
                                   beam-selection CUDA extension and its calls, and
                                   the request and step times, which it then forgets
        GET  /pilot/config         the server's build_info and environment, read
                                   once after the measured requests
        POST /pilot/profile/start  cudaProfilerStart (profiled processes only)
        POST /pilot/profile/stop   cudaProfilerStop
  - "profile" entries exist only in the profiled processes and push NVTX ranges
    only while a capture is active: bench/request|phase|section per request, or,
    under load (BENCH_PROFILE=steps), bench/step|phase|section per engine step.

The server log lists every applied patch at startup.
"""

import contextlib
import importlib
import importlib.metadata
import inspect
import os
import runpy
import sys
import threading
import time
from collections import Counter
from pathlib import Path

# Only the patch machinery: more modules in this process put a ~220 ms stall
# into one round of measured requests at concurrency 4 (seen when it imported
# the adapters).
from bench.engines.patching import (  # PYTHONPATH has /benchmark-tools
    Patch,
    apply,
    configuration,
)

# What the profiled process marks: each request ("requests") or, under load, each
# engine step ("steps"); "0" in the latency runs.
MARKS = os.environ.get("BENCH_PROFILE", "0")
PROFILE = MARKS != "0"
PACKAGES = (
    "torch triton flashinfer-python nvidia-cutlass-dsl quack-kernels transformers"
)


# ---- every difference from the native server ------------------------------
def patches():
    http = "gr_inference.gr_serving.http:GRHTTPServingAdapter"
    executor = "gr_inference.gr_serving.continuous:GRContinuousServingExecutor"
    return [
        Patch(
            f"{http}._dispatch",
            pilot_routes,
            "adds /pilot/runtime, /pilot/config and /pilot/profile/start|stop; "
            "other routes go to the original dispatcher",
        ),
        Patch(
            f"{http}._handle_sglang_generate",
            request_range,
            "notes when the handler starts and returns; NVTX request range when "
            "profiling, in-flight range under load",
        ),
        Patch(
            f"{http}._handle_sglang_generate_async",
            request_range,
            "notes when the awaited handler starts and returns; NVTX request range "
            "when profiling, in-flight range under load",
            needs="asgi",
        ),
        Patch(
            f"{executor}._run_prefill",
            step_times(decode=False),
            "notes a prefill step's requests, start and end",
        ),
        Patch(
            f"{executor}._run_decode_batch",
            step_times(decode=True),
            "notes a decode step's requests, rows, start and end",
        ),
        Patch(
            f"{executor}._run_prefill",
            phase_range(decode=False),
            "NVTX phase `prefill`",
            profile_only=True,
        ),
        Patch(
            f"{executor}._run_decode_batch",
            phase_range(decode=True),
            "NVTX phase `decode_<step>`",
            profile_only=True,
        ),
        Patch(
            f"{executor}._profile_section",
            named_section,
            "NVTX section named by SID-GR's own profile sections",
            profile_only=True,
        ),
        Patch(
            f"{executor}._forward_prefill",
            section("model.forward_prefill"),
            "NVTX section `model.forward_prefill`",
            profile_only=True,
        ),
        Patch(
            f"{executor}._store_single_prefill_result",
            section("prefill.initialize_generation"),
            "NVTX section `prefill.initialize_generation`",
            profile_only=True,
        ),
    ]


# ---- state the hooks share --------------------------------------------------
_capturing = False
_scope = threading.local()  # request IDs of the phase running on this thread
# Handler threads and the worker add records; /pilot/runtime takes them.
_records = threading.Lock()
_requests: dict[str, list[int]] = {}  # rid: [handler start, return] in ns
_steps: list[dict] = []


@contextlib.contextmanager
def nvtx(kind, label):
    if not _capturing or label is None:
        yield
        return
    import torch

    torch.cuda.nvtx.range_push(f"bench/{kind}:{label}")
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def inflight(rid):
    """Marking steps, also each request's time in the handler: requests overlap
    there, so a start/end range, which unlike push and pop need not nest."""
    if not (_capturing and MARKS == "steps" and rid is not None):
        yield
        return
    import torch

    window = torch.cuda.nvtx.range_start(f"bench/inflight:{rid}")
    try:
        yield
    finally:
        torch.cuda.nvtx.range_end(window)


def section_label(name):
    rid = getattr(_scope, "rid", None)
    return None if rid is None else f"{rid}:{name}"


# ---- always-on hook: the /pilot routes ---------------------------------------
def pilot_routes(original):
    from gr_inference.gr_serving.http import GRHTTPResponse

    def wrapped(self, method, route, payload):
        if (method, route) == ("GET", ("pilot", "runtime")):
            return GRHTTPResponse(200, runtime(self))
        if (method, route) == ("GET", ("pilot", "config")):
            snapshot = configuration(self.build_info, "build_info", os.environ)
            return GRHTTPResponse(200, snapshot)
        if method == "POST" and route in (
            ("pilot", "profile", "start"),
            ("pilot", "profile", "stop"),
        ):
            return GRHTTPResponse(200, profile(self, route[2] == "start"))
        return original(self, method, route, payload)

    return wrapped


def _executor(serving):
    facade = serving.facade
    return getattr(facade, "facade", facade).executor


def _idle(serving):
    """Hold the worker lock so snapshots and profiler calls never overlap a request."""
    return getattr(serving.facade, "_lock", contextlib.nullcontext())


def _class(value):
    return type(value).__name__ if value is not None else None


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime(serving):
    """Facts read from the running executor; the client checks and reports them."""
    with _idle(serving):
        executor = _executor(serving)
        backends = Counter()
        for layer in executor.engine.model.layers:
            backend = layer.prefill_runner.attention._backend
            backends[_class(getattr(backend, "_backend", None) or backend)] += 1
        decode = executor.engine.decode_engine.attention
        decode = getattr(decode, "backend", None) or getattr(decode, "_backend", None)
        weights = executor.engine.model.embed_tokens.weight.dtype
        prefill_graph = executor.prefill_cuda_graph_runner
        decode_graph = executor.decode_cuda_graph_runner
        # A disabled graph runner has no status; None keeps that distinct from 0.
        prefill_status = prefill_graph.status() if prefill_graph is not None else {}
        decode_status = decode_graph.status() if decode_graph is not None else {}
        selection, selection_calls = beam_selection()
        # Under the worker lock: a tick, with the steps it records, has finished.
        records = taken_records()
    return {
        "pid": os.getpid(),
        "gr_inference": os.path.dirname(sys.modules["gr_inference"].__file__),
        "packages": {name: _version(name) for name in PACKAGES.split()},
        "attention": {
            "prefill": ", ".join(
                f"{name} x{count}" for name, count in backends.items()
            ),
            "decode": _class(decode),
        },
        "dtype": str(weights).removeprefix("torch."),
        "cuda_graphs": {
            "prefill": prefill_graph is not None,
            "decode": decode_graph is not None,
        },
        "counters": {
            "prefill.graph_replays": prefill_status.get("prefill_cuda_graph_replays"),
            "prefill.graph_captures": prefill_status.get("prefill_cuda_graph_captures"),
            "decode.graph_replays": decode_status.get("decode_cuda_graph_replays"),
            "decode.graph_captures": decode_status.get("decode_cuda_graph_captures"),
            "decode.eager_fallbacks": decode_status.get(
                "decode_cuda_graph_fallback_eager_count"
            ),
            # gr-inference-internal only: decode model steps run without a graph,
            # and mixed-step batches rerun as one batch per step.
            "decode.eager_model_steps": getattr(
                executor, "decode_cuda_graph_eager_model_steps", None
            ),
            "decode.mixed_step_fallbacks": getattr(
                executor, "mixed_step_decode_fallbacks", None
            ),
            **selection_calls,
        },
        "beam_selection": selection,
        **records,
        "patches": [patch.target for patch in applied],
    }


# gr-inference-internal's beam-selection CUDA extension; SID-GR has none.
BEAM_SELECTION = "gr_inference.gr_kernels.beam_selection"


def beam_selection():
    """(status, counters) of gr-inference-internal's beam-selection CUDA
    extension, or (None, {}) without one. The engine builds it at first use and
    does without it if the build fails. A call either ran on it ("<name>_cuda")
    or was declined or failed, and the engine did that step in torch."""
    try:
        module = importlib.import_module(BEAM_SELECTION)
    except ModuleNotFoundError as missing:  # the module, not one it imports
        if not f"{BEAM_SELECTION}.".startswith(f"{missing.name}."):
            raise
        return None, {}
    status, calls = module.extension_status(), module.call_counts()
    ran = sum(count for call, count in calls.items() if call.endswith("_cuda"))
    return (
        {
            "loaded": status["loaded"],
            "failed": status["failed"],
            "error": status["error"],
            "build": status["source"],  # "jit" or "prebuilt" once loaded
        },
        {
            "beam_selection.cuda_calls": ran,
            "beam_selection.fallbacks": sum(calls.values()) - ran,
        },
    )


def taken_records():
    """The request and step times gathered since the last call."""
    with _records:
        records = {"requests": dict(_requests), "steps": list(_steps)}
        _requests.clear()
        _steps.clear()
    return records


def profile(serving, start):
    import torch

    global _capturing
    if not PROFILE:
        raise ValueError("profiling requires BENCH_PROFILE=requests or steps")
    with _idle(serving):
        device = _executor(serving).engine.model.embed_tokens.weight.device
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            cudart = torch.cuda.cudart()
            code = cudart.cudaProfilerStart() if start else cudart.cudaProfilerStop()
        if int(code) != 0:
            raise RuntimeError(
                f"cudaProfiler{'Start' if start else 'Stop'} returned {code}"
            )
        _capturing = start
    return {"capturing": _capturing}


# ---- always-on hooks: request and step times ----------------------------------
def request_range(original):
    """The same hook for a handler that runs on its own thread and for one the
    ASGI server awaits on its event loop."""

    @contextlib.contextmanager
    def handled(payload):
        rid = payload.get("rid") or payload.get("request_id")
        started = time.perf_counter_ns()
        try:  # requests overlap under load: no request window there, but inflight
            with nvtx("request", rid if MARKS == "requests" else None), inflight(rid):
                yield
        finally:
            if rid is not None:
                with _records:
                    _requests[str(rid)] = [started, time.perf_counter_ns()]

    if inspect.iscoroutinefunction(original):

        async def awaited(self, payload, **options):
            with handled(payload):
                return await original(self, payload, **options)

        return awaited

    def wrapped(self, payload):
        with handled(payload):
            return original(self, payload)

    return wrapped


def step_times(decode):
    def make(original):
        def wrapped(self, *args):
            # _run_prefill(request_ids) or _run_decode_batch(decode_batch, states)
            ids = args[0].request_ids if decode else args[0]
            started = time.perf_counter_ns()
            try:
                return original(self, *args)
            except Exception as exc:
                # gr-inference-internal abandons a mixed-step decode batch before
                # it selects beams and reruns its requests as one batch per step;
                # those reruns are the requests' steps.
                if type(exc).__name__ == "_MixedStepDecodeFallback":
                    ids = ()
                raise
            finally:
                step = {
                    "kind": "decode" if decode else "prefill",
                    "requests": [str(rid) for rid in ids],
                    "rows": len(ids) * args[0].beam_width if decode else len(ids),
                    "tokens": None,
                    "scheduled_ns": started,
                    "launched_ns": started,
                    "finished_ns": time.perf_counter_ns(),
                }
                if ids:  # _run_prefill returns at once without requests
                    with _records:
                        _steps.append(step)

        return wrapped

    return make


# ---- profile-only hooks ------------------------------------------------------
_windows = iter(range(1, 1 << 62))  # step window numbers


def phase_range(decode):
    """NVTX phase per request (prefill, decode_<k>) or, marking steps, a step window
    with its kind (prefill, decode) as the phase."""

    def make(original):
        def wrapped(self, *args):
            ids = args[0].request_ids if decode else args[0]
            if MARKS == "steps":
                if not ids:  # _run_prefill returns at once without requests
                    return original(self, *args)
                unit, name = str(next(_windows)), "decode" if decode else "prefill"
                window = nvtx("step", unit)
            else:
                unit = ",".join(map(str, ids))
                name = f"decode_{args[0].step + 1}" if decode else "prefill"
                window = contextlib.nullcontext()
            previous, _scope.rid = getattr(_scope, "rid", None), unit
            try:
                with window, nvtx("phase", f"{unit}:{name}"):
                    return original(self, *args)
            finally:
                _scope.rid = previous

        return wrapped

    return make


def named_section(original):  # _profile_section(name) returns a context manager
    @contextlib.contextmanager
    def wrapped(self, name):
        with original(self, name), nvtx("section", section_label(name)):
            yield

    return wrapped


def section(label):
    def make(original):
        def wrapped(self, *args, **kwargs):
            with nvtx("section", section_label(label)):
                return original(self, *args, **kwargs)

        return wrapped

    return make


# ---- launch --------------------------------------------------------------------
applied: list[Patch] = []


def features():
    """The code paths of this engine revision that PATCHES entries need: "asgi"
    when its HTTP server is ASGI (GRHTTPServingAdapter.handle_async)."""
    from gr_inference.gr_serving.http import GRHTTPServingAdapter

    return frozenset({"asgi"} if hasattr(GRHTTPServingAdapter, "handle_async") else ())


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: sid_gr_server.py .../tools/serve_qwen3_gr_http.py [args]"
        )
    entrypoint = Path(sys.argv[1]).resolve()
    if entrypoint.name != "serve_qwen3_gr_http.py":
        # A package probe or install from the script: run it as python3 would.
        os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
    # Running this file puts bench/engines/ first on sys.path; drop it so a
    # module there, such as sglang.py, cannot shadow one SID-GR imports.
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.path[:0] = [str(entrypoint.parent), str(entrypoint.parent.parent / "src")]
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    applied.extend(apply(patches(), features(), PROFILE))
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
