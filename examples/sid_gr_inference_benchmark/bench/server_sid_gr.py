#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run SID-GR's own HTTP server with the benchmark's declared patches.

scripts/serve_qwen3_gr_http.sh execs $PYTHON_BIN; the runner points it here:

    server_sid_gr.py .../tools/serve_qwen3_gr_http.py [args]

Compared with the native server (serve_qwen3_gr_http.sh with plain python3):

Same
  - The server: serve_qwen3_gr_http.py from the mounted source, run unchanged in
    this process with runpy, with the arguments and environment the script
    builds. Requests go to SID-GR's own handlers; there is no proxy or extra hop.

Different, and nothing else
  - PATCHES: the methods listed there. Hooks follow the rules in
    bench/patching.py: original arguments and results, no GPU sync.
  - The one "always" entry adds four routes and leaves every other route to
    the original dispatcher (about 85 ns per request):
        GET  /pilot/runtime        packages, attention backends, CUDA graph counters
        GET  /pilot/config         the server's build_info and environment, read
                                   once after the measured requests
        POST /pilot/profile/start  cudaProfilerStart (BENCH_PROFILE=1 only)
        POST /pilot/profile/stop   cudaProfilerStop
  - "profile" entries exist only in the profiled process (BENCH_PROFILE=1) and
    push NVTX ranges bench/request|phase|section only while a capture is active.

The server log lists every applied patch at startup.
"""

import contextlib
import importlib.metadata
import os
import runpy
import sys
import threading
from collections import Counter
from pathlib import Path

# Only bench.patching: more modules in this process put a ~220 ms stall into one
# round of measured requests at concurrency 4 (seen when it imported bench.adapters).
from bench.patching import (  # PYTHONPATH has /benchmark-tools
    Patch,
    apply,
    configuration,
)

PROFILE = os.environ.get("BENCH_PROFILE") == "1"
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
            "NVTX request range",
            profile_only=True,
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
    return {
        "pid": os.getpid(),
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
            "decode.eager_fallbacks": decode_status.get(
                "decode_cuda_graph_fallback_eager_count"
            ),
        },
        "patches": [patch.target for patch in applied],
    }


def profile(serving, start):
    import torch

    global _capturing
    if not PROFILE:
        raise ValueError("profiling requires BENCH_PROFILE=1")
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


# ---- profile-only hooks ------------------------------------------------------
def request_range(original):
    def wrapped(self, payload):
        with nvtx("request", payload.get("rid") or payload.get("request_id")):
            return original(self, payload)

    return wrapped


def phase_range(decode):
    def make(original):
        def wrapped(self, *args):
            ids = args[0].request_ids if decode else args[0]
            rid = ",".join(map(str, ids))
            name = f"decode_{args[0].step + 1}" if decode else "prefill"
            previous, _scope.rid = getattr(_scope, "rid", None), rid
            try:
                with nvtx("phase", f"{rid}:{name}"):
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


def main():
    entrypoint = Path(sys.argv[1]).resolve()
    if entrypoint.name != "serve_qwen3_gr_http.py":
        raise SystemExit(
            "usage: server_sid_gr.py .../tools/serve_qwen3_gr_http.py [args]"
        )
    # Running this file puts bench/ first on sys.path; drop it so bench/trace.py
    # or bench/config.py cannot shadow a module SID-GR imports.
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.path[:0] = [str(entrypoint.parent), str(entrypoint.parent.parent / "src")]
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    applied.extend(apply(patches(), frozenset(), PROFILE))  # no entry needs a feature
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
