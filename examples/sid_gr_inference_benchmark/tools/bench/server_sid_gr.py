#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PYTHON_BIN for scripts/serve_qwen3_gr_http.sh: the native SID-GR server plus
benchmark endpoints.

GET  /pilot/runtime        packages, resolved attention backends, CUDA graph counters
POST /pilot/profile/start  cudaProfilerStart (BENCH_PROFILE=1 only)
POST /pilot/profile/stop   cudaProfilerStop

With BENCH_PROFILE=1, NVTX ranges bench/request|phase|section are pushed only
while a capture is active. They mark CPU launch scopes and never synchronize.
"""

import contextlib
import functools
import importlib.metadata
import os
import runpy
import sys
import threading
from collections import Counter
from pathlib import Path

PROFILE = os.environ.get("BENCH_PROFILE") == "1"
PACKAGES = (
    "torch triton flashinfer-python nvidia-cutlass-dsl quack-kernels transformers"
)
_capturing = False
_scope = threading.local()  # request IDs of the phase running on this thread


def _executor(adapter):
    facade = adapter.facade
    return getattr(facade, "facade", facade).executor


def _idle(adapter):
    """Hold the worker lock so snapshots and profiler calls never overlap a request."""
    return getattr(adapter.facade, "_lock", contextlib.nullcontext())


def _class(value):
    return type(value).__name__ if value is not None else None


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime(adapter):
    with _idle(adapter):
        executor = _executor(adapter)
        backends = Counter()
        for layer in executor.engine.model.layers:
            backend = layer.prefill_runner.attention._backend
            backends[_class(getattr(backend, "_backend", None) or backend)] += 1
        decode = executor.engine.decode_engine.attention
        decode = getattr(decode, "backend", None) or getattr(decode, "_backend", None)
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
        "cuda_graphs": {
            "prefill": prefill_graph is not None,
            "decode": decode_graph is not None,
        },
        "counters": {
            "prefill.graph_replays": prefill_status.get("prefill_cuda_graph_replays"),
            "decode.graph_replays": decode_status.get("decode_cuda_graph_replays"),
            "decode.eager_fallbacks": decode_status.get(
                "decode_cuda_graph_fallback_eager_count"
            ),
        },
    }


def profile(adapter, start):
    import torch

    global _capturing
    if not PROFILE:
        raise ValueError("profiling requires BENCH_PROFILE=1")
    with _idle(adapter):
        device = _executor(adapter).engine.model.embed_tokens.weight.device
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


def wrap(owner, name, make):
    original = getattr(owner, name)
    setattr(owner, name, functools.wraps(original)(make(original)))


def install_markers(adapter_class, executor_class):
    def request(original):
        def wrapped(self, payload):
            with nvtx("request", payload.get("rid") or payload.get("request_id")):
                return original(self, payload)

        return wrapped

    def phase(original, decode):
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

    wrap(adapter_class, "_handle_sglang_generate", request)
    wrap(executor_class, "_run_prefill", lambda f: phase(f, decode=False))
    wrap(executor_class, "_run_decode_batch", lambda f: phase(f, decode=True))
    wrap(executor_class, "_profile_section", named_section)
    wrap(executor_class, "_forward_prefill", section("model.forward_prefill"))
    initialize = section("prefill.initialize_generation")
    wrap(executor_class, "_store_single_prefill_result", initialize)


def install():
    from gr_inference.gr_serving.continuous import GRContinuousServingExecutor
    from gr_inference.gr_serving.http import GRHTTPResponse, GRHTTPServingAdapter

    def dispatch(original):
        def wrapped(self, method, route, payload):
            if (method, route) == ("GET", ("pilot", "runtime")):
                return GRHTTPResponse(200, runtime(self))
            if method == "POST" and route in (
                ("pilot", "profile", "start"),
                ("pilot", "profile", "stop"),
            ):
                return GRHTTPResponse(200, profile(self, route[2] == "start"))
            return original(self, method, route, payload)

        return wrapped

    wrap(GRHTTPServingAdapter, "_dispatch", dispatch)
    if PROFILE:
        install_markers(GRHTTPServingAdapter, GRContinuousServingExecutor)


def main():
    entrypoint = Path(sys.argv[1]).resolve()
    if entrypoint.name != "serve_qwen3_gr_http.py":
        raise SystemExit(
            "usage: server_sid_gr.py .../tools/serve_qwen3_gr_http.py [args]"
        )
    sys.path[:0] = [str(entrypoint.parent), str(entrypoint.parent.parent / "src")]
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    install()
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
