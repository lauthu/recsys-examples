# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constants and helpers shared by the host runner and the container client."""

import http.client
import json
import math
import statistics
from pathlib import Path

# NVTX ranges: bench/request:<rid>, bench/phase:<rid>:<phase>, bench/section:<rid>:<name>
MARKER = "bench"
# JIT and model caches; the runner points each at a per-image directory.
CACHE_ENV = """XDG_CACHE_HOME HF_HOME TORCH_HOME TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR
    TORCHINDUCTOR_CACHE_DIR CUDA_CACHE_PATH FLASHINFER_WORKSPACE_BASE TVM_FFI_CACHE_DIR
    CUTE_DSL_CACHE_DIR QUACK_CACHE_DIR SGLANG_CACHE_DIR SGLANG_JIT_CACHE_DIR
    NUMBA_CACHE_DIR MPLCONFIGDIR""".split()


def fail(message):
    raise ValueError(message)


def call(port, method, path, body=None, timeout=60, text=False):
    """HTTP request to a local server; returns parsed JSON, or the raw body if text."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        connection.request(method, path, data, headers)
        response = connection.getresponse()
        payload = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"{method} {path}: HTTP {response.status} {payload[:300]!r}")
    return payload if text else json.loads(payload)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, indent=2):
    """Write atomically so an interrupted run never leaves truncated JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=indent, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def quantile(values, q):
    """Linear interpolation between order statistics at position (n - 1) * q."""
    return interpolate(sorted(values), q)


def interpolate(ordered, q):
    """quantile() of values already sorted ascending."""
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values):
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }
