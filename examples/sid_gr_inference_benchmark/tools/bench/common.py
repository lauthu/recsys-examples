# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constants and helpers shared by the host runner and the container client."""

import json
import math
import statistics
from pathlib import Path

ENGINES = ("sid-gr", "sglang-beam-search-new", "sglang-sidtrie")
# NVTX ranges: bench/request:<rid>, bench/phase:<rid>:<phase>, bench/section:<rid>:<name>
MARKER = "bench"
SIDTRIE_FILE = "trie_output_head.safetensors"


def is_sglang(engine):
    return engine.startswith("sglang")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Write atomically so an interrupted run never leaves truncated JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def quantile(values, q):
    """Linear interpolation between order statistics at position (n - 1) * q."""
    ordered = sorted(values)
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
