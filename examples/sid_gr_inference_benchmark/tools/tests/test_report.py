# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import xml.etree.ElementTree as ET

import pytest
from bench import report, sunburst
from bench.common import quantile, summarize

WORK = [
    ["prefill", "model", "linear", 6_000_000],
    ["prefill", "model", "attention", 3_000_000],
    ["decode_1", "model", "linear", 2_000_000],
    ["decode_2", "beam_selection", "selection", 1_000_000],
    ["unknown", "unknown", "other", 0],
]
TRACE = {
    "requests": 2,
    "window_ns": 30_000_000,
    "kernel_ns": 12_000_000,
    "kernels": 40,
    "unknown_kernels": 0,
    "work": WORK,
    "busy_ns": {
        "kernel": 11_000_000,
        "overlap": 1_000_000,
        "memory": 0,
        "idle": 18_000_000,
    },
}


def test_statistics():
    assert quantile([1, 2, 3, 4, 5], 0.95) == pytest.approx(4.8)
    assert summarize([4.0, 1.0, 3.0, 2.0]) == {
        "n": 4,
        "mean": 2.5,
        "p50": 2.5,
        "p95": pytest.approx(3.85),
        "p99": pytest.approx(3.97),
        "max": 4.0,
    }


def test_sunburst_is_valid_svg_with_one_slice_per_node():
    svg = sunburst.render(WORK, 2, "title", "subtitle")
    root = ET.fromstring(svg.split("\n", 1)[1])
    paths = root.findall("{http://www.w3.org/2000/svg}path")
    # 3 phases, 3 sections and 4 families; the zero-time unknown phase is dropped
    assert len(paths) == 3 + 3 + 4
    assert "6.000 ms" in svg  # 12 ms of kernel time over 2 requests


def context(engine, **overrides):
    runtime = {
        "packages": {"torch": "2.9.1"},
        "attention": {"prefill": "A x28", "decode": "B"},
        "cuda_graphs": {"prefill": True, "decode": False},
    }
    clean = [
        {
            "run": f"clean_{i}",
            "latency_ms": [10.0, 12.0, 14.0, 16.0],
            "duration_s": 0.05,
            "gpu": "GPU",
            "driver": "1.0",
            "cpu": "CPU",
            "runtime": runtime,
            "catalog_beams": 8,
            "counters": {
                "prefill.graph_replays": 4,
                "forward.prefill.graph": 4,
                "forward.prefill.eager": 0,
                "forward.decode.graph": 0,
                "forward.decode.eager": 8,
            },
        }
        for i in (1, 2)
    ]
    config = {
        "engine": engine,
        "env": {"SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "true"},
        "workload": {
            "context_length": 2048,
            "beam_width": 256,
            "output_tokens": 3,
            "seed": 42,
        },
        "runs": {"clean": 2, "requests": 4, "warmups": 1, "profile_requests": 2},
    }
    ctx = {
        "config": config,
        "clean": clean,
        "profile": {**clean[0], "catalog_beams": 4},
        "trace": TRACE,
        "catalog": None,
        "model": "Qwen3-0.6B",
        "source": "repo abc",
        "tools": "abc",
        "image": "sha256:1",
        "cpus": {"server": "0-9", "client": "10"},
        "command": "python3 tools/run_benchmark.py",
    }
    return {**ctx, **overrides}


def test_sid_gr_report_with_a_disabled_graph_runner():
    text = report.render(context("sid-gr"))
    assert "mean 13.000 ms, p99 16.000 ms over 8 requests" in text
    assert (
        "| clean_1 | 4 | 13.000 | 13.000 | 15.700 | 15.940 | 16.000 | 80.00 |" in text
    )
    assert (
        "| **All** | 8 | 13.000 | 13.000 | 16.000 | 16.000 | 16.000 | 80.00 |" in text
    )
    assert "prefill: 1.00 replays/request; decode: disabled" in text
    assert "| prefill | model | 4.500 | 75.0% |" in text
    assert "| No GPU activity | 9.000 | 60.0% |" in text


def test_sidtrie_report_states_observed_paths_and_catalog():
    catalog = {"name": "items.jsonl", "items": 300, "sha256": "f" * 64}
    text = report.render(context("sglang-sidtrie", catalog=catalog))
    assert "prefill: 8 of 8 forwards as graph replays" in text
    assert "decode: 0 of 16 forwards as graph replays" in text
    assert "shared-context attention on" in text
    assert "all 20 returned beams are catalog items" in text
    assert "catalog of 300 items" in text
