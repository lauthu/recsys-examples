# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest
from bench import report, sunburst
from bench.common import quantile, summarize
from bench.config import Runs, Solution, Source, Workload

WORK = [
    ["prefill", "model", "linear", 6_000_000],
    ["prefill", "model", "attention", 3_000_000],
    ["decode_1", "model", "linear", 2_000_000],
    ["decode_2", "beam_selection", "topk", 1_000_000],
    ["unknown", "unknown", "other", 0],
]
GAPS = [
    ["decode_1", "model", "idle", 3_000_000],
    ["no phase", "no section", "idle", 15_000_000],
]
TRACE = {
    "requests": 2,
    "window_ns": 30_000_000,
    "kernel_ns": 12_000_000,
    "kernels": 40,
    "unknown_kernels": 0,
    "work": WORK,
    "gaps": GAPS,
    "top": [
        ["void cutlass::Kernel2<gemm_bf16>(T1::Params)", "linear", 8_000_000, 40],
        ["void at::native::mbtopk::gatherTopK<float>(a | b)", "topk", 1_000_000, 4],
    ],
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
    svg = sunburst.render(WORK, GAPS, 2, "title", "subtitle")
    root = ET.fromstring(svg.split("\n", 1)[1])
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    # 4 phases, 4 sections and 6 families; the zero-time unknown phase is dropped
    assert len(paths) == 4 + 4 + 6
    assert "15.000 ms" in svg  # 12 ms of kernel and 18 ms of idle time over 2 requests
    assert "decode_1 / model / no GPU activity: 1.500 ms/request" in svg


def embedded(svg):
    root = ET.fromstring(svg.split("\n", 1)[1])
    script = root.find("{http://www.w3.org/2000/svg}script").text
    data = script.split("var DATA = ", 1)[1].split(", NS = ", 1)[0]
    return script, json.loads(data)


def test_sunburst_embeds_the_tree_its_script_zooms_through():
    script, data = embedded(sunburst.render(WORK, GAPS, 2, "title", "<subtitle>"))
    assert data["requests"] == 2 and data["total"] == 30_000_000
    phases = data["tree"]
    names = [phase["n"] for phase in phases]
    assert names == ["prefill", "decode_1", "decode_2", "no phase"]
    # Every node's time is the sum of its children's, so a zoomed view adds up.
    for phase in phases:
        assert phase["v"] == sum(c["v"] for c in phase["c"])
        for section in phase["c"]:
            assert section["v"] == sum(c["v"] for c in section["c"])
    assert "mouseenter" in script and "Escape" in script


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_sunburst_script_is_valid_javascript(tmp_path):
    script, _ = embedded(sunburst.render(WORK, GAPS, 2, "title", "subtitle"))
    (tmp_path / "script.js").write_text(script)
    check = subprocess.run(["node", "--check", str(tmp_path / "script.js")])
    assert check.returncode == 0


MANIFEST = {
    "source": {
        "url": "https://github.com/example/engine",
        "commit": "c" * 40,
        "head": "d" * 40,
        "enforced": True,
        "differences": [],
    },
    "image": {
        "id": "sha256:" + "1" * 64,
        "tags": ["bench:engine"],
        "repo_digests": [],
        "labels": {"org.opencontainers.image.version": "nightly-1"},
    },
    "model": {
        "id": "Qwen/Qwen3-0.6B",
        "revision": "e" * 40,
        "sha256": {"config.json": "f" * 64},
    },
    "benchmark_code": "abc123",
    "recipe": "images/Dockerfile.engine",
    "configs": {"scenario": "1" * 64, "solution": "2" * 64},
    "nsys": "NVIDIA Nsight Systems version 2026.4",
}


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
    scenario = SimpleNamespace(
        workload=Workload(
            context_length=2048, beam_width=256, output_tokens=3, seed=42
        ),
        runs=Runs(clean=2, requests=4, warmups=1, profile_requests=2),
    )
    solution = Solution(
        engine=engine,
        source=Source(url="https://github.com/example/engine", commit="c" * 40),
        env={"SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "true"},
        server_settings=None if engine == "sid-gr" else {"page_size": 1},
    )
    ctx = {
        "scenario": scenario,
        "solution": solution,
        "clean": clean,
        "profile": {**clean[0], "catalog_beams": 4},
        "trace": TRACE,
        "catalog": None,
        "model": "Qwen3-0.6B",
        "manifest": MANIFEST,
        "cpus": {"server": "0-9", "client": "10"},
        "command": "python3 run_benchmark.py",
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
    assert "| prefill | model | 4.500 | 0.000 | 4.500 | 30.0% |" in text
    assert "| decode_1 | model | 1.000 | 1.500 | 2.500 | 16.7% |" in text
    assert "| no phase | no section | 0.000 | 7.500 | 7.500 | 50.0% |" in text
    assert "| **Total** |  | 6.000 | 9.000 | 15.000 | 100.0% |" in text
    assert "| No GPU activity | 9.000 | 60.0% |" in text
    assert "| `cutlass::Kernel2<gemm_bf16>(T1::Params)` | linear | 4.000 | 20 |" in text
    assert "gatherTopK<float>(a \\| b)" in text  # a pipe must not split the row


def test_sidtrie_report_states_observed_paths_and_catalog():
    catalog = {"name": "items.jsonl", "items": 300, "sha256": "f" * 64}
    text = report.render(context("sglang-sidtrie", catalog=catalog))
    assert "prefill: 8 of 8 forwards as graph replays" in text
    assert "decode: 0 of 16 forwards as graph replays" in text
    assert "shared-context attention on" in text
    assert "all 20 returned beams are catalog items" in text
    assert "catalog of 300 items" in text


def test_report_prints_the_pins_that_reproduce_the_run():
    text = report.render(context("sid-gr"))
    assert "https://github.com/example/engine @ `cccccccccccc`" in text
    assert f"`sha256:{'1' * 64}`" in text and "nightly-1" in text
    assert "`images/Dockerfile.engine`" in text
    assert "Qwen/Qwen3-0.6B @ `eeeeeeeeeeee`" in text
    assert "NVIDIA Nsight Systems version 2026.4" in text
    assert '"benchmark_code": "abc123"' in text  # the JSON copy


def test_report_copies_the_solution_settings_as_yaml():
    text = report.render(context("sglang-sidtrie"))
    assert "```yaml\nenv:\n  SGLANG_BEAM_SHARED_CONTEXT_ATTENTION: 'true'\n" in text
    assert "server_settings:\n  page_size: 1\n```" in text
    assert "scenario `111111111111`, solution `222222222222`" in text


def test_report_flags_an_unenforced_pin():
    unpinned = {
        **MANIFEST["source"],
        "enforced": False,
        "differences": ["python differs"],
    }
    text = report.render(context("sid-gr", manifest={**MANIFEST, "source": unpinned}))
    assert "pin not enforced: python differs" in text
