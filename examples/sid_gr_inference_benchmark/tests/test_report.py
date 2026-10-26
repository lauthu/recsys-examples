# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import jinja2
import pytest
from bench import report, results, sunburst, template
from bench.common import quantile, summarize
from bench.config import Model, Runs, Scenario, Solution, Source, Workload

ROOT = Path(__file__).resolve().parents[1]
SVG = "{http://www.w3.org/2000/svg}"

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
    "windows": 2,
    "window_ns": 30_000_000,
    "kernel_ns": 12_000_000,
    "kernels": 40,
    "launches": {"kernel": 30, "graph": 4},
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


def test_bootstrap_is_seeded_and_covers_the_estimate():
    runs = [("clean_1", [10.0, 12.0, 14.0, 16.0], 0.05)] * 2
    values = report.bootstrap(runs, seed="s", resamples=500)
    assert values == report.bootstrap(runs, seed="s", resamples=500)
    low, high = report.interval(values["mean"])
    assert 10.0 < low < 13.0 < high < 16.0
    assert report.interval(values["rate"]) == (80.0, 80.0)  # equal durations


def test_bootstrap_includes_the_variation_between_processes():
    # Each process answers every request in the same time, but the two differ:
    # resampling only the requests would give a zero-width interval.
    runs = [("clean_1", [10.0] * 50, 1.0), ("clean_2", [20.0] * 50, 2.0)]
    values = report.bootstrap(runs, seed="s")
    assert report.interval(values["mean"]) == (10.0, 20.0)
    assert report.interval(values["p99"]) == (10.0, 20.0)
    assert report.interval(values["rate"]) == (25.0, 50.0)  # 50 per 2 s .. per 1 s


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


ENVIRONMENT = {
    "BENCH_PROFILE": "0",
    "SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "true",
    "HF_TOKEN": "redacted",
}
SETTINGS = {"page_size": 1, "dtype": "bfloat16", "port": 30001, "policy": "fcfs"}
CONFIG = {
    "source": "ServerArgs.resolved_dict()",
    "settings": SETTINGS,
    "environment": ENVIRONMENT,
}
PROFILED = {
    **CONFIG,
    "settings": {**SETTINGS, "port": 30002},  # per run, not listed
    "environment": {
        **ENVIRONMENT,
        "HOSTNAME": "other-container",
        "BENCH_PROFILE": "1",
        "CUDA_INJECTION64_PATH": "/nsys/inject.so",
    },
}


def levels(*concurrency, latency=(10.0, 12.0, 14.0, 16.0)):
    """A run's measured levels, each with the same latencies: 1 ms outside the
    server, 1 ms queued, 4 ms prefill and 4 ms decode, the rest returning; one
    4 ms prefill step and two 2 ms decode steps per request."""
    n = len(latency)
    phases = {"outside": [1.0] * n, "queue": [1.0] * n, "prefill": [4.0] * n}
    phases.update(decode=[4.0] * n, **{"return": [ms - 10.0 for ms in latency]})
    steps = [["prefill", 1, 1, 2048, 0.5, 4.0]] + [
        ["decode", 1, 256, None, 0.2, 2.0]
    ] * 2
    return [
        {
            "concurrency": k,
            "latency_ms": list(latency),
            "duration_s": 0.05,
            "phases_ms": phases,
            "steps": steps * n,
        }
        for k in concurrency
    ]


SCENARIO = Scenario(
    model=Model(
        id="Qwen/Qwen3-0.6B",
        revision="e" * 40,
        sha256={"config.json": "f" * 64},
        dtype="bfloat16",
    ),
    workload=Workload(
        context_length=2048, beam_width=256, output_tokens=3, concurrency=1, seed=42
    ),
    runs=Runs(clean=2, requests=4, warmups=1, profile_requests=2),
)


def saved(engine, **overrides):
    """One engine's saved results (bench/results.py), measured at concurrency 1."""
    runtime = {
        "packages": {"torch": "2.9.1"},
        "attention": {"prefill": "A x28", "decode": "B"},
        "dtype": "bfloat16",
        "cuda_graphs": {"prefill": True, "decode": False},
        "patches": ["m:Runner.forward"],
    }
    clean = [
        {
            "run": f"clean_{i}",
            "order": [1],
            "levels": levels(1),
            "gpu": "GPU",
            "driver": "1.0",
            "cpu": "CPU",
            "runtime": runtime,
            "config": CONFIG,
            "server_command": ["taskset", "--cpu-list", "0", "python3", "server"],
            "catalog_beams": 8,
            "counters": {
                "prefill.graph_replays": 4,
                "forward.prefill.graph": 4,
                "forward.prefill.eager": 0,
                "forward.decode.graph": 0,
                "forward.decode.eager": 8,
                "forward.decode.rows": 2048,
            },
        }
        for i in (1, 2)
    ]
    solution = Solution(
        engine=engine,
        source=Source(url="https://github.com/example/engine", commit="c" * 40),
        env={"SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "true"},
        server_settings=None if engine == "sid-gr" else {"page_size": 1},
    )
    found = {
        "name": engine,
        "solutions": [engine],
        "reference": engine,
        "scenario": SCENARIO,
        "solution": solution,
        "clean": clean,
        "profile": {
            **clean[0],
            "catalog_beams": 4,
            "runtime": {**runtime, "patches": ["m:Runner.forward", "m:Batch.run"]},
            "config": PROFILED,
            "server_command": ["nsys", "profile", "taskset", "python3", "server"],
        },
        "trace": TRACE,
        "load": None,
        "catalog": None,
        "model": "Qwen3-0.6B",
        "checkpoint_dtype": "float32",
        "manifest": MANIFEST,
        "cpus": {"server": "0-9", "client": "10"},
        "added": {
            "env": {"BENCH_PROFILE": "0"},
            "server_settings": {"dtype": "bfloat16", "port": "PORT"},
        },
        "command": "python3 run_benchmark.py",
        "lengths": [2048] * 4,
        "outputs": {"1": {"r": [[[1, 2, 3], -0.5]]}},
    }
    return {**found, **overrides}


def at(found, *concurrency, **changes):
    """found measured at these concurrency levels, its workload also changed by
    changes."""
    workload = dataclasses.replace(
        found["scenario"].workload,
        concurrency=list(concurrency) if len(concurrency) > 1 else concurrency[0],
        **changes,
    )
    found["scenario"] = dataclasses.replace(found["scenario"], workload=workload)
    for result in found["clean"]:
        result["levels"] = levels(*concurrency)
    return found


def text_of(found):
    return report.render(found)["report.md"]


def section(text, start, end="\n## "):
    """The text from the heading start to the next heading at level end."""
    return text.split(start, 1)[1].split(end, 1)[0]


LOAD_TRACE = {
    "windows": 3,
    "window_ns": 30_000_000,
    "span_ns": 40_000_000,
    "kernel_ns": 24_000_000,
    "busy_ns": {
        "kernel": 24_000_000,
        "overlap": 0,
        "memory": 3_000_000,
        "idle": 3_000_000,
    },
    "work": [
        ["decode", "model", "attention", 16_000_000],
        ["prefill", "model", "linear", 8_000_000],
    ],
    "gaps": [
        ["decode", "model", "idle", 2_000_000],
        ["prefill", "no section", "idle", 1_000_000],
    ],
    "idle": [[30_000_000, 10_000_000, "between steps"]],
}
LOAD_STEPS = [["prefill", 4, 4, 8192, 0.5, 10.0]] + [
    ["decode", 4, 1024, None, 0.2, 10.0]
] * 2


def under_load(found):
    """found measured at concurrency 1 and 4, with a capture under load at 4."""
    result = {
        "levels": [{"concurrency": 4, "latency_ms": [30.0] * 4, "steps": LOAD_STEPS}],
        "server_command": ["nsys", "profile", "python3", "server", "--steps"],
    }
    found = at(found, 1, 4)
    found["load"] = {"trace": LOAD_TRACE, "result": result}
    return found


def test_report_opens_with_how_it_was_measured():
    text = text_of(saved("sglang"))
    assert text.startswith("# SGLang (public) · beam-search benchmark\n\n")
    scope = (
        "Qwen3-0.6B, bfloat16 · 2,048-token prompts · 256 beams · 3 output tokens · GPU"
    )
    assert f"\n{scope}\n" in text
    method = section(text, "## How it was measured")
    assert "At concurrency *k*, *k* clients each send a request" in method
    assert "Measured at concurrency 1." in method
    assert "8 per concurrency level: 4 in each of 2 separate server starts." in method
    assert "Each level begins with 1 unmeasured warmup request." in method
    assert "(2,000 resamples)." in method
    assert "from one more server start under Nsight Systems" in method


def test_report_tables_throughput_and_latency_with_intervals():
    found = saved("sglang")
    for result, latency in zip(found["clean"], ([10.0] * 4, [20.0] * 4)):
        result["levels"] = levels(1, latency=latency)
    part = section(text_of(found), "## Results")
    header = "| Concurrency | Throughput (requests/s) | Mean latency (ms) | p50 (ms) |"
    assert header in part
    # Each server start answers in its own constant time: only resampling the
    # starts moves the mean, from one's to the other's.
    assert "| 1 | 80.0 (80.0–80.0) | 15.0 (10.0–20.0) | 15.0 | 20.0 | 20.0 |" in part
    assert "| 1 | 2 | 4 | 80.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |" in part
    assert "load.svg" not in part  # one level, no line to draw


def test_report_charts_throughput_and_latency_by_concurrency():
    files = report.render(at(saved("sglang"), 1, 2))
    text = files["report.md"]
    assert "Measured at concurrency 1 and 2." in text
    assert "each of which measures every level in turn" in text
    assert "![Throughput and latency by concurrency](load.svg)" in text
    assert re.search(
        r"\| 2 \| 80\.0 \(80\.0–80\.0\) \| 13\.0 \([\d.]+–[\d.]+\) \|", text
    )
    assert "measures the levels in a different order" in text
    root = ET.fromstring(files["load.svg"].split("\n", 1)[1])
    assert len(root.findall(f".//{SVG}polyline")) == 4  # throughput; mean, p95, p99
    assert (
        "Throughput at concurrency 1: 80.0 requests/s (95% interval"
        in files["load.svg"]
    )
    assert "p99 at concurrency 2: 16.0 ms<" in files["load.svg"]


def test_report_splits_the_latency_and_tables_the_steps():
    files = report.render(saved("sglang"))
    part = section(files["report.md"], "## Where the time goes")
    columns = "| Queue (ms) | Prefill (ms) | Decode (ms) | HTTP and response (ms) | Latency (ms) |"
    assert columns in part
    # 1 ms outside the server and 3 ms returning make HTTP and response.
    assert "| 1 | 1.0 | 4.0 | 4.0 | 4.0 | 13.0 |" in part
    # 2 runs x 4 requests: 8 prefill steps of 4 ms, 16 decode steps of 2 ms, in 2 x 50 ms.
    assert "| 1 | 4.0 | 1.00 | 2.0 | 1.00 | 64% |" in part
    assert (
        "| 1 | prefill | 8 | 4.00 | 4.00 | 0.50 | 1.00 | 1.0 | 2,048.0 | 32.0% |"
        in part
    )
    assert "| 1 | decode | 16 | 2.00 | 2.00 | 0.20 | 1.00 | 256.0 | – | 32.0% |" in part
    assert "*k* × 256 rows" in part
    assert "Concurrency 1: HTTP and response 4.0 ms (31%)" in files["breakdown.svg"]


def test_report_states_how_each_capture_was_taken():
    text = text_of(saved("sid-gr"))
    gpu = section(text, "## GPU profile")
    row = "| One request at a time | 1 | 2 | 6.0 | 60.0% of the server's time |"
    assert row + " [profile.nsys-rep](profile.nsys-rep) |" in gpu
    assert "Under load" not in gpu  # one level
    assert "How the capture was taken:" in gpu
    assert "the client sends 1 warmup request at the capture's concurrency" in gpu
    assert "(`--capture-range=cudaProfilerApi`)" in gpu
    assert "```bash\nnsys profile taskset python3 server\n```" in gpu


def test_report_splits_one_request_by_phase_and_section():
    text = text_of(saved("sid-gr"))
    assert "prefill: 1.00 replays/request; decode: disabled" in text
    assert "| Prefill | 4.50 | 0.00 | 4.50 | 30.0% |" in text
    assert "| Decode step 1 | 1.00 | 1.50 | 2.50 | 16.7% |" in text
    assert "| Outside the phases | 0.00 | 7.50 | 7.50 | 50.0% |" in text
    assert "| **Total** | **6.00** | **9.00** | **15.00** | **100.0%** |" in text
    assert "| Decode step 1 | model | 1.000 | 1.500 | 2.500 | 16.7% |" in text
    assert "| No GPU activity | 9.000 | 60.0% |" in text
    assert "15 kernel launches and 2 CUDA graph launches" in text
    assert "| `cutlass::Kernel2<gemm_bf16>(T1::Params)` | linear | 4.000 | 20 |" in text
    assert "gatherTopK<float>(a \\| b)" in text  # a pipe must not split the row


def test_report_shows_the_gpu_profile_under_load():
    text = text_of(under_load(saved("sglang")))
    gpu = section(text, "## GPU profile")
    row = "| Under load | 4 | 4 | 6.0 | 10.0% of the steps' time |"
    assert row + " [load_profile.nsys-rep](load_profile.nsys-rep) |" in gpu
    assert "How each capture was taken:" in gpu
    assert "4 clients keep 4 requests in flight" in gpu
    assert "over the 4 requests they served" in gpu
    assert "```bash\nnsys profile python3 server --steps\n```" in gpu
    load = section(text, "### Under load, concurrency 4")
    # Kernel ms per step of its kind, and per request over the 4 requests.
    assert "| Prefill | 1 | 4.0 | 10.0 | 8.0 | 1.0 | 2.0 |" in load
    assert "| Decode | 2 | 4.0 | 10.0 | 8.0 | 1.0 | 4.0 |" in load
    assert "| **All steps** | **3** | | | | | **6.0** |" in load
    assert "The 3 steps took 30.0 ms of the 40.0 ms" in load
    assert "no activity for 3.0 ms, and for 10.0 ms no step was running" in load
    assert "| Decode | model | 8.000 | 1.000 |" in load
    assert "| 30.000 | 10.000 | between steps |" in load


def test_report_says_why_there_is_no_profile_under_load():
    text = text_of(at(saved("sglang"), 1, 4))
    assert "### Under load\n\nNot captured: the overlap scheduler launches" in text
    assert "Under load" not in text_of(saved("sglang"))


def test_vedantjh2_report_states_observed_paths_and_catalog():
    catalog = {"name": "items.jsonl", "items": 300, "sha256": "f" * 64}
    text = text_of(saved("vedantjh2-sglang", catalog=catalog))
    assert "prefill: 8 of 8 forwards as graph replays" in text
    assert "decode: 0 of 16 forwards as graph replays" in text
    assert "shared-context attention on" in text
    assert "all 20 returned beams are catalog items" in text
    assert " · catalog of 300 items · " in text
    assert "| Trie output head (observed) | compact trie output head" in text
    text = text_of(saved("vedantjh2-sglang"))
    row = (
        "| Trie output head (observed) | off: no catalog; shared-context attention on |"
    )
    assert row in text


def test_report_states_the_observed_overlap_scheduler():
    found = saved("sglang")
    for result in found["clean"]:
        result["runtime"] = {**result["runtime"], "overlap": True}
    text = text_of(found)
    assert "| Overlap scheduler (observed) | on |" in text
    assert "runs steps side by side, so it can exceed 100%" in text
    assert "Overlap scheduler" not in text_of(saved("sid-gr"))


def test_report_shows_latency_by_prompt_length():
    found = saved("sglang", lengths=[100, 200, 300, 500])
    text = text_of(at(found, 1, context_length=500, min_context_length=100))
    assert "Qwen3-0.6B, bfloat16 · prompts of 100 to 500 tokens · 256 beams" in text
    assert "lengths drawn uniformly from 100 to 500 tokens |" in text
    part = section(text, "### Latency by prompt length")
    assert "| Prompt tokens | Requests per start | Mean at concurrency 1 (ms) |" in part
    assert "| 100–199 | 1 | 10.0 |" in part and "| 400–500 | 1 | 16.0 |" in part
    # The profiled requests are the first measured ones.
    assert "so the times are per request (prompts of 100 and 200 tokens)" in text


def test_report_prints_the_pins_that_reproduce_the_run():
    text = text_of(saved("sid-gr"))
    row = "| Model | Qwen/Qwen3-0.6B @ `eeeeeeeeeeee`; float32 checkpoint, weights "
    assert row + "bfloat16 (observed) |" in text
    assert "https://github.com/example/engine @ `cccccccccccc`" in text
    assert f"`sha256:{'1' * 64}`" in text and "| Image base | nightly-1 |" in text
    assert "| Image recipe | `images/Dockerfile.engine` |" in text
    assert "| Nsight Systems | NVIDIA Nsight Systems version 2026.4 |" in text
    assert '"benchmark_code": "abc123"' in text  # the JSON copy
    assert "scenario `111111111111`, solution `222222222222`" in text


def test_report_shows_the_solution_settings_and_what_the_benchmark_adds():
    text = text_of(saved("vedantjh2-sglang"))
    solution, added = text.split("Added by the benchmark")
    assert "```yaml\nenv:\n  SGLANG_BEAM_SHARED_CONTEXT_ATTENTION: 'true'\n" in solution
    assert "server_settings:\n  page_size: 1\n```" in solution
    assert "env:\n  BENCH_PROFILE: '0'\nserver_settings:\n  dtype: bfloat16\n" in added


def test_report_holds_the_full_configuration_the_server_reported():
    text = text_of(saved("sglang"))
    part = text.split("Full server configuration (observed)</summary>")[1]
    part = part.split("</details>")[0]
    assert "Settings (ServerArgs.resolved_dict()):" in part
    assert "(`PORT` differs per run by design):\nnone.\n" in part
    assert "policy: fcfs" in part and "HF_TOKEN: redacted" in part
    assert "The capture of one request at a time ran in a server start" in part
    assert '- `env.BENCH_PROFILE`: `"0"` → `"1"`' in part
    assert '- `env.CUDA_INJECTION64_PATH`: `unset` → `"/nsys/inject.so"`' in part
    assert "settings.port" not in part and "HOSTNAME" not in part


def test_report_lists_settings_the_engine_changed():
    changed = {**CONFIG, "settings": {**SETTINGS, "page_size": 4}}
    clean = [{**run, "config": changed} for run in saved("sglang")["clean"]]
    text = text_of(saved("sglang", clean=clean))
    assert "| `server_settings.page_size` | `1` | `4` |" in text
    del changed["settings"]["dtype"]
    text = text_of(saved("sglang", clean=clean))
    assert '| `server_settings.dtype` | `"bfloat16"` | `unset` |' in text


def test_report_lists_the_observed_server_patches():
    text = text_of(saved("sglang"))
    assert "Changes to the engine's own HTTP server (observed)" in text
    patched = "The measured starts patched:\n\n- `m:Runner.forward`\n\n"
    assert patched + "The captures also patched:\n\n- `m:Batch.run`\n" in text


def test_report_flags_an_unenforced_pin():
    unpinned = {
        **MANIFEST["source"],
        "enforced": False,
        "differences": ["python differs"],
    }
    text = text_of(saved("sid-gr", manifest={**MANIFEST, "source": unpinned}))
    assert "pin not enforced: python differs" in text


def test_a_missing_value_fails_the_render():
    with pytest.raises(jinja2.UndefinedError):
        template.render("report.md.j2")


def test_saved_results_render_the_same_report(tmp_path):
    results.save(tmp_path, under_load(saved("sglang")))
    found = results.load(tmp_path)
    assert found["scenario"] == under_load(saved("sglang"))["scenario"]
    assert found["solution"] == saved("sglang")["solution"]
    assert report.render(found) == report.render(under_load(saved("sglang")))


def test_old_results_fail_with_their_format(tmp_path):
    results.save(tmp_path, saved("sglang"))
    original = results.FORMAT
    try:
        results.FORMAT = original + 1
        with pytest.raises(ValueError, match="has format 1, not 2"):
            results.load(tmp_path)
    finally:
        results.FORMAT = original


def test_rendering_is_deterministic_across_processes(tmp_path):
    """The same results render the same bytes, whatever the hash seed."""
    results.save(tmp_path, under_load(saved("sglang")))
    script = "import sys; from pathlib import Path; from bench import results; "
    script += "results.render(Path(sys.argv[1]))"
    rendered = []
    for seed in ("1", "2"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        command = [sys.executable, "-c", script, str(tmp_path)]
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        rendered.append({p.name: p.read_bytes() for p in sorted(tmp_path.iterdir())})
    assert rendered[0] == rendered[1]
    names = {
        "results.json.gz",
        "report.md",
        "load.svg",
        "breakdown.svg",
        "sunburst.svg",
    }
    assert set(rendered[0]) == names


def test_the_render_tool_renders_every_engine_and_the_comparison(tmp_path):
    names = ["sglang", "vedantjh2"]
    for name, engine in zip(names, ("sglang", "vedantjh2-sglang")):
        (tmp_path / name).mkdir()
        found = saved(engine, name=name, solutions=names, reference="sglang")
        results.save(tmp_path / name, found)
    tool = [sys.executable, str(ROOT / "tools/render_reports.py")]
    done = subprocess.run(tool + [str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "sglang/report.md").is_file()
    assert (tmp_path / "vedantjh2/sunburst.svg").is_file()
    comparison = (tmp_path / "comparison.md").read_text()
    assert (
        "[sglang](sglang/report.md) and [vedantjh2](vedantjh2/report.md)" in comparison
    )
    missing = subprocess.run(
        tool + [str(tmp_path / "sglang/none")], capture_output=True, text=True
    )
    assert missing.returncode != 0 and "no results.json.gz" in missing.stderr
