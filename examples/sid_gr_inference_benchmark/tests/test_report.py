# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
import re
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


def test_report_tables_the_confidence_intervals():
    ctx = context("sglang")
    for result, latency in zip(ctx["clean"], ([10.0] * 4, [20.0] * 4)):
        result["levels"] = levels(1, latency=latency)
    text = report.render(ctx)
    section = text.split("### 95% confidence intervals")[1].split("## GPU profile")[0]
    assert "| Mean ms | p95 ms | p99 ms | Requests/s |" in section
    assert "| 15.000 [10.00, 20.00] | 20.000 [10.00, 20.00] |" in section
    assert "mean 15.000 ms (95% CI 10.00 to 20.00)" in text
    assert "2,000 resamples, each drawing the server processes" in section
    assert "which 2 processes show only roughly" in section


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


def context(engine, **overrides):
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
    scenario = SimpleNamespace(
        workload=Workload(
            context_length=2048, beam_width=256, output_tokens=3, concurrency=1, seed=42
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
    }
    return {**ctx, **overrides}


def at(ctx, workload, *concurrency):
    """ctx measured at these concurrency levels under this workload."""
    ctx["scenario"] = SimpleNamespace(workload=workload, runs=ctx["scenario"].runs)
    for result in ctx["clean"]:
        result["levels"] = levels(*concurrency)
    return ctx


def test_sid_gr_report_with_a_disabled_graph_runner():
    text = report.render(context("sid-gr"))
    summary = (
        r"mean 13\.000 ms \(95% CI \d+\.\d\d to \d+\.\d\d\), p99 16\.000 ms over 8"
    )
    assert re.search(summary, text)
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
    assert "15 kernel launches and 2 CUDA graph launches" in text
    assert "| `cutlass::Kernel2<gemm_bf16>(T1::Params)` | linear | 4.000 | 20 |" in text
    assert "gatherTopK<float>(a \\| b)" in text  # a pipe must not split the row


def test_vedantjh2_report_states_observed_paths_and_catalog():
    catalog = {"name": "items.jsonl", "items": 300, "sha256": "f" * 64}
    text = report.render(context("vedantjh2-sglang", catalog=catalog))
    assert "prefill: 8 of 8 forwards as graph replays" in text
    assert "decode: 0 of 16 forwards as graph replays" in text
    assert "shared-context attention on" in text
    assert "all 20 returned beams are catalog items" in text
    assert "catalog of 300 items" in text
    assert "| Trie output head (observed) | compact trie output head" in text
    text = report.render(context("vedantjh2-sglang"))
    row = (
        "| Trie output head (observed) | off: no catalog; shared-context attention on |"
    )
    assert row in text


def test_report_states_the_observed_overlap_scheduler():
    ctx = context("sglang")
    for result in ctx["clean"]:
        result["runtime"] = {**result["runtime"], "overlap": True}
    assert "| Overlap scheduler (observed) | on |" in report.render(ctx)
    assert "Overlap scheduler" not in report.render(context("sid-gr"))


def test_report_states_the_requests_in_flight():
    text = report.render(context("sglang"))
    assert "80.00 requests/s at concurrency 1." in text
    assert "; requests go one at a time." in text
    assert "decode: 0 of 16 forwards as graph replays, 256 rows each" in text
    ctx = context("sglang")
    workload = dataclasses.replace(ctx["scenario"].workload, concurrency=4)
    text = report.render(at(ctx, workload, 4))
    assert "at concurrency 4." in text
    assert "1 warmup + 4 measured requests per run, 4 in flight |" in text
    assert "4 requests are in flight, as each of 4 clients sends its next" in text
    assert "one at a time after 1 warmups, so it shows one request's GPU time" in text


def test_report_tables_every_concurrency_level():
    ctx = context("sglang")
    workload = dataclasses.replace(ctx["scenario"].workload, concurrency=[2, 1])
    text = report.render(at(ctx, workload, 1, 2))
    assert (
        "- **Latency and throughput** over 2 server processes × 4 requests per level"
        in text
    )
    found = (
        r"concurrency 1: mean 13\.000 ms \(95% CI [\d.]+ to [\d.]+\), p99 16\.000 ms, "
    )
    assert re.search(found + r"80\.00 requests/s; concurrency 2:", text)
    assert "| Concurrency | Mean ms | p95 ms | p99 ms | Requests/s |" in text
    assert "| Concurrency | Run | Requests | Mean ms |" in text
    assert (
        "| 1 | clean_1 | 4 | 13.000 |" in text
        and "| 2 | **All** | 8 | 13.000 |" in text
    )
    assert "measured requests per run and level, 1 and 2 in flight |" in text
    assert "each server process measures concurrency 1 and 2 in turn" in text
    assert "so it shows one request's GPU time" in text


def test_report_shows_latency_by_prompt_length():
    ctx = context("sglang", lengths=[100, 200, 300, 500])
    workload = dataclasses.replace(
        ctx["scenario"].workload, context_length=500, min_context_length=100
    )
    text = report.render(at(ctx, workload, 1))
    assert "Qwen3-0.6B · prompts of 100 to 500 tokens · beam width 256" in text
    assert "prompt lengths drawn uniformly from 100 to 500 tokens;" in text
    section = text.split("### Latency by prompt length")[1].split("## GPU profile")[0]
    assert "| Prompt tokens | Requests | Mean ms at concurrency 1 |" in section
    assert (
        "| 100–199 | 1 | 10.000 |" in section and "| 400–500 | 1 | 16.000 |" in section
    )
    # The profiled requests are the first measured ones.
    assert (
        "the capture holds 2 requests sent one at a time after 1 warmups (prompts of 100 and 200 tokens)"
        in text
    )


def test_report_prints_the_pins_that_reproduce_the_run():
    text = report.render(context("sid-gr"))
    row = "| Model | Qwen/Qwen3-0.6B @ `eeeeeeeeeeee`; float32 checkpoint, weights "
    assert row + "bfloat16 (observed) |" in text
    assert "https://github.com/example/engine @ `cccccccccccc`" in text
    assert f"`sha256:{'1' * 64}`" in text and "nightly-1" in text
    assert "`images/Dockerfile.engine`" in text
    assert "Qwen/Qwen3-0.6B @ `eeeeeeeeeeee`" in text
    assert "NVIDIA Nsight Systems version 2026.4" in text
    assert '"benchmark_code": "abc123"' in text  # the JSON copy


def test_report_shows_the_solution_settings_and_what_the_benchmark_adds():
    text = report.render(context("vedantjh2-sglang"))
    solution, added = text.split("Added by the benchmark")
    assert "```yaml\nenv:\n  SGLANG_BEAM_SHARED_CONTEXT_ATTENTION: 'true'\n" in solution
    assert "server_settings:\n  page_size: 1\n```" in solution
    assert "env:\n  BENCH_PROFILE: '0'\nserver_settings:\n  dtype: bfloat16\n" in added
    assert "scenario `111111111111`, solution `222222222222`" in text


def test_report_holds_the_full_configuration_the_server_reported():
    text = report.render(context("sglang"))
    section = text.split("Full server configuration (observed)</summary>")[1]
    section = section.split("</details>")[0]
    assert "Settings (ServerArgs.resolved_dict()):" in section
    assert "\n\nnone\n" in section  # nothing set differs; PORT is skipped
    assert "policy: fcfs" in section and "HF_TOKEN: redacted" in section
    assert '- `env.BENCH_PROFILE`: `"0"` → `"1"`' in section
    assert '- `env.CUDA_INJECTION64_PATH`: `unset` → `"/nsys/inject.so"`' in section
    assert "```bash\nnsys profile taskset python3 server\n```" in section
    assert "settings.port" not in section and "HOSTNAME" not in section


def test_report_lists_settings_the_engine_changed():
    clean = context("sglang")["clean"]
    changed = {**CONFIG, "settings": {**SETTINGS, "page_size": 4}}
    clean = [{**run, "config": changed} for run in clean]
    text = report.render(context("sglang", clean=clean))
    assert "| `server_settings.page_size` | `1` | `4` |" in text
    del changed["settings"]["dtype"]
    text = report.render(context("sglang", clean=clean))
    assert '| `server_settings.dtype` | `"bfloat16"` | `unset` |' in text


def test_report_lists_the_observed_server_patches():
    text = report.render(context("sglang"))
    assert "Changes to the engine's own HTTP server (observed)" in text
    assert "latency runs patched:\n\n- `m:Runner.forward`\n\nThe profiled run" in text
    assert "The profiled run also patched:\n\n- `m:Batch.run`" in text


def test_report_flags_an_unenforced_pin():
    unpinned = {
        **MANIFEST["source"],
        "enforced": False,
        "differences": ["python differs"],
    }
    text = report.render(context("sid-gr", manifest={**MANIFEST, "source": unpinned}))
    assert "pin not enforced: python differs" in text


def test_report_splits_the_latency_and_tables_the_steps():
    text = report.render(context("sglang"))
    section = text.split("## Where the time goes")[1].split("## GPU profile")[0]
    assert (
        "| Outside server ms | Queue ms | Prefill ms | Decode ms | Return ms | Latency ms |"
        in section
    )
    assert "| 1.000 | 1.000 | 4.000 | 4.000 | 3.000 | 13.000 |" in section
    # 2 runs x 4 requests: 8 prefill steps of 4 ms, 16 decode steps of 2 ms, in 2 x 50 ms.
    assert (
        "| prefill | 8 | 4.000 | 4.000 | 0.500 | 1.00 | 1.0 | 2,048.0 | 32.0% |"
        in section
    )
    assert (
        "| decode | 16 | 2.000 | 2.000 | 0.200 | 1.00 | 256.0 | – | 32.0% |" in section
    )
    assert "a decode step of k requests has k × 256 rows" in section


LOAD_TRACE = {
    "windows": 3,
    "window_ns": 30_000_000,
    "span_ns": 40_000_000,
    "kernel_ns": 24_000_000,
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


def loaded(ctx, load):
    """ctx measured at concurrency 1 and 4, with this capture under load."""
    workload = dataclasses.replace(ctx["scenario"].workload, concurrency=[1, 4])
    at(ctx, workload, 1, 4)
    ctx["load"] = load
    return ctx


def test_report_shows_the_gpu_profile_under_load():
    steps = [["prefill", 4, 4, 8192, 0.5, 10.0]] + [
        ["decode", 4, 1024, None, 0.2, 10.0]
    ] * 2
    result = {"levels": [{"concurrency": 4, "latency_ms": [30.0] * 4, "steps": steps}]}
    text = report.render(
        loaded(context("sglang"), {"trace": LOAD_TRACE, "result": result})
    )
    section = text.split("## GPU profile under load")[1].split("## Setup")[0]
    assert "at concurrency 4: each of 4 clients" in section
    assert "the capture holds 3 steps. Trace: [load_profile.nsys-rep]" in section
    # Kernel ms each over the step kind's steps, per request over the 4 requests.
    assert "| prefill | 1 | 4.00 | 10.000 | 8.000 | 2.000 | 1.000 |" in section
    assert "| decode | 2 | 4.00 | 10.000 | 8.000 | 4.000 | 1.000 |" in section
    assert (
        "6.000 ms in all at concurrency 4, against 6.000 ms one request at a time"
        in section
    )
    assert "The steps took 30.000 ms of the 40.000 ms" in section
    assert "no activity for 3.000 ms, and 10.000 ms passed between steps" in section
    assert "| decode | model | 8.000 | 1.000 |" in section
    assert "| 30.000 | 10.000 | between steps |" in section


def test_report_says_why_there_is_no_profile_under_load():
    text = report.render(loaded(context("sglang"), None))
    assert "Not captured: the overlap scheduler launches each step" in text
    assert "GPU profile under load" not in report.render(context("sglang"))
