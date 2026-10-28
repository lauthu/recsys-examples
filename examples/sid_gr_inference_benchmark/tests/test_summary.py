# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import re
import xml.etree.ElementTree as ET

from bench.outputs import chart, summary
from tests.test_report import LOAD_TRACE, SVG, at, levels, saved, section

A, B, C, D = ([n, 2, 3] for n in (1, 2, 3, 4))


def beams(*paths):
    return [(path, -float(i)) for i, path in enumerate(paths)]


def test_identical_outputs_agree_completely():
    outputs = {"r": beams(A, B, C)}
    assert summary.parity(outputs, outputs) == {
        "top1": 1.0,
        "overlap": 1.0,
        "overlap_min": 1.0,
        "rank": 1.0,
        "score_delta": 0.0,
    }


def test_parity_separates_order_membership_and_score():
    reference = {"r": beams(A, B, C, D), "s": beams(A, B, C, D)}
    candidate = {
        "r": [(B, -0.5), (A, -1.0), (C, -2.0), ([9, 9, 9], -3.0)],  # swap and replace
        "s": beams(A, B, C, D),
    }
    found = summary.parity(reference, candidate)
    assert found["top1"] == 0.5  # request r starts with B
    assert found["overlap"] == (0.75 + 1.0) / 2  # three of four beams shared
    assert found["overlap_min"] == 0.75  # the worst request, r
    assert found["rank"] == (0.25 + 1.0) / 2  # only C is still at its rank
    assert found["score_delta"] == 1.0  # A: -0.0 versus -1.0


def with_outputs(engine, paths, mean_ms, concurrency=(1,)):
    """mean_ms: one latency for every level, or {level: latency}."""
    found = at(saved(engine), *concurrency)
    per_level = (
        mean_ms if isinstance(mean_ms, dict) else dict.fromkeys(concurrency, mean_ms)
    )
    for run in found["rounds"]:
        run["levels"] = [levels(k, latency=[per_level[k]] * 4)[0] for k in concurrency]
    found["outputs"] = {str(k): {"r": beams(*paths)} for k in concurrency}
    return found


def text_of(everything, names, reference):
    return summary.render(everything, names, reference)["summary.md"]


def test_summary_tables_each_engine_against_the_reference():
    reference = with_outputs("sid-gr", [A, B], 10.0)
    other = with_outputs("sglang-beam-search-new", [B, A], 15.0)
    names = ["sid_gr", "sglang_beam_search_new"]
    text = text_of([reference, other], names, "sid_gr")
    links = "[sid_gr](sid_gr/report.md) and "
    assert links + "[sglang_beam_search_new](sglang_beam_search_new/report.md)" in text
    header = (
        "| Engine | Throughput (requests/s) | Mean latency (ms) | p50 (ms) | p95 (ms) |"
    )
    assert header + " p99 (ms) | Mean vs sid_gr |" in text
    # Every request takes the same time, so each interval is a single point.
    row = "| sid_gr | 80.0 (80.0–80.0) | 10.0 (10.0–10.0) | 10.0 | 10.0 | 10.0 |"
    assert row + " reference |" in text
    assert "| +50.0% (+50.0% to +50.0%) |" in text
    assert "### Concurrency" not in text  # one level, one table
    assert "## Output agreement with sid_gr" in text
    assert "| sglang_beam_search_new | 0.0% | 100.0% (100.0%) | 0.0% | 1 |" in text


def test_the_change_against_the_reference_is_paired_by_request():
    # Latencies vary from request to request, so each engine's own interval has
    # width; the candidate takes exactly twice as long on every request, so the
    # paired change is +100% in every resample.
    reference = with_outputs("sglang", [A, B], 10.0)
    candidate = with_outputs("vedantjh2-sglang", [A, B], 10.0)
    for ref_run, run, offset in zip(reference["rounds"], candidate["rounds"], (0, 5)):
        latency = [10.0 + offset + i for i in range(20)]
        ref_run["levels"] = levels(1, latency=latency)
        run["levels"] = levels(1, latency=[2 * ms for ms in latency])
    text = text_of([reference, candidate], ["sglang", "vedantjh2"], "sglang")
    assert "| +100.0% (+100.0% to +100.0%) |" in text
    mean = re.search(
        r"\| vedantjh2 \| [\d.]+ \([\d.–]+\) \| [\d.]+ \(([\d.]+)–([\d.]+)\)", text
    )
    assert float(mean[1]) < float(mean[2])  # the engine's own mean is uncertain


def test_the_reference_can_be_any_engine():
    first = with_outputs("sid-gr", [A, B], 10.0)
    second = with_outputs("sglang", [B, A], 15.0)
    text = text_of([first, second], ["sid_gr", "sglang"], "sglang")
    assert "## Output agreement with sglang" in text
    # sid_gr is faster
    assert "| -33.3% (-33.3% to -33.3%) |" in text and " reference |" in text
    assert "| sid_gr | SID-GR | example/engine `cccccccccccc` |" in text


def test_summary_with_several_levels_tables_and_charts_each():
    first = with_outputs("sglang", [A, B], {1: 10.0, 4: 20.0}, concurrency=(1, 4))
    second = with_outputs("vedantjh2-sglang", [B, A], 20.0, concurrency=(1, 4))
    files = summary.render([first, second], ["sglang", "vedantjh2"], "sglang")
    text = files["summary.md"]
    assert "### Concurrency 1" in text and "### Concurrency 4" in text
    # Against the reference at the same level: +100% at 1, +0% at 4.
    assert text.count("| +100.0% (+100.0% to +100.0%) |") == 1
    assert "| +0.0% (+0.0% to +0.0%) |" in section(text, "### Concurrency 4")
    assert "![Throughput and latency by concurrency](load.svg)" in text
    assert "| vedantjh2 | 4 | 0.0% | 100.0% (100.0%) | 0.0% | 1 |" in text
    root = ET.fromstring(files["load.svg"].split("\n", 1)[1])
    assert len(root.findall(f".//{SVG}polyline")) == 6  # 2 engines, 3 panels
    assert "vedantjh2 at concurrency 4: 20.0 ms" in files["load.svg"]


def test_one_level_has_no_chart():
    files = summary.render([with_outputs("sglang", [A, B], 10.0)], ["sglang"], "sglang")
    assert set(files) == {"summary.md"}
    assert "load.svg" not in files["summary.md"]


def test_summary_splits_the_latency_and_tables_the_steps():
    first = with_outputs("sglang", [A, B], {1: 10.0, 4: 20.0}, concurrency=(1, 4))
    second = with_outputs("vedantjh2-sglang", [B, A], 20.0, concurrency=(1, 4))
    text = text_of([first, second], ["sglang", "vedantjh2"], "sglang")
    part = section(text, "## Where the time goes")
    columns = "| Engine | Concurrency | Queue (ms) | Prefill (ms) | Decode (ms) |"
    assert columns + " HTTP and response (ms) | Latency (ms) |" in part
    assert "| sglang | 4 | 1.0 | 4.0 | 4.0 | 11.0 | 20.0 |" in part
    # 4 requests a run, each with one 4 ms prefill and two 2 ms decode steps, in 50 ms.
    assert "| sglang | 4 | 4.0 | 1.00 | 2.0 | 1.00 | 64% |" in section(
        text, "## Batching"
    )


def test_gpu_table_adds_each_requests_gpu_time_under_load():
    result = {"levels": [{"concurrency": 4, "latency_ms": [30.0] * 8, "steps": []}]}
    first = with_outputs("sglang", [A, B], 10.0, concurrency=(1, 4))
    first["load"] = {"trace": LOAD_TRACE, "result": result}
    second = with_outputs("vedantjh2-sglang", [B, A], 20.0, concurrency=(1, 4))
    unmarked = {k: v for k, v in LOAD_TRACE.items() if k != "inflight"}
    second["load"] = {"trace": unmarked, "result": result}  # an older capture
    text = text_of([first, second], ["sglang", "vedantjh2"], "sglang")
    part = section(text, "## GPU time per request")
    columns = "a request's time in the server (ms) | Kernels running in it (ms) |"
    assert f"| At concurrency 4: {columns} No GPU activity in it (ms) |" in part
    # Each request's own 30 ms, not the kernel time over the requests served
    assert "| sglang | 6.0 | 60.0% | 30.0 | 22.0 | 7.0 |" in part
    assert "| vedantjh2 | 6.0 | 60.0% | – | – | – |" in part
    assert '"–": not captured' in part


def test_the_gpu_table_names_the_profiled_concurrency():
    result = {"levels": [{"concurrency": 2, "latency_ms": [30.0] * 8, "steps": []}]}
    found = with_outputs("sglang", [A, B], 10.0, concurrency=(1, 2, 4))
    runs = dataclasses.replace(found["scenario"].runs, profile_concurrency=2)
    found["scenario"] = dataclasses.replace(found["scenario"], runs=runs)
    found["load"] = {"trace": LOAD_TRACE, "result": result}
    part = section(text_of([found], ["sglang"], "sglang"), "## GPU time per request")
    assert "| At concurrency 2: a request's time in the server (ms) |" in part
    assert "| sglang | 6.0 | 60.0% | 30.0 | 22.0 | 7.0 |" in part


def variable(engine, offset):
    """Two rounds at concurrency 1 and 4 with prompts of 100, 200, 300 and 500
    tokens, answered in 10, 12, 14 and 16 ms plus offset."""
    found = with_outputs(engine, [A, B], 10.0, concurrency=(1, 4))
    workload = dataclasses.replace(
        found["scenario"].workload, context_length=500, min_context_length=100
    )
    found["scenario"] = dataclasses.replace(found["scenario"], workload=workload)
    found["lengths"] = [100, 200, 300, 500]
    for run in found["rounds"]:
        latency = [ms + offset for ms in (10.0, 12.0, 14.0, 16.0)]
        run["levels"] = levels(1, 4, latency=latency)
    return found


def test_summary_tables_and_charts_latency_by_prompt_length():
    everything = [variable("sglang", 0.0), variable("vedantjh2-sglang", 5.0)]
    files = summary.render(everything, ["sglang", "vedantjh2"], "sglang")
    part = section(files["summary.md"], "## Latency by prompt length")
    columns = "| Concurrency | Prompt tokens | Requests per round |"
    assert columns + " sglang (ms) | vedantjh2 (ms) |" in part
    assert "| 1 | 100–199 | 1 | 10.0 | 15.0 |" in part
    assert "| 4 | 400–500 | 1 | 16.0 | 21.0 |" in part
    assert "![Latency by prompt length at concurrency 1](lengths.svg)" in part
    svg = files["lengths.svg"]
    assert "vedantjh2 with 400–500-token prompts: 21.0 ms" in svg
    root = ET.fromstring(svg.split("\n", 1)[1])
    assert len(root.findall(f".//{SVG}polyline")) == 2  # one line per engine
    fixed = summary.render([with_outputs("sglang", [A, B], 10.0)], ["sglang"], "sglang")
    assert "lengths.svg" not in fixed and "by prompt length" not in fixed["summary.md"]


def test_chart_assigns_colors_in_order_and_never_cycles():
    series = {
        f"e{i}": [(10.0 + i, None, None), (15.0 + i, None, None)] for i in range(8)
    }
    svg = chart.lines("t", "s", [1, 2], [("p", "ms", series)])
    assert 'class="line l7"' in svg and "l8" not in svg
    series["e8"] = series["e0"]  # a ninth engine has no slot of its own
    assert chart.lines("t", "s", [1, 2], [("p", "ms", series)]) is None


def test_chart_styles_define_every_slot_in_light_and_dark():
    light, dark = chart.STYLE.split("@media (prefers-color-scheme: dark)")
    assert chart.SLOTS == 8
    for i in range(chart.SLOTS):
        assert f".l{i} {{" in light and f".l{i} {{" in dark
        assert f".d{i} {{" in light and f".d{i} {{" in dark


def test_chart_ticks_are_clean():
    assert chart.ticks(9.3) == [0, 2, 4, 6, 8, 10]
    assert chart.ticks(43.0)[-1] == 50


def test_charts_make_room_for_their_title_and_subtitle():
    # SVG text doesn't wrap: a long line widens the chart instead of running past
    # its edge, and short ones keep the usual width.
    series = {"e": [(1.0, None, None), (2.0, None, None)]}
    subtitle = (
        "Qwen3-0.6B · prompts of 256 to 2,048 tokens · 256 beams · 3 output tokens"
    )
    title = (
        "SGLang beam-search-new, with a much longer name: where a request's time goes"
    )

    def width(svg):
        return float(ET.fromstring(svg.split("\n", 1)[1]).get("width"))

    assert width(chart.lines("t", "s", [1, 2], [("p", "ms", series)])) == 504
    wide = chart.lines("t", subtitle, [1, 2], [("p", "ms", series)])
    assert width(wide) - 60 - 24 >= 0.55 * 12 * len(subtitle)  # from x=60, 12 px
    assert width(chart.stacked("t", "s", "ms", "ms", ["x"], [("r", [1.0])])) == 760
    bars = chart.stacked(title, "s", "ms", "ms", ["x"], [("r", [1.0])])
    assert width(bars) - 116 - 24 >= 0.6 * 16 * len(title)  # from x=116, 16 px bold


def test_charts_are_valid_svg_whatever_their_text():
    svg = chart.stacked(
        "a <b> & c", "s", "ms per request", "ms", ["x", "y"], [("r", [1.0, 2.0])]
    )
    root = ET.fromstring(svg.split("\n", 1)[1])
    assert root.get("aria-label") == "a <b> & c"
    assert "r: y 2.0 ms (67%)" in svg and root.find(f"{SVG}style") is not None
