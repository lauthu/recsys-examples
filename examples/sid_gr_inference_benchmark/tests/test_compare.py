# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import xml.etree.ElementTree as ET

from bench import chart, compare
from tests.test_report import MANIFEST, at, context, levels

A, B, C, D = ([n, 2, 3] for n in (1, 2, 3, 4))


def beams(*paths):
    return [(path, -float(i)) for i, path in enumerate(paths)]


def test_identical_outputs_agree_completely():
    outputs = {"r": beams(A, B, C)}
    assert compare.parity(outputs, outputs) == {
        "top1": 100.0,
        "overlap": 100.0,
        "overlap_min": 100.0,
        "rank": 100.0,
        "score_delta": 0.0,
    }


def test_parity_separates_order_membership_and_score():
    reference = {"r": beams(A, B, C, D), "s": beams(A, B, C, D)}
    candidate = {
        "r": [(B, -0.5), (A, -1.0), (C, -2.0), ([9, 9, 9], -3.0)],  # swap and replace
        "s": beams(A, B, C, D),
    }
    found = compare.parity(reference, candidate)
    assert found["top1"] == 50.0  # request r starts with B
    assert found["overlap"] == (75.0 + 100.0) / 2  # three of four beams shared
    assert found["overlap_min"] == 75.0  # the worst request, r
    assert found["rank"] == (25.0 + 100.0) / 2  # only C is still at its rank
    assert found["score_delta"] == 1.0  # A: -0.0 versus -1.0


def with_outputs(engine, paths, mean_ms, concurrency=(1,)):
    """mean_ms: one latency for every level, or {level: latency}."""
    ctx = context(engine)
    if len(concurrency) > 1:
        workload = dataclasses.replace(
            ctx["scenario"].workload, concurrency=list(concurrency)
        )
        at(ctx, workload, *concurrency)
    per_level = (
        mean_ms if isinstance(mean_ms, dict) else dict.fromkeys(concurrency, mean_ms)
    )
    for run in ctx["clean"]:
        run["levels"] = [levels(k, latency=[per_level[k]] * 4)[0] for k in concurrency]
    ctx["outputs"] = {str(k): {"r": beams(*paths)} for k in concurrency}
    ctx["manifest"] = MANIFEST
    return ctx


def test_comparison_table_uses_the_first_engine_as_reference():
    reference = with_outputs("sid-gr", [A, B], 10.0)
    other = with_outputs("sglang-beam-search-new", [B, A], 15.0)
    text = compare.render(
        [reference, other], ["sid_gr", "sglang_beam_search_new"], "sid_gr"
    )
    # Every request takes the same time, so each interval is a single point.
    row = "| [sid_gr](sid_gr/report.md) (SID-GR) | 10.000 [10.00, 10.00] | 10.000 |"
    assert row in text and "| 10.000 [10.00, 10.00] | reference |" in text
    assert "| +50.0% [+50.0%, +50.0%] |" in text
    assert "| Engine | Mean ms | p50 ms | p95 ms | p99 ms | Mean vs reference |" in text
    assert "| sglang_beam_search_new | 0.0% | 100.0% (100.0%) | 0.0% |" in text
    assert "Output agreement with sid_gr" in text


def test_the_change_against_the_reference_is_paired_by_request():
    # Latencies vary from request to request, so each engine's own interval has
    # width; the candidate takes exactly twice as long on every request, so the
    # paired change is +100% in every resample.
    reference = with_outputs("sglang", [A, B], 10.0)
    candidate = with_outputs("vedantjh2-sglang", [A, B], 10.0)
    for ref_run, run, offset in zip(reference["clean"], candidate["clean"], (0, 5)):
        latency = [10.0 + offset + i for i in range(20)]
        ref_run["levels"] = levels(1, latency=latency)
        run["levels"] = levels(1, latency=[2 * ms for ms in latency])
    text = compare.render([reference, candidate], ["sglang", "vedantjh2"], "sglang")
    assert "| +100.0% [+100.0%, +100.0%] |" in text
    row = text.split("[vedantjh2](vedantjh2/report.md)")[1].split("\n")[0]
    low, high = (float(x) for x in row.split("[")[1].split("]")[0].split(", "))
    assert low < high  # the engine's own mean is uncertain


def test_the_reference_can_be_any_engine():
    first = with_outputs("sid-gr", [A, B], 10.0)
    second = with_outputs("sglang", [B, A], 15.0)
    text = compare.render([first, second], ["sid_gr", "sglang"], "sglang")
    assert "Output agreement with sglang" in text
    # sid_gr is faster
    assert "| -33.3% [-33.3%, -33.3%] |" in text and "| reference |" in text
    assert "## Versions" in text and "example/engine `cccccccccccc`" in text


def test_comparison_with_several_levels_tables_each_and_charts_them():
    first = with_outputs("sglang", [A, B], {1: 10.0, 4: 20.0}, concurrency=(1, 4))
    second = with_outputs("vedantjh2-sglang", [B, A], 20.0, concurrency=(1, 4))
    names = ["sglang", "vedantjh2"]
    text = compare.render([first, second], names, "sglang")
    assert "### Concurrency 1" in text and "### Concurrency 4" in text
    # Against the reference at the same level: +100% at 1, +0% at 4.
    level_4 = text.split("### Concurrency 4")[1].split("## ")[0]
    assert text.count("| +100.0% [+100.0%, +100.0%] |") == 1
    assert "| +0.0% [+0.0%, +0.0%] |" in level_4
    assert "## Throughput and p99 by concurrency" in text
    assert "| sglang | 80.00 · 10.0 ms | 80.00 · 20.0 ms |" in text
    assert "| vedantjh2 | 4 | 0.0% | 100.0% (100.0%) | 0.0% |" in text
    assert "![p99 latency against throughput](throughput.svg)" in text
    svg = compare.chart_svg([first, second], names)
    root = ET.fromstring(svg.split("\n", 1)[1])
    lines = root.findall(".//{http://www.w3.org/2000/svg}polyline")
    assert len(lines) == 2 and "each line runs through concurrency 1 and 4" in svg
    assert "vedantjh2 · concurrency 4: 80.00 requests/s, p99 20.0 ms" in svg


def test_one_level_has_no_chart():
    ctx = with_outputs("sglang", [A, B], 10.0)
    assert compare.chart_svg([ctx], ["sglang"]) is None
    text = compare.render([ctx], ["sglang"], "sglang")
    assert "Throughput and p99" not in text and "throughput.svg" not in text


def test_chart_assigns_colors_in_order_and_never_cycles():
    curves = {f"e{i}": [(1, 10.0 + i, 20.0), (2, 15.0 + i, 30.0)] for i in range(8)}
    svg = chart.render(curves, "t", "s")
    assert 'class="line l7"' in svg and "l8" not in svg
    curves["e8"] = curves["e0"]  # a ninth engine has no slot of its own
    assert chart.render(curves, "t", "s") is None


def test_chart_ticks_are_clean_and_end_labels_never_collide():
    assert chart.ticks(9.3) == [0, 2, 4, 6, 8, 10]
    assert chart.ticks(43.0)[-1] == 50
    same = [(1, 10.0, 20.0), (2, 15.0, 30.0)]
    svg = chart.render({"a": same, "b": same}, "t", "s")
    assert svg.count('class="label"') == 1  # the legend still names both


def test_comparison_has_a_service_level_table():
    first = with_outputs("sglang", [A, B], {1: 10.0, 4: 20.0}, concurrency=(1, 4))
    second = with_outputs("vedantjh2-sglang", [B, A], 20.0, concurrency=(1, 4))
    text = compare.render([first, second], ["sglang", "vedantjh2"], "sglang")
    section = text.split("## Service level")[1].split("## ")[0]
    assert (
        "| Engine | Concurrency | Requests/s | Queue ms | Prefill ms | Decode ms |"
        in section
    )
    assert (
        "| sglang | 4 | 80.00 [80.00, 80.00] | 1.000 | 4.000 | 4.000 | 4.000 (1.00) | 2.000 (1.00) |"
        in section
    )
    # The request-level tables keep latency only.
    assert "| Engine | Mean ms | p50 ms | p95 ms | p99 ms | Mean vs reference |" in text
