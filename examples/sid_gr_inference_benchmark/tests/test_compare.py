# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from bench import compare
from tests.test_report import MANIFEST, context

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


def with_outputs(engine, paths, mean_ms):
    ctx = context(engine)
    for run in ctx["clean"]:
        run["latency_ms"] = [mean_ms] * 4
    ctx["outputs"] = {"r": beams(*paths)}
    ctx["manifest"] = MANIFEST
    return ctx


def test_comparison_table_uses_the_first_engine_as_reference():
    reference = with_outputs("sid-gr", [A, B], 10.0)
    other = with_outputs("sglang-beam-search-new", [B, A], 15.0)
    text = compare.render(
        [reference, other], ["sid_gr", "sglang_beam_search_new"], "sid_gr"
    )
    assert "| [sid_gr](sid_gr/report.md) (SID-GR) | 10.000 |" in text
    assert "| reference |" in text and "| +50.0% |" in text
    assert "| sglang_beam_search_new | 0.0% | 100.0% (100.0%) | 0.0% |" in text
    assert "Output agreement with sid_gr" in text


def test_the_reference_can_be_any_engine():
    first = with_outputs("sid-gr", [A, B], 10.0)
    second = with_outputs("sglang", [B, A], 15.0)
    text = compare.render([first, second], ["sid_gr", "sglang"], "sglang")
    assert "Output agreement with sglang" in text
    assert "| -33.3% |" in text and "| reference |" in text  # sid_gr is faster
    assert "## Versions" in text and "example/engine `cccccccccccc`" in text
