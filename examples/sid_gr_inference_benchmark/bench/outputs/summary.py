# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""summary.md and its chart, across the engines measured in one invocation,
from their saved results; templates/summary.md.j2 holds its words and tables.

All engines receive identical requests at every concurrency level, so beams are
compared request by request with a reference engine (public SGLang by default),
at the same concurrency, using the first round of each.
"""

from bench import engines
from bench.config import load_concurrency
from bench.outputs import chart, template
from bench.outputs.report import (
    PARTS,
    RESAMPLES,
    interval,
    length_rows,
    level_records,
    parts,
    phase_means,
    scope,
    step_summary,
    summaries,
)


def parity(reference, candidate):
    """Compare {request_id: [(tokens, score), ...]} outputs of two engines."""
    top1, overlap, rank, delta = [], [], [], 0.0
    for rid, expected in reference.items():
        actual = candidate[rid]
        ref_tokens = [tuple(tokens) for tokens, _ in expected]
        got_tokens = [tuple(tokens) for tokens, _ in actual]
        top1.append(ref_tokens[0] == got_tokens[0])
        overlap.append(len(set(ref_tokens) & set(got_tokens)) / len(ref_tokens))
        rank.append(
            sum(a == b for a, b in zip(ref_tokens, got_tokens)) / len(ref_tokens)
        )
        scores = {tuple(tokens): score for tokens, score in actual}
        for tokens, score in expected:
            if tuple(tokens) in scores:
                delta = max(delta, abs(score - scores[tuple(tokens)]))
    n = len(top1)
    return {
        "top1": sum(top1) / n,
        "overlap": sum(overlap) / n,
        "overlap_min": min(overlap),
        "rank": sum(rank) / n,
        "score_delta": delta,
    }


def change(level, base):
    """(relative change of the mean against the reference's at the same level,
    low, high): the 95% interval comes from the paired bootstrap, since both
    engines' resamples draw the same requests."""
    ratios = [a / b - 1 for a, b in zip(level["means"], base["means"])]
    return (level["stats"]["mean"] / base["stats"]["mean"] - 1, *interval(ratios))


def level_view(everything, names, reference):
    """Per concurrency level, each engine's statistics, parts and steps."""
    ref = everything[names.index(reference)]
    levels = []
    for k in ref["scenario"].workload.levels:
        base, rows = summaries(ref)[k], []
        for name, results in zip(names, everything):
            found = summaries(results)[k]
            records = level_records(results["rounds"])[k]
            steps = step_summary(records)
            rows.append(
                {
                    "name": name,
                    **found["stats"],
                    "rate": found["rate"],
                    "ci": found["ci"],
                    "change": None if name == reference else change(found, base),
                    "parts": parts(phase_means(records)),
                    "steps": steps,
                    "busy": sum(step["share"] for step in steps.values()),
                }
            )
        levels.append({"k": k, "engines": rows})
    return levels


def length_view(everything):
    """Mean latency per prompt-length bin at each concurrency, one value per
    engine; None with one prompt length. Every engine gets the same prompts,
    so the bins hold the same requests."""
    per_engine = [length_rows(results) for results in everything]
    if per_engine[0] is None:
        return None
    rows = []
    for i, k in enumerate(everything[0]["scenario"].workload.levels):
        for b, first in enumerate(per_engine[0]):
            rows.append(
                {
                    "k": k,
                    "low": first["low"],
                    "high": first["high"],
                    "requests": first["requests"],
                    "means": [engine[b]["means"][i] for engine in per_engine],
                }
            )
    return rows


def gpu_rows(everything, names):
    """Kernel time per request of each engine's captures."""
    rows = []
    for name, results in zip(names, everything):
        trace, load = results["trace"], results["load"]
        loaded = None
        if load:  # amortized over the requests the steps served
            served = len(load["result"]["levels"][0]["latency_ms"])
            loaded = load["trace"]["kernel_ns"] / served / 1e6
        rows.append(
            {
                "name": name,
                "single": trace["kernel_ns"] / trace["windows"] / 1e6,
                "idle_share": trace["busy_ns"]["idle"] / trace["window_ns"],
                "loaded": loaded,
            }
        )
    return rows


def view(everything, names, reference):
    """Everything summary.md shows, as plain numbers and names; everything
    holds each engine's saved results, in command-line order."""
    first = everything[0]
    workload = first["scenario"].workload
    ref = everything[names.index(reference)]
    agreement = []
    for name, results in zip(names, everything):
        for k in workload.levels:
            found = parity(ref["outputs"][str(k)], results["outputs"][str(k)])
            agreement.append({"name": name, "k": k, **found})
    return {
        "engines": [
            {"name": name, "title": engines.get(results["solution"].engine).title}
            for name, results in zip(names, everything)
        ],
        "reference": reference,
        "model": first["model"],
        "dtype": first["rounds"][0]["runtime"]["dtype"],
        "gpu": first["rounds"][0]["gpu"],
        "driver": first["rounds"][0]["driver"],
        "workload": workload,
        "levels": workload.levels,
        "load_concurrency": load_concurrency(first["scenario"]),
        "runs": first["scenario"].runs,
        "catalog": first["catalog"],
        "resamples": RESAMPLES,
        "results": level_view(everything, names, reference),
        "lengths": length_view(everything),
        "gpu_rows": gpu_rows(everything, names),
        "agreement": agreement,
        "versions": [
            {
                "name": name,
                "title": engines.get(results["solution"].engine).title,
                "source": results["manifest"]["source"],
                "image": results["manifest"]["image"]["id"],
            }
            for name, results in zip(names, everything)
        ],
        "model_pin": first["manifest"]["model"],
    }


def load_chart(found):
    """Throughput, mean latency and p99 against concurrency, one line per
    engine; None with one level or more engines than color slots."""
    levels = found["results"]
    if len(levels) < 2:
        return None
    names = [engine["name"] for engine in found["engines"]]

    def series(value):
        return {
            name: [value(level["engines"][i]) for level in levels]
            for i, name in enumerate(names)
        }

    return chart.lines(
        "Throughput and latency by concurrency",
        scope(found),
        found["levels"],
        [
            (
                "Throughput (requests/s)",
                "requests/s",
                series(lambda row: (row["rate"], *row["ci"]["rate"])),
            ),
            (
                "Mean latency (ms)",
                "ms",
                series(lambda row: (row["mean"], *row["ci"]["mean"])),
            ),
            ("p99 latency (ms)", "ms", series(lambda row: (row["p99"], None, None))),
        ],
    )


def length_chart(found):
    """Mean latency against prompt length at the lowest concurrency, one line
    per engine, where length matters most; None with one prompt length or
    more engines than color slots."""
    rows = found["lengths"]
    if not rows:
        return None
    k = found["levels"][0]
    rows = [row for row in rows if row["k"] == k]
    labels = [f"{row['low']:,}–{row['high']:,}" for row in rows]
    series = {
        engine["name"]: [(row["means"][i], None, None) for row in rows]
        for i, engine in enumerate(found["engines"])
    }
    return chart.lines(
        f"Latency by prompt length at concurrency {k}",
        scope(found),
        labels,
        [("Mean latency (ms)", "ms", series)],
        axis="Prompt tokens",
        where="with {}-token prompts",
    )


def render(everything, names, reference):
    """{file name: text} of summary.md and its chart."""
    found = view(everything, names, reference)
    files = {}
    for name, draw in (("load.svg", load_chart), ("lengths.svg", length_chart)):
        svg = draw(found)
        if svg:
            files[name] = svg
    text = template.render("summary.md.j2", **found, charts=files.keys(), parts=PARTS)
    return {"summary.md": text, **files}
