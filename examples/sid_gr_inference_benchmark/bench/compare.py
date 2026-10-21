# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""comparison.md: one table across the engines measured in a single invocation.

All engines receive identical requests, so beams are compared request by request
with a reference engine (public SGLang by default), using the first clean run of each.
"""

from bench import engines
from bench.common import summarize
from bench.report import table


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
        "top1": 100 * sum(top1) / n,
        "overlap": 100 * sum(overlap) / n,
        "overlap_min": 100 * min(overlap),
        "rank": 100 * sum(rank) / n,
        "score_delta": delta,
    }


def pooled_mean(ctx):
    return summarize([ms for run in ctx["clean"] for ms in run["latency_ms"]])["mean"]


def render(contexts, names, reference):
    """contexts are the per-engine report contexts, in command-line order."""
    ref = contexts[names.index(reference)]
    first = contexts[0]
    workload, catalog = first["scenario"].workload, first["catalog"]
    machine = first["clean"][0]
    scope = (
        f"{first['model']} · context {workload.context_length:,} · beam width "
        f"{workload.beam_width} · {workload.output_tokens} output tokens"
    )
    if catalog:
        scope += f" · catalog of {catalog['items']:,} items"

    base = pooled_mean(ref)
    performance, agreement, versions = [], [], []
    for name, ctx in zip(names, contexts):
        pooled = [ms for run in ctx["clean"] for ms in run["latency_ms"]]
        duration = sum(run["duration_s"] for run in ctx["clean"])
        stats = summarize(pooled)
        trace = ctx["trace"]
        per_request = trace["kernel_ns"] / trace["requests"] / 1e6
        idle = 100 * trace["busy_ns"]["idle"] / trace["window_ns"]
        change = (
            f"{100 * (stats['mean'] / base - 1):+.1f}%"
            if name != reference
            else "reference"
        )
        title = engines.get(ctx["solution"].engine).title
        link = f"[{name}]({name}/report.md)"
        performance.append(
            [
                f"{link} ({title})",
                f"{stats['mean']:.3f}",
                f"{stats['p50']:.3f}",
                f"{stats['p99']:.3f}",
                f"{stats['n'] / duration:.2f}",
                change,
                f"{per_request:.3f}",
                f"{idle:.1f}%",
            ]
        )
        found = parity(ref["outputs"], ctx["outputs"])
        agreement.append(
            [
                name,
                f"{found['top1']:.1f}%",
                f"{found['overlap']:.1f}% ({found['overlap_min']:.1f}%)",
                f"{found['rank']:.1f}%",
                f"{found['score_delta']:.4g}",
            ]
        )

        manifest = ctx["manifest"]
        source, image = manifest["source"], manifest["image"]
        pinned = "yes" if source["enforced"] else "**no**"
        versions.append(
            [
                name,
                f"{source['url'].removeprefix('https://github.com/')} `{source['commit'][:12]}`",
                f"`{image['id'].removeprefix('sha256:')[:12]}`",
                pinned,
            ]
        )

    return f"""# Beam-search engine comparison

{scope} · {machine['gpu']} (driver {machine['driver']})

## Latency and GPU time

{table(["Engine", "Mean ms", "p50 ms", "p99 ms", "Requests/s", "Mean vs reference", "Kernel ms/request", "No GPU activity"], performance, right=range(1, 8))}

Each engine ran the same {first['scenario'].runs.requests} requests over HTTP, {workload.concurrency} in flight, in {first['scenario'].runs.clean} separate server processes; the reference is `{reference}`. Kernel time and GPU idle share come from each engine's separate Nsight run; see its report for the definitions.

## Output agreement with {reference}

{table(["Engine", "Top-1 match", "Beam overlap (worst request)", "Rank match", "Max score Δ"], agreement, right=range(1, 5))}

Compared over the measured requests of each engine's first clean run. Top-1 match: identical best beam. Beam overlap: share of the reference's beams that the engine also returned, in any order, as the mean over requests and, in parentheses, the lowest request. Rank match: share of ranks with identical tokens. Max score Δ: largest cumulative log-probability difference among shared beams; engines normalize over different token sets with a catalog, so it can be nonzero when the beams agree.

## Versions

{table(["Engine", "Source", "Image", "Pins enforced"], versions)}

Model {first['manifest']['model']['id']} @ `{first['manifest']['model']['revision'][:12]}`; the full manifest is in each engine's report.
"""
