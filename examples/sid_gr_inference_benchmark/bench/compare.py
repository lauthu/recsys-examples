# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""comparison.md: tables across the engines measured in a single invocation.

All engines receive identical requests at every concurrency level, so beams are
compared request by request with a reference engine (public SGLang by default),
at the same level, using the first clean run of each.
"""

from bench import adapters, chart
from bench.report import level_runs, listing, pooled, prompt_text, table

CHART = "throughput.svg"  # p99 latency against requests/s, with several levels


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


def levels(ctx):
    """{concurrency: (pooled stats, requests/s)} of one engine, ascending."""
    return {k: pooled(runs) for k, runs in level_runs(ctx["clean"]).items()}


def curves(contexts, names):
    """{engine: [(concurrency, requests/s, p99 ms)]}, the data of the chart."""
    return {
        name: [(k, rate, stats["p99"]) for k, (stats, rate) in levels(ctx).items()]
        for name, ctx in zip(names, contexts)
    }


def chart_svg(contexts, names):
    """throughput.svg with several levels and at most one engine per color slot;
    None otherwise."""
    workload = contexts[0]["scenario"].workload
    if len(workload.levels) < 2:
        return None
    subtitle = (
        f"{prompt_text(workload).capitalize()} · beam width {workload.beam_width}"
    )
    subtitle += f" · each line runs through concurrency {listing(workload.levels)}"
    return chart.render(
        curves(contexts, names), "p99 latency against throughput", subtitle
    )


def render(contexts, names, reference):
    """contexts are the per-engine report contexts, in command-line order."""
    ref = contexts[names.index(reference)]
    first = contexts[0]
    workload, catalog = first["scenario"].workload, first["catalog"]
    machine = first["clean"][0]
    scope = (
        f"{first['model']} · {prompt_text(workload)} · beam width "
        f"{workload.beam_width} · {workload.output_tokens} output tokens"
    )
    if catalog:
        scope += f" · catalog of {catalog['items']:,} items"
    measured = {name: levels(ctx) for name, ctx in zip(names, contexts)}
    several = len(workload.levels) > 1

    latency = []
    for k in workload.levels:
        base = measured[reference][k][0]["mean"]
        rows = []
        for name, ctx in zip(names, contexts):
            stats, rate = measured[name][k]
            change = (
                "reference"
                if name == reference
                else f"{100 * (stats['mean'] / base - 1):+.1f}%"
            )
            title = adapters.get(ctx["solution"].engine).title
            rows.append(
                [
                    f"[{name}]({name}/report.md) ({title})",
                    f"{stats['mean']:.3f}",
                    f"{stats['p50']:.3f}",
                    f"{stats['p99']:.3f}",
                    f"{rate:.2f}",
                    change,
                ]
            )
        headers = ["Engine", "Mean ms", "p50 ms", "p99 ms", "Requests/s"]
        text = table([*headers, "Mean vs reference"], rows, right=range(1, 6))
        latency.append(f"### Concurrency {k}\n\n{text}" if several else text)

    gpu, agreement, versions = [], [], []
    for name, ctx in zip(names, contexts):
        trace = ctx["trace"]
        per_request = trace["kernel_ns"] / trace["requests"] / 1e6
        idle = 100 * trace["busy_ns"]["idle"] / trace["window_ns"]
        gpu.append([name, f"{per_request:.3f}", f"{idle:.1f}%"])
        for k in workload.levels:
            found = parity(ref["outputs"][str(k)], ctx["outputs"][str(k)])
            agreement.append(
                [name]
                + ([k] if several else [])
                + [
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

    throughput, figure = "", ""
    if len(names) <= len(chart.SERIES):
        figure = f"\n\n![p99 latency against throughput]({CHART})"
    if several:
        rows = [
            [name]
            + [
                f"{rate:.2f} · {stats['p99']:.1f} ms"
                for stats, rate in measured[name].values()
            ]
            for name in names
        ]
        columns = ["Engine"] + [f"Concurrency {k}" for k in workload.levels]
        throughput = f"""## Throughput and p99 by concurrency

{table(columns, rows, right=range(1, len(columns)))}

Requests/s · p99 ms at each level. In a closed loop, throughput rises with the clients until the GPU saturates, and from then on only latency grows.{figure}

"""
    agree_headers = ["Engine"] + (["Concurrency"] if several else [])
    agree_headers += ["Top-1 match", "Beam overlap (worst request)", "Rank match"]
    agree_headers += ["Max score Δ"]
    in_flight = (
        f"at each of concurrency {listing(workload.levels)}"
        if several
        else f"{workload.levels[0]} in flight"
    )
    runs = first["scenario"].runs
    latency_tables = "\n\n".join(latency)

    return f"""# Beam-search engine comparison

{scope} · {machine['gpu']} (driver {machine['driver']})

## Latency and throughput

{latency_tables}

Each engine ran the same {runs.requests} requests over HTTP, {in_flight}, in {runs.clean} separate server processes; the reference is `{reference}`, at the same level.

{throughput}## GPU time per request

{table(["Engine", "Kernel ms/request", "No GPU activity"], gpu, right=(1, 2))}

From each engine's separate Nsight run, which sends its requests one at a time; see its report for the definitions.

## Output agreement with {reference}

{table(agree_headers, agreement, right=range(1, len(agree_headers)))}

Compared over the measured requests of each engine's first clean run{', at each level' if several else ''}. Top-1 match: identical best beam. Beam overlap: share of the reference's beams that the engine also returned, in any order, as the mean over requests and, in parentheses, the lowest request. Rank match: share of ranks with identical tokens. Max score Δ: largest cumulative log-probability difference among shared beams; engines normalize over different token sets with a catalog, so it can be nonzero when the beams agree.

## Versions

{table(["Engine", "Source", "Image", "Pins enforced"], versions)}

Model {first['manifest']['model']['id']} @ `{first['manifest']['model']['revision'][:12]}`; the full manifest is in each engine's report.
"""
