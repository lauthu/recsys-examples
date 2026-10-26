# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""comparison.md: tables across the engines measured in a single invocation.

All engines receive identical requests at every concurrency level, so beams are
compared request by request with a reference engine (public SGLang by default),
at the same level, using the first clean run of each.
"""

from bench import adapters, chart
from bench.report import (
    RESAMPLES,
    interval,
    level_records,
    listing,
    phase_means,
    prompt_text,
    step_summary,
    summaries,
    table,
    with_interval,
)

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
    return {k: (s["stats"], s["rate"]) for k, s in summaries(ctx).items()}


def step_cell(summary):
    """Mean step ms and, in parentheses, requests per step."""
    if summary is None:
        return "–"
    return f"{summary['mean']:.3f} ({summary['requests']:.2f})"


def change_cell(level, base):
    """Mean against the reference's at the same level, with the 95% interval of
    the paired bootstrap: both engines' resamples draw the same requests."""
    ratio = level["stats"]["mean"] / base["stats"]["mean"] - 1
    ratios = [a / b - 1 for a, b in zip(level["means"], base["means"])]
    low, high = interval(ratios)
    return f"{100 * ratio:+.1f}% [{100 * low:+.1f}%, {100 * high:+.1f}%]"


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
        base = summaries(ref)[k]
        rows = []
        for name, ctx in zip(names, contexts):
            level = summaries(ctx)[k]
            stats, ci = level["stats"], level["ci"]
            change = "reference" if name == reference else change_cell(level, base)
            title = adapters.get(ctx["solution"].engine).title
            rows.append(
                [
                    f"[{name}]({name}/report.md) ({title})",
                    with_interval(stats["mean"], ci["mean"], 3),
                    f"{stats['p50']:.3f}",
                    with_interval(stats["p95"], ci["p95"], 3),
                    with_interval(stats["p99"], ci["p99"], 3),
                    change,
                ]
            )
        headers = ["Engine", "Mean ms", "p50 ms", "p95 ms", "p99 ms"]
        text = table([*headers, "Mean vs reference"], rows, right=range(1, 6))
        latency.append(f"### Concurrency {k}\n\n{text}" if several else text)

    service = []
    for name, ctx in zip(names, contexts):
        records = level_records(ctx["clean"])
        for k in workload.levels:
            level, means = summaries(ctx)[k], phase_means(records[k])
            steps = step_summary(records[k])
            service.append(
                [name]
                + ([k] if several else [])
                + [with_interval(level["rate"], level["ci"]["rate"], 2)]
                + [f"{means[phase]:.3f}" for phase in ("queue", "prefill", "decode")]
                + [step_cell(steps.get(kind)) for kind in ("prefill", "decode")]
            )

    gpu, agreement, versions = [], [], []
    for name, ctx in zip(names, contexts):
        trace = ctx["trace"]
        per_request = trace["kernel_ns"] / trace["windows"] / 1e6
        idle = 100 * trace["busy_ns"]["idle"] / trace["window_ns"]
        gpu.append([name, f"{per_request:.3f}", f"{idle:.1f}%"])
        if several:  # amortized over the requests the steps served under load
            load = ctx["load"]
            served = load and len(load["result"]["levels"][0]["latency_ms"])
            gpu[-1].append(
                f"{load['trace']['kernel_ns'] / served / 1e6:.3f}" if load else "–"
            )
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
    top = workload.levels[-1]
    gpu_headers = ["Engine", "Kernel ms/request", "No GPU activity"]
    loaded = ""
    if several:
        gpu_headers.append(f"Kernel ms/request at concurrency {top}")
        loaded = (
            f" At concurrency {top}, another Nsight run marks engine steps instead: "
            "their kernel time over the requests they served ('–' for an engine whose "
            "overlap scheduler leaves steps indistinguishable)."
        )
    service_headers = ["Engine"] + (["Concurrency"] if several else [])
    service_headers += ["Requests/s", "Queue ms", "Prefill ms", "Decode ms"]
    service_headers += ["Prefill step ms", "Decode step ms"]
    between = (
        f"With {runs.clean} processes, the intervals show the variation between "
        "processes only roughly;"
        if runs.clean > 1
        else "With one process, the intervals cannot show the variation between "
        "processes;"
    )
    latency_tables = "\n\n".join(latency)

    return f"""# Beam-search engine comparison

{scope} · {machine['gpu']} (driver {machine['driver']})

## Request level

{latency_tables}

Each engine ran the same {runs.requests} requests over HTTP, {in_flight}, in {runs.clean} separate server processes; the reference is `{reference}`, at the same level. Brackets hold 95% intervals from a two-level bootstrap ({RESAMPLES:,} resamples of the server processes and then of the requests within each). Every engine's resamples draw the same requests, so the interval of the change against the reference is paired. {between} p99 rests on the slowest 1% of the pooled requests.

## Service level

{table(service_headers, service, right=range(1, len(service_headers)))}

Requests/s with its 95% interval as above. Queue, prefill and decode are the mean time per request, from the server's own handler and step times: queue runs from the handler's start to the start of the request's first prefill step, prefill is that step, and decode runs from its end to the end of the request's last decode step, including other requests' steps in between. A step's time is the engine's, from choosing its batch to processing its results, with the requests it carried in parentheses. Each engine's report splits the rest of the latency and tables every step kind.

{throughput}## GPU time per request

{table(gpu_headers, gpu, right=range(1, len(gpu_headers)))}

From each engine's separate Nsight run, which sends its requests one at a time; see its report for the definitions.{loaded}

## Output agreement with {reference}

{table(agree_headers, agreement, right=range(1, len(agree_headers)))}

Compared over the measured requests of each engine's first clean run{', at each level' if several else ''}. Top-1 match: identical best beam. Beam overlap: share of the reference's beams that the engine also returned, in any order, as the mean over requests and, in parentheses, the lowest request. Rank match: share of ranks with identical tokens. Max score Δ: largest cumulative log-probability difference among shared beams; engines normalize over different token sets with a catalog, so it can be nonzero when the beams agree.

## Versions

{table(["Engine", "Source", "Image", "Pins enforced"], versions)}

Model {first['manifest']['model']['id']} @ `{first['manifest']['model']['revision'][:12]}`; the full manifest is in each engine's report.
"""
