# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""report.md from the run results and the trace summary.

Every number is measured or observed at runtime; solution settings are copied
verbatim from the configuration rather than interpreted. The context carries the
typed Scenario and Solution from bench.config.
"""

import json
from collections import defaultdict

import yaml
from bench import engines
from bench.common import summarize
from bench.sunburst import phase_key
from bench.trace import NO_PHASE, NO_SECTION


def short(name, limit=90):
    """Kernel symbol without the return type and template arguments."""
    name = name.removeprefix("void ")
    name = name if len(name) <= limit else name[: limit - 1] + "…"
    return name.replace("|", "\\|").replace("`", "")


def table(headers, rows, right=()):
    align = ["---:" if i in right else "---" for i in range(len(headers))]
    lines = [headers, align, *rows]
    return "\n".join("| " + " | ".join(map(str, line)) + " |" for line in lines)


def latency_row(name, latencies, duration):
    stats = summarize(latencies)
    values = [f"{stats[key]:.3f}" for key in ("mean", "p50", "p95", "p99", "max")]
    return [name, stats["n"], *values, f"{stats['n'] / duration:.2f}"], stats


def latency_rows(clean):
    rows = [latency_row(r["run"], r["latency_ms"], r["duration_s"])[0] for r in clean]
    pooled = [ms for result in clean for ms in result["latency_ms"]]
    duration = sum(result["duration_s"] for result in clean)
    row, total = latency_row("**All**", pooled, duration)
    return rows + [row], total, total["n"] / duration


def setup_rows(ctx):
    """Hardware, software and workload; runtime rows are read from the server."""
    solution, clean, catalog = ctx["solution"], ctx["clean"], ctx["catalog"]
    engine = engines.get(solution.engine)
    workload, runs = ctx["scenario"].workload, ctx["scenario"].runs
    machine, cpus, runtime = clean[0], ctx["cpus"], clean[0]["runtime"]
    manifest = ctx["manifest"]
    model = manifest["model"]
    packages = [f"{name} {v}" for name, v in runtime["packages"].items() if v]
    cpu = f"{machine['cpu']}; server CPUs {cpus['server']}, client CPUs "
    attention = runtime["attention"]
    backends = f"prefill {attention['prefill']}; decode {attention['decode']}"
    inputs = f"synthetic token IDs (seed {workload.seed}, special tokens excluded)"
    inputs += f"; {runs.warmups} warmup + {runs.requests} measured requests"
    rows = [
        ["GPU", f"{machine['gpu']} (driver {machine['driver']})"],
        ["CPU", cpu + cpus["client"]],
        ["Engine", f"{engine.title}, {source_text(manifest['source'])}"],
        ["Image", f"`{manifest['image']['id']}`"],
        ["Packages", ", ".join(packages)],
        ["Attention (observed)", backends],
        ["CUDA graphs (observed)", engine.graph_usage(clean)],
        ["Model", f"{model['id']} @ `{model['revision'][:12]}`, bfloat16"],
        ["Workload", inputs + " per run"],
    ]
    rows += engine.setup_rows(solution)
    if catalog:
        checked = sum(result["catalog_beams"] for result in clean + [ctx["profile"]])
        text = f"{catalog['name']}: {catalog['items']:,} items, sha256 "
        text += f"`{catalog['sha256'][:12]}`; all {checked:,} returned beams are "
        rows.append(["Catalog", text + "catalog items"])
    return rows


def source_text(source):
    text = f"{source['url']} @ `{source['commit'][:12]}`"
    if not source["enforced"]:
        text += (
            " **(pin not enforced: "
            + "; ".join(source["differences"] or ["no difference"])
            + ")**"
        )
    return text


def reproduce_rows(manifest):
    source, image, model = manifest["source"], manifest["image"], manifest["model"]
    labels = image["labels"]
    base = labels.get("org.opencontainers.image.version") or labels.get(
        "org.opencontainers.image.revision", "none recorded"
    )
    digests = (
        ", ".join(f"`{d}`" for d in image["repo_digests"]) or "none (built locally)"
    )
    files = ", ".join(
        f"{name} `{digest[:12]}`" for name, digest in model["sha256"].items()
    )
    return [
        ["Benchmark code", manifest["benchmark_code"]],
        [
            "Engine source",
            f"{source_text(source)}; checkout HEAD `{source['head'][:12]}`",
        ],
        [
            "Image",
            f"`{image['id']}`; tags {', '.join(image['tags']) or 'none'}; digests {digests}",
        ],
        ["Image base", base],
        ["Image recipe", f"`{manifest['recipe']}`"],
        ["Model", f"{model['id']} @ `{model['revision']}`; SHA-256 {files}"],
        [
            "Config SHA-256",
            f"scenario `{manifest['configs']['scenario'][:12]}`, solution `{manifest['configs']['solution'][:12]}`",
        ],
        ["Nsight Systems", manifest["nsys"] or "unknown"],
    ]


def render(ctx):
    solution, clean, trace = ctx["solution"], ctx["clean"], ctx["trace"]
    engine = engines.get(solution.engine)
    workload, runs = ctx["scenario"].workload, ctx["scenario"].runs
    requests = trace["requests"]

    def per_request(ns):
        return ns / requests / 1e6

    latency, total, throughput = latency_rows(clean)
    sections, gaps = defaultdict(int), defaultdict(int)
    for phase, section, _, ns in trace["work"]:
        sections[(phase, section)] += ns
    for phase, section, _, ns in trace["gaps"]:
        gaps[(phase, section)] += ns
    kernel_ns, window_ns = trace["kernel_ns"], trace["window_ns"]
    gap_ns = sum(gaps.values())
    (top_phase, top_section), top_ns = max(sections.items(), key=lambda item: item[1])
    idle = trace["busy_ns"]["idle"]
    catalog = ctx["catalog"]
    scope = (
        f"{ctx['model']} · context {workload.context_length:,} · beam width "
        f"{workload.beam_width} · {workload.output_tokens} output tokens"
    )
    if catalog:
        scope += f" · catalog of {catalog['items']:,} items"

    def share(ns, whole):
        return [f"{per_request(ns):.3f}", f"{100 * ns / whole:.1f}%"]

    def time_row(kernel, gap):
        share = 100 * (kernel + gap) / (kernel_ns + gap_ns)
        times = (per_request(ns) for ns in (kernel, gap, kernel + gap))
        return [*(f"{ms:.3f}" for ms in times), f"{share:.1f}%"]

    keys = set(sections) | set(gaps)
    totals = {key: sections[key] + gaps[key] for key in keys}
    order = sorted(keys, key=lambda key: (phase_key(key[0]), -totals[key]))
    section_rows = [[*key, *time_row(sections[key], gaps[key])] for key in order]
    section_rows.append(["**Total**", "", *time_row(kernel_ns, gap_ns)])
    busy = trace["busy_ns"]
    window_rows = [
        ["One kernel running", *share(busy["kernel"], window_ns)],
        ["Kernels overlapping", *share(busy["overlap"], window_ns)],
        ["Memory copy/set only", *share(busy["memory"], window_ns)],
        ["No GPU activity", *share(busy["idle"], window_ns)],
        ["**Server window**", *share(window_ns, window_ns)],
    ]

    kernel_rows = [
        [
            f"`{short(name)}`",
            family,
            f"{per_request(ns):.3f}",
            f"{launches / requests:g}",
        ]
        for name, family, ns, launches in trace["top"]
    ]
    setup = setup_rows(ctx)
    settings = {"env": solution.env}
    if solution.server_settings is not None:
        settings["server_settings"] = solution.server_settings

    return f"""# {engine.title} HTTP beam-search benchmark

{scope}

- **Latency:** mean {total['mean']:.3f} ms, p99 {total['p99']:.3f} ms over {total['n']:,} requests ({runs.clean} server process{'es' if runs.clean > 1 else ''} × {runs.requests}), {throughput:.2f} requests/s at concurrency 1.
- **GPU profile:** {per_request(kernel_ns):.3f} ms of kernel time per request; the largest section is {top_phase} `{top_section}` ({per_request(top_ns):.3f} ms, {100 * top_ns / kernel_ns:.1f}%). No GPU activity is traced for {per_request(idle):.3f} ms ({100 * idle / window_ns:.1f}%) of the {per_request(window_ns):.3f} ms server window.

## Latency

{table(["Run", "Requests", "Mean ms", "p50 ms", "p95 ms", "p99 ms", "Max ms", "Requests/s"], latency, right=range(1, 8))}

The client times each request from JSON encoding to the parsed response, one request at a time over a new loopback HTTP connection. Every run is a separate server process; warmup requests are excluded. Percentiles interpolate linearly between order statistics.

## GPU profile

A separate server process ran under Nsight Systems; the capture holds {requests} requests sent after {runs.warmups} warmups. Trace: [profile.nsys-rep](profile.nsys-rep).

![Server time per request](sunburst.svg)

Open [sunburst.svg](sunburst.svg) directly in a browser to hover for details and click a slice to zoom in.

{table(["Phase", "Section", "Kernel ms", "Non-kernel ms", "Total ms", "Share"], section_rows, right=(2, 3, 4, 5))}

{table(["Server window per request", "ms", "Share"], window_rows, right=(1, 2))}

{table(["Longest-running kernels", "Family", "ms/request", "Launches/request"], kernel_rows, right=(2, 3))}

Families come from kernel names only: GEMMs of different layers (QKV, MLP, LM head) look alike, so they share `linear`; phase and section show where a kernel ran. Open the sunburst to see families per phase and section.

The server window is the request handler (NVTX `bench/request`). Times are per request. Kernel time adds up kernel durations, so concurrent kernels count separately and the total exceeds the server window by their overlap. A kernel gets its phase and section from the NVTX range around its launch call; {trace['unknown_kernels']} of {trace['kernels']:,} kernels had no unambiguous launch range and are listed as `unknown`. Non-kernel time (memory copy/set only, or no GPU activity) gets the innermost phase and section range, on any thread, that covers it; outside every range it is `{NO_PHASE}` / `{NO_SECTION}`. "No GPU activity" means no traced kernel or memory operation, which alone does not prove CPU overhead. Profiled timings are separate from the latency runs.

## Setup

{table(["", ""], setup)}

<details><summary>Solution settings passed to the server</summary>

```yaml
{yaml.safe_dump(settings, sort_keys=False).rstrip()}
```

</details>

## Reproduce

{table(["", ""], reproduce_rows(ctx["manifest"]))}

<details><summary>Manifest (JSON)</summary>

```json
{json.dumps(ctx["manifest"], indent=2)}
```

</details>

```bash
{ctx['command']}
```
"""
