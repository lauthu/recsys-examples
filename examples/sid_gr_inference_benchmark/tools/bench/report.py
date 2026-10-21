# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""report.md from the run results and the trace summary.

Every number is measured or observed at runtime; engine settings are copied
verbatim from the configuration rather than interpreted.
"""

import json
from collections import defaultdict

from bench.common import summarize
from bench.sunburst import phase_key

ENGINE_NAMES = {
    "sid-gr": "SID-GR",
    "sglang-beam-search-new": "SGLang beam-search-new",
    "sglang-sidtrie": "SGLang SIDTrie",
}


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


def graph_usage(engine, clean):
    """Observed CUDA graph use summed over the clean runs' measured windows."""
    counts = defaultdict(int)
    for result in clean:
        for key, value in result["counters"].items():
            counts[key] += value
    requests = sum(len(result["latency_ms"]) for result in clean)
    if engine == "sid-gr":
        enabled = clean[0]["runtime"]["cuda_graphs"]
        parts = [
            f"{phase}: {counts[f'{phase}.graph_replays'] / requests:.2f} replays/request"
            if enabled[phase]
            else f"{phase}: disabled"
            for phase in ("prefill", "decode")
        ]
        if enabled["decode"]:
            parts[-1] += f", {counts['decode.eager_fallbacks']} eager fallbacks"
        return "; ".join(parts)
    parts = []
    for phase in ("prefill", "decode"):
        graph, eager = (
            counts[f"forward.{phase}.graph"],
            counts[f"forward.{phase}.eager"],
        )
        parts.append(
            f"{phase}: {graph:,} of {graph + eager:,} forwards as graph replays"
        )
    return "; ".join(parts)


def setup_rows(ctx):
    """Hardware, software and workload; runtime rows are read from the server."""
    config, clean, catalog = ctx["config"], ctx["clean"], ctx["catalog"]
    engine, workload, runs = config["engine"], config["workload"], config["runs"]
    machine, cpus, runtime = clean[0], ctx["cpus"], clean[0]["runtime"]
    packages = [f"{name} {v}" for name, v in runtime["packages"].items() if v]
    cpu = f"{machine['cpu']}; server CPUs {cpus['server']}, client CPUs "
    attention = runtime["attention"]
    backends = f"prefill {attention['prefill']}; decode {attention['decode']}"
    inputs = f"synthetic token IDs (seed {workload['seed']}, special tokens excluded)"
    inputs += f"; {runs['warmups']} warmup + {runs['requests']} measured requests"
    rows = [
        ["GPU", f"{machine['gpu']} (driver {machine['driver']})"],
        ["CPU", cpu + cpus["client"]],
        ["Engine", f"{ENGINE_NAMES[engine]}, {ctx['source']}"],
        ["Image", f"`{ctx['image']}`"],
        ["Packages", ", ".join(packages)],
        ["Attention (observed)", backends],
        ["CUDA graphs (observed)", graph_usage(engine, clean)],
        ["Model", f"{ctx['model']}, bfloat16"],
        ["Workload", inputs + " per run"],
    ]
    if engine == "sglang-sidtrie":
        shared = config["env"].get("SGLANG_BEAM_SHARED_CONTEXT_ATTENTION") == "true"
        text = "compact trie output head, full-vocabulary normalizer and device beam "
        text += "history on every measured request; shared-context attention "
        rows.append(["SIDTrie (observed)", text + ("on" if shared else "off")])
    if catalog:
        checked = sum(result["catalog_beams"] for result in clean + [ctx["profile"]])
        text = f"{catalog['name']}: {catalog['items']:,} items, sha256 "
        text += f"`{catalog['sha256'][:12]}`; all {checked:,} returned beams are "
        rows.append(["Catalog", text + "catalog items"])
    return rows + [["Benchmark tools", ctx["tools"]]]


def render(ctx):
    config, clean, trace = ctx["config"], ctx["clean"], ctx["trace"]
    engine, workload, runs = config["engine"], config["workload"], config["runs"]
    requests = trace["requests"]

    def per_request(ns):
        return ns / requests / 1e6

    latency, total, throughput = latency_rows(clean)
    sections = defaultdict(int)
    for phase, section, _, ns in trace["work"]:
        sections[(phase, section)] += ns
    kernel_ns, window_ns = trace["kernel_ns"], trace["window_ns"]
    (top_phase, top_section), top_ns = max(sections.items(), key=lambda item: item[1])
    idle = trace["busy_ns"]["idle"]
    catalog = ctx["catalog"]
    scope = (
        f"{ctx['model']} · context {workload['context_length']:,} · beam width "
        f"{workload['beam_width']} · {workload['output_tokens']} output tokens"
    )
    if catalog:
        scope += f" · catalog of {catalog['items']:,} items"

    def share(ns, whole):
        return [f"{per_request(ns):.3f}", f"{100 * ns / whole:.1f}%"]

    order = sorted(sections.items(), key=lambda x: (phase_key(x[0][0]), -x[1]))
    section_rows = [[*key, *share(ns, kernel_ns)] for key, ns in order]
    section_rows.append(["**Total**", "", *share(kernel_ns, kernel_ns)])
    busy = trace["busy_ns"]
    window_rows = [
        ["One kernel running", *share(busy["kernel"], window_ns)],
        ["Kernels overlapping", *share(busy["overlap"], window_ns)],
        ["Memory copy/set only", *share(busy["memory"], window_ns)],
        ["No GPU activity", *share(busy["idle"], window_ns)],
        ["**Server window**", *share(window_ns, window_ns)],
    ]

    setup = setup_rows(ctx)
    settings = {"env": config["env"]}
    if "server_settings" in config:
        settings["server_settings"] = config["server_settings"]

    return f"""# {ENGINE_NAMES[engine]} HTTP beam-search benchmark

{scope}

- **Latency:** mean {total['mean']:.3f} ms, p99 {total['p99']:.3f} ms over {total['n']:,} requests ({runs['clean']} server process{'es' if runs['clean'] > 1 else ''} × {runs['requests']}), {throughput:.2f} requests/s at concurrency 1.
- **GPU profile:** {per_request(kernel_ns):.3f} ms of kernel time per request; the largest section is {top_phase} `{top_section}` ({per_request(top_ns):.3f} ms, {100 * top_ns / kernel_ns:.1f}%). No GPU activity is traced for {per_request(idle):.3f} ms ({100 * idle / window_ns:.1f}%) of the {per_request(window_ns):.3f} ms server window.

## Latency

{table(["Run", "Requests", "Mean ms", "p50 ms", "p95 ms", "p99 ms", "Max ms", "Requests/s"], latency, right=range(1, 8))}

The client times each request from JSON encoding to the parsed response, one request at a time over a new loopback HTTP connection. Every run is a separate server process; warmup requests are excluded. Percentiles interpolate linearly between order statistics.

## GPU profile

A separate server process ran under Nsight Systems; the capture holds {requests} requests sent after {runs['warmups']} warmups. Trace: [profile.nsys-rep](profile.nsys-rep).

![GPU kernel time per request](sunburst.svg)

{table(["Phase", "Section", "Kernel ms/request", "Share"], section_rows, right=(2, 3))}

{table(["Server window per request", "ms", "Share"], window_rows, right=(1, 2))}

The server window is the request handler (NVTX `bench/request`). Kernel time adds up kernel durations, so concurrent kernels count separately. A kernel gets its phase and section from the NVTX range around its launch call; {trace['unknown_kernels']} of {trace['kernels']:,} kernels had no unambiguous launch range and are listed as `unknown`. "No GPU activity" means no traced kernel or memory operation, which alone does not prove CPU overhead. Profiled timings are separate from the latency runs.

## Setup

{table(["", ""], setup)}

<details><summary>Engine settings passed to the server</summary>

```json
{json.dumps(settings, indent=2)}
```

</details>

## Reproduce

```bash
{ctx['command']}
```
"""
