# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""report.md from the run results and the trace summary.

Every number is measured or observed at runtime; solution settings are copied
verbatim from the configuration rather than interpreted. The context carries the
typed Scenario and Solution from bench.config, the settings the benchmark adds
to the solution's (run_benchmark.added_settings), and, in each run's result, the
server's full settings and environment as it reported them (GET /pilot/config).
"""

import json
import math
import random
import shlex
import statistics
from collections import defaultdict

import yaml
from bench import adapters
from bench.common import PHASES, interpolate, quantile, summarize
from bench.sunburst import phase_key
from bench.trace import NO_PHASE, NO_SECTION

RESAMPLES = 2000  # bootstrap resamples per engine and concurrency level


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


def level_runs(clean):
    """{concurrency: [(run, latencies, duration)]}, ascending, over the clean runs."""
    levels = defaultdict(list)
    for result in clean:
        for level in result["levels"]:
            entry = (result["run"], level["latency_ms"], level["duration_s"])
            levels[level["concurrency"]].append(entry)
    return dict(sorted(levels.items()))


def pooled(runs):
    """(stats, requests/s) of one level's runs, pooled."""
    stats = summarize([ms for _, latencies, _ in runs for ms in latencies])
    return stats, stats["n"] / sum(duration for _, _, duration in runs)


def bootstrap(runs, seed, resamples=RESAMPLES):
    """{statistic: resampled values} of one level's runs: each resample draws the
    server processes with replacement and then the requests within each drawn
    process, so process-to-process variation widens the intervals too. The draws
    depend only on the seed and the run and request counts, so every engine of an
    invocation gets the same draws: request i is the same request for each, which
    pairs their comparison."""
    rng = random.Random(seed)
    values = {"mean": [], "p95": [], "p99": [], "rate": []}
    for _ in range(resamples):
        sample, seconds = [], 0.0
        for r in rng.choices(range(len(runs)), k=len(runs)):
            _, latencies, duration = runs[r]
            n = len(latencies)
            sample += [latencies[i] for i in rng.choices(range(n), k=n)]
            seconds += duration
        sample.sort()
        values["mean"].append(math.fsum(sample) / len(sample))
        values["p95"].append(interpolate(sample, 0.95))
        values["p99"].append(interpolate(sample, 0.99))
        values["rate"].append(len(sample) / seconds)
    return values


def interval(values):
    """The 95% percentile interval of bootstrap values."""
    return quantile(values, 0.025), quantile(values, 0.975)


def summaries(ctx):
    """{concurrency: {"stats", "rate", "ci", "means"}} of one engine, pooled over
    its clean runs, with 95% intervals; computed once per context."""
    if "summaries" not in ctx:
        found = {}
        for k, runs in level_runs(ctx["clean"]).items():
            stats, rate = pooled(runs)
            values = bootstrap(runs, seed=f"bootstrap/{k}")
            ci = {name: interval(v) for name, v in values.items()}
            found[k] = {"stats": stats, "rate": rate, "ci": ci, "means": values["mean"]}
        ctx["summaries"] = found
    return ctx["summaries"]


def with_interval(value, bounds, digits):
    return f"{value:.{digits}f} [{bounds[0]:.2f}, {bounds[1]:.2f}]"


def level_records(clean):
    """{concurrency: [each clean run's record of that level]}, ascending."""
    found = defaultdict(list)
    for result in clean:
        for level in result["levels"]:
            found[level["concurrency"]].append(level)
    return dict(sorted(found.items()))


def phase_means(records):
    """{phase: mean ms per request} over one level's records."""
    return {
        name: statistics.fmean(
            ms for level in records for ms in level["phases_ms"][name]
        )
        for name in PHASES
    }


def step_summary(records):
    """{kind: step statistics} over one level's records. A step record is
    [kind, requests, rows, prompt tokens, scheduling ms, step ms]; share is the
    part of the level's wall time its steps took."""
    wall_ms = 1000 * sum(level["duration_s"] for level in records)
    found = {}
    for kind in ("prefill", "decode"):
        steps = [
            step for level in records for step in level["steps"] if step[0] == kind
        ]
        if not steps:
            continue
        mean = {}
        for i, name in ((1, "requests"), (2, "rows"), (3, "tokens"), (4, "schedule")):
            values = [step[i] for step in steps if step[i] is not None]
            mean[name] = statistics.fmean(values) if values else None
        ms = [step[5] for step in steps]
        found[kind] = {
            "steps": len(steps),
            "mean": statistics.fmean(ms),
            "p95": quantile(ms, 0.95),
            **mean,
            "share": sum(ms) / wall_ms,
        }
    return found


def phase_rows(ctx):
    """Mean ms of each phase of a request, and the mean latency, per level."""
    rows, records = [], level_records(ctx["clean"])
    for k, level in summaries(ctx).items():
        means = phase_means(records[k])
        cells = [f"{means[name]:.3f}" for name in PHASES]
        rows.append(
            ([k] if len(records) > 1 else [])
            + cells
            + [f"{level['stats']['mean']:.3f}"]
        )
    return rows


def step_rows(ctx):
    """One row per level and step kind: count, time, batch size and share."""
    rows, records = [], level_records(ctx["clean"])

    def mean(value, digits=1):
        return "–" if value is None else f"{value:,.{digits}f}"

    for k, found in ((k, step_summary(r)) for k, r in records.items()):
        for kind, s in found.items():
            rows.append(
                ([k] if len(records) > 1 else [])
                + [
                    kind,
                    f"{s['steps']:,}",
                    f"{s['mean']:.3f}",
                    f"{s['p95']:.3f}",
                    f"{s['schedule']:.3f}",
                    mean(s["requests"], 2),
                    mean(s["rows"]),
                    mean(s["tokens"]),
                    f"{100 * s['share']:.1f}%",
                ]
            )
    return rows


def load_rows(ctx):
    """Per step kind of the capture under load: steps, requests each, wall and
    kernel ms each, kernel ms per request (amortized over the batch) and ms with
    no GPU activity each. Wall ms comes from the server's step times."""
    trace, (level,) = ctx["load"]["trace"], ctx["load"]["result"]["levels"]
    requests, rows = len(level["latency_ms"]), []
    for kind in ("prefill", "decode"):
        steps = [step for step in level["steps"] if step[0] == kind]
        if not steps:
            continue
        kernel = sum(ns for phase, _, _, ns in trace["work"] if phase == kind)
        idle = sum(
            ns
            for phase, _, state, ns in trace["gaps"]
            if (phase, state) == (kind, "idle")
        )
        rows.append(
            [
                kind,
                f"{len(steps):,}",
                f"{statistics.fmean(step[1] for step in steps):.2f}",
                f"{statistics.fmean(step[5] for step in steps):.3f}",
                f"{kernel / len(steps) / 1e6:.3f}",
                f"{kernel / requests / 1e6:.3f}",
                f"{idle / len(steps) / 1e6:.3f}",
            ]
        )
    return rows


def load_section(ctx):
    """## GPU profile under load, when the scenario has several levels."""
    load, levels = ctx["load"], ctx["scenario"].workload.levels
    if load is None:
        if len(levels) == 1:
            return ""
        return (
            "\n\n## GPU profile under load\n\nNot captured: the overlap scheduler "
            "launches each step before the previous step's results are processed, "
            "so a trace cannot tell the steps apart."
        )
    trace, (level,) = load["trace"], load["result"]["levels"]
    k, requests = level["concurrency"], len(level["latency_ms"])
    one = ctx["trace"]
    single = one["kernel_ns"] / one["windows"] / 1e6
    loaded = trace["kernel_ns"] / requests / 1e6
    between = trace["span_ns"] - trace["window_ns"]
    in_steps = sum(ns for _, _, state, ns in trace["gaps"] if state == "idle")
    columns = ["Step", "Steps", "Requests each", "Wall ms each", "Kernel ms each"]
    columns += ["Kernel ms per request", "No GPU activity ms each"]
    sections, gaps = defaultdict(int), defaultdict(int)
    for phase, section, _, ns in trace["work"]:
        sections[(phase, section)] += ns
    for phase, section, _, ns in trace["gaps"]:
        gaps[(phase, section)] += ns
    counts = defaultdict(int)
    for step in level["steps"]:
        counts[step[0]] += 1
    section_rows = []
    for phase, section in sorted(
        set(sections) | set(gaps), key=lambda key: (phase_key(key[0]), -sections[key])
    ):
        steps = counts.get(phase) or len(level["steps"])  # unknown: per any step
        section_rows.append(
            [
                phase,
                section,
                f"{sections[(phase, section)] / steps / 1e6:.3f}",
                f"{gaps[(phase, section)] / steps / 1e6:.3f}",
            ]
        )
    idle_rows = [
        [f"{start / 1e6:.3f}", f"{ns / 1e6:.3f}", where]
        for start, ns, where in trace["idle"]
    ]
    return f"""

## GPU profile under load

Another server process ran under Nsight Systems at concurrency {k}: each of {k} clients sent its next request when its last returned, {requests} measured requests after the warmups. The server marked each engine step instead of each request, since requests share steps under load; the capture holds {trace['windows']:,} steps. Trace: [load_profile.nsys-rep](load_profile.nsys-rep).

{table(columns, load_rows(ctx), right=range(1, len(columns)))}

Kernel time per request is the steps' kernel time over the {requests} requests they served: {loaded:.3f} ms in all at concurrency {k}, against {single:.3f} ms one request at a time (GPU profile above). The steps took {trace['window_ns'] / 1e6:.3f} ms of the {trace['span_ns'] / 1e6:.3f} ms from the first step's start to the last one's end; inside them the GPU had no activity for {in_steps / 1e6:.3f} ms, and {between / 1e6:.3f} ms passed between steps with no step running.

{table(["Step", "Section", "Kernel ms per step", "Non-kernel ms per step"], section_rows, right=(2, 3))}

Longest stretches with no GPU activity, from the first step's start:

{table(["Starts at ms", "Lasts ms", "Inside"], idle_rows, right=(0, 1))}"""


def interval_rows(ctx):
    """Mean, p95, p99 and requests/s with their 95% intervals, per level."""
    rows, several = [], len(summaries(ctx)) > 1
    for k, level in summaries(ctx).items():
        stats, ci = level["stats"], level["ci"]
        cells = [with_interval(stats[s], ci[s], 3) for s in ("mean", "p95", "p99")]
        cells.append(with_interval(level["rate"], ci["rate"], 2))
        rows.append(([k] if several else []) + cells)
    return rows


def latency_rows(clean):
    """Each run's row and the pooled one per level; the level column only when
    there are several."""
    levels, rows = level_runs(clean), []
    for concurrency, runs in levels.items():
        prefix = [concurrency] if len(levels) > 1 else []
        rows += [prefix + latency_row(*run)[0] for run in runs]
        all_ms = [ms for _, latencies, _ in runs for ms in latencies]
        duration = sum(d for _, _, d in runs)
        rows.append(prefix + latency_row("**All**", all_ms, duration)[0])
    return rows, {k: pooled(runs) for k, runs in levels.items()}


def listing(values):
    """1; 1 and 2; 1, 2 and 4."""
    values = [f"{value:,}" for value in values]
    return (
        values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]
    )


def prompt_text(workload):
    """The prompt lengths, for the scope line of reports and comparisons."""
    if workload.min_context_length is None:
        return f"context {workload.context_length:,}"
    low, high = workload.min_context_length, workload.context_length
    return f"prompts of {low:,} to {high:,} tokens"


def length_bins(workload):
    """Four equal-width [low, high) bins over the prompt lengths; the last includes
    context_length."""
    low, high = workload.min_context_length, workload.context_length + 1
    edges = [low + (high - low) * i // 4 for i in range(5)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def length_rows(clean, lengths, workload):
    """Mean latency per prompt-length bin (rows) and concurrency level (columns),
    pooled over the clean runs; every level sends the same prompts in order."""
    levels, rows = level_runs(clean), []
    for low, high in length_bins(workload):
        inside = [i for i, n in enumerate(lengths) if low <= n < high]
        cells = []
        for runs in levels.values():
            ms = [latencies[i] for _, latencies, _ in runs for i in inside]
            cells.append(f"{summarize(ms)['mean']:.3f}" if ms else "–")
        rows.append([f"{low:,}–{high - 1:,}", len(inside), *cells])
    return rows


def setup_rows(ctx):
    """Hardware, software and workload; runtime rows are read from the server."""
    solution, clean, catalog = ctx["solution"], ctx["clean"], ctx["catalog"]
    adapter = adapters.get(solution.engine)
    workload, runs = ctx["scenario"].workload, ctx["scenario"].runs
    machine, cpus, runtime = clean[0], ctx["cpus"], clean[0]["runtime"]
    manifest = ctx["manifest"]
    model = manifest["model"]
    packages = [f"{name} {v}" for name, v in runtime["packages"].items() if v]
    cpu = f"{machine['cpu']}; server CPUs {cpus['server']}, client CPUs "
    attention = runtime["attention"]
    backends = f"prefill {attention['prefill']}; decode {attention['decode']}"
    inputs = f"synthetic token IDs (seed {workload.seed}, special tokens excluded)"
    if workload.min_context_length is not None:
        inputs += (
            f", prompt lengths drawn uniformly from {workload.min_context_length:,}"
        )
        inputs += f" to {workload.context_length:,} tokens"
    several = len(workload.levels) > 1
    inputs += f"; {runs.warmups} warmup + {runs.requests} measured requests per run"
    inputs += f"{' and level' if several else ''}, {listing(workload.levels)} in flight"
    rows = [
        ["GPU", f"{machine['gpu']} (driver {machine['driver']})"],
        ["CPU", cpu + cpus["client"]],
        ["Engine", f"{adapter.title}, {source_text(manifest['source'])}"],
        ["Image", f"`{manifest['image']['id']}`"],
        ["Packages", ", ".join(packages)],
        ["Attention (observed)", backends],
        ["CUDA graphs (observed)", adapter.graph_usage(clean)],
    ]
    if "overlap" in runtime:  # SGLang's scheduler; SID-GR has its own worker
        overlap = "on" if runtime["overlap"] else "off"
        rows.append(["Overlap scheduler (observed)", overlap])
    weights = (
        f"{ctx['checkpoint_dtype']} checkpoint, weights {runtime['dtype']} (observed)"
    )
    rows += [
        ["Model", f"{model['id']} @ `{model['revision'][:12]}`; {weights}"],
        ["Workload", inputs],
    ]
    rows += adapter.setup_rows(solution, catalog)
    if catalog:
        checked = sum(result["catalog_beams"] for result in clean + [ctx["profile"]])
        text = f"{catalog['name']}: {catalog['items']:,} items, sha256 "
        text += f"`{catalog['sha256'][:12]}`; all {checked:,} returned beams are "
        rows.append(["Catalog", text + "catalog items"])
    return rows


UNSET = object()  # a setting the server does not report


def cell(value):
    """One value as compact JSON, safe inside a table cell."""
    text = "unset" if value is UNSET else json.dumps(value)
    return "`" + text.replace("|", "\\|") + "`"


def changed_by_engine(ctx, observed):
    """Rows for what the solution or the benchmark set that the server runs with a
    different value; PORT differs per run by design."""
    solution, added = ctx["solution"], ctx["added"]
    settings = {**(solution.server_settings or {}), **added.get("server_settings", {})}
    rows = []
    for kind, wanted, running in (
        ("server_settings", settings, observed["settings"]),
        ("env", {**solution.env, **added["env"]}, observed["environment"]),
    ):
        for key, value in wanted.items():
            found = running.get(key, UNSET)
            if value != "PORT" and found != value:
                rows.append([f"`{kind}.{key}`", cell(value), cell(found)])
    return rows


def profile_differences(clean, profile, added):
    """Settings and environment variables in which the profiled process differed,
    except the port and the container's hostname, which differ in every run."""
    lines = []
    for kind, key, wanted in (
        ("settings", "settings", added.get("server_settings", {})),
        ("env", "environment", {**added["env"], "HOSTNAME": "PORT"}),
    ):
        before, after = clean["config"][key], profile["config"][key]
        per_run = {name for name, value in wanted.items() if value == "PORT"}
        for name in sorted((before.keys() | after.keys()) - per_run):
            old, new = before.get(name, UNSET), after.get(name, UNSET)
            if old != new:
                lines.append(f"- `{kind}.{name}`: {cell(old)} → {cell(new)}")
    return lines


def configuration_section(ctx):
    """The server's own report of its settings and environment, if it gave one."""
    clean, profile = ctx["clean"][0], ctx["profile"]
    observed = clean.get("config")
    if not observed:
        return ""
    rows = changed_by_engine(ctx, observed)
    changed = table(["Setting", "Set to", "Running with"], rows) if rows else "none"
    profiled = ""
    if profile.get("config"):
        lines = profile_differences(clean, profile, ctx["added"]) or ["- nothing"]
        profiled = (
            "\n\nThe profiled process differed from the first clean run in (port and "
            "container hostname aside):\n\n"
            + "\n".join(lines)
            + "\n\nand ran as:\n\n```bash\n"
            + shlex.join(profile["server_command"])
            + "\n```"
        )
    source = observed["source"]
    settings = yaml.safe_dump(observed["settings"], sort_keys=True).rstrip()
    environment = yaml.safe_dump(observed["environment"], sort_keys=True).rstrip()
    return f"""

<details><summary>Full server configuration (observed)</summary>

Read from the server once, after the first clean run's measured requests: its settings ({source}) and its process environment. Text values of names with the word TOKEN, SECRET, PASSWORD or KEY are redacted.

Changed by the engine, that is set by the solution or the benchmark but running with another value (`PORT` differs per run by design):

{changed}

Settings ({source}):

```yaml
{settings}
```

Environment:

```yaml
{environment}
```{profiled}

</details>"""


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
    adapter = adapters.get(solution.engine)
    workload, runs = ctx["scenario"].workload, ctx["scenario"].runs
    requests = trace["windows"]

    def per_request(ns):
        return ns / requests / 1e6

    latency, levels = latency_rows(clean)
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
        f"{ctx['model']} · {prompt_text(workload)} · beam width "
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
    busy, launches = trace["busy_ns"], trace["launches"]
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
    latency_patches = clean[0]["runtime"].get("patches", [])
    profile_patches = ctx["profile"]["runtime"].get("patches", [])
    added = [p for p in profile_patches if p not in latency_patches]
    patch_lines = "\n".join(f"- `{p}`" for p in latency_patches) or "- none"
    patch_lines += "\n\nThe profiled run also patched:\n\n"
    patch_lines += "\n".join(f"- `{p}`" for p in added) or "- nothing else"
    top = workload.levels[-1]
    if len(levels) > 1:
        in_flight = (
            f"each server process measures concurrency {listing(workload.levels)} in "
            "turn, rotating their order between processes, and at concurrency k each "
            "of k clients sends its next request when its last returns"
        )
    elif top == 1:
        in_flight = "requests go one at a time"
    else:
        in_flight = (
            f"{top} requests are in flight, as each of {top} clients sends its next "
            "one when its last returns"
        )
    sequential = (
        ""
        if top == 1
        else ", so it shows one request's GPU time, not how the latency runs batch"
        + ("; the profile under load below does" if ctx["load"] else "")
    )
    if workload.min_context_length is not None:
        sizes = listing(ctx["lengths"][:requests])
        sequential = f" (prompts of {sizes} tokens){sequential}"
    measured = f"({runs.clean} server process{'es' if runs.clean > 1 else ''} × {runs.requests})"
    found = summaries(ctx)

    def mean_text(k, stats):
        low, high = found[k]["ci"]["mean"]
        return f"mean {stats['mean']:.3f} ms (95% CI {low:.2f} to {high:.2f})"

    if len(levels) > 1:
        summary = "; ".join(
            f"concurrency {k}: {mean_text(k, s)}, p99 {s['p99']:.3f} ms, "
            f"{rate:.2f} requests/s"
            for k, (s, rate) in levels.items()
        )
        summary = f"- **Latency and throughput** over {measured.strip('()')} requests per level: {summary}."
    else:
        ((k, (stats, rate)),) = levels.items()
        summary = (
            f"- **Latency:** {mean_text(k, stats)}, p99 {stats['p99']:.3f} ms over "
            f"{stats['n']:,} requests {measured}, {rate:.2f} requests/s at concurrency {top}."
        )
    headers = [
        "Run",
        "Requests",
        "Mean ms",
        "p50 ms",
        "p95 ms",
        "p99 ms",
        "Max ms",
        "Requests/s",
    ]
    if len(levels) > 1:
        headers = ["Concurrency", *headers]
    latency_table = table(headers, latency, right=range(len(headers) - 7, len(headers)))
    columns = ["Mean ms", "p95 ms", "p99 ms", "Requests/s"]
    columns = (["Concurrency"] if len(levels) > 1 else []) + columns
    pooled_n = next(iter(levels.values()))[0]["n"]
    between = (
        f"so they include the variation between server processes, which "
        f"{runs.clean} processes show only roughly"
        if runs.clean > 1
        else "but one server process cannot show the variation between processes"
    )
    intervals = (
        "\n\n### 95% confidence intervals\n\n"
        + table(columns, interval_rows(ctx), right=range(len(columns)))
        + f"\n\nEach cell is the pooled value and, in brackets, its 95% interval from a "
        f"two-level bootstrap: {RESAMPLES:,} resamples, each drawing the server "
        f"processes with replacement and then the requests within each drawn process, "
        f"{between}. p99 rests on the slowest 1% of the {pooled_n:,} pooled requests"
        f"{' per level' if len(levels) > 1 else ''}."
    )
    several = len(levels) > 1
    phase_columns = ["Outside server ms", "Queue ms", "Prefill ms", "Decode ms"]
    phase_columns = (["Concurrency"] if several else []) + phase_columns
    phase_columns += ["Return ms", "Latency ms"]
    step_columns = ["Step", "Steps", "Mean ms", "p95 ms", "Scheduling ms"]
    step_columns = (["Concurrency"] if several else []) + step_columns
    step_columns += ["Requests each", "Rows each", "Prompt tokens each", "Wall time"]
    breakdown = f"""## Where the time goes

{table(phase_columns, phase_rows(ctx), right=range(len(phase_columns)))}

Mean per request, pooled over the server processes; the phases add up to the latency. The client's send and reply times and the server's own handler and step times share one clock (`time.perf_counter_ns`). **Outside server** is the latency minus the time the server's request handler took: the client, the connection and HTTP. **Queue** runs from the handler's start to the start of the request's first prefill step. **Prefill** is that step, shared with the requests batched into it. **Decode** runs from the end of the prefill to the end of the request's last decode step, including other requests' steps that ran in between. **Return** runs from there to the handler's return.

{table(step_columns, step_rows(ctx), right=[i for i, c in enumerate(step_columns) if c != "Step"])}

The measured requests' engine steps, pooled over the server processes. A step runs from when the scheduler starts choosing its batch (SGLang's `get_next_batch_to_run`; SID-GR's `_run_prefill` or `_run_decode_batch` call) to when its results are processed; scheduling is the part before the launch. Rows count every beam: a decode step of k requests has k × {workload.beam_width} rows. Wall time is the steps' share of the time from the first send to the last reply of each level; with the overlap scheduler off, steps never overlap."""
    by_length = ""
    if workload.min_context_length is not None:
        rows = length_rows(clean, ctx["lengths"][: runs.requests], workload)
        columns = ["Prompt tokens", "Requests"]
        columns += [f"Mean ms at concurrency {k}" for k in levels]
        by_length = (
            "\n\n### Latency by prompt length\n\n"
            + table(columns, rows, right=range(1, len(columns)))
            + "\n\nEach level sends the same prompts; a bin's mean pools its requests "
            "over the server processes."
        )
    settings = {"env": solution.env}
    if solution.server_settings is not None:
        settings["server_settings"] = solution.server_settings

    return f"""# {adapter.title} HTTP beam-search benchmark

{scope}

{summary}
- **GPU profile:** {per_request(kernel_ns):.3f} ms of kernel time per request; the largest section is {top_phase} `{top_section}` ({per_request(top_ns):.3f} ms, {100 * top_ns / kernel_ns:.1f}%). No GPU activity is traced for {per_request(idle):.3f} ms ({100 * idle / window_ns:.1f}%) of the {per_request(window_ns):.3f} ms server window.

## Latency

{latency_table}

The client times each request from JSON encoding to the parsed response, each over a new loopback HTTP connection; {in_flight}. Every run is a separate server process; warmup requests are excluded. Percentiles interpolate linearly between order statistics.{intervals}{by_length}

{breakdown}

## GPU profile

A separate server process ran under Nsight Systems; the capture holds {requests} requests sent one at a time after {runs.warmups} warmups{sequential}. Trace: [profile.nsys-rep](profile.nsys-rep).

![Server time per request](sunburst.svg)

Open [sunburst.svg](sunburst.svg) directly in a browser to hover for details and click a slice to zoom in.

{table(["Phase", "Section", "Kernel ms", "Non-kernel ms", "Total ms", "Share"], section_rows, right=(2, 3, 4, 5))}

{table(["Server window per request", "ms", "Share"], window_rows, right=(1, 2))}

Launch calls per request: {launches['kernel'] / requests:g} kernel launches and {launches['graph'] / requests:g} CUDA graph launches; the kernels inside a graph need no call of their own.

{table(["Longest-running kernels", "Family", "ms/request", "Launches/request"], kernel_rows, right=(2, 3))}

Families come from kernel names only: GEMMs of different layers (QKV, MLP, LM head) look alike, so they share `linear`; phase and section show where a kernel ran. Open the sunburst to see families per phase and section.

The server window is the request handler (NVTX `bench/request`). Times are per request. Kernel time adds up kernel durations, so concurrent kernels count separately and the total exceeds the server window by their overlap. A kernel gets its phase and section from the NVTX range around its launch call; {trace['unknown_kernels']} of {trace['kernels']:,} kernels had no unambiguous launch range and are listed as `unknown`. Non-kernel time (memory copy/set only, or no GPU activity) gets the innermost phase and section range, on any thread, that covers it; outside every range it is `{NO_PHASE}` / `{NO_SECTION}`. "No GPU activity" means no traced kernel or memory operation, which alone does not prove CPU overhead. Profiled timings are separate from the latency runs.{load_section(ctx)}

## Setup

{table(["", ""], setup)}

<details><summary>Server settings and environment</summary>

From the solution file, passed verbatim:

```yaml
{yaml.safe_dump(settings, sort_keys=False).rstrip()}
```

Added by the benchmark for this scenario and machine (`PORT` is chosen per run). The rest of the server's environment is the container's, without its `GR_*`, `SGLANG_*` and `MODEL_*` variables:

```yaml
{yaml.safe_dump(ctx["added"], sort_keys=False).rstrip()}
```

</details>{configuration_section(ctx)}

<details><summary>Changes to the engine's own HTTP server (observed)</summary>

The launcher runs the engine's own server and patches only these methods, as reported by the server; each hook calls the original unchanged. `PATCHES` in `bench/server_*.py` says what each one adds. The latency runs patched:

{patch_lines}

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
