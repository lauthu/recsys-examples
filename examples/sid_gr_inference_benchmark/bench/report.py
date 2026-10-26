# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""report.md and its charts, from one run's saved results (bench/results.py).

view() reduces the results to plain numbers and names; templates/report.md.j2
holds every word and table of the report, and bench/chart.py and
bench/sunburst.py draw its charts, so the same results always render the same
files. Every number is measured or observed at runtime; solution settings are
copied verbatim from the configuration rather than interpreted. The results
carry the Scenario and Solution from bench.config, the settings the benchmark
adds to the solution's (run_benchmark.added_settings), and, in each run's
result, the server's full settings and environment as it reported them
(GET /pilot/config).
"""

import functools
import json
import math
import random
import re
import statistics
from collections import defaultdict

from bench import adapters, chart, sunburst, template
from bench.common import PHASES, interpolate, quantile, summarize
from bench.sunburst import phase_key
from bench.trace import NO_PHASE, NO_SECTION

RESAMPLES = 2000  # bootstrap resamples per engine and concurrency level
# Where a request's time goes, from its PHASES: "HTTP and response" joins the
# time outside the server's handler and the handler's return after the last step.
PARTS = ("Queue", "Prefill", "Decode", "HTTP and response")
CHARTS = ("load.svg", "breakdown.svg", "sunburst.svg")


def short(name, limit=90):
    """Kernel symbol without the return type, cut at limit characters."""
    name = name.removeprefix("void ")
    return name if len(name) <= limit else name[: limit - 1] + "…"


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


@functools.cache
def summary(k, runs):
    """One level's pooled statistics and requests/s with 95% intervals, and the
    resampled means; runs is ((run, latencies, duration), ...) as tuples, so the
    same measurements are bootstrapped once, for the report and the comparison."""
    stats, rate = pooled(runs)
    values = bootstrap(runs, seed=f"bootstrap/{k}")
    ci = {name: interval(v) for name, v in values.items()}
    return {"stats": stats, "rate": rate, "ci": ci, "means": values["mean"]}


def summaries(results):
    """{concurrency: summary} of one engine, pooled over its clean runs."""
    return {
        k: summary(k, tuple((run, tuple(ms), seconds) for run, ms, seconds in runs))
        for k, runs in level_runs(results["clean"]).items()
    }


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


def parts(means):
    """Mean ms of each of PARTS, from phase_means."""
    return [
        means["queue"],
        means["prefill"],
        means["decode"],
        means["outside"] + means["return"],
    ]


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


def level_view(results):
    """Per concurrency level: the pooled latency statistics and throughput with
    their 95% intervals, where the time went and the engine's steps."""
    records, rows = level_records(results["clean"]), []
    for k, found in summaries(results).items():
        steps = step_summary(records[k])
        rows.append(
            {
                "k": k,
                **found["stats"],
                "rate": found["rate"],
                "ci": found["ci"],
                "parts": parts(phase_means(records[k])),
                "steps": steps,
                "busy": sum(step["share"] for step in steps.values()),
            }
        )
    return rows


def start_rows(clean):
    """Each server start's statistics, per level."""
    rows = []
    for k, runs in level_runs(clean).items():
        for run, latencies, duration in runs:
            stats = summarize(latencies)
            start = int(run.rsplit("_", 1)[1])
            rows.append(
                {"k": k, "start": start, "rate": stats["n"] / duration, **stats}
            )
    return rows


def length_bins(workload):
    """Four equal-width [low, high) bins over the prompt lengths; the last includes
    context_length."""
    low, high = workload.min_context_length, workload.context_length + 1
    edges = [low + (high - low) * i // 4 for i in range(5)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def length_rows(results):
    """Mean latency per prompt-length bin and level, pooled over the clean runs;
    every level sends the same prompts in order. None with one prompt length."""
    workload, runs = results["scenario"].workload, results["scenario"].runs
    if workload.min_context_length is None:
        return None
    lengths, levels = results["lengths"][: runs.requests], level_runs(results["clean"])
    rows = []
    for low, high in length_bins(workload):
        inside = [i for i, n in enumerate(lengths) if low <= n < high]
        means = []
        for level in levels.values():
            ms = [latencies[i] for _, latencies, _ in level for i in inside]
            means.append(statistics.fmean(ms) if ms else None)
        rows.append(
            {"low": low, "high": high - 1, "requests": len(inside), "means": means}
        )
    return rows


def phase_label(phase):
    """How the report names a phase of the trace: Prefill, Decode step 2, ..."""
    decode = re.fullmatch(r"decode_(\d+)", phase)
    if decode:
        return f"Decode step {decode[1]}"
    return {NO_PHASE: "Outside the phases"}.get(phase, phase.capitalize())


def by_section(trace):
    """{(phase, section): [kernel ns, non-kernel ns]} in phase order, then by
    time, largest first."""
    found = defaultdict(lambda: [0, 0])
    for phase, section, _, ns in trace["work"]:
        found[(phase, section)][0] += ns
    for phase, section, _, ns in trace["gaps"]:
        found[(phase, section)][1] += ns
    order = sorted(found, key=lambda key: (phase_key(key[0]), -sum(found[key]), key))
    return {key: found[key] for key in order}


def profile_view(results):
    """The capture of requests sent one at a time, per request: by phase and
    section, the server window's GPU states, launches and the longest kernels."""
    trace, workload = results["trace"], results["scenario"].workload
    requests, window, busy = trace["windows"], trace["window_ns"], trace["busy_ns"]

    def ms(ns):
        return ns / requests / 1e6

    sections = by_section(trace)
    total = sum(map(sum, sections.values()))

    def row(name, kernel, other, **extra):
        times = {"kernel": ms(kernel), "other": ms(other), "total": ms(kernel + other)}
        return {"name": name, **times, "share": (kernel + other) / total, **extra}

    phases = defaultdict(lambda: [0, 0])
    for (phase, _), (kernel, other) in sections.items():
        phases[phase][0] += kernel
        phases[phase][1] += other
    states = (
        ("One kernel running", "kernel"),
        ("Kernels overlapping", "overlap"),
        ("Memory copy or set only", "memory"),
        ("No GPU activity", "idle"),
    )
    lengths = None
    if workload.min_context_length is not None:
        lengths = results["lengths"][:requests]
    return {
        "requests": requests,
        "lengths": lengths,
        "kernel": ms(trace["kernel_ns"]),
        "window": ms(window),
        "idle_share": busy["idle"] / window,
        "phases": [row(phase_label(p), *times) for p, times in phases.items()],
        "sections": [
            row(phase_label(p), *times, section=section)
            for (p, section), times in sections.items()
        ],
        "total": row("Total", *map(sum, zip(*sections.values()))),
        "states": [
            {"name": name, "ms": ms(busy[key]), "share": busy[key] / window}
            for name, key in states
        ],
        "launches": {
            kind: count / requests for kind, count in trace["launches"].items()
        },
        "kernels": [
            {
                "name": short(name),
                "family": family,
                "ms": ms(ns),
                "launches": n / requests,
            }
            for name, family, ns, n in trace["top"]
        ],
        "unknown": trace["unknown_kernels"],
        "count": trace["kernels"],
        "command": results["profile"]["server_command"],
    }


def load_view(results):
    """The capture under load: per step kind and section, and the longest
    stretches with no GPU activity; None without one."""
    load = results["load"]
    if load is None:
        return None
    trace, (level,) = load["trace"], load["result"]["levels"]
    requests = len(level["latency_ms"])
    steps = defaultdict(list)
    for step in level["steps"]:
        steps[step[0]].append(step)
    kernel, idle = defaultdict(int), defaultdict(int)
    for phase, _, _, ns in trace["work"]:
        kernel[phase] += ns
    for phase, _, state, ns in trace["gaps"]:
        if state == "idle":
            idle[phase] += ns
    kinds = []
    for kind in ("prefill", "decode"):
        found = steps.get(kind)
        if not found:
            continue
        n = len(found)
        kinds.append(
            {
                "name": kind.capitalize(),
                "steps": n,
                "requests": statistics.fmean(step[1] for step in found),
                "wall": statistics.fmean(step[5] for step in found),
                "kernel": kernel[kind] / n / 1e6,
                "idle": idle[kind] / n / 1e6,
                "per_request": kernel[kind] / requests / 1e6,
            }
        )
    sections = []
    for (phase, section), (kernel_ns, other_ns) in by_section(trace).items():
        n = len(steps.get(phase, ())) or len(level["steps"])  # unknown: per any step
        sections.append(
            {
                "step": phase_label(phase),
                "section": section,
                "kernel": kernel_ns / n / 1e6,
                "other": other_ns / n / 1e6,
            }
        )
    window, busy = trace["window_ns"], trace["busy_ns"]
    return {
        "k": level["concurrency"],
        "requests": requests,
        "steps": trace["windows"],
        "kinds": kinds,
        "kernel": trace["kernel_ns"] / requests / 1e6,
        "idle_share": busy["idle"] / window,
        "window": window / 1e6,
        "span": trace["span_ns"] / 1e6,
        "idle": busy["idle"] / 1e6,
        "between": (trace["span_ns"] - window) / 1e6,
        "sections": sections,
        "stretches": [
            {"start": start / 1e6, "ms": ns / 1e6, "where": where}
            for start, ns, where in trace["idle"]
        ],
        "command": load["result"]["server_command"],
    }


UNSET = object()  # a setting the server does not report


def as_json(value):
    """One value as compact JSON, or unset."""
    return "unset" if value is UNSET else json.dumps(value)


def changed_by_engine(results, observed):
    """[(name, set to, running with)] for what the solution or the benchmark set
    that the server runs with a different value; PORT differs per run by design."""
    solution, added = results["solution"], results["added"]
    settings = {**(solution.server_settings or {}), **added.get("server_settings", {})}
    rows = []
    for kind, wanted, running in (
        ("server_settings", settings, observed["settings"]),
        ("env", {**solution.env, **added["env"]}, observed["environment"]),
    ):
        for key, value in wanted.items():
            found = running.get(key, UNSET)
            if value != "PORT" and found != value:
                rows.append((f"{kind}.{key}", as_json(value), as_json(found)))
    return rows


def profile_differences(clean, profile, added):
    """[(name, before, after)] for the settings and environment variables in
    which a profiled process differed, except the port and the container's
    hostname, which differ in every run."""
    found = []
    for kind, key, wanted in (
        ("settings", "settings", added.get("server_settings", {})),
        ("env", "environment", {**added["env"], "HOSTNAME": "PORT"}),
    ):
        before, after = clean["config"][key], profile["config"][key]
        per_run = {name for name, value in wanted.items() if value == "PORT"}
        for name in sorted((before.keys() | after.keys()) - per_run):
            old, new = before.get(name, UNSET), after.get(name, UNSET)
            if old != new:
                found.append((f"{kind}.{name}", as_json(old), as_json(new)))
    return found


def configuration_view(results):
    """The server's own report of its settings and environment, and how the
    profiled processes differed; None if the server gave none."""
    clean = results["clean"][0]
    observed = clean.get("config")
    if not observed:
        return None
    runs = [("profile", results["profile"])]
    if results["load"]:
        runs.append(("load", results["load"]["result"]))
    profiled = [
        {
            "name": name,
            "differences": profile_differences(clean, result, results["added"]),
        }
        for name, result in runs
        if result.get("config")
    ]
    return {
        "source": observed["source"],
        "changed": changed_by_engine(results, observed),
        "settings": dict(sorted(observed["settings"].items())),
        "environment": dict(sorted(observed["environment"].items())),
        "profiled": profiled,
    }


def setup_view(results):
    """Hardware, software and workload; runtime values are read from the server."""
    solution, clean, catalog = results["solution"], results["clean"], results["catalog"]
    adapter = adapters.get(solution.engine)
    machine, runtime, manifest = clean[0], clean[0]["runtime"], results["manifest"]
    checked = None
    if catalog:
        checked = sum(r["catalog_beams"] for r in clean + [results["profile"]])
    profile_patches = results["profile"]["runtime"].get("patches", [])
    patches = runtime.get("patches", [])
    settings = {"env": solution.env}
    if solution.server_settings is not None:
        settings["server_settings"] = solution.server_settings
    return {
        "gpu": machine["gpu"],
        "driver": machine["driver"],
        "cpu": machine["cpu"],
        "cpus": results["cpus"],
        "source": manifest["source"],
        "image": manifest["image"]["id"],
        "packages": [f"{name} {v}" for name, v in runtime["packages"].items() if v],
        "attention": runtime["attention"],
        "graphs": adapter.graph_usage(clean),
        "overlap": runtime.get("overlap"),  # SGLang's scheduler; SID-GR has none
        "model": manifest["model"],
        "checkpoint_dtype": results["checkpoint_dtype"],
        "dtype": runtime["dtype"],
        "engine_rows": adapter.setup_rows(solution, catalog),
        "catalog": catalog,
        "checked": checked,
        "settings": settings,
        "added": results["added"],
        "patches": patches,
        "profile_patches": [p for p in profile_patches if p not in patches],
    }


def view(results):
    """Everything report.md shows, as plain numbers and names."""
    scenario, solution = results["scenario"], results["solution"]
    workload = scenario.workload
    return {
        "engine": adapters.get(solution.engine).title,
        "model": results["model"],
        "dtype": results["clean"][0]["runtime"]["dtype"],
        "gpu": results["clean"][0]["gpu"],
        "workload": workload,
        "levels": workload.levels,
        "runs": scenario.runs,
        "catalog": results["catalog"],
        "resamples": RESAMPLES,
        "results": level_view(results),
        "starts": start_rows(results["clean"]),
        "lengths": length_rows(results),
        "overlap": results["clean"][0]["runtime"].get("overlap"),
        "profile": profile_view(results),
        "load": load_view(results),
        "no_phase": NO_PHASE,
        "no_section": NO_SECTION,
        "setup": setup_view(results),
        "configuration": configuration_view(results),
        "manifest": results["manifest"],
        "command": results["command"],
    }


def scope(found):
    """The chart subtitle: model, prompts and beams."""
    workload = found["workload"]
    prompts = f"{workload.context_length:,}-token prompts"
    if workload.min_context_length is not None:
        prompts = f"prompts of {workload.min_context_length:,} to {workload.context_length:,} tokens"
    return f"{found['model']} · {prompts} · {workload.beam_width} beams · {workload.output_tokens} output tokens"


def charts(found, results):
    """{file name: SVG}: throughput and latency against concurrency (with
    several levels), where the time goes per level, and the profile's sunburst."""
    rows, files = found["results"], {}
    subtitle = scope(found)
    if len(rows) > 1:
        latency = {  # the mean with its interval; p95 and p99 as measured
            "Mean": [(row["mean"], *row["ci"]["mean"]) for row in rows],
            "p95": [(row["p95"], None, None) for row in rows],
            "p99": [(row["p99"], None, None) for row in rows],
        }
        rate = [(row["rate"], *row["ci"]["rate"]) for row in rows]
        files["load.svg"] = chart.lines(
            f"{found['engine']}: throughput and latency by concurrency",
            subtitle,
            found["levels"],
            [
                ("Throughput (requests/s)", "requests/s", {"Throughput": rate}),
                ("Latency (ms)", "ms", latency),
            ],
        )
    files["breakdown.svg"] = chart.stacked(
        f"{found['engine']}: where a request's time goes",
        subtitle,
        "Mean ms per request",
        "ms",
        PARTS,
        [(f"Concurrency {row['k']}", row["parts"]) for row in rows],
    )
    trace = results["trace"]
    files["sunburst.svg"] = sunburst.render(
        trace["work"],
        trace["gaps"],
        trace["windows"],
        f"{found['engine']} · {found['model']}",
        f"Server time per request, mean of {trace['windows']} profiled requests sent "
        "one at a time: phase › section › kernel family or non-kernel time",
    )
    return files


def render(results):
    """{file name: text} of report.md and its charts."""
    found = view(results)
    files = charts(found, results)
    text = template.render("report.md.j2", **found, charts=files.keys(), parts=PARTS)
    return {"report.md": text, **files}
