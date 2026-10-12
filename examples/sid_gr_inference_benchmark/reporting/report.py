"""Readable reports from recorded results; no execution or GPU dependencies."""

import copy
import shlex
import statistics
from pathlib import Path
from string import Template

from harness.artifacts import digest, read, write
from harness.config import ROOT
from reporting.sol import estimate
from scenarios.validation import compare


def number(value):
    return "—" if value is None else f"{value:.3f}"


def table(headers, rows, right_align=()):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| "
            + " | ".join(
                "---:" if i in right_align else "---" for i in range(len(headers))
            )
            + " |",
            *["| " + " | ".join(clean(x) for x in row) + " |" for row in rows],
        ]
    )


def details(title, content):
    return f"<details>\n<summary>{title}</summary>\n\n{content}\n\n</details>"


def comparison(data, cases):
    """One row per workload, with candidate latencies alongside each other."""
    names = list(dict.fromkeys(r["solution"] for r in data))
    records = {(r["case"], r["solution"]): r for r in data}
    rows = []
    for case_id in sorted(
        {r["case"] for r in data},
        key=lambda key: (
            cases[key]["context_length"],
            cases[key]["batch_size"],
            cases[key]["beam_width"],
            key,
        ),
    ):
        case = cases[case_id]
        candidates = [records.get((case_id, name), {}) for name in names]
        best = min(
            (r["median_ms"] for r in candidates if "median_ms" in r), default=None
        )
        values = []
        for record in candidates:
            value = (
                number(record["median_ms"])
                if "median_ms" in record
                else record.get("status", "—")
            )
            if best is not None and record.get("median_ms") == best:
                value = f"**{value}**"
            values.append(value)
        rows.append(
            [case["context_length"], case["batch_size"], case["beam_width"], *values]
        )
    return table(
        ["Context", "Batch", "Beam", *names], rows, right_align=range(3 + len(names))
    )


def parity_notice(data):
    mismatches = [
        r for r in data if (r.get("agreement") or {}).get("top1_exact") is False
    ]
    summary = table(
        ["Exact", "Different", "Top-1 mismatches", "Unavailable"],
        [
            [
                sum(r["parity"] == "exact" for r in data),
                sum(r["parity"] == "different" for r in data),
                len(mismatches),
                sum(r["parity"] == "unavailable" for r in data),
            ]
        ],
        right_align=range(4),
    )
    if mismatches:
        summary += "\n\n" + table(
            ["Top-1 mismatch case", "Candidate"],
            [[r["case"], r["solution"]] for r in mismatches],
        )
    return summary


def parity_table(data):
    rows = []
    for record in data:
        agreement = record.get("agreement") or {}
        top1 = agreement.get("top1_exact")
        score = agreement.get("max_score_difference")
        rows.append(
            [
                record["case"].split("/", 1)[-1],
                record["solution"],
                record["parity"],
                "Match" if top1 is True else "**Mismatch**" if top1 is False else "—",
                number(100 * agreement["min_overlap"]) if agreement else "—",
                f"{score:.3g}" if score is not None else "—",
            ]
        )
    return table(
        ["Case", "Candidate", "Parity", "Top-1", "Overlap %", "Max score Δ"],
        rows,
        right_align=(4, 5),
    )


def stats(samples):
    if not samples or any(x <= 0 for x in samples):
        raise ValueError("Latency samples must be positive")
    quartiles = (
        statistics.quantiles(samples, n=4, method="inclusive")
        if len(samples) > 1
        else [samples[0]] * 3
    )
    return statistics.median(samples), quartiles[2] - quartiles[0]


def render(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    if manifest["schema_version"] != 2:
        raise ValueError(
            "Use the run's original checkout to regenerate reports for an older artifact layout"
        )
    cfg, jobs = manifest["config"], manifest["jobs"]
    cases = {r["case"]["id"]: r for r in manifest["cases"]}
    records = {
        (j["case_id"], j["solution"]): read(directory / j["directory"] / "result.json")
        for j in jobs
    }
    versions, data = {}, []
    for job in jobs:
        key = job["case_id"], job["solution"]
        result, case = records[key], cases[job["case_id"]]["case"]
        baseline = records.get((key[0], cfg["baseline"]), {})
        row = {
            "scenario": case["name"],
            "case": key[0],
            "solution": key[1],
            "status": result["status"],
            "measurement": case["measurement"],
            "parity": "unavailable",
        }
        if result["status"] != "success" or result.get("profiled", False):
            data.append(row)
            continue
        median, iqr = stats(result["samples_ms"])
        base_ms = (
            statistics.median(baseline["samples_ms"])
            if baseline.get("status") == "success" and not baseline.get("profiled")
            else None
        )
        delta = 100 * (median / base_ms - 1) if base_ms else None
        agreement = None
        if base_ms and baseline["workload_sha256"] == result["workload_sha256"]:
            checks = [
                c
                for output in result["outputs"]
                for c in compare(baseline["outputs"][0], output)
            ]
            errors = [
                c["max_score_difference"]
                for c in checks
                if c["max_score_difference"] is not None
            ]
            agreement = {
                "top1_exact": all(c["top1_exact"] for c in checks),
                "ordered_exact": all(c["ordered_exact"] for c in checks),
                "min_overlap": min(c["beam_overlap"] for c in checks),
                "max_score_difference": max(errors, default=None),
            }
        bound = estimate(
            case,
            cases[key[0]]["model"]["config"],
            cfg["hardware_config"],
            result["environment"]["gpu"],
            median,
        )
        req = read(directory / job["directory"] / "request.json")
        source = req.get("source") or result.get("adapter", {}).get("source") or {}
        packages = result["environment"]["packages"]
        versions[key[1]] = [
            key[1],
            source.get("commit", source.get("tree_sha256", "installed"))[:16],
            str(source.get("dirty", False)),
            ", ".join(
                f"{k} {v}"
                for k, v in packages.items()
                if k in {"torch", "sglang", "triton"}
            ),
        ]
        row.update(
            median_ms=median,
            iqr_ms=iqr,
            delta_pct=delta,
            agreement=agreement,
            requests_per_second=case["batch_size"] * 1000 / median,
            parity=(
                "exact"
                if agreement["ordered_exact"] and agreement["max_score_difference"] == 0
                else "different"
            )
            if agreement
            else "unavailable",
            speed_of_light=bound,
        )
        data.append(row)
    write(directory / "summary.json", data)
    succeeded = sum(r["status"] == "success" for r in data)
    message = table(
        ["Status", "Jobs finished", "Successful", "Other outcomes", "Mode"],
        [
            [
                manifest["status"],
                f"{len(data)}/{sum(manifest['scenario_jobs'].values())}",
                succeeded,
                len(data) - succeeded,
                manifest["mode"],
            ]
        ],
    )
    if manifest.get("reason"):
        message += "\n\n> " + manifest["reason"]
    bounds, unavailable = [], set()
    for row in data:
        bound = row.get("speed_of_light", {})
        if "floor_ms" in bound:
            bounds.append(
                [
                    row["case"],
                    row["solution"],
                    number(bound["floor_ms"]),
                    number(bound["bound_over_measured_pct"]),
                    number(bound["gap_factor"]),
                ]
            )
        elif bound:
            unavailable.add(bound["reason"])
    sol_table = (
        table(["Case", "Candidate", "Floor ms", "Floor/measured %", "Gap ×"], bounds)
        if bounds
        else ""
    )
    if unavailable:
        sol_table += "\n\n" + table(
            ["Estimate", "Reason"],
            [["Unavailable", reason] for reason in sorted(unavailable)],
        )
    if "dense_bf16_tflops" in cfg["hardware_config"]:
        sol_table += "\n\n" + table(
            ["Dense BF16 TFLOP/s", "Source"],
            [
                [
                    cfg["hardware_config"]["dense_bf16_tflops"],
                    cfg["hardware_config"]["source"],
                ]
            ],
        )
    comparisons, measurements, validation = [], [], []
    for scenario in manifest["scenario_jobs"]:
        rows = sorted(
            (r for r in data if r["scenario"] == scenario),
            key=lambda r: (r["case"], r["solution"] != cfg["baseline"], r["solution"]),
        )
        if not rows:
            continue
        comparisons.append(
            f"### {scenario}\n\n"
            + comparison(rows, {key: value["case"] for key, value in cases.items()})
        )
        measurements.append(
            f"### {scenario}\n\n"
            + table(
                [
                    "Case",
                    "Candidate",
                    "Status",
                    "Median ms",
                    "IQR ms",
                    "Δ latency %",
                    "Requests/s",
                ],
                [
                    [
                        r["case"].split("/", 1)[-1],
                        r["solution"],
                        r["status"],
                        *[
                            number(r.get(key))
                            for key in [
                                "median_ms",
                                "iqr_ms",
                                "delta_pct",
                                "requests_per_second",
                            ]
                        ],
                    ]
                    for r in rows
                ],
                right_align=(3, 4, 5, 6),
            )
        )
        validation.append(f"### {scenario}\n\n" + parity_table(rows))
    failures = [
        [
            job["case_id"],
            job["solution"],
            records[job["case_id"], job["solution"]].get("reason", job["status"]),
        ]
        for job in jobs
        if job["status"] != "success"
    ]
    archive = directory / "raw_artifacts.tar.gz"
    diagnostic_links = [
        [f"[{path.parent.name}]({path.relative_to(directory).as_posix()})"]
        for path in sorted((directory / "diagnostics").glob("*/diagnosis.md"))
    ]
    rendered = Template((ROOT / "reporting/templates/e2e.md").read_text()).substitute(
        run_id=manifest["run_id"],
        summary=message,
        hardware=hardware_table(manifest["hardware"]),
        baseline=cfg["baseline"],
        attention=(
            "> **Smoke run:** small-sample validation only; excluded from scenario rankings.\n\n"
            if manifest["mode"] == "smoke"
            else ""
        )
        + parity_notice(data),
        failures=(
            "### Failed or unsupported jobs\n\n"
            + table(["Case", "Candidate", "Reason"], failures)
        )
        if failures
        else "",
        protocols=table(
            ["Scenario", "Warmups", "Measurements"],
            sorted(
                {
                    (
                        r["scenario"],
                        r["measurement"]["warmup_runs"],
                        r["measurement"]["repeats"],
                    )
                    for r in data
                }
            ),
        ),
        leaderboards=(
            "Smoke runs do not update scenario leaderboards."
            if manifest["mode"] == "smoke"
            else "\n".join(
                f"- [{name}](../leaderboards/{name}.md)"
                for name in manifest["scenario_jobs"]
            )
        ),
        results="\n\n".join(comparisons) or "No finished jobs yet.",
        measurements="\n\n".join(measurements),
        validation="\n\n".join(validation) or "No finished jobs to compare.",
        sol=sol_table.strip() or "No successful measurements to estimate.",
        versions=table(
            ["Solution", "Source commit/tree", "Dirty", "Key packages"],
            versions.values(),
        ),
        artifacts=(
            "[Run archive](raw_artifacts.tar.gz)"
            if archive.is_file()
            else "[Manifest](manifest.json) · [Summary JSON](summary.json)"
        ),
        extract=f"tar -xzf {shlex.quote(str(archive))} -C {shlex.quote(str(directory))}\n"
        if archive.is_file()
        else "",
        directory=shlex.quote(str(directory)),
        diagnostics=table(["Saved GPU diagnosis"], diagnostic_links)
        if diagnostic_links
        else "",
    )
    (directory / "report.md").write_text(rendered)
    leaderboards(directory.parent)
    return data


def hardware_table(hardware):
    cpu = hardware.get("cpu")
    rows = [
        ["GPU", hardware["name"]],
        [
            "GPU memory / driver",
            f"{hardware.get('memory_mib', 'unrecorded')} MiB / {hardware.get('driver', 'unrecorded')}",
        ],
    ]
    if cpu:
        rows.extend(
            [
                ["CPU", cpu["model"]],
                [
                    "CPU topology / affinity",
                    f"{cpu['sockets']} sockets · {cpu['physical_cores']} physical cores · {cpu['logical_cpus']} logical CPUs · {len(cpu['affinity'])} allowed CPUs",
                ],
            ]
        )
    else:
        rows.append(["CPU", "unrecorded"])
    return table(["Hardware", "Recorded configuration"], rows)


def leaderboards(root):
    """Rebuild stable scenario boards from the latest measured result per row."""
    groups = {}
    # Chronological run IDs make regeneration of old reports unable to roll back
    # newer results. Failed/unsupported attempts also supersede older successes.
    for path in sorted(root.glob("*/manifest.json")):
        manifest = read(path)
        summary = path.parent / "summary.json"
        if (
            manifest["schema_version"] != 2
            or manifest["mode"] != "e2e"
            or not summary.is_file()
        ):
            continue
        cases = {r["case"]["id"]: r for r in manifest["cases"]}
        for record in read(summary):
            scenario = record["scenario"]
            spec = copy.deepcopy(manifest["config"]["scenario_configs"][scenario])
            # Installation paths are not workload semantics; checkpoint bytes are.
            spec["model"].pop("path", None)
            spec["inputs"].pop("path", None)
            revision = digest(
                {"scenario": spec, "model": cases[record["case"]]["model"]["sha256"]}
            )
            hardware = {k: v for k, v in manifest["hardware"].items() if k != "uuid"}
            key = scenario, revision, digest(hardware)
            group = groups.setdefault(
                key, {"hardware": hardware, "rows": {}, "cases": {}}
            )
            group["cases"][record["case"]] = cases[record["case"]]["case"]
            group["rows"][(record["case"], record["solution"])] = (
                record,
                manifest,
                path.parent.name,
            )
    target = root / "leaderboards"
    target.mkdir(exist_ok=True)
    boards = {}
    for (scenario, revision, _), group in sorted(groups.items()):
        rows, data = [], []
        for record, manifest, run_dir in sorted(
            group["rows"].values(),
            key=lambda item: (
                item[0]["case"],
                item[0].get("median_ms", float("inf")),
                item[0]["solution"],
            ),
        ):
            data.append(record)
            rows.append(
                [
                    record["case"].split("/", 1)[-1],
                    record["solution"],
                    record["status"],
                    number(record.get("requests_per_second")),
                    manifest["config"]["baseline"],
                    f"[{manifest['run_id']}](../{run_dir}/report.md) ({manifest['status']})",
                ]
            )
        section = (
            f"## {group['hardware']['name']}\n\nScenario/checkpoint revision: `{revision[:12]}`.\n\n"
            f"**CPU:** {group['hardware'].get('cpu', {}).get('model', 'unrecorded')} · "
            f"**Driver:** {group['hardware'].get('driver', 'unrecorded')}\n\n"
            + parity_notice(data)
            + "\n\n"
            + details("Hardware configuration", hardware_table(group["hardware"]))
            + "\n\n### Latency comparison\n\n"
            + comparison(data, group["cases"])
            + "\n\n"
            + details(
                "Output parity, including reference repeat consistency",
                parity_table(data),
            )
            + "\n\n"
            + details(
                "Throughput, status and source runs",
                table(
                    [
                        "Case",
                        "Candidate",
                        "Status",
                        "Requests/s",
                        "Reference",
                        "Source run",
                    ],
                    rows,
                ),
            )
        )
        boards.setdefault(scenario, []).append(section)
    for scenario, sections in boards.items():
        text = Template(
            (ROOT / "reporting/templates/leaderboard.md").read_text()
        ).substitute(scenario=scenario, sections="\n\n".join(sections))
        (target / f"{scenario}.md").write_text(text)
