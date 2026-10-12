"""Readable reports from recorded results; no execution or GPU dependencies."""

import statistics
from pathlib import Path
from string import Template

from harness.artifacts import read, write
from harness.config import ROOT
from reporting.sol import estimate
from scenarios.validation import compare


def number(value):
    return "—" if value is None else f"{value:.3f}"


def table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(clean(x) for x in row) + " |" for row in rows],
        ]
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
    timing, validity, bounds, versions, data = [], [], [], {}, []
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
            timing.append([*key, result["status"], "—", "—", "—", "—"])
            validity.append(
                [*key, result.get("reason", "Excluded profiled run"), "—", "—", "—"]
            )
            data.append(row)
            continue
        median, iqr = stats(result["samples_ms"])
        base_ms = (
            statistics.median(baseline["samples_ms"])
            if baseline.get("status") == "success" and not baseline.get("profiled")
            else None
        )
        delta = 100 * (median / base_ms - 1) if base_ms else None
        timing.append(
            [
                *key,
                "success",
                number(median),
                number(iqr),
                number(delta),
                number(case["batch_size"] * 1000 / median),
            ]
        )
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
        validity.append(
            [
                *key,
                "structural pass",
                str(agreement["top1_exact"]) if agreement else "—",
                number(100 * agreement["min_overlap"]) if agreement else "—",
                f"{agreement['max_score_difference']:.3g}"
                if agreement and agreement["max_score_difference"] is not None
                else "—",
            ]
        )
        bound = estimate(
            case,
            cases[key[0]]["model"]["config"],
            cfg["hardware_config"],
            result["environment"]["gpu"],
            median,
        )
        bounds.append(
            [
                *key,
                number(bound.get("floor_ms")),
                number(bound.get("bound_over_measured_pct")),
                number(bound.get("gap_factor")),
                bound.get("reason", "partial matrix-compute floor"),
            ]
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
    message = f"{len(data)}/{sum(manifest['scenario_jobs'].values())} jobs finished; {succeeded} successful. Run status: {manifest['status']}. Mode: {manifest['mode']}."
    if manifest.get("reason"):
        message += " " + manifest["reason"]
    limitations = "- Frameworks run serially, with case order rotating the solution order; clocks are not locked by this suite.\n- GPU memory is unavailable until a comparable process-tree collector is configured.\n- Historical reports use a different timing boundary and are not merged into this leaderboard.\n- Structural validity does not imply exact beam agreement.\n- Small sample counts are smoke evidence, not a regression conclusion."
    sol_table = table(
        ["Case", "Solution", "Floor ms", "Floor/measured %", "Gap ×", "Coverage"],
        bounds,
    )
    if "dense_bf16_tflops" in cfg["hardware_config"]:
        sol_table += f"\n\nConfigured dense BF16 ceiling: {cfg['hardware_config']['dense_bf16_tflops']} TFLOP/s. Source: {cfg['hardware_config']['source']}"
    hardware = manifest["hardware"]
    rendered = Template((ROOT / "reporting/templates/e2e.md").read_text()).substitute(
        run_id=manifest["run_id"],
        summary=message,
        hardware=f"{hardware.get('name', 'unavailable')}; driver {hardware.get('driver', 'unavailable')}; {hardware.get('memory_mib', 'unavailable')} MiB",
        baseline=cfg["baseline"],
        measurement="Offline batches on one GPU; protocol is recorded per scenario below.",
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
        leaderboards="\n".join(
            f"- [{name}](leaderboards/{name}.md)" for name in manifest["scenario_jobs"]
        ),
        results=table(
            [
                "Case",
                "Solution",
                "Status",
                "Median ms",
                "IQR ms",
                "Δ latency %",
                "Requests/s",
            ],
            timing,
        ),
        validation=table(
            [
                "Case",
                "Solution",
                "Validity",
                "Top-1 exact",
                "Min overlap %",
                "Max score Δ",
            ],
            validity,
        ),
        sol=sol_table,
        versions=table(
            ["Solution", "Source commit/tree", "Dirty", "Key packages"],
            versions.values(),
        ),
        limitations=limitations,
        reproduce="Use the frozen `jobs/*/*/*/request.json` and `command.json` for exact execution settings.\n\n"
        "```bash\npython bench.py report --run "
        + str(directory)
        + "\npython bench.py diagnose --run "
        + str(directory)
        + " --case <case-id>\n```\n"
        "The manifest records checkpoint hashes, source changes, image IDs and shared input checksums.",
    )
    (directory / "report.md").write_text(rendered)
    leaderboards(directory, manifest, data)
    return data


def leaderboards(directory, manifest, data):
    """Refresh each scenario's status within this run, without mixing runs or SKUs."""
    target = directory / "leaderboards"
    target.mkdir(exist_ok=True)
    for scenario, planned in manifest["scenario_jobs"].items():
        records = [r for r in data if r["scenario"] == scenario]
        rows = []
        for r in sorted(
            records, key=lambda r: (r["case"], r.get("median_ms", float("inf")))
        ):
            agreement = r.get("agreement") or {}
            rows.append(
                [
                    r["case"],
                    r["solution"],
                    r["status"],
                    number(r.get("median_ms")),
                    number(r.get("requests_per_second")),
                    r["parity"],
                    agreement.get("top1_exact", "—"),
                    number(agreement.get("min_overlap")),
                    number(agreement.get("max_score_difference")),
                ]
            )
        text = (
            f"# {scenario} — {manifest['run_id']}\n\n"
            f"Run status: {manifest['status']}. Mode: {manifest['mode']}. "
            f"{len(records)}/{planned} jobs finished.\n\n"
            f"Hardware: {manifest['hardware']['name']} ({manifest['config']['hardware']}). "
            f"UUID: {manifest['hardware']['uuid']}.\n\n"
            + table(
                [
                    "Case",
                    "Candidate",
                    "Status",
                    "Median ms",
                    "Requests/s",
                    "Parity",
                    "Top-1 exact",
                    "Min overlap",
                    "Max score Δ",
                ],
                rows,
            )
            + "\n\nParity is exact only when ordered beams and scores match the baseline exactly. "
            "Different results remain visible; unavailable means no comparable baseline. "
            "The baseline row checks repeat consistency, not independent accuracy. "
            "Requests/s is completed batch size divided by median latency.\n\n"
            "[Full report](../report.md) · [Frozen run](../manifest.json) · [Raw metrics](../summary.json)\n"
        )
        (target / f"{scenario}.md").write_text(text)
