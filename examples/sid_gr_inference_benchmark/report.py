"""Readable reports from recorded results; no execution or GPU dependencies."""

from pathlib import Path
import statistics
from string import Template

from artifacts import read, write
from config import ROOT
from sol import estimate
from validation import compare


def number(value):
    return "—" if value is None else f"{value:.3f}"


def table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(clean(x) for x in row) + " |" for row in rows]])


def stats(samples):
    if not samples or any(x <= 0 for x in samples):
        raise ValueError("Latency samples must be positive")
    quartiles = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [samples[0]] * 3
    return statistics.median(samples), quartiles[2] - quartiles[0]


def render(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    cfg, jobs = manifest["config"], manifest["jobs"]
    cases = {r["case"]["id"]: r for r in manifest["cases"]}
    records = {(j["case_id"], j["solution"]): read(directory / j["directory"] / "result.json") for j in jobs}
    timing, validity, bounds, versions, data = [], [], [], {}, []
    for job in jobs:
        key = job["case_id"], job["solution"]
        result, case = records[key], cases[job["case_id"]]["case"]
        baseline = records.get((key[0], cfg["baseline"]), {})
        row = {"case": key[0], "solution": key[1], "status": result["status"]}
        if result["status"] != "success" or result.get("profiled", False):
            timing.append([*key, result["status"], "—", "—", "—", "—"])
            validity.append([*key, result.get("reason", "Excluded profiled run"), "—", "—", "—"])
            data.append(row)
            continue
        median, iqr = stats(result["samples_ms"])
        base_ms = statistics.median(baseline["samples_ms"]) if baseline.get("status") == "success" and not baseline.get("profiled") else None
        delta = 100 * (median / base_ms - 1) if base_ms else None
        timing.append([*key, "success", number(median), number(iqr), number(delta), number(case["batch_size"] * 1000 / median)])
        agreement = None
        if base_ms and baseline["workload_sha256"] == result["workload_sha256"]:
            checks = [c for output in result["outputs"] for c in compare(baseline["outputs"][0], output)]
            errors = [c["max_score_difference"] for c in checks if c["max_score_difference"] is not None]
            agreement = {"top1_exact": all(c["top1_exact"] for c in checks), "ordered_exact": all(c["ordered_exact"] for c in checks),
                         "min_overlap": min(c["beam_overlap"] for c in checks), "max_score_difference": max(errors, default=None)}
        validity.append([*key, "structural pass", str(agreement["top1_exact"]) if agreement else "—",
                         number(100 * agreement["min_overlap"]) if agreement else "—",
                         f"{agreement['max_score_difference']:.3g}" if agreement and agreement["max_score_difference"] is not None else "—"])
        bound = estimate(case, cases[key[0]]["model"]["config"], cfg.get("hardware"), result["environment"]["gpu"], median)
        bounds.append([*key, number(bound.get("floor_ms")), number(bound.get("bound_over_measured_pct")), number(bound.get("gap_factor")), bound.get("reason", "partial matrix-compute floor")])
        req = read(directory / job["directory"] / "request.json")
        source = req.get("source") or result.get("adapter", {}).get("source") or {}
        packages = result["environment"]["packages"]
        versions[key[1]] = [key[1], source.get("commit", source.get("tree_sha256", "installed"))[:16],
                           str(source.get("dirty", False)), ", ".join(f"{k} {v}" for k, v in packages.items() if k in {"torch", "sglang", "triton"})]
        row.update(median_ms=median, iqr_ms=iqr, delta_pct=delta, agreement=agreement, speed_of_light=bound)
        data.append(row)
    write(directory / "summary.json", data)
    succeeded = sum(r["status"] == "success" for r in data)
    message = f"{succeeded}/{len(data)} jobs completed successfully. Run status: {manifest['status']}."
    if manifest.get("reason"):
        message += " " + manifest["reason"]
    limitations = "- Frameworks run serially, with case order rotating the solution order; clocks are not locked by this suite.\n- GPU memory is unavailable until a comparable process-tree collector is configured.\n- Historical reports use a different timing boundary and are not merged into this leaderboard.\n- Structural validity does not imply exact beam agreement.\n- Small sample counts are smoke evidence, not a regression conclusion."
    sol_table = table(["Case", "Solution", "Floor ms", "Floor/measured %", "Gap ×", "Coverage"], bounds)
    if cfg.get("hardware"):
        sol_table += f"\n\nConfigured dense BF16 ceiling: {cfg['hardware']['dense_bf16_tflops']} TFLOP/s. Source: {cfg['hardware']['source']}"
    hardware = manifest["hardware"]
    rendered = Template((ROOT / "reports/template.md").read_text()).substitute(
        run_id=manifest["run_id"], summary=message,
        hardware=f"{hardware.get('name', 'unavailable')}; driver {hardware.get('driver', 'unavailable')}; {hardware.get('memory_mib', 'unavailable')} MiB", baseline=cfg["baseline"],
        measurement=f"{cfg['measurement']['warmup_runs']} warmups, {cfg['measurement']['repeats']} measurements per case; offline, single GPU",
        results=table(["Case", "Solution", "Status", "Median ms", "IQR ms", "Δ latency %", "Requests/s"], timing),
        validation=table(["Case", "Solution", "Validity", "Top-1 exact", "Min overlap %", "Max score Δ"], validity),
        sol=sol_table,
        versions=table(["Solution", "Source commit/tree", "Dirty", "Key packages"], versions.values()), limitations=limitations,
        reproduce="Use the frozen `jobs/*/*/*/request.json` and `command.json` for exact execution settings.\n\n"
                  "```bash\npython bench.py report --run " + str(directory) + "\npython bench.py diagnose --run " + str(directory) + " --case <case-id>\n```\n"
                  "The manifest records checkpoint hashes, source changes, image IDs and shared input checksums.")
    (directory / "report.md").write_text(rendered)
    return data


def leaderboard(directory, target):
    directory, target = Path(directory), Path(target)
    data = render(directory)
    manifest = read(directory / "manifest.json")
    rows = [[r["case"], r["solution"], r["status"], number(r.get("median_ms")), number(r.get("delta_pct"))] for r in data]
    target.write_text("# SID-GR leaderboard\n\n" + f"Run: {manifest['run_id']}. GPU: {manifest['hardware'].get('name')}.\n\n"
                      + table(["Case", "Solution", "Status", "Median ms", "Δ latency %"], rows)
                      + f"\n\nFull report: [{manifest['run_id']}]({directory / 'report.md'}).\n\nOnly this compatible run is included; previous run directories remain intact.\n")
