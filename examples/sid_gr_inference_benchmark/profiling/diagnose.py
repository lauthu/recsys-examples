"""Capture GPU timelines for a case from an existing end-to-end run."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from string import Template

from harness.artifacts import model_info, read, source_info, suite_hash, write
from harness.config import ROOT
from harness.runner import launch
from harness.runtime import cpu_info, gpu_info
from profiling.timeline import summarize
from reporting.report import number, table


def run(directory, case_id):
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    if suite_hash(ROOT) != manifest["suite_sha256"]:
        raise ValueError(
            "Suite code changed since this run; rerun e2e before diagnosis"
        )
    hardware = gpu_info(manifest["config"]["execution"]["gpu"])
    hardware["cpu"] = cpu_info()
    if hardware != manifest["hardware"]:
        raise ValueError(
            "GPU, driver or CPU configuration changed since the source benchmark"
        )
    jobs = [
        j
        for j in manifest["jobs"]
        if j["case_id"] == case_id and j["status"] == "success"
    ]
    if not jobs:
        raise ValueError("No successful e2e jobs for the selected case")
    output = (
        directory
        / "diagnostics"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:6]
        )
    )
    output.mkdir(parents=True)
    timelines, measured, records = [], [], []
    for job in jobs:
        original = directory / job["directory"]
        request, result = read(original / "request.json"), read(
            original / "result.json"
        )
        if source_info(request["solution"]["source"].get("path")) != request["source"]:
            raise ValueError(
                f"{job['solution']}: source changed; rerun e2e before diagnosis"
            )
        if model_info(request["model"]["path"])["sha256"] != request["model"]["sha256"]:
            raise ValueError("Checkpoint changed since the source benchmark")
        request["expected_environment"], request["expected_adapter"] = (
            result["environment"],
            result["adapter"],
        )
        request["measurement"] = {
            "warmup_runs": max(2, request["measurement"]["warmup_runs"]),
            "repeats": 1,
        }
        target = output / job["solution"] / "timeline"
        capture = launch(request, target, profile=True)
        summary = (
            summarize(target / "timeline.sqlite")
            if capture["status"] == "success"
            else capture
        )
        write(target / "summary.json", summary)
        records.append({"solution": job["solution"], "summary": summary})
        if summary["status"] == "success":
            rows = [
                [r["name"], r["calls"], number(r["gpu_ms"])]
                for r in summary["kernels"][:10]
            ]
            timelines.append(
                f"**{job['solution']}**\n\n"
                + table(["Kernel", "Calls", "GPU ms"], rows)
            )
            measured.append(
                f"{job['solution']}: {summary['gpu_active_union_ms']:.3f} ms GPU-active union in {summary['measured_window_ms']:.3f} ms measured window."
            )
        else:
            timelines.append(
                f"**{job['solution']}:** {summary.get('reason', summary['status'])}"
            )
    write(output / "diagnosis.json", records)
    report = Template(
        (ROOT / "reporting/templates/diagnosis.md").read_text()
    ).substitute(
        case_id=case_id,
        original=f"[{manifest['run_id']}]({directory / 'report.md'})",
        timelines="\n\n".join(timelines),
        measured=" ".join(measured)
        or "No usable timeline collected; consult capture status and logs.",
        reproduce=f"```bash\npython bench.py diagnose --run {directory} --case {case_id}\n```\n",
    )
    (output / "diagnosis.md").write_text(report)
    print(f"Diagnosis: {output / 'diagnosis.md'}")
    return output, records
