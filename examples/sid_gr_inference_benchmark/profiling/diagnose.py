"""Capture GPU timelines for a case from an existing end-to-end run."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from harness.artifacts import model_info, read, source_info, suite_hash, write
from harness.config import ROOT
from harness.runner import launch
from harness.runtime import cpu_info, gpu_info
from profiling.timeline import summarize
from reporting.diagnosis import render
from reporting.report import render as render_run


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
    records = []
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
    write(output / "diagnosis.json", records)
    render(output)
    render_run(directory)
    print(f"Diagnosis: {output / 'diagnosis.md'}")
    return output, records
