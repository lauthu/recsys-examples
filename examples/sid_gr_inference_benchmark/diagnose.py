"""Optional diagnostics for selected immutable end-to-end requests."""

from datetime import datetime, timezone
from pathlib import Path
from string import Template
import uuid

from artifacts import model_info, read, source_info, suite_hash, write
from config import ROOT
from profiling.timeline import summarize
from report import number, table
from runner import launch
from runtime import gpu_info


def run(directory, case_id, *, timeline=True, kernels=(), counters=()):
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    if suite_hash(ROOT) != manifest["suite_sha256"]:
        raise ValueError("Suite code changed since this run; rerun e2e before diagnosis")
    gpu = gpu_info(manifest["config"]["execution"]["gpu"])
    if gpu != manifest["hardware"]:
        raise ValueError("GPU/driver changed since the source benchmark")
    jobs = [j for j in manifest["jobs"] if j["case_id"] == case_id and j["status"] == "success"]
    if not jobs:
        raise ValueError("No successful e2e jobs for the selected case")
    output = directory / "diagnostics" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    timelines, operators, measured, records = [], [], [], []
    for job in jobs:
        original = directory / job["directory"]
        request, result = read(original / "request.json"), read(original / "result.json")
        if source_info(request["solution"]["source"].get("path")) != request.get("source"):
            raise ValueError(f"{job['solution']}: source changed; rerun e2e before diagnosis")
        if model_info(request["model"]["path"])["sha256"] != request["model"]["sha256"]:
            raise ValueError("Checkpoint changed since the source benchmark")
        request["expected_environment"], request["expected_adapter"] = result["environment"], result["adapter"]
        request["measurement"] = {"warmup_runs": max(2, request["measurement"]["warmup_runs"]), "repeats": 1}
        if timeline:
            target = output / job["solution"] / "timeline"
            capture = launch(request, target, profile="timeline")
            summary = summarize(target / "timeline.sqlite") if capture["status"] == "success" else capture
            write(target / "summary.json", summary)
            if summary["status"] == "success":
                rows = [[r["name"], r["calls"], number(r["gpu_ms"])] for r in summary["kernels"][:10]]
                timelines.append(f"**{job['solution']}**\n\n" + table(["Kernel", "Calls", "GPU ms"], rows))
                measured.append(f"{job['solution']}: {summary['gpu_active_union_ms']:.3f} ms GPU-active union in {summary['measured_window_ms']:.3f} ms measured window.")
            else:
                timelines.append(f"**{job['solution']}:** {summary.get('reason', summary['status'])}")
            records.append({"solution": job["solution"], "kind": "timeline", "summary": summary})
        for operation in sorted(set(kernels) | set(counters)):
            if operation not in request["solution"].get("kernels", []):
                operators.append(f"{job['solution']}/{operation}: unavailable; no operator entry point exposed.")
                records.append({"solution": job["solution"], "kind": "kernel", "result": {
                    "status": "unsupported", "operation": operation, "reason": "No operator entry point exposed"}})
                continue
            if operation in kernels:
                target = output / job["solution"] / operation
                result = launch(request, target, kernel=operation)
                if result["status"] == "success":
                    rows = [[r["implementation"], number(r["eager_gpu_ms"] * 1000), number(r["cuda_graph_ms"] * 1000), r["peak_extra_tensor_bytes"], f"{r['max_abs_error']:.3g}"] for r in result["records"]]
                    operators.append(f"**{job['solution']}/{operation}**\n\n" + table(["Implementation", "Eager GPU µs", "Graph GPU µs", "Extra tensor bytes", "Max error"], rows))
                else:
                    operators.append(f"{job['solution']}/{operation}: {result.get('reason', result['status'])}")
                records.append({"solution": job["solution"], "kind": "kernel", "result": result})
            if operation in counters:
                result = launch(request, output / job["solution"] / (operation + "_counters"), profile="counters", kernel=operation)
                detail = result.get("reason", f"See {result.get('counter_report', 'native counter report')} and counters.csv.")
                operators.append(f"{job['solution']}/{operation} counters: {result['status']}. {detail} Timings from this capture are excluded from operator comparisons.")
                records.append({"solution": job["solution"], "kind": "counters", "result": result})
    write(output / "diagnosis.json", records)
    report = Template((ROOT / "reports/diagnosis_template.md").read_text()).substitute(
        case_id=case_id, original=f"[{manifest['run_id']}]({directory / 'report.md'})",
        timelines="\n\n".join(timelines) or "Not requested.", kernels="\n\n".join(operators) or "Not requested.",
        sol=f"See the same workload's [unprofiled speed-of-light estimate]({directory / 'report.md'}#speed-of-light-estimate). Counter utilization is not substituted for an end-to-end physical bound.",
        measured=" ".join(measured) or "No usable timeline collected; consult capture status and logs.",
        reproduce=f"Original frozen requests: `{directory / 'jobs'}`. Diagnostic requests, commands, logs and native profiler files are alongside `diagnosis.json`.\n\n"
                  f"```bash\npython bench.py diagnose --run {directory} --case {case_id}" + (" --no-timeline" if not timeline else "") + (" --kernels " + " ".join(kernels) if kernels else "") + (" --counters " + " ".join(counters) if counters else "") + "\n```\n")
    (output / "diagnosis.md").write_text(report)
    print(f"Diagnosis: {output / 'diagnosis.md'}")
    return output, records
