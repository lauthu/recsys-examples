"""Resolve cases once, then run each solution serially on the selected GPU."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import fnmatch
import importlib
from pathlib import Path
import uuid

from artifacts import digest, model_info, read, source_info, suite_hash, write
from config import ROOT, cases
import runtime
from validation import Unsupported
from workload import build


def select_cases(cfg, pattern=None, smoke=False):
    selected = [c for c in cases(cfg) if pattern is None or fnmatch.fnmatchcase(c["id"], pattern)]
    if not selected:
        raise ValueError(f"No cases match {pattern!r}")
    return selected[:1] if smoke else selected


def launch(request, directory, *, profile=None, kernel=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    req, output = directory / "request.json", directory / "result.json"
    write(req, request)
    try:
        cmd, env, container = runtime.command(req, output, request, profile=profile, kernel=kernel)
        write(directory / "command.json", {"argv": cmd})
        status = runtime.execute(cmd, env, directory / "worker.log", request["execution"]["timeout_seconds"], container)
        if status["timed_out"]:
            result = {"status": "timed_out", "reason": "Worker exceeded timeout"}
        elif output.exists():
            result = read(output)
            if status["returncode"] != 0 and result["status"] == "success":
                result = {"status": "failed", "reason": f"Process exited {status['returncode']} after writing results"}
        else:
            result = {"status": "failed", "reason": f"Worker exited {status['returncode']} without a result; see worker.log"}
    except (OSError, ValueError) as exc:
        result = {"status": "failed", "reason": str(exc)}
    result.update(case_id=request["case"]["id"], solution=request["solution"]["name"], workload_sha256=request["workload_sha256"])
    write(output, result)
    return result


def e2e(cfg, selected):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    directory = Path(cfg["output_dir"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "run_id": run_id, "mode": "e2e", "config": cfg,
                "suite_sha256": suite_hash(ROOT), "hardware": runtime.gpu_info(cfg["execution"]["gpu"]),
                "cases": [], "jobs": [], "status": "running"}
    write(directory / "manifest.json", manifest)
    model_cache, solution_cache = {}, {}
    try:
        for index, case in enumerate(selected):
            model_path = case["model"].get("path")
            if not model_path or not Path(model_path).is_dir():
                raise ValueError(f"Missing checkpoint for {case['name']}. Set its model-path environment variable; see README.")
            if model_path not in model_cache:
                print(f"Hashing checkpoint: {model_path}", flush=True)
                model_cache[model_path] = model_info(model_path)
            model = model_cache[model_path]
            workload = build(case, model["config"])
            workload_path = directory / "workloads" / (case["id"].replace("/", "_") + ".json")
            write(workload_path, workload)
            case_fingerprint = digest({"case": case, "model": model["sha256"], "workload": workload})
            manifest["cases"].append({"case": case, "model": model, "fingerprint": case_fingerprint})
            names = cfg["solutions"][index % len(cfg["solutions"]):] + cfg["solutions"][:index % len(cfg["solutions"])]
            for name in names:
                solution = copy.deepcopy(cfg["solution_configs"][name])
                job_dir = directory / "jobs" / name / case["id"]
                job_dir.mkdir(parents=True)
                request = {"case": case, "solution": solution, "model": model, "run_dir": str(directory),
                           "suite_sha256": manifest["suite_sha256"], "workload_path": str(workload_path),
                           "workload_sha256": digest(workload), "measurement": cfg["measurement"], "execution": cfg["execution"]}
                print(f"{case['id']} / {name}", flush=True)
                try:
                    if not solution["enabled"]:
                        raise Unsupported(solution.get("reason", "Adapter not implemented"))
                    importlib.import_module(solution["adapter"]).check_support(case, solution)
                    if name not in solution_cache:
                        runtime.freeze(solution)
                        source = source_info(solution["source"].get("path"))
                        expected = solution["source"].get("expected_commit")
                        if expected and (not source or source.get("commit") != expected or source.get("dirty")):
                            raise ValueError(f"{name}: source does not match clean expected_commit")
                        solution_cache[name] = (solution, source)
                    request["solution"], request["source"] = copy.deepcopy(solution_cache[name])
                    result = launch(request, job_dir)
                except Unsupported as exc:
                    result = {"status": "unsupported", "reason": str(exc)}
                    write(job_dir / "result.json", result)
                except Exception as exc:
                    result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                    write(job_dir / "result.json", result)
                manifest["jobs"].append({"case_id": case["id"], "solution": name, "directory": str(job_dir.relative_to(directory)), "status": result["status"]})
                write(directory / "manifest.json", manifest)
                print(f"  {result['status']}: {result.get('reason', 'completed')}", flush=True)
                if result["status"] in {"failed", "timed_out"} and not cfg["execution"]["continue_on_error"]:
                    raise RuntimeError(result.get("reason", "Worker failed"))
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", reason=str(exc))
        raise
    finally:
        write(directory / "manifest.json", manifest)
        from report import render
        render(directory)
        print(f"Report: {directory / 'report.md'}", flush=True)
    return directory, manifest
