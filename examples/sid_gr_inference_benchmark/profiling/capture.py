"""Run profilers inside the solution environment, then retain native artifacts."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artifacts import read, write


def counter_report(stem):
    for suffix in [".ncu-rep", ".ncu-repz"]:
        path = Path(str(stem) + suffix)
        if path.is_file():
            return path
    raise ValueError("Nsight Compute did not produce a native report")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", choices=["timeline", "counters"], required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kernel")
    args = p.parse_args()
    tool = "nsys" if args.tool == "timeline" else "ncu"
    if not shutil.which(tool):
        write(args.output, {"status": "unsupported", "reason": f"{tool} is not installed in this solution environment"})
        return 0
    directory = Path(args.output).parent
    worker = [sys.executable, str(ROOT / "worker.py"), "--request", args.request, "--output", args.output]
    if args.kernel:
        worker += ["--kernel", args.kernel]
    if args.tool == "timeline":
        report = str(directory / "timeline")
        cmd = [tool, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--cuda-flush-interval=100",
               "--sample=none", "--force-overwrite=true", "--output", report,
               *worker, "--profile"]
    else:
        if args.kernel != "beam_logsumexp":
            write(args.output, {"status": "unsupported", "reason": "Counter selection is implemented for beam_logsumexp only"})
            return 0
        report = str(directory / "counters")
        cmd = [tool, "--target-processes", "all", "--section", "SpeedOfLight", "--section", "MemoryWorkloadAnalysis",
               "--kernel-name", "regex:.*(partial_logsumexp|merge_logsumexp).*", "--launch-count", "2",
               "--cache-control", "none", "--clock-control", "none", "--force-overwrite", "--export", report, *worker]
    write(directory / "profiler_command.json", {"argv": cmd})
    code = subprocess.run(cmd).returncode
    if code:
        write(args.output, {"status": "failed", "reason": f"{tool} exited {code}; see worker.log", "profiled": True})
        return 1
    result = read(args.output)
    if result["status"] == "success" and args.tool == "timeline":
        subprocess.run([tool, "export", "--type", "sqlite", "--force-overwrite=true", "--output", report + ".sqlite", report + ".nsys-rep"], check=True)
    elif result["status"] == "success":
        native = counter_report(report)
        with (directory / "counters.csv").open("w") as stream:
            subprocess.run([tool, "--import", str(native), "--csv", "--page", "raw"], stdout=stream, check=True)
        result["counter_report"] = native.name
    result["profiled"] = True
    write(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
