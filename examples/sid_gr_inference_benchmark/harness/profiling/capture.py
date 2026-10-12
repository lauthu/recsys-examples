"""Run Nsight Systems inside the solution environment and export its timeline."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from harness.artifacts import read, write


def capture(request_path, output_path):
    if not shutil.which("nsys"):
        return {
            "status": "unsupported",
            "reason": "nsys is not installed in this solution environment",
        }
    directory = Path(output_path).parent
    report = str(directory / "timeline")
    cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--cuda-graph-trace=node",
        "--cuda-flush-interval=100",
        "--sample=none",
        "--force-overwrite=true",
        "--output",
        report,
        sys.executable,
        "-m",
        "harness.worker",
        "--request",
        str(request_path),
        "--output",
        str(output_path),
        "--profile",
    ]
    write(directory / "profiler_command.json", {"argv": cmd})
    subprocess.run(cmd, check=True)
    result = read(output_path)
    if result["status"] == "success":
        subprocess.run(
            [
                "nsys",
                "export",
                "--type",
                "sqlite",
                "--force-overwrite=true",
                "--output",
                report + ".sqlite",
                report + ".nsys-rep",
            ],
            check=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = capture(args.request, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        result = {
            "status": "failed",
            "reason": f"Nsight capture failed: {exc}; see worker.log",
        }
    result["profiled"] = True
    write(args.output, result)
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
