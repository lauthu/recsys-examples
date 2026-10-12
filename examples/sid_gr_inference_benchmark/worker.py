"""Isolated GPU worker: common timing and validation for every engine."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import sys
import time
import traceback

from artifacts import digest, read, write
from validation import Unsupported, check


def environment(torch):
    packages = {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in importlib.metadata.distributions()
    }
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "engine_env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("GR_INFERENCE_", "SGLANG_"))
            and not k.endswith("_CACHE_DIR")
        },
    }


def run(request, profile=False):
    from pathlib import Path

    from artifacts import suite_hash

    if suite_hash(Path(__file__).parent) != request["suite_sha256"]:
        raise ValueError("Benchmark implementation changed after request creation")
    case, solution = request["case"], request["solution"]
    adapter_module = importlib.import_module(solution["adapter"])
    adapter_module.check_support(case, solution)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this solution's runtime")
    torch.set_grad_enabled(False)
    torch.manual_seed(case["inputs"].get("seed", 42))
    env = environment(torch)
    if request.get("expected_environment") and env != request["expected_environment"]:
        raise ValueError("Runtime environment drifted from the original e2e run")
    workload = read(request["workload_path"])
    if digest(workload) != request["workload_sha256"]:
        raise ValueError("Shared input workload checksum changed")
    started = time.perf_counter()
    adapter = adapter_module.Adapter(request)
    setup_ms = (time.perf_counter() - started) * 1000
    samples, outputs, profiling_started = [], [], False
    try:
        provenance = adapter.provenance()
        if (
            request.get("expected_adapter")
            and provenance != request["expected_adapter"]
        ):
            raise ValueError(
                "Adapter source or kernel dependency drifted from the e2e run"
            )
        for _ in range(request["measurement"]["warmup_runs"]):
            warm = adapter.generate(workload)
            adapter.synchronize()
            check(warm, workload, case, request["model"]["config"]["vocab_size"])
        if profile and hasattr(adapter, "profile_start"):
            adapter.profile_start()
            profiling_started = True
        for _ in range(request["measurement"]["repeats"]):
            adapter.synchronize()
            if profile:
                torch.cuda.nvtx.range_push("sid_bench/measured")
            start = time.perf_counter()
            try:
                result = adapter.generate(workload)
                adapter.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
            finally:
                if profile:
                    torch.cuda.nvtx.range_pop()
            check(result, workload, case, request["model"]["config"]["vocab_size"])
            samples.append(elapsed)
            outputs.append(result)
        return {
            "status": "success",
            "samples_ms": samples,
            "outputs": outputs,
            "setup_ms": setup_ms,
            "environment": env,
            "adapter": provenance,
            "validation": "all warmup and measured outputs structurally valid",
            "timing_scope": "CPU token IDs through completed CPU beam IDs/scores; includes adapter conversion",
            "profiled": profile,
        }
    finally:
        try:
            if profiling_started:
                adapter.profile_stop()
        finally:
            adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    try:
        request = read(args.request)
        result = run(request, args.profile)
    except Unsupported as exc:
        result = {"status": "unsupported", "reason": str(exc)}
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    write(args.output, result)
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
