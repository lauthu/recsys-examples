"""Launch one isolated solution; terminate its children on timeout."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import uuid
from pathlib import Path

from harness.config import ROOT


def freeze(solution):
    runtime = solution["runtime"]
    if runtime["kind"] == "docker":
        # The image must already exist. Never silently pull a moving tag.
        image = json.loads(
            subprocess.check_output(
                ["docker", "image", "inspect", runtime["image"]], text=True
            )
        )[0]
        runtime["requested_image"] = runtime["image"]
        runtime["image"] = image["Id"]
        runtime["repo_digests"] = image.get("RepoDigests", [])


def command(request_path, output_path, request, *, profile=False):
    solution, execution = request["solution"], request["execution"]
    runtime = solution["runtime"]
    job_dir = Path(output_path).parent
    cache = Path(request["run_dir"]) / "cache" / solution["name"]
    cache.mkdir(parents=True, exist_ok=True)
    env = {
        # Docker's numeric --user may have no passwd entry; getpass needs USER.
        "USER": os.environ.get("USER", str(os.getuid())),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "XDG_CACHE_HOME": str(cache),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "HF_HOME": str(cache / "huggingface"),
        "SGLANG_CACHE_DIR": str(cache / "sglang"),
        "SGLANG_JIT_CACHE_DIR": str(cache / "sglang_jit"),
        "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
        "QUACK_CACHE_DIR": str(cache / "quack"),
        "CUTE_DSL_CACHE_DIR": str(cache / "cute"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "TVM_FFI_CACHE_DIR": str(cache / "tvm_ffi"),
        **solution.get("env", {}),
        "CUDA_VISIBLE_DEVICES": execution["gpu"],
    }
    worker = [
        runtime["python"],
        "-m",
        "profiling.capture" if profile else "harness.worker",
        "--request",
        str(request_path),
        "--output",
        str(output_path),
    ]
    container = None
    if runtime["kind"] == "docker":
        container = "sid-bench-" + uuid.uuid4().hex[:12]
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container,
            "--gpus",
            f"device={execution['gpu']}",
            "--cpuset-cpus",
            ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))),
            "--shm-size",
            "4g",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--entrypoint",
            worker[0],
        ]
        mounts = {
            str(ROOT.parents[1]): "ro",
            str(Path(request["model"]["path"]).resolve()): "ro",
        }
        if solution["source"].get("path"):
            mounts[solution["source"]["path"]] = "ro"
        for path in runtime.get("mounts", []):
            mounts[path] = "ro"
        # The original request/workload may live outside this diagnosis directory.
        mounts[request["run_dir"]] = "ro"
        mounts[str(cache.resolve())] = "rw"
        mounts[str(job_dir.resolve())] = "rw"
        for path, mode in mounts.items():
            cmd += [
                "--mount",
                f"type=bind,source={path},target={path},readonly"
                if mode == "ro"
                else f"type=bind,source={path},target={path}",
            ]
        for key, value in env.items():
            cmd += ["--env", f"{key}={value}"]
        cmd += ["--workdir", str(ROOT), runtime["image"], *worker[1:]]
        return cmd, os.environ.copy(), container
    env = {**os.environ, **env}
    return worker, env, container


def execute(cmd, env, log, timeout, container=None):
    with Path(log).open("w") as stream:
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            return {"returncode": code, "timed_out": False}
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if container:
                subprocess.run(
                    ["docker", "rm", "-f", container],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            if isinstance(exc, KeyboardInterrupt):
                raise
            return {"returncode": process.returncode, "timed_out": True}


def gpu_info(gpu):
    if not isinstance(gpu, str) or not gpu.startswith("GPU-"):
        raise ValueError(
            "Set execution.gpu (BENCHMARK_GPU) to a full GPU UUID from nvidia-smi -L"
        )
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        name, identity, driver, memory = [p.strip() for p in result.strip().split(",")]
        if identity != gpu:
            raise ValueError("Use the full GPU UUID, not an abbreviation")
        return {
            "name": name,
            "uuid": identity,
            "driver": driver,
            "memory_mib": float(memory),
        }
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot identify selected GPU {gpu}: {exc}") from exc


def cpu_info():
    """Record host topology and the CPUs available to the benchmark process."""
    info = json.loads(
        subprocess.check_output(
            ["lscpu", "--json"], text=True, env={**os.environ, "LC_ALL": "C"}
        )
    )
    fields = {row["field"].rstrip(":"): row["data"] for row in info["lscpu"]}
    return {
        "model": fields["Model name"],
        "sockets": int(fields["Socket(s)"]),
        "physical_cores": int(fields["Socket(s)"]) * int(fields["Core(s) per socket"]),
        "logical_cpus": int(fields["CPU(s)"]),
        "affinity": sorted(os.sched_getaffinity(0)),
    }
