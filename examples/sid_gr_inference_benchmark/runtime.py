"""Launch one isolated solution; terminate its children on timeout."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import uuid
from pathlib import Path

from config import ROOT


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
        **solution.get("env", {}),
    }
    worker = [
        runtime["python"],
        str(ROOT / "worker.py"),
        "--request",
        str(request_path),
        "--output",
        str(output_path),
    ]
    if profile:
        worker = [
            runtime["python"],
            str(ROOT / "profiling/capture.py"),
            "--request",
            str(request_path),
            "--output",
            str(output_path),
        ]
    container = None
    if runtime["kind"] == "docker":
        # Session-visible ordinals can differ from the Docker host's ordinals.
        gpu = gpu_info(execution["gpu"]).get("uuid")
        if not gpu:
            raise ValueError(
                f"Cannot resolve GPU {execution['gpu']!r} to a UUID for Docker"
            )
        container = "sid-bench-" + uuid.uuid4().hex[:12]
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container,
            "--gpus",
            f"device={gpu}",
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
    env = {**os.environ, **env, "CUDA_VISIBLE_DEVICES": str(execution["gpu"])}
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
        return {
            "name": name,
            "uuid": identity,
            "driver": driver,
            "memory_mib": float(memory),
        }
    except (OSError, ValueError, subprocess.CalledProcessError):
        return {"name": None, "uuid": None}
