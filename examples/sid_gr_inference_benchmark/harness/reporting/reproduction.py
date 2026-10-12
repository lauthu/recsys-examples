"""Show saved source identities beside machine-local configuration values."""

import os
from pathlib import Path
from urllib.parse import quote

from harness.config import ROOT


def source_link(source):
    identity = source.get("commit", source.get("tree_sha256", "unrecorded"))
    remote = source.get("remote_url")
    if remote and source.get("commit"):
        path = source.get("subdirectory", ".")
        url = f"{remote.removesuffix('.git')}/tree/{identity}"
        if path != ".":
            url += "/" + quote(path)
        label = f"[{identity[:12]}]({url})"
    else:
        label = f"`{identity}`"
    return label + (" + local changes" if source.get("dirty") else "")


def configuration_rows(cfg, cases, requests, results, directory):
    """Return environment, artifact and engine-option rows from frozen metadata."""
    origins, artifacts = {}, []

    def artifact(solution, component, identity, path=None):
        row = [solution, component, identity]
        if row not in artifacts:
            artifacts.append(row)
        if path:
            origins[str(Path(path).resolve())] = identity

    for record in cases:
        model = record["model"]
        if model.get("repo_id") and model.get("revision"):
            identity = (
                f"[{model['repo_id']}@{model['revision'][:12]}]"
                f"(https://huggingface.co/{model['repo_id']}/tree/{model['revision']})"
            )
        else:
            identity = f"Checkpoint SHA256 `{model['sha256']}`"
        artifact("All", "Model", identity, model.get("path"))

    for request, result in zip(requests, results):
        name = request["solution"]["name"]
        source = request.get("source") or result.get("adapter", {}).get("source")
        if source:
            artifact(name, "Engine source", source_link(source), source.get("path"))
        kernel = result.get("adapter", {}).get("decode_kernel")
        if kernel:
            artifact(name, "GR decode kernels", source_link(kernel), kernel.get("path"))
        runtime = request["solution"]["runtime"]
        if runtime["kind"] == "docker":
            image = runtime["image"]
            identity = f"`{image}`"
            if runtime.get("repo_digests"):
                identity += " · " + ", ".join(
                    f"`{value}`" for value in runtime["repo_digests"]
                )
            if runtime.get("build_recipe"):
                recipe = os.path.relpath(ROOT / runtime["build_recipe"], directory)
                identity += f" · [Build recipe]({recipe})"
            artifact(name, "Container image", identity)
            origins[runtime.get("requested_image", image)] = identity

    values, options = {}, []
    for name in cfg["solutions"]:
        solution = cfg["solution_configs"][name]
        for key, value in {
            **cfg.get("environment", {}).get(name, {}),
            **solution.get("env", {}),
        }.items():
            values.setdefault((key, value), []).append(name)
        for key, value in solution["engine"].items():
            options.append(
                [
                    name,
                    f"`{key}`",
                    f"`{str(value).lower() if isinstance(value, bool) else value}`",
                ]
            )
    environment = [
        [
            f"`{key}`",
            ", ".join(scopes),
            f"`{value}`" if value else "(unset)",
            origins.get(value, origins.get(str((ROOT / value).resolve()), "—"))
            if value
            else "—",
        ]
        for (key, value), scopes in sorted(values.items())
    ]
    return environment, artifacts, options
