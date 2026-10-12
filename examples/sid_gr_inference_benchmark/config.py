"""Small, strict YAML configuration and deterministic case expansion."""

from __future__ import annotations

import copy
import itertools
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


class UniqueLoader(yaml.SafeLoader):
    pass


def mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)


def expand(value):
    if isinstance(value, str):

        def replace(match):
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ValueError(f"Set environment variable {name}")

        return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)(?::-([^}]*))?\}", replace, value)
    if isinstance(value, list):
        return [expand(x) for x in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def fields(value, required=(), *, optional=(), label="configuration"):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(value) - set(required) - set(optional)
    missing = set(required) - set(value)
    if unknown or missing:
        raise ValueError(
            f"{label}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
        )


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def read_yaml(path):
    with Path(path).open() as stream:
        try:
            result = expand(yaml.load(stream, Loader=UniqueLoader))
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return result


def resolve(path, base=ROOT):
    return str((Path(base) / Path(path).expanduser()).resolve())


def scenario(name):
    path = ROOT / "scenarios" / f"{name}.yml"
    cfg = read_yaml(path)
    fields(
        cfg, ["name", "model", "inputs", "generation", "prefix_cache"], label=str(path)
    )
    if cfg["name"] != name:
        raise ValueError(f"Scenario name must match filename: {name}")
    fields(
        cfg["model"],
        ["id", "revision", "dtype", "quantization", "kv_cache_dtype"],
        optional=["path"],
        label="model",
    )
    fields(
        cfg["inputs"],
        ["type", "context_lengths", "batch_sizes"],
        optional=["seed", "path"],
        label="inputs",
    )
    fields(
        cfg["generation"],
        [
            "beam_widths",
            "output_tokens",
            "return_beams",
            "ignore_eos",
            "length_penalty",
            "constraints",
            "score",
        ],
        label="generation",
    )
    if cfg["inputs"]["type"] not in {"synthetic_tokens", "jsonl"}:
        raise ValueError("inputs.type must be synthetic_tokens or jsonl")
    if cfg["inputs"]["type"] == "jsonl" and not cfg["inputs"].get("path"):
        raise ValueError("jsonl inputs require path")
    integer(cfg["inputs"].get("seed", 42), "seed", 0)
    integer(cfg["generation"]["output_tokens"], "output_tokens")
    for section, key in [
        ("inputs", "context_lengths"),
        ("inputs", "batch_sizes"),
        ("generation", "beam_widths"),
    ]:
        values = cfg[section][key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list of unique integers")
        for value in values:
            integer(value, key)
        if len(set(values)) != len(values):
            raise ValueError(f"{key} contains duplicates")
    for obj, key in [(cfg, "prefix_cache"), (cfg["generation"], "ignore_eos")]:
        if type(obj[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if cfg["model"].get("path"):
        cfg["model"]["path"] = resolve(cfg["model"]["path"])
    if cfg["inputs"].get("path"):
        cfg["inputs"]["path"] = resolve(cfg["inputs"]["path"])
    return cfg


def solution(name):
    cfg = read_yaml(ROOT / "solutions" / f"{name}.yml")
    fields(cfg, ["name", "adapter", "source", "runtime", "engine"], optional=["env"])
    if cfg["name"] != name:
        raise ValueError(f"Solution name must match filename: {name}")
    fields(cfg["source"], [], optional=["path", "expected_commit"])
    fields(cfg["runtime"], ["kind", "python"], optional=["image", "mounts"])
    if cfg["runtime"]["kind"] not in {"local", "docker"}:
        raise ValueError("runtime.kind must be local or docker")
    if cfg["runtime"]["kind"] == "docker" and not cfg["runtime"].get("image"):
        raise ValueError(f"{name}: Docker runtime requires an image")
    if not isinstance(cfg["engine"], dict) or not isinstance(cfg.get("env", {}), dict):
        raise ValueError("engine and env must be mappings")
    if cfg["source"].get("path"):
        cfg["source"]["path"] = resolve(cfg["source"]["path"])
    cfg["runtime"]["mounts"] = [
        resolve(p) for p in cfg["runtime"].get("mounts", []) if p
    ]
    cfg["env"] = {k: str(v) for k, v in cfg.get("env", {}).items() if v != ""}
    return cfg


def load(path):
    cfg = read_yaml(path)
    fields(
        cfg,
        [
            "scenarios",
            "solutions",
            "baseline",
            "measurement",
            "execution",
            "output_dir",
        ],
        optional=["hardware"],
    )
    for key in ["scenarios", "solutions"]:
        if (
            not isinstance(cfg[key], list)
            or not cfg[key]
            or any(
                not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", x)
                for x in cfg[key]
            )
        ):
            raise ValueError(f"{key} must contain simple scenario/solution names")
        if len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"{key} must be a nonempty list of unique names")
    if cfg["baseline"] not in cfg["solutions"]:
        raise ValueError("baseline must be one of solutions")
    fields(cfg["measurement"], ["warmup_runs", "repeats"])
    integer(cfg["measurement"]["warmup_runs"], "warmup_runs", 0)
    integer(cfg["measurement"]["repeats"], "repeats")
    fields(cfg["execution"], ["gpu", "timeout_seconds", "continue_on_error"])
    integer(cfg["execution"]["timeout_seconds"], "timeout_seconds")
    if type(cfg["execution"]["continue_on_error"]) is not bool:
        raise ValueError("continue_on_error must be a boolean")
    if cfg.get("hardware"):
        fields(cfg["hardware"], ["gpu_name", "dense_bf16_tflops", "source"])
        peak = cfg["hardware"]["dense_bf16_tflops"]
        if (
            isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not 0 < peak < float("inf")
        ):
            raise ValueError("dense_bf16_tflops must be finite and positive")
    cfg["output_dir"] = resolve(cfg["output_dir"], Path(path).resolve().parent)
    cfg["solution_configs"] = {name: solution(name) for name in cfg["solutions"]}
    cfg["scenario_configs"] = {name: scenario(name) for name in cfg["scenarios"]}
    return cfg


def cases(cfg):
    for name, spec in cfg["scenario_configs"].items():
        for context, batch, width in itertools.product(
            spec["inputs"]["context_lengths"],
            spec["inputs"]["batch_sizes"],
            spec["generation"]["beam_widths"],
        ):
            case = copy.deepcopy(spec)
            case.update(
                id=f"{name}/ctx{context}_b{batch}_w{width}",
                context_length=context,
                batch_size=batch,
                beam_width=width,
            )
            yield case
