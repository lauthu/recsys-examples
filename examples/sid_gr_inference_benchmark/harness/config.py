"""Small, strict YAML configuration and deterministic case expansion."""

from __future__ import annotations

import copy
import itertools
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


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


def expand(value, environment=None):
    if isinstance(value, str):

        def replace(match):
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                selected = os.environ[name]
            elif default is not None:
                selected = default
            else:
                raise ValueError(f"Set environment variable {name}")
            if environment is not None:
                environment[name] = selected
            return selected

        return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)(?::-([^}]*))?\}", replace, value)
    if isinstance(value, list):
        return [expand(x, environment) for x in value]
    if isinstance(value, dict):
        return {k: expand(v, environment) for k, v in value.items()}
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


def read_yaml(path, environment=None):
    with Path(path).open() as stream:
        try:
            result = expand(yaml.load(stream, Loader=UniqueLoader), environment)
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return result


def resolve(path, base=ROOT):
    return str((Path(base) / Path(path).expanduser()).resolve())


def scenario(name, environment=None):
    path = ROOT / "config/scenarios" / f"{name}.yml"
    cfg = read_yaml(path, environment)
    fields(
        cfg,
        ["name", "model", "inputs", "generation", "measurement"],
        label=str(path),
    )
    if cfg["name"] != name:
        raise ValueError(f"Scenario name must match filename: {name}")
    fields(
        cfg["model"],
        ["id", "revision", "dtype", "quantization", "kv_cache_dtype"],
        optional=["path"],
        label="model",
    )
    if not isinstance(cfg["model"]["id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", cfg["model"]["id"]
    ):
        raise ValueError("model.id must be a Hugging Face repository ID")
    if not isinstance(cfg["model"]["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", cfg["model"]["revision"]
    ):
        raise ValueError("model.revision must be a full immutable commit SHA")
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
    fields(cfg["measurement"], ["warmup_runs", "repeats"])
    integer(cfg["measurement"]["warmup_runs"], "warmup_runs", 0)
    integer(cfg["measurement"]["repeats"], "repeats")
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
    if type(cfg["generation"]["ignore_eos"]) is not bool:
        raise ValueError("ignore_eos must be a boolean")
    if cfg["model"].get("path"):
        cfg["model"]["path"] = resolve(cfg["model"]["path"])
    if cfg["inputs"].get("path"):
        cfg["inputs"]["path"] = resolve(cfg["inputs"]["path"])
    return cfg


def solution(name, environment=None):
    cfg = read_yaml(ROOT / "config/solutions" / f"{name}.yml", environment)
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
    environment = {}
    cfg = read_yaml(path, environment)
    fields(
        cfg,
        [
            "scenarios",
            "solutions",
            "baseline",
            "hardware",
            "execution",
            "output_dir",
        ],
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
    fields(cfg["execution"], ["gpu", "timeout_seconds", "continue_on_error"])
    integer(cfg["execution"]["timeout_seconds"], "timeout_seconds")
    if type(cfg["execution"]["continue_on_error"]) is not bool:
        raise ValueError("continue_on_error must be a boolean")
    name = cfg["hardware"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("hardware must name a profile in config/hardware/")
    hardware = read_yaml(ROOT / "config/hardware" / f"{name}.yml", environment)
    fields(
        hardware, ["gpu_name", "source", "cpu_cores"], optional=["dense_bf16_tflops"]
    )
    integer(hardware["cpu_cores"], "hardware.cpu_cores")
    if not all(
        isinstance(hardware[k], str) and hardware[k].strip()
        for k in ["gpu_name", "source"]
    ):
        raise ValueError("hardware requires a GPU name and specification source")
    if "dense_bf16_tflops" in hardware:
        peak = hardware["dense_bf16_tflops"]
        if (
            isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not 0 < peak < float("inf")
        ):
            raise ValueError("dense_bf16_tflops must be finite and positive")
    cfg["hardware_config"] = hardware
    profile = Path(path).resolve()
    cfg["run_profile"] = str(
        profile.relative_to(ROOT) if profile.is_relative_to(ROOT) else profile
    )
    cfg["output_dir"] = resolve(cfg["output_dir"], Path(path).resolve().parent)
    cfg["scenario_configs"] = {
        name: scenario(name, environment) for name in cfg["scenarios"]
    }
    cfg["solution_configs"], cfg["environment"] = {}, {}
    for name in cfg["solutions"]:
        values = {}
        cfg["solution_configs"][name] = solution(name, values)
        cfg["environment"][name] = {**environment, **values}
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
