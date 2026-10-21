# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration, checkpoint and workload checks on the host.

Two kinds of YAML files: a scenario (configs/scenarios/: what is measured,
shared by every solution in a comparison) and a solution (configs/solutions/:
how one engine is launched). Each file loads into the frozen dataclasses below;
unknown or missing keys, wrong types and out-of-range values fail before any
container starts. Engine-specific rules live in the engine's `validate`.
"""

import dataclasses
import random
import re
import types
import typing
from dataclasses import dataclass
from typing import Any

import yaml
from bench import engines
from bench.common import read_json
from bench.engines import SIDTRIE_FILE, fail

SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
MANAGED_ENV = set(
    """XDG_CACHE_HOME HF_HOME TORCH_HOME TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR
    TORCHINDUCTOR_CACHE_DIR CUDA_CACHE_PATH FLASHINFER_WORKSPACE_BASE TVM_FFI_CACHE_DIR
    CUTE_DSL_CACHE_DIR QUACK_CACHE_DIR SGLANG_CACHE_DIR SGLANG_JIT_CACHE_DIR
    NUMBA_CACHE_DIR MPLCONFIGDIR
    BENCH_PROFILE BENCH_ENGINE PYTHONPATH PYTHON_BIN HOME CUDA_VISIBLE_DEVICES
    MODEL_DIR GR_MODEL_DIR GR_DECODE_ATTEN_ROOT GR_HTTP_HOST GR_HTTP_PORT GR_DEVICE
    GR_CONTEXT_LEN GR_BEAM_WIDTH GR_DECODE_STEPS GR_MAX_BATCH_SIZE GR_DECODE_BACKEND
    GR_BEAM_KV_POOL_CAPACITY GR_CONTEXT_KV_POOL_CAPACITY GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS
    GR_SUPPRESS_SPECIAL_TOKENS_ON_IGNORE_EOS GR_CATALOG_JSONL GR_ALLOW_CATALOG_RELOAD
    SGLANG_PROFILE_V2 SGLANG_BEAM_TRIE_TOPK_LOGPROB""".split()
)


def check(condition, message):
    if not condition:
        fail(message)


def at_least(config, name, **minimums):
    for key, minimum in minimums.items():
        check(getattr(config, key) >= minimum, f"{name}.{key} must be >= {minimum}")


# ---- scenario ------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """Hub checkpoint pin; the runner hashes the local files against it."""

    id: str
    revision: str  # full Hub commit ID
    sha256: dict[str, str]  # config.json, weights and index file -> SHA-256

    def __post_init__(self):
        check(self.id, "model.id must be a Hub model ID such as Qwen/Qwen3-0.6B")
        check(
            SHA1.fullmatch(self.revision),
            "model.revision must be a full 40-character Hub commit ID",
        )
        check(
            "config.json" in self.sha256,
            "model.sha256 must map file names, including config.json, to SHA-256",
        )
        check(
            all(SHA256.fullmatch(value) for value in self.sha256.values()),
            "model.sha256 values must be 64-character lowercase SHA-256 digests",
        )


@dataclass(frozen=True)
class Workload:
    context_length: int  # prompt tokens per request
    beam_width: int
    output_tokens: int  # generated tokens per beam: one from prefill, the rest decode
    seed: int  # seeds the synthetic token IDs

    def __post_init__(self):
        minimums = dict(context_length=1, beam_width=1, output_tokens=2, seed=0)
        at_least(self, "workload", **minimums)


@dataclass(frozen=True)
class Runs:
    clean: int  # separate server processes measured for latency
    requests: int  # measured requests per clean process
    warmups: int  # unmeasured requests before them
    profile_requests: int  # requests captured by the profiled process

    def __post_init__(self):
        minimums = dict(clean=1, requests=1, warmups=0, profile_requests=1)
        at_least(self, "runs", **minimums)


@dataclass(frozen=True)
class Cpus:
    server: list[int]
    client: list[int]

    def __post_init__(self):
        for key in ("server", "client"):
            cpus = getattr(self, key)
            check(
                cpus and len(set(cpus)) == len(cpus) and min(cpus) >= 0,
                f"cpus.{key} must be a nonempty list of distinct CPU numbers",
            )
        check(
            not set(self.server) & set(self.client),
            "cpus.server and cpus.client must not overlap",
        )


@dataclass(frozen=True)
class Scenario:
    """What is measured; shared by every solution in a run."""

    model: Model
    workload: Workload
    runs: Runs
    cpus: Cpus
    gpu: str  # one device index such as "0", or a GPU UUID

    def __post_init__(self):
        check(
            re.fullmatch(r"\d+|GPU-[\w-]+", self.gpu),
            'gpu must be one device index such as "0" or a GPU UUID',
        )


# ---- solution ------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    """Engine source pin; the checkout must match it (see provenance)."""

    url: str
    commit: str  # full commit ID
    # Branch the commit was taken from; images/build.sh names dev-branch images by it.
    branch: str | None = None

    def __post_init__(self):
        check(self.url.startswith("https://"), "source.url must be an https:// URL")
        check(self.branch != "", "source.branch must not be empty")
        check(
            SHA1.fullmatch(self.commit),
            "source.commit must be a full 40-character lowercase commit ID",
        )


@dataclass(frozen=True)
class TrieConfig:
    """SIDTrie token layout: depth d uses [token_start + d * size, ... + size)."""

    token_start: int
    codebook_size: int

    def __post_init__(self):
        at_least(self, "trie_config", token_start=0, codebook_size=1)


@dataclass(frozen=True)
class Solution:
    """How one engine is launched."""

    engine: str  # a name in engines.ENGINES
    source: Source
    env: dict[str, str]  # server environment, passed verbatim
    # SGLang ServerArgs, passed verbatim; SGLang validates names and values.
    server_settings: dict[str, Any] | None = None
    trie_config: TrieConfig | None = None  # sglang-sidtrie only

    def __post_init__(self):
        engines.get(self.engine)
        managed = sorted(MANAGED_ENV & self.env.keys())
        check(not managed, f"env sets runner-managed variables {managed}")


# ---- loading -------------------------------------------------------------
class _Loader(yaml.SafeLoader):
    """Safe YAML that rejects duplicate keys instead of keeping the last one."""

    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        duplicates = sorted({str(key) for key in keys if keys.count(key) > 1})
        if duplicates:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate keys {duplicates}", node.start_mark
            )
        return super().construct_mapping(node, deep)


def read_yaml(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return yaml.load(stream, Loader=_Loader)
    except yaml.YAMLError as exc:
        fail(f"{path}: {exc}")


def parse(cls, value, name):
    """Build dataclass `cls` from a mapping, checking keys and field types."""
    check(isinstance(value, dict), f"{name} must be a mapping")
    fields = {field.name: field for field in dataclasses.fields(cls)}
    required = {
        key for key, field in fields.items() if field.default is dataclasses.MISSING
    }
    unknown, missing = sorted(value.keys() - fields.keys()), sorted(
        required - value.keys()
    )
    check(not unknown, f"{name} has unknown keys {unknown}")
    check(not missing, f"{name} is missing keys {missing}")
    hints = typing.get_type_hints(cls)
    return cls(
        **{
            key: convert(hints[key], item, f"{name}.{key}")
            for key, item in value.items()
        }
    )


def convert(kind, value, name):
    """Check `value` against a field type: scalars, list, dict, dataclass, X | None."""
    origin, args = typing.get_origin(kind), typing.get_args(kind)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        (kind,) = [arg for arg in args if arg is not type(None)]
        return convert(kind, value, name)
    if dataclasses.is_dataclass(kind):
        return parse(kind, value, name)
    if origin is list:
        check(isinstance(value, list), f"{name} must be a list")
        return [convert(args[0], item, f"{name}[{i}]") for i, item in enumerate(value)]
    if origin is dict:
        check(isinstance(value, dict), f"{name} must be a mapping")
        check(all(type(key) is str for key in value), f"{name} keys must be strings")
        return {
            key: convert(args[1], item, f"{name}.{key}") for key, item in value.items()
        }
    if kind is Any:
        return value
    # Exact types: YAML reads `true` as bool and `1` as int, never as str.
    hint = ' (quote it, for example "1")' if kind is str else ""
    check(
        type(value) is kind,
        f"{name} must be {kind.__name__}, not {type(value).__name__}{hint}",
    )
    return value


def load_scenario(path):
    return parse(Scenario, read_yaml(path), f"scenario {path.name}")


def load_solution(path, workload):
    """How one engine is launched; returns (engine, Solution)."""
    solution = parse(Solution, read_yaml(path), f"solution {path.name}")
    engine = engines.get(solution.engine)
    sections = {
        key
        for key in ("server_settings", "trie_config")
        if getattr(solution, key) is not None
    }
    check(
        sections == set(engine.sections),
        f"{engine.name} solutions take exactly the sections "
        f"{sorted(engine.sections)} besides engine, source and env",
    )
    engine.validate(solution, workload)
    return engine, solution


def load_model(model, workload):
    config = read_json(model / "config.json")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        fail("the model checkpoint must be bfloat16")
    names = [p.name for p in model.iterdir() if p.suffix == ".safetensors"]
    if not set(names) - {SIDTRIE_FILE}:
        fail(f"{model} has no safetensors weights")
    length = workload.context_length + workload.output_tokens
    if length > config.get("max_position_embeddings", length):
        fail("context_length + output_tokens exceeds max_position_embeddings")
    if workload.beam_width > config["vocab_size"]:
        fail("beam_width exceeds the vocabulary size")
    return config


def make_workload(scenario, model_config):
    """Seeded random token IDs without special tokens; identical for every engine."""
    workload, runs = scenario.workload, scenario.runs
    special = set()
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = model_config.get(key)
        special.update(value if isinstance(value, list) else [value])
    allowed = [t for t in range(model_config["vocab_size"]) if t not in special]
    rng = random.Random(workload.seed)
    count = runs.warmups + max(runs.requests, runs.profile_requests)
    rows = [rng.choices(allowed, k=workload.context_length) for _ in range(count)]
    return [
        {"request_id": f"request-{i}", "input_ids": row} for i, row in enumerate(rows)
    ]
