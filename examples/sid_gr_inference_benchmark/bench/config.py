# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The configuration, as typed classes, and the checkpoint and workload checks.

Two kinds of YAML files: a scenario (configs/scenarios/: what is measured,
shared by every solution in an invocation) and a solution (configs/solutions/:
how one engine is configured). The CPUs and GPU of an invocation come from the
command line (Resources). Each loads into the frozen dataclasses below, so
unknown or missing keys, wrong types and out-of-range values fail before any
container starts. For each solution, the runner writes the three, with two
facts it reads from the checkpoint and the catalog, as RunConfig to
/work/run_config.json: the container reads that file, never the YAML files.
Engine-specific rules live in the engine's adapter (bench/engines).
"""

import dataclasses
import random
import re
import types
import typing
from dataclasses import dataclass
from typing import Any

from bench import engines
from bench.common import fail, read_json, write_json
from bench.engines.sglang import TRIE_HEAD_FILE

SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
# torch's names, which the servers report back as the weights' dtype.
DTYPES = ("bfloat16", "float16", "float32")


def check(condition, message):
    if not condition:
        fail(message)


def at_least(config, name, **minimums):
    for key, minimum in minimums.items():
        check(getattr(config, key) >= minimum, f"{name}.{key} must be >= {minimum}")


# ---- scenario ------------------------------------------------------------
@dataclass(frozen=True)
class SidCodebooks:
    """Where semantic-ID tokens sit in the vocabulary: the token at depth d lies
    in [token_start + d * codebook_size, token_start + (d + 1) * codebook_size)."""

    token_start: int
    codebook_size: int

    def __post_init__(self):
        at_least(self, "model.sid_codebooks", token_start=0, codebook_size=1)


@dataclass(frozen=True)
class Model:
    """Hub checkpoint pin; the runner hashes the local files against it."""

    id: str
    revision: str  # full Hub commit ID
    sha256: dict[str, str]  # config.json, weights and index file -> SHA-256
    dtype: str  # weights and activations while serving; the checkpoint may differ
    # Read by engines that decode a catalog through an SID trie output head.
    sid_codebooks: SidCodebooks | None = None

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
        check(self.dtype in DTYPES, f"model.dtype must be one of {', '.join(DTYPES)}")


@dataclass(frozen=True)
class Workload:
    context_length: int  # prompt tokens per request; the longest with min_context_length
    beam_width: int
    output_tokens: int  # generated tokens per beam: one from prefill, the rest decode
    # Requests in flight, as a closed loop of this many clients; a list measures
    # each level in turn.
    concurrency: int | list[int]
    seed: int  # seeds the synthetic token IDs and lengths
    # Variable lengths: each prompt's length is drawn uniformly from
    # [min_context_length, context_length].
    min_context_length: int | None = None

    def __post_init__(self):
        minimums = dict(context_length=1, beam_width=1, output_tokens=2, seed=0)
        at_least(self, "workload", **minimums)
        levels = self.concurrency
        levels = levels if isinstance(levels, list) else [levels]
        check(
            levels and min(levels) >= 1 and len(set(levels)) == len(levels),
            "workload.concurrency must be >= 1, or a list of distinct such values",
        )
        if self.min_context_length is not None:
            check(
                1 <= self.min_context_length <= self.context_length,
                "workload.min_context_length must be between 1 and context_length",
            )

    @property
    def levels(self):
        """The concurrency levels, ascending."""
        levels = self.concurrency
        return sorted(levels if isinstance(levels, list) else [levels])


@dataclass(frozen=True)
class Runs:
    # Rounds of measurement. Each round starts a new container and server and
    # measures every concurrency in turn; the results pool the rounds.
    rounds: int
    requests: int  # measured requests per concurrency in each round
    # Unmeasured requests per client before each concurrency's measured ones, so
    # concurrency k starts with k times as many and runs full before measuring.
    warmups: int
    profile_requests: int  # requests each GPU profile captures
    # The concurrency the GPU profile under load captures; default: the largest.
    profile_concurrency: int | None = None

    def __post_init__(self):
        minimums = dict(rounds=1, requests=1, warmups=0, profile_requests=1)
        at_least(self, "runs", **minimums)

    def warmups_at(self, concurrency):
        """Unmeasured requests before the measured ones at this concurrency."""
        return self.warmups * concurrency


@dataclass(frozen=True)
class CatalogPin:
    """The catalog every beam is constrained to; --catalog gives its path."""

    sha256: str

    def __post_init__(self):
        check(
            SHA256.fullmatch(self.sha256),
            "catalog.sha256 must be a 64-character lowercase SHA-256 digest",
        )


@dataclass(frozen=True)
class Scenario:
    """What is measured; shared by every solution in a run."""

    model: Model
    workload: Workload
    runs: Runs
    catalog: CatalogPin | None = None

    def __post_init__(self):
        chosen = self.runs.profile_concurrency
        check(
            chosen is None or chosen in self.workload.levels,
            "runs.profile_concurrency must be one of workload.concurrency",
        )


# ---- command line ----------------------------------------------------------
@dataclass(frozen=True)
class Resources:
    """CPUs and GPU of one invocation: they belong to the machine, not to what
    is measured, so they come from the command line."""

    server_cpus: list[int]
    client_cpus: list[int]
    gpu: str  # one device index such as "0", or a GPU UUID

    def __post_init__(self):
        for key in ("server_cpus", "client_cpus"):
            cpus = getattr(self, key)
            check(
                cpus and len(set(cpus)) == len(cpus) and min(cpus) >= 0,
                f"--{key.replace('_', '-')} must be distinct CPU numbers",
            )
        check(
            not set(self.server_cpus) & set(self.client_cpus),
            "--server-cpus and --client-cpus must not overlap",
        )
        check(
            re.fullmatch(r"\d+|GPU-[\w-]+", self.gpu),
            "--gpu must be one device index such as 0 or a GPU UUID",
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
class Solution:
    """How one engine is configured; nothing in it depends on the model, the
    workload or the machine."""

    engine: str  # a name in bench.engines.ENGINES
    source: Source
    env: dict[str, str]  # server environment, passed verbatim
    # SGLang ServerArgs, passed verbatim; SGLang validates names and values.
    server_settings: dict[str, Any] | None = None

    def __post_init__(self):
        managed = sorted(engines.get(self.engine).managed_env & self.env.keys())
        check(not managed, f"env sets runner-managed variables {managed}")


@dataclass(frozen=True)
class RunConfig:
    """One solution's run as the container reads it: the scenario, the solution
    and the resources, with the vocabulary size from the checkpoint and the
    item count of the catalog, if the scenario pins one."""

    scenario: Scenario
    solution: Solution
    resources: Resources
    vocab_size: int
    catalog_items: int | None = None

    @property
    def workload(self):
        return self.scenario.workload

    def write(self, path):
        write_json(path, dataclasses.asdict(self))

    @classmethod
    def read(cls, path):
        return parse(cls, read_json(path), "run config")


# ---- loading -------------------------------------------------------------
def read_yaml(path):
    """Safe YAML that rejects duplicate keys instead of keeping the last one."""
    import yaml  # here, not above: the container reads only JSON, without PyYAML

    class Loader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
            duplicates = sorted({str(key) for key in keys if keys.count(key) > 1})
            if duplicates:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate keys {duplicates}", node.start_mark
                )
            return super().construct_mapping(node, deep)

    try:
        with open(path, encoding="utf-8") as stream:
            return yaml.load(stream, Loader=Loader)
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
    """Check `value` against a field type: scalars, list, dict, dataclass, unions."""
    origin, args = typing.get_origin(kind), typing.get_args(kind)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        kinds = [arg for arg in args if arg is not type(None)]
        if len(kinds) > 1:  # such as int | list[int]: the YAML type picks one
            names = " or ".join(getattr(k, "__name__", str(k)) for k in kinds)
            kinds = [k for k in kinds if (typing.get_origin(k) or k) is type(value)]
            check(kinds, f"{name} must be {names}, not {type(value).__name__}")
        return convert(kinds[0], value, name)
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


def load_solution(path, scenario):
    """How one engine is configured; returns (its adapter, Solution)."""
    solution = parse(Solution, read_yaml(path), f"solution {path.name}")
    adapter = engines.get(solution.engine)
    sections = {"server_settings"} if solution.server_settings is not None else set()
    check(
        sections == set(adapter.sections),
        f"{adapter.name} solutions take exactly the sections "
        f"{sorted(adapter.sections)} besides engine, source and env",
    )
    adapter.validate(solution, scenario)
    return adapter, solution


def load_model(model, workload):
    config = read_json(model / "config.json")
    names = [p.name for p in model.iterdir() if p.suffix == ".safetensors"]
    if not set(names) - {TRIE_HEAD_FILE}:
        fail(f"{model} has no safetensors weights")
    length = workload.context_length + workload.output_tokens
    if length > config.get("max_position_embeddings", length):
        fail("context_length + output_tokens exceeds max_position_embeddings")
    if workload.beam_width > config["vocab_size"]:
        fail("beam_width exceeds the vocabulary size")
    return config


def load_concurrency(scenario):
    """The concurrency the GPU profile under load captures."""
    return scenario.runs.profile_concurrency or max(scenario.workload.levels)


def block_layout(scenario):
    """(warmup rows, rows) of each concurrency's block of requests (a level):
    every block holds the warmups the largest concurrency sends, then the
    measured requests, which so start at the same row in every block."""
    runs = scenario.runs
    warmups = runs.warmups_at(max(scenario.workload.levels))
    return warmups, warmups + max(runs.requests, runs.profile_requests)


def make_workload(scenario, model_config):
    """Seeded random token IDs without special tokens, identical for every engine.

    Each concurrency gets its own block of requests (block_layout), in order, with new IDs and tokens, so a server never sees a request twice
    and cannot answer one level from another. With min_context_length, the
    seeded lengths are the same in every block.
    """
    workload = scenario.workload
    special = set()
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = model_config.get(key)
        special.update(value if isinstance(value, list) else [value])
    allowed = [t for t in range(model_config["vocab_size"]) if t not in special]
    count = block_layout(scenario)[1]
    lengths = [workload.context_length] * count
    if workload.min_context_length is not None:
        draw = random.Random(f"{workload.seed}/lengths")
        low, high = workload.min_context_length, workload.context_length
        lengths = [draw.randint(low, high) for _ in range(count)]
    rng = random.Random(workload.seed)
    return [
        {
            "request_id": f"request-{i}",
            "input_ids": rng.choices(allowed, k=lengths[i % count]),
        }
        for i in range(count * len(workload.levels))
    ]
