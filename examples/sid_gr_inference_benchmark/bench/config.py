# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration, checkpoint and workload checks on the host.

Two kinds of JSON files: one benchmark file (what is measured, shared by every
engine in a comparison) and one file per engine (how that engine is launched).
"""

import random
import re

from bench import engines, provenance
from bench.common import read_json
from bench.engines import SIDTRIE_FILE, fail

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


def require(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        fail(f"{name} must be an integer >= {minimum}")


def require_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        fail(f"{name} needs exactly the keys {sorted(keys)}")


def load_benchmark(path):
    """What is measured: workload, run counts, CPU split and GPU."""
    config = read_json(path)
    require_keys(
        config, ("model", "workload", "runs", "cpus", "gpu"), "the benchmark config"
    )
    provenance.check_model_pin(config["model"])
    workload, runs, cpus = config["workload"], config["runs"], config["cpus"]
    minimums = {"context_length": 1, "beam_width": 1, "output_tokens": 2, "seed": 0}
    require_keys(workload, minimums, "workload")
    for key, minimum in minimums.items():
        require(workload[key], f"workload.{key}", minimum)
    minimums = {"clean": 1, "requests": 1, "warmups": 0, "profile_requests": 1}
    require_keys(runs, minimums, "runs")
    for key, minimum in minimums.items():
        require(runs[key], f"runs.{key}", minimum)
    require_keys(cpus, ("server", "client"), "cpus")
    for key, value in cpus.items():
        if not isinstance(value, list) or not value or len(set(value)) != len(value):
            fail(f"cpus.{key} must be a nonempty list of distinct CPU numbers")
        for cpu in value:
            require(cpu, f"cpus.{key} entry", 0)
    if set(cpus["server"]) & set(cpus["client"]):
        fail("cpus.server and cpus.client must not overlap")
    gpu = config["gpu"]
    if type(gpu) is not str or not re.fullmatch(r"\d+|GPU-[\w-]+", gpu):
        fail('gpu must be one device index such as "0" or a GPU UUID')
    return config


def load_engine(path, workload):
    """How one engine is launched; returns (engine, config)."""
    config = read_json(path)
    engine = engines.get(config.get("engine"))
    require_keys(config, engine.keys, f"the {engine.name} config")
    provenance.check_pin(config["source"])
    env = config["env"]
    if not isinstance(env, dict) or any(type(v) is not str for v in env.values()):
        fail('env values must be strings, for example "1" rather than true')
    if MANAGED_ENV & env.keys():
        fail(f"env sets runner-managed variables {sorted(MANAGED_ENV & env.keys())}")
    engine.validate(config, workload)
    return engine, config


def load_model(model, workload):
    config = read_json(model / "config.json")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        fail("the model checkpoint must be bfloat16")
    names = [p.name for p in model.iterdir() if p.suffix == ".safetensors"]
    if not set(names) - {SIDTRIE_FILE}:
        fail(f"{model} has no safetensors weights")
    length = workload["context_length"] + workload["output_tokens"]
    if length > config.get("max_position_embeddings", length):
        fail("context_length + output_tokens exceeds max_position_embeddings")
    if workload["beam_width"] > config["vocab_size"]:
        fail("beam_width exceeds the vocabulary size")
    return config


def make_workload(benchmark, model_config):
    """Seeded random token IDs without special tokens; identical for every engine."""
    workload, runs = benchmark["workload"], benchmark["runs"]
    special = set()
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = model_config.get(key)
        special.update(value if isinstance(value, list) else [value])
    allowed = [t for t in range(model_config["vocab_size"]) if t not in special]
    rng, length = random.Random(workload["seed"]), workload["context_length"]
    count = runs["warmups"] + max(runs["requests"], runs["profile_requests"])
    rows = [rng.choices(allowed, k=length) for _ in range(count)]
    return [
        {"request_id": f"request-{i}", "input_ids": row} for i, row in enumerate(rows)
    ]
