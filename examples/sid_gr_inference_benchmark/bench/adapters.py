# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the benchmark talks to each engine: one adapter class per engine.

An adapter is plain logic and holds no settings. It reads the solution (how
the engine is configured) and the scenario (what is measured) on the host, and
plan.json, which the runner writes from both plus the CPUs and GPU of the run,
in the container. The runner, the client, the launchers and the report ask the
adapter instead of branching on the engine name. To add an engine, subclass
`Adapter`, fill in the attributes and methods below and list it in `ADAPTERS`;
nothing else changes.
Standard library only: this module is imported on the host and in the container.

Host side   solution checks, catalog and model preparation, mounts; these
            methods get the typed Solution and Scenario from bench.config.
Container   these methods get plan.json.
Server side launch command, readiness and profiler endpoints, request payload.
Response    beam parsing, per-run observed checks, setup rows for the report.
"""

import array
import json
import math
import pickle
import struct
import sys

from bench.common import CACHE_ENV, call, fail

TRIE_HEAD_FILE = "trie_output_head.safetensors"
SID_GR_SCRIPT = "/source/examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
SHARED = "SGLANG_BEAM_SHARED_CONTEXT_ATTENTION"
TOPK = "SGLANG_BEAM_TRIE_TOPK_LOGPROB"  # "true": normalize over the codebook only
# Image recipe of the SGLang dev branches, which images/build.sh builds from source.
FROM_SOURCE = (
    "docker/Dockerfile of the source, target framework_final (images/build.sh)"
)
# Settings the container only sees through the plan, never from the image.
AMBIENT = ("GR_", "SGLANG_", "MODEL_")


class Adapter:
    name: str  # the engine name a solution gives as `engine`
    title: str
    source_label: str  # what --source must be, for error messages
    entry: str  # file that must exist under --source
    # How the image is built: a Dockerfile under images/, or, for a dev branch
    # built from source by images/build.sh, the engine's own recipe.
    recipe: str
    mount = "/source"  # where --source appears in the container
    fingerprint: tuple[str, ...] = (
        "examples/sid-gr-inference",
        "corelib/gr_decode_atten",
    )
    # Optional solution sections this engine requires; it rejects the others.
    sections: tuple[str, ...] = ()
    # Environment the runner or the adapter sets; a solution must not.
    managed_env = frozenset(
        CACHE_ENV
        + "HOME PYTHONPATH CUDA_VISIBLE_DEVICES BENCH_PROFILE BENCH_ENGINE".split()
    )
    # The engine's code paths that launcher patches target (patching.Patch.needs).
    features: frozenset[str] = frozenset()
    ready = "/pilot/ready"
    # With a catalog, the engine serves a generated copy of the checkpoint.
    model_view = False
    catalog_supported = True
    # The catalog's native storage format: generated file name, or None to read JSONL.
    catalog_file: str | None = None

    # ---- host side -------------------------------------------------------
    def validate(self, solution, scenario):
        """Engine-specific checks of a Solution against the Scenario; raise ValueError."""

    def convert_catalog(self, paths, model, vocab_size):
        """Bytes of the engine's native catalog file (only if catalog_file is set);
        `model` is the scenario's Model."""
        raise NotImplementedError

    # ---- server side -----------------------------------------------------
    def server(self, plan, port, profile):
        """Return (argv, env) that start the server. env holds what the benchmark
        sets; the client adds it to the container's environment without AMBIENT."""
        raise NotImplementedError

    def server_args(self, plan, port):
        """SGLang ServerArgs this adapter sets over the solution's server_settings;
        None for an engine configured only through its environment."""
        return None

    def environment(self, plan, profile):
        # CPU thread pools match the server CPUs unless the solution says otherwise.
        threads = str(len(plan["server_cpus"]))
        env = {"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
        env.update(plan["env"], BENCH_PROFILE="1" if profile else "0")
        return env

    def profile(self, port, start):
        """Start or stop the cudaProfilerApi capture range in the server process."""
        raise NotImplementedError

    def payload(self, row, workload):
        raise NotImplementedError

    def beams(self, response, rid):
        """[(token_ids, cumulative logprob)] in the server's order."""
        raise NotImplementedError

    def catalog_status(self, port, runtime):
        raise NotImplementedError

    def check_source(self, runtime):
        """Raise if the server is not running the mounted source."""

    def check_counters(self, plan, counters, requests, runtime):
        """Raise if the measured requests did not take the requested code path."""

    # ---- report ----------------------------------------------------------
    def graph_usage(self, clean):
        raise NotImplementedError

    def setup_rows(self, solution, catalog):
        """Extra (label, text) rows for the report's setup table."""
        return []


class SidGrAdapter(Adapter):
    name = "sid-gr"
    recipe = "images/Dockerfile.sid-gr"
    title = "SID-GR"
    source_label = "a recsys-examples checkout"
    entry = "examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
    ready = "/ready"
    # Set from the plan below; the model and the catalog come only from the runner.
    managed_env = Adapter.managed_env | frozenset(
        """PYTHON_BIN MODEL_DIR GR_MODEL_DIR GR_DECODE_ATTEN_ROOT GR_HTTP_HOST
        GR_HTTP_PORT GR_CONTEXT_LEN GR_BEAM_WIDTH GR_DECODE_STEPS GR_MAX_BATCH_SIZE
        GR_BEAM_KV_POOL_CAPACITY GR_CONTEXT_KV_POOL_CAPACITY
        GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS GR_CATALOG_JSONL GR_ALLOW_CATALOG_RELOAD""".split()
    )

    def validate(self, solution, scenario):
        if scenario.model.dtype != "bfloat16":
            fail(
                "sid-gr builds its model in bfloat16 and has no dtype setting; "
                f"the scenario asks for model.dtype {scenario.model.dtype}"
            )

    def server(self, plan, port, profile):
        workload = plan["workload"]
        # Every request in flight shares one batch; one deployment serves every level.
        batch = max(plan["levels"])
        env = self.environment(plan, profile)
        env.update(
            {
                "PYTHON_BIN": "/benchmark-tools/bench/server_sid_gr.py",
                "MODEL_DIR": "/model",
                "GR_DECODE_ATTEN_ROOT": "/source/corelib/gr_decode_atten",
                "GR_HTTP_HOST": "127.0.0.1",
                "GR_HTTP_PORT": str(port),
                "GR_CONTEXT_LEN": str(workload["context_length"]),
                "GR_BEAM_WIDTH": str(workload["beam_width"]),
                "GR_DECODE_STEPS": str(workload["output_tokens"] - 1),
                "GR_MAX_BATCH_SIZE": str(batch),
                "GR_BEAM_KV_POOL_CAPACITY": str(batch),
                "GR_CONTEXT_KV_POOL_CAPACITY": str(batch),
                "GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS": ",".join(
                    str(size) for size in range(1, batch + 1)
                ),
            }
        )
        if plan["catalog"]:
            env["GR_CATALOG_JSONL"] = plan["catalog"]["server_path"]
        return ["bash", SID_GR_SCRIPT], env

    def profile(self, port, start):
        call(port, "POST", f"/pilot/profile/{'start' if start else 'stop'}")

    def payload(self, row, workload):
        params = {
            "n": workload["beam_width"],
            "max_new_tokens": workload["output_tokens"],
            "ignore_eos": True,
        }
        return {
            "input_ids": row["input_ids"],
            "rid": row["request_id"],
            "stream": False,
            "sampling_params": params,
        }

    def beams(self, response, rid):
        meta = response["meta_info"]
        if meta.get("request_id") != rid:
            fail(f"response does not belong to {rid}")
        return [
            (b["output_ids"], float(b["meta_info"]["sequence_score"]))
            for b in meta["beam_results"]
        ]

    def catalog_status(self, port, runtime):
        status = call(port, "GET", "/catalog/status")["catalog"] or {}
        return {
            "active": bool(status),
            "source": status.get("source"),
            "items": status.get("item_count"),
        }

    def check_counters(self, plan, counters, requests, runtime):
        # SID-GR keys prefill graphs by prompt length and captures a new one on first
        # sight, so with variable lengths captures would land in measured latency.
        captures = counters.get("prefill.graph_captures")
        if captures:
            raise RuntimeError(
                f"SID-GR captured {captures} prefill CUDA graphs during measured "
                "requests; for variable prompt lengths set "
                'GR_INFERENCE_DISABLE_PREFILL_CUDA_GRAPH: "1" in the solution'
            )

    def graph_usage(self, clean):
        counts = _sum_counters(clean)
        requests = sum(measured_requests(result) for result in clean)
        enabled = clean[0]["runtime"]["cuda_graphs"]
        parts = [
            (
                f"{phase}: {counts[f'{phase}.graph_replays'] / requests:.2f} replays/request"
                if enabled[phase]
                else f"{phase}: disabled"
            )
            for phase in ("prefill", "decode")
        ]
        if enabled["decode"]:
            parts[-1] += f", {counts['decode.eager_fallbacks']} eager fallbacks"
        return "; ".join(parts)


class SGLangAdapter(Adapter):
    source_label = "an SGLang checkout"
    entry = "python/sglang/srt/entrypoints/http_server.py"
    mount = "/sglang"
    fingerprint: tuple[str, ...] = ("python",)
    sections: tuple[str, ...] = ("server_settings",)
    managed_env = Adapter.managed_env | {"SGLANG_PROFILE_V2"}
    # Set by server_args; a solution must not.
    managed_settings = frozenset(
        """model_path host port dtype random_seed cuda_graph_bs_decode
        cuda_graph_bs_prefill max_running_requests beam_search_constraint_dict""".split()
    )

    def validate(self, solution, scenario):
        settings = solution.server_settings
        managed = sorted(self.managed_settings & settings.keys())
        if managed:
            fail(f"server_settings must not set {managed}: the runner sets them")
        if settings.get("tp_size", 1) != 1:
            fail("server_settings need tp_size 1: phase markers need one process")

    def server(self, plan, port, profile):
        env = self.environment(plan, profile)
        env.update(PYTHONPATH="/sglang/python:/benchmark-tools", BENCH_ENGINE=self.name)
        env["SGLANG_PROFILE_V2"] = "0"  # the cudaProfilerApi control path
        argv = [sys.executable, "-m", "bench.server_sglang"]
        return argv + ["--plan", "/work/plan.json", "--port", str(port)], env

    def server_args(self, plan, port):
        """Where the model and the port are, and what follows from the scenario.

        One deployment serves every concurrency level, so it is sized for the
        largest, top. CUDA graph sizes: k requests in flight decode k * beam_width
        rows; prefill sizes come from prefill_tokens. Request slots: every beam
        holds one, and beam-search-new also keeps the request's own; with too few,
        requests queue or the server fails.
        """
        workload, top = plan["workload"], max(plan["levels"])
        return {
            "model_path": "/model",
            "host": "127.0.0.1",
            "port": port,
            "dtype": plan["dtype"],
            "random_seed": workload["seed"],
            "cuda_graph_bs_decode": [
                workload["beam_width"] * k for k in range(1, top + 1)
            ],
            "cuda_graph_bs_prefill": prefill_tokens(workload, top),
            "max_running_requests": top * (workload["beam_width"] + 1),
        }

    def profile(self, port, start):
        # Native endpoints answer with plain text.
        body = {"activities": ["CUDA_PROFILER"]} if start else None
        call(
            port,
            "POST",
            "/start_profile" if start else "/stop_profile",
            body,
            text=True,
        )

    def payload(self, row, workload):
        params = {
            "n": workload["beam_width"],
            "max_new_tokens": workload["output_tokens"],
            "ignore_eos": True,
            **self.sampling(workload),
        }
        return {
            "input_ids": row["input_ids"],
            "rid": row["request_id"],
            "stream": False,
            "sampling_params": params,
        }

    def sampling(self, workload):
        return {}

    def beams(self, response, rid):
        meta = response["meta_info"]
        if meta.get("id") != rid:
            fail(f"response does not belong to {rid}")
        result = []
        for beam in meta["beam_results"]:
            tokens, mean = beam["output_ids"], float(
                beam["meta_info"]["sequence_score"]
            )
            # SGLang reports mean logprob per generated token; scores here are sums.
            result.append((tokens, mean * len(tokens)))
        return result

    def catalog_status(self, port, runtime):
        return runtime.get("catalog")

    def check_source(self, runtime):
        if not runtime["sglang"].startswith("/sglang/"):
            raise RuntimeError(f"imported {runtime['sglang']}, not the mounted SGLang")

    def graph_usage(self, clean):
        counts = _sum_counters(clean)
        parts = []
        for phase in ("prefill", "decode"):
            graph, eager = (
                counts[f"forward.{phase}.graph"],
                counts[f"forward.{phase}.eager"],
            )
            parts.append(
                f"{phase}: {graph:,} of {graph + eager:,} forwards as graph replays"
            )
        decodes = counts["forward.decode.graph"] + counts["forward.decode.eager"]
        if counts["forward.decode.rows"]:  # the beams of every request batched
            rows = counts["forward.decode.rows"] / decodes
            each = (
                f"{rows:g} rows each"
                if rows.is_integer()
                else f"{rows:.1f} rows on average"
            )
            parts[-1] += f", {each}"
        return "; ".join(parts)


class BeamSearchNewAdapter(SGLangAdapter):
    name = "sglang-beam-search-new"
    recipe = FROM_SOURCE
    title = "SGLang beam-search-new"
    features = frozenset({"legacy_beam"})  # the branch's use_beam_search scheduler
    catalog_file = "catalog.pkl"

    def validate(self, solution, scenario):
        super().validate(solution, scenario)
        if not solution.server_settings.get("enable_beam_search"):
            fail("sglang-beam-search-new needs server_settings.enable_beam_search true")
        # The branch turns the overlap scheduler off for beam search; say so.
        if not solution.server_settings.get("disable_overlap_schedule"):
            fail(
                "sglang-beam-search-new needs server_settings.disable_overlap_schedule "
                "true: the branch runs beam search without the overlap scheduler"
            )

    def server_args(self, plan, port):
        args = super().server_args(plan, port)
        if plan["catalog"]:
            args["beam_search_constraint_dict"] = plan["catalog"]["server_source"]
        return args

    def convert_catalog(self, paths, model, vocab_size):
        """Pickled token trie read by this branch's prefix-constrained trie."""
        root = {"t": False, "c": {}}
        nodes = 1
        for tokens in sorted(paths):
            node = root
            for token in tokens:
                if token not in node["c"]:
                    node["c"][token] = {"t": False, "c": {}}
                    nodes += 1
                node = node["c"][token]
            node["t"] = True
        lengths = {len(path) for path in paths}
        return pickle.dumps(
            {
                "tree": root,
                "num_sequences": len(paths),
                "num_nodes": nodes,
                "num_edges": nodes - 1,
                "max_depth": max(lengths),
                "min_depth": min(lengths),
            },
            protocol=4,
        )

    def sampling(self, workload):
        return {"use_beam_search": True}


class NativeSGLangAdapter(SGLangAdapter):
    """SGLang's native `beam_width` API: public SGLang and the vedantjh2 fork."""

    features = frozenset({"native_beam"})  # BeamCoordinator and BeamGroup

    def sampling(self, workload):
        return {"beam_width": workload["beam_width"]}

    def check_counters(self, plan, counters, requests, runtime):
        # Scores are the mean log-probability times the length only at penalty 1.
        if runtime["length_penalty"] != 1.0:
            raise RuntimeError(
                f"length penalty {runtime['length_penalty']} breaks mean-to-sum score conversion"
            )


class PublicSGLangAdapter(NativeSGLangAdapter):
    name = "sglang"
    recipe = "images/Dockerfile.sglang"
    title = "SGLang (public)"
    catalog_supported = False


class Vedantjh2SGLangAdapter(NativeSGLangAdapter):
    name = "vedantjh2-sglang"
    recipe = FROM_SOURCE
    title = "vedantjh2/sglang (trie-head)"
    features = NativeSGLangAdapter.features | {"trie_head", "shared_context"}
    model_view = True
    catalog_file = TRIE_HEAD_FILE

    def validate(self, solution, scenario):
        super().validate(solution, scenario)
        legacy = {"enable_beam_search", "enable_beam_cascade_attention"}
        legacy &= solution.server_settings.keys()
        if legacy:
            fail(
                f"vedantjh2-sglang uses the native beam_width API; remove {sorted(legacy)}"
            )
        for name in (SHARED, TOPK):
            if solution.env.get(name, "false") not in ("true", "false"):
                fail(f'{name} must be "true" or "false"')
        if scenario.catalog is None:  # native beam search: no trie output head
            return
        if scenario.model.sid_codebooks is None:
            fail(
                "vedantjh2-sglang with a catalog needs model.sid_codebooks in the "
                "scenario: where the SID tokens sit in the vocabulary"
            )
        if scenario.workload.output_tokens != 3:
            fail(
                "vedantjh2-sglang with a catalog needs workload.output_tokens 3: its "
                "trie output head takes exactly three SID codebooks"
            )

    def convert_catalog(self, paths, model, vocab_size):
        layout = model.sid_codebooks
        return trie_output_head(
            paths,
            token_start=layout.token_start,
            codebook_size=layout.codebook_size,
            vocab_size=vocab_size,
        )

    def shared(self, env):
        return env.get(SHARED, "false") == "true"

    def codebook_normalized(self, env):
        return env.get(TOPK, "false") == "true"

    def check_counters(self, plan, counters, requests, runtime):
        """Every measured request must take the path that was requested: the
        trie output head exactly when there is a catalog, with the requested
        normalization."""
        decode = counters["forward.decode.graph"] + counters["forward.decode.eager"]
        shared = self.shared(plan["env"])
        codebook = self.codebook_normalized(plan["env"])
        problems = []
        try:
            super().check_counters(plan, counters, requests, runtime)
        except RuntimeError as exc:
            problems.append(str(exc))
        trie = ("compact_decode", "exact_normalizer", "device_history")
        if not plan["catalog"]:
            if any(counters[f"vedantjh2.{name}"] for name in trie):
                problems.append("the trie output head ran without a catalog")
        else:
            if decode <= 0 or counters["vedantjh2.compact_decode"] != decode:
                problems.append(
                    "not every decode forward used the compact trie output head"
                )
            if codebook and counters["vedantjh2.exact_normalizer"]:
                problems.append(
                    f"the full-vocabulary normalizer ran although {TOPK} is true"
                )
            if not codebook and counters["vedantjh2.exact_normalizer"] != decode:
                problems.append(
                    "not every decode forward kept the full-vocabulary normalizer"
                )
            if counters["vedantjh2.device_history"] != requests:
                problems.append("not every request finalized beams with device history")
        if shared and (
            counters["forward.decode.graph"]
            or counters["vedantjh2.shared_attention"] < counters["forward.decode.eager"]
        ):
            problems.append(
                "shared-context attention did not run in every decode forward"
            )
        if not shared and counters["vedantjh2.shared_attention"]:
            problems.append("shared-context attention ran although it is disabled")
        if problems:
            raise RuntimeError("vedantjh2-sglang check failed: " + "; ".join(problems))

    def setup_rows(self, solution, catalog):
        if catalog:
            normalizer = (
                "codebook-only"
                if self.codebook_normalized(solution.env)
                else "full-vocabulary"
            )
            text = f"compact trie output head, {normalizer} normalization and device "
            text += "beam history on every measured request"
        else:
            text = "off: no catalog"
        state = "on" if self.shared(solution.env) else "off"
        return [
            (
                "Trie output head (observed)",
                f"{text}; shared-context attention {state}",
            )
        ]


def trie_output_head(paths, *, token_start, codebook_size, vocab_size):
    """trie_output_head.safetensors for the vedantjh2-sglang trie output head.

    Depth d of a three-token path must lie in codebook range
    [token_start + d * codebook_size, token_start + (d + 1) * codebook_size).
    The file stores range-local codes as three-level CSR int32 indices; the
    fork reads exactly three codebooks.
    """
    if token_start < 0 or codebook_size <= 0:
        raise ValueError("sid_codebooks needs token_start >= 0 and codebook_size > 0")
    if token_start + 3 * codebook_size > vocab_size:
        raise ValueError("sid_codebooks ranges exceed the model vocabulary")
    tree = {}
    for path in paths:
        if len(path) != 3:
            raise ValueError("trie output head paths must have exactly three tokens")
        codes = [
            token - token_start - depth * codebook_size
            for depth, token in enumerate(path)
        ]
        if any(not 0 <= code < codebook_size for code in codes):
            raise ValueError(
                f"token path {path} is outside the model.sid_codebooks ranges"
            )
        tree.setdefault(codes[0], {}).setdefault(codes[1], []).append(codes[2])
    roots, level1, level2 = sorted(tree), [], []
    level1_offsets, level2_offsets = [0], [0]
    for a in roots:
        for b in sorted(tree[a]):
            level1.append(b)
            level2.extend(sorted(tree[a][b]))
            level2_offsets.append(len(level2))
        level1_offsets.append(len(level1))
    arrays = {
        "root_token_ids": roots,
        "level1_offsets": level1_offsets,
        "level1_token_ids": level1,
        "level2_offsets": level2_offsets,
        "level2_token_ids": level2,
    }
    header = {
        "__metadata__": {
            "format_version": "1",
            "token_start": str(token_start),
            "codebook_size": str(codebook_size),
            "num_codebooks": "3",
        }
    }
    payload = bytearray()
    for name, values in arrays.items():
        start = len(payload)
        packed = array.array("i", values)
        if sys.byteorder != "little":
            packed.byteswap()
        payload += packed.tobytes()
        header[name] = {
            "dtype": "I32",
            "shape": [len(values)],
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + bytes(payload)


def prefill_tokens(workload, top):
    """Prefill CUDA graph sizes in tokens for up to `top` requests in flight.

    Fixed lengths: k prompts of context_length for k = 1..top. Variable lengths:
    1.5x steps from min_context_length up to top * context_length, so a batch pads
    by less than 1.5x; SGLang replays a prefill graph only below 2x padding and
    runs the batch eagerly otherwise.
    """
    longest, shortest = workload["context_length"], workload.get("min_context_length")
    if shortest is None:
        return [longest * k for k in range(1, top + 1)]
    sizes, size = [], shortest
    while size < top * longest:
        sizes.append(size)
        size = math.ceil(size * 1.5)
    return sizes + [top * longest]


def measured_requests(result):
    """Measured requests of one run's result.json, over every concurrency level."""
    return sum(len(level["latency_ms"]) for level in result["levels"])


def _sum_counters(clean):
    totals = {}
    for result in clean:
        for key, value in result["counters"].items():
            totals[key] = totals.get(key, 0) + value
    return _Counts(totals)


class _Counts(dict):
    def __missing__(self, key):
        return 0


ADAPTERS = {
    adapter.name: adapter
    for adapter in (
        SidGrAdapter(),
        PublicSGLangAdapter(),
        BeamSearchNewAdapter(),
        Vedantjh2SGLangAdapter(),
    )
}


def get(engine):
    """The adapter of an engine name, as a solution gives it in `engine`."""
    if engine not in ADAPTERS:
        fail(f"engine must be one of {', '.join(ADAPTERS)}")
    return ADAPTERS[engine]
