# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Everything the benchmark needs to know about one engine, in one class.

The runner (host), the client (container) and the report ask the engine instead
of branching on its name. To add an engine, subclass `Engine`, fill in the
attributes and methods below and list it in `ENGINES`; nothing else changes.
Standard library only: this module is imported on the host and in the container.

Host side   solution sections and checks, catalog and model preparation, mounts;
            these methods get the typed Solution and Workload from bench.config.
Container   these methods get plan.json, which the runner generates from them.
Server side launch command, readiness and profiler endpoints, request payload.
Response    beam parsing, per-run observed checks, setup rows for the report.
"""

import array
import json
import os
import pickle
import struct
import sys

from bench.common import call

TRIE_HEAD_FILE = "trie_output_head.safetensors"
SID_GR_SCRIPT = "/source/examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
SHARED = "SGLANG_BEAM_SHARED_CONTEXT_ATTENTION"
# Image recipe of the SGLang dev branches, which images/build.sh builds from source.
FROM_SOURCE = (
    "docker/Dockerfile of the source, target framework_final (images/build.sh)"
)
# Settings the container only sees through the plan, never from the image.
AMBIENT = ("GR_", "SGLANG_", "MODEL_")


def fail(message):
    raise ValueError(message)


class Engine:
    name: str
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
    ready = "/pilot/ready"
    model_view = False  # the engine serves a generated copy of the checkpoint
    catalog_supported = True
    catalog_required = False
    # The catalog's native storage format: generated file name, or None to read JSONL.
    catalog_file: str | None = None

    # ---- host side -------------------------------------------------------
    def validate(self, solution, workload):
        """Engine-specific checks of a Solution against the Workload; raise ValueError."""

    def convert_catalog(self, paths, solution, vocab_size):
        """Bytes of the engine's native catalog file (only if catalog_file is set)."""
        raise NotImplementedError

    # ---- server side -----------------------------------------------------
    def server(self, plan, port, profile):
        """Return (argv, env) that start the server."""
        raise NotImplementedError

    def environment(self, plan, profile):
        env = {k: v for k, v in os.environ.items() if not k.startswith(AMBIENT)}
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

    def setup_rows(self, solution):
        """Extra (label, text) rows for the report's setup table."""
        return []


class SidGr(Engine):
    name = "sid-gr"
    recipe = "images/Dockerfile.sid-gr"
    title = "SID-GR"
    source_label = "a recsys-examples checkout"
    entry = "examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
    ready = "/ready"

    def server(self, plan, port, profile):
        workload = plan["workload"]
        batch = workload["concurrency"]  # every request in flight shares one batch
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
                "GR_DEVICE": "cuda",
                "GR_DECODE_BACKEND": "real",
                "GR_SUPPRESS_SPECIAL_TOKENS_ON_IGNORE_EOS": "0",
            }
        )
        if plan["catalog"]:
            env["GR_CATALOG_JSONL"] = plan["catalog"]["server_path"]
        return ["bash", SID_GR_SCRIPT, "--no-executor-sync-timing"], env

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

    def graph_usage(self, clean):
        counts = _sum_counters(clean)
        requests = sum(len(result["latency_ms"]) for result in clean)
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


class SGLang(Engine):
    source_label = "an SGLang checkout"
    entry = "python/sglang/srt/entrypoints/http_server.py"
    mount = "/sglang"
    fingerprint: tuple[str, ...] = ("python",)
    sections: tuple[str, ...] = ("server_settings",)

    def validate(self, solution, workload):
        settings = solution.server_settings
        # The launcher sets these; graph sizes follow the workload (graph_sizes).
        managed = {"model_path", "host", "port", "beam_search_constraint_dict"}
        managed |= {"cuda_graph_bs_decode", "cuda_graph_bs_prefill"}
        if managed & settings.keys():
            fail(f"server_settings must not set {sorted(managed & settings.keys())}")
        if settings.get("tp_size", 1) != 1:
            fail("server_settings need tp_size 1: phase markers need one process")
        # Every beam holds a request slot, and beam-search-new keeps the request's
        # own slot too; with too few, requests queue or the server fails.
        slots = workload.concurrency * (workload.beam_width + 1)
        if settings.get("max_running_requests", slots) < slots:
            fail(
                "server_settings.max_running_requests must be at least "
                f"concurrency x (beam_width + 1) = {slots}"
            )

    def server(self, plan, port, profile):
        env = self.environment(plan, profile)
        env.update(PYTHONPATH="/sglang/python:/benchmark-tools", BENCH_ENGINE=self.name)
        env["SGLANG_PROFILE_V2"] = "0"  # the cudaProfilerApi control path
        argv = [sys.executable, "-m", "bench.server_sglang"]
        return argv + ["--plan", "/work/plan.json", "--port", str(port)], env

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
            parts[-1] += f", {counts['forward.decode.rows'] / decodes:g} rows each"
        return "; ".join(parts)


class BeamSearchNew(SGLang):
    name = "sglang-beam-search-new"
    recipe = FROM_SOURCE
    title = "SGLang beam-search-new"
    catalog_file = "catalog.pkl"

    def validate(self, solution, workload):
        super().validate(solution, workload)
        if not solution.server_settings.get("enable_beam_search"):
            fail("sglang-beam-search-new needs server_settings.enable_beam_search true")
        # The branch turns the overlap scheduler off for beam search; say so.
        if not solution.server_settings.get("disable_overlap_schedule"):
            fail(
                "sglang-beam-search-new needs server_settings.disable_overlap_schedule "
                "true: the branch runs beam search without the overlap scheduler"
            )

    def convert_catalog(self, paths, solution, vocab_size):
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


class Native(SGLang):
    """SGLang's native `beam_width` API: public SGLang and the vedantjh2 fork."""

    def sampling(self, workload):
        return {"beam_width": workload["beam_width"]}

    def check_counters(self, plan, counters, requests, runtime):
        # Scores are the mean log-probability times the length only at penalty 1.
        if runtime["length_penalty"] != 1.0:
            raise RuntimeError(
                f"length penalty {runtime['length_penalty']} breaks mean-to-sum score conversion"
            )


class SGLangPublic(Native):
    name = "sglang"
    recipe = "images/Dockerfile.sglang"
    title = "SGLang (public)"
    catalog_supported = False


class Vedantjh2SGLang(Native):
    name = "vedantjh2-sglang"
    recipe = FROM_SOURCE
    title = "vedantjh2/sglang (trie-head)"
    sections = ("server_settings", "trie_config")
    model_view = True
    catalog_required = True
    catalog_file = TRIE_HEAD_FILE

    def validate(self, solution, workload):
        super().validate(solution, workload)
        legacy = {"enable_beam_search", "enable_beam_cascade_attention"}
        legacy &= solution.server_settings.keys()
        if legacy:
            fail(
                f"vedantjh2-sglang uses the native beam_width API; remove {sorted(legacy)}"
            )
        if workload.output_tokens != 3:
            fail(
                "vedantjh2-sglang needs workload.output_tokens 3 (three SID codebooks)"
            )
        if solution.env.get(SHARED, "false") not in ("true", "false"):
            fail(f'{SHARED} must be "true" or "false"')

    def convert_catalog(self, paths, solution, vocab_size):
        trie = solution.trie_config
        return trie_output_head(
            paths,
            token_start=trie.token_start,
            codebook_size=trie.codebook_size,
            vocab_size=vocab_size,
        )

    def server(self, plan, port, profile):
        argv, env = super().server(plan, port, profile)
        # Keep exact full-vocabulary normalization; the approximation changes scores.
        env["SGLANG_BEAM_TRIE_TOPK_LOGPROB"] = "false"
        return argv, env

    def shared(self, env):
        return env.get(SHARED, "false") == "true"

    def check_counters(self, plan, counters, requests, runtime):
        """Every measured request must take the trie-head path that was requested."""
        decode = counters["forward.decode.graph"] + counters["forward.decode.eager"]
        shared = self.shared(plan["env"])
        problems = []
        try:
            super().check_counters(plan, counters, requests, runtime)
        except RuntimeError as exc:
            problems.append(str(exc))
        if decode <= 0 or counters["vedantjh2.compact_decode"] != decode:
            problems.append(
                "not every decode forward used the compact trie output head"
            )
        if counters["vedantjh2.exact_normalizer"] != decode:
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

    def setup_rows(self, solution):
        text = "compact trie output head, full-vocabulary normalizer and device beam "
        state = "on" if self.shared(solution.env) else "off"
        return [
            (
                "Trie output head (observed)",
                f"{text}history on every measured request; shared-context attention {state}",
            )
        ]


def trie_output_head(paths, *, token_start, codebook_size, vocab_size):
    """trie_output_head.safetensors for the vedantjh2-sglang trie output head.

    Depth d of a three-token path must lie in codebook range
    [token_start + d * codebook_size, token_start + (d + 1) * codebook_size).
    The file stores range-local codes as three-level CSR int32 indices.
    """
    if token_start < 0 or codebook_size <= 0:
        raise ValueError("trie_config needs token_start >= 0 and codebook_size > 0")
    if token_start + 3 * codebook_size > vocab_size:
        raise ValueError("trie_config codebook ranges exceed the model vocabulary")
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
                f"token path {path} is outside the trie_config codebook ranges"
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


def graph_sizes(workload):
    """SGLang CUDA graph sizes: k requests in flight decode k * beam_width rows
    and prefill k * context_length tokens."""
    sizes = range(1, workload["concurrency"] + 1)
    return {
        "cuda_graph_bs_decode": [workload["beam_width"] * k for k in sizes],
        "cuda_graph_bs_prefill": [workload["context_length"] * k for k in sizes],
    }


def _sum_counters(clean):
    totals = {}
    for result in clean:
        for key, value in result["counters"].items():
            totals[key] = totals.get(key, 0) + value
    return _Counts(totals)


class _Counts(dict):
    def __missing__(self, key):
        return 0


ENGINES = {
    engine.name: engine
    for engine in (SidGr(), SGLangPublic(), BeamSearchNew(), Vedantjh2SGLang())
}


def get(name):
    if name not in ENGINES:
        fail(f"engine must be one of {', '.join(ENGINES)}")
    return ENGINES[name]
