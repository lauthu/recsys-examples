# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Everything the benchmark needs to know about one engine, in one class.

The runner (host), the client (container) and the report ask the engine instead
of branching on its name. To add an engine, subclass `Engine`, fill in the
attributes and methods below and list it in `ENGINES`; nothing else changes.
Standard library only: this module is imported on the host and in the container.

Host side   config keys and validation, catalog and model preparation, mounts.
Server side launch command, readiness and profiler endpoints, request payload.
Response    beam parsing, per-run observed checks, setup rows for the report.
"""

import os
import sys

from bench import catalog as catalogs
from bench.common import call

SIDTRIE_FILE = "trie_output_head.safetensors"
SID_GR_SCRIPT = "/source/examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
SHARED = "SGLANG_BEAM_SHARED_CONTEXT_ATTENTION"
# Settings the container only sees through the plan, never from the image.
AMBIENT = ("GR_", "SGLANG_", "MODEL_")


def fail(message):
    raise ValueError(message)


class Engine:
    name: str
    title: str
    source_label: str  # what --source must be, for error messages
    entry: str  # file that must exist under --source
    mount = "/source"  # where --source appears in the container
    fingerprint: tuple[str, ...] = (
        "examples/sid-gr-inference",
        "corelib/gr_decode_atten",
    )
    keys: tuple[str, ...] = ("engine", "env")  # exactly the keys an engine config has
    ready = "/pilot/ready"
    model_view = False  # the engine serves a generated copy of the checkpoint
    catalog_supported = True
    catalog_required = False
    # The catalog's native storage format: generated file name, or None to read JSONL.
    catalog_file: str | None = None

    # ---- host side -------------------------------------------------------
    def validate(self, config, workload):
        """Engine-specific configuration checks; raise ValueError."""

    def convert_catalog(self, paths, config, vocab_size):
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

    def setup_rows(self, config):
        """Extra (label, text) rows for the report's setup table."""
        return []


class SidGr(Engine):
    name = "sid-gr"
    title = "SID-GR"
    source_label = "a recsys-examples checkout"
    entry = "examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
    ready = "/ready"

    def server(self, plan, port, profile):
        workload = plan["workload"]
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
                "GR_MAX_BATCH_SIZE": "1",
                "GR_BEAM_KV_POOL_CAPACITY": "1",
                "GR_CONTEXT_KV_POOL_CAPACITY": "1",
                "GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS": "1",
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
    keys: tuple[str, ...] = ("engine", "env", "server_settings")

    def validate(self, config, workload):
        settings = config["server_settings"]
        managed = {"model_path", "host", "port", "beam_search_constraint_dict"}
        if managed & settings.keys():
            fail(f"server_settings must not set {sorted(managed & settings.keys())}")
        if settings.get("tp_size", 1) != 1 or not settings.get(
            "disable_overlap_schedule"
        ):
            fail("server_settings need tp_size 1 and disable_overlap_schedule true")
        # Graph sizes are fixed by the workload; a mismatch silently runs eager.
        for key, expected in (
            ("cuda_graph_bs_decode", workload["beam_width"]),
            ("cuda_graph_bs_prefill", workload["context_length"]),
        ):
            if key in settings and settings[key] != [expected]:
                fail(f"server_settings.{key} must be [{expected}] for this workload")

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
        return "; ".join(parts)


class BeamSearchNew(SGLang):
    name = "sglang-beam-search-new"
    title = "SGLang beam-search-new"
    catalog_file = "catalog.pkl"

    def validate(self, config, workload):
        super().validate(config, workload)
        if not config["server_settings"].get("enable_beam_search"):
            fail("sglang-beam-search-new needs server_settings.enable_beam_search true")

    def convert_catalog(self, paths, config, vocab_size):
        return catalogs.sglang_trie(paths)

    def sampling(self, workload):
        return {"use_beam_search": True}


class Native(SGLang):
    """SGLang's native `beam_width` API: public SGLang and the SIDTrie fork."""

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
    title = "SGLang (public)"
    catalog_supported = False


class SidTrie(Native):
    name = "sglang-sidtrie"
    title = "SGLang SIDTrie"
    keys = ("engine", "env", "server_settings", "trie_config")
    model_view = True
    catalog_required = True
    catalog_file = SIDTRIE_FILE

    def validate(self, config, workload):
        super().validate(config, workload)
        legacy = {"enable_beam_search", "enable_beam_cascade_attention"}
        legacy &= config["server_settings"].keys()
        if legacy:
            fail(
                f"sglang-sidtrie uses the native beam_width API; remove {sorted(legacy)}"
            )
        if workload["output_tokens"] != 3:
            fail("sglang-sidtrie needs workload.output_tokens 3 (three SID codebooks)")
        if config["env"].get(SHARED, "false") not in ("true", "false"):
            fail(f'{SHARED} must be "true" or "false"')
        trie = config["trie_config"]
        if not isinstance(trie, dict) or set(trie) != {"token_start", "codebook_size"}:
            fail("trie_config needs exactly the keys ['codebook_size', 'token_start']")
        for key, minimum in (("token_start", 0), ("codebook_size", 1)):
            if type(trie[key]) is not int or trie[key] < minimum:
                fail(f"trie_config.{key} must be an integer >= {minimum}")

    def convert_catalog(self, paths, config, vocab_size):
        return catalogs.sidtrie(paths, **config["trie_config"], vocab_size=vocab_size)

    def server(self, plan, port, profile):
        argv, env = super().server(plan, port, profile)
        # Keep exact full-vocabulary normalization; the approximation changes scores.
        env["SGLANG_BEAM_TRIE_TOPK_LOGPROB"] = "false"
        return argv, env

    def shared(self, env):
        return env.get(SHARED, "false") == "true"

    def check_counters(self, plan, counters, requests, runtime):
        """Every measured request must take the native SIDTrie path that was requested."""
        decode = counters["forward.decode.graph"] + counters["forward.decode.eager"]
        shared = self.shared(plan["env"])
        problems = []
        try:
            super().check_counters(plan, counters, requests, runtime)
        except RuntimeError as exc:
            problems.append(str(exc))
        if decode <= 0 or counters["sidtrie.compact_decode"] != decode:
            problems.append(
                "not every decode forward used the compact trie output head"
            )
        if counters["sidtrie.exact_normalizer"] != decode:
            problems.append(
                "not every decode forward kept the full-vocabulary normalizer"
            )
        if counters["sidtrie.device_history"] != requests:
            problems.append("not every request finalized beams with device history")
        if shared and (
            counters["forward.decode.graph"]
            or counters["sidtrie.shared_attention"] < counters["forward.decode.eager"]
        ):
            problems.append(
                "shared-context attention did not run in every decode forward"
            )
        if not shared and counters["sidtrie.shared_attention"]:
            problems.append("shared-context attention ran although it is disabled")
        if problems:
            raise RuntimeError("SIDTrie check failed: " + "; ".join(problems))

    def setup_rows(self, config):
        text = "compact trie output head, full-vocabulary normalizer and device beam "
        state = "on" if self.shared(config["env"]) else "off"
        return [
            (
                "SIDTrie (observed)",
                f"{text}history on every measured request; shared-context attention {state}",
            )
        ]


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
    for engine in (SidGr(), SGLangPublic(), BeamSearchNew(), SidTrie())
}


def get(name):
    if name not in ENGINES:
        fail(f"engine must be one of {', '.join(ENGINES)}")
    return ENGINES[name]
