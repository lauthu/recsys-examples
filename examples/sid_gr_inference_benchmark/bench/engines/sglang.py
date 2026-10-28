# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang and its forks: public SGLang, beam-search-new and vedantjh2-sglang.
All three run SGLang's own HTTP server in the launcher sglang_server.py,
configured by the solution's server_settings plus what server_args() sets."""

import array
import json
import math
import pickle
import struct
import sys

from bench.common import call, fail
from bench.engines.adapter import Adapter, sum_counters

TRIE_HEAD_FILE = "trie_output_head.safetensors"
SHARED = "SGLANG_BEAM_SHARED_CONTEXT_ATTENTION"
TOPK = "SGLANG_BEAM_TRIE_TOPK_LOGPROB"  # "true": normalize over the codebook only
# The beam width b5706db99 captures shared-context decode graphs for
# (configs/solutions/vedantjh2_sglang_shared_graphs.yaml)
SHARED_WIDTH = "SGLANG_BEAM_SHARED_CONTEXT_WIDTH"
DENSE = "SGLANG_BEAM_TRIE_DENSE_HEAD"  # "true": the trie masks the dense LM head
# Image recipe of the SGLang dev branches, which images/build.sh builds from source.
FROM_SOURCE = (
    "docker/Dockerfile of the source, target framework_final (images/build.sh)"
)


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

    def overlaps_steps(self, solution):
        return not solution.server_settings.get("disable_overlap_schedule", False)

    def server(self, config, port, profile):
        env = self.environment(config, profile)
        env.update(PYTHONPATH="/sglang/python:/benchmark-tools", BENCH_ENGINE=self.name)
        env["SGLANG_PROFILE_V2"] = "0"  # the cudaProfilerApi control path
        argv = [sys.executable, "-m", "bench.engines.sglang_server"]
        return argv + ["--config", "/work/run_config.json", "--port", str(port)], env

    def server_args(self, config, port):
        """Where the model and the port are, and what follows from the scenario.

        One deployment serves every concurrency, so it is sized for the largest,
        top. CUDA graph sizes: k requests in flight decode k * beam_width rows;
        prefill sizes come from prefill_tokens. Request slots: every beam holds
        one, and beam-search-new also keeps the request's own; with too few,
        requests queue or the server fails.
        """
        workload = config.workload
        top = max(workload.levels)
        return {
            "model_path": "/model",
            "host": "127.0.0.1",
            "port": port,
            "dtype": config.scenario.model.dtype,
            "random_seed": workload.seed,
            "cuda_graph_bs_decode": [
                workload.beam_width * k for k in range(1, top + 1)
            ],
            "cuda_graph_bs_prefill": prefill_tokens(workload, top),
            "max_running_requests": top * (workload.beam_width + 1),
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
            "n": workload.beam_width,
            "max_new_tokens": workload.output_tokens,
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

    def graph_usage(self, rounds):
        counts = sum_counters(rounds)
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

    def server_args(self, config, port):
        args = super().server_args(config, port)
        if config.catalog_items is not None:
            args["beam_search_constraint_dict"] = self.catalog_source
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
        return {"beam_width": workload.beam_width}

    def check_counters(self, config, counters, requests, runtime):
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
        for name in (SHARED, TOPK, DENSE):
            if solution.env.get(name, "false") not in ("true", "false"):
                fail(f'{name} must be "true" or "false"')
        width = solution.env.get(SHARED_WIDTH)
        if width is not None and (
            not self.shared(solution.env)
            or width != str(scenario.workload.beam_width)
            or scenario.catalog is None
        ):
            fail(
                f"{SHARED_WIDTH} captures shared-context decode graphs only with "
                f'{SHARED} "true", a catalog and the scenario\'s beam width'
            )
        if scenario.catalog is None:  # native beam search: no trie output head
            if self.dense_head(solution.env):
                fail(f"{DENSE} needs a catalog: the trie masks the dense LM head")
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

    def shared_graphs(self, env):
        return SHARED_WIDTH in env

    def dense_head(self, env):
        return env.get(DENSE, "false") == "true"

    def codebook_normalized(self, env):
        return env.get(TOPK, "false") == "true"

    def check_counters(self, config, counters, requests, runtime):
        """Every measured request must take the path that was requested: the
        trie output head exactly when there is a catalog, with the requested
        normalization."""
        decode = counters["forward.decode.graph"] + counters["forward.decode.eager"]
        shared = self.shared(config.solution.env)
        codebook = self.codebook_normalized(config.solution.env)
        problems = []
        try:
            super().check_counters(config, counters, requests, runtime)
        except RuntimeError as exc:
            problems.append(str(exc))
        trie = ("compact_decode", "exact_normalizer", "device_history")
        if config.catalog_items is None:
            if any(counters[f"vedantjh2.{name}"] for name in trie):
                problems.append("the trie output head ran without a catalog")
        elif self.dense_head(config.solution.env):
            if (
                counters["vedantjh2.compact_decode"]
                or counters["vedantjh2.exact_normalizer"]
            ):
                problems.append(
                    f"the compact trie output head ran although {DENSE} is true"
                )
            if counters["vedantjh2.device_history"] != requests:
                problems.append("not every request finalized beams with device history")
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
        if shared and self.shared_graphs(config.solution.env):
            if decode <= 0 or counters["forward.decode.graph"] != decode:
                problems.append(
                    "not every decode forward replayed a shared-context CUDA graph"
                )
        elif shared and (
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
            head = (
                "dense LM head masked by the trie"
                if self.dense_head(solution.env)
                else "compact trie output head"
            )
            text = f"{head}, {normalizer} normalization and device "
            text += "beam history on every measured request"
        else:
            text = "off: no catalog"
        state = "on" if self.shared(solution.env) else "off"
        if self.shared_graphs(solution.env):
            state += f" in CUDA graphs at width {solution.env[SHARED_WIDTH]}"
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
    longest, shortest = workload.context_length, workload.min_context_length
    if shortest is None:
        return [longest * k for k in range(1, top + 1)]
    sizes, size = [], shortest
    while size < top * longest:
        sizes.append(size)
        size = math.ceil(size * 1.5)
    return sizes + [top * longest]
