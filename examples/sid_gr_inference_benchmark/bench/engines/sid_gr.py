# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SID-GR and gr-inference-internal, the repository SID-GR was imported from:
configured through their environment and served by their own HTTP server,
which runs in the launcher sid_gr_server.py."""

from bench.common import call, fail
from bench.engines.adapter import Adapter, measured_requests, sum_counters

SCRIPT = "/source/examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
INTERNAL_SCRIPT = "/source/scripts/serve_qwen3_gr_http.sh"
LAUNCHER = "/benchmark-tools/bench/engines/sid_gr_server.py"  # the script's $PYTHON_BIN


class SidGrAdapter(Adapter):
    name = "sid-gr"
    recipe = "images/Dockerfile.sid-gr"
    title = "SID-GR"
    source_label = "a recsys-examples checkout"
    entry = "examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
    ready = "/ready"
    # Set from the RunConfig below; the model and the catalog come only from the runner.
    managed_env = Adapter.managed_env | frozenset(
        """PYTHON_BIN MODEL_DIR GR_MODEL_DIR GR_DECODE_ATTEN_ROOT GR_HTTP_HOST
        GR_HTTP_PORT GR_CONTEXT_LEN GR_BEAM_WIDTH GR_DECODE_STEPS GR_MAX_BATCH_SIZE
        GR_BEAM_KV_POOL_CAPACITY GR_CONTEXT_KV_POOL_CAPACITY
        GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS GR_CATALOG_JSONL GR_ALLOW_CATALOG_RELOAD""".split()
    )

    def validate(self, solution, scenario):
        if scenario.model.dtype != "bfloat16":
            fail(
                f"{self.name} builds its model in bfloat16 and has no dtype setting; "
                f"the scenario asks for model.dtype {scenario.model.dtype}"
            )

    def server(self, config, port, profile):
        env = self.environment(config, profile)
        env.update(self.shape(config, port))
        env["GR_DECODE_ATTEN_ROOT"] = "/source/corelib/gr_decode_atten"
        if config.catalog_items is not None:
            env["GR_CATALOG_JSONL"] = self.catalog_source
        return ["bash", SCRIPT], env

    def shape(self, config, port):
        """The launcher, the model, the port and the request shape, which the
        engine's script turns into server options."""
        workload = config.workload
        # Every request in flight shares one batch; one deployment serves every
        # concurrency, so it is sized for the largest.
        batch = max(workload.levels)
        return {
            "PYTHON_BIN": LAUNCHER,
            "MODEL_DIR": "/model",
            "GR_HTTP_HOST": "127.0.0.1",
            "GR_HTTP_PORT": str(port),
            "GR_CONTEXT_LEN": str(workload.context_length),
            "GR_BEAM_WIDTH": str(workload.beam_width),
            "GR_DECODE_STEPS": str(workload.output_tokens - 1),
            "GR_MAX_BATCH_SIZE": str(batch),
            "GR_BEAM_KV_POOL_CAPACITY": str(batch),
            "GR_CONTEXT_KV_POOL_CAPACITY": str(batch),
            "GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS": ",".join(
                str(size) for size in range(1, batch + 1)
            ),
        }

    def profile(self, port, start):
        call(port, "POST", f"/pilot/profile/{'start' if start else 'stop'}")

    def payload(self, row, workload):
        params = {
            "n": workload.beam_width,
            "max_new_tokens": workload.output_tokens,
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

    def check_source(self, runtime):
        # An image may also hold an installed copy of the engine.
        if not runtime["gr_inference"].startswith(f"{self.mount}/"):
            raise RuntimeError(
                f"imported {runtime['gr_inference']}, not the mounted {self.title}"
            )

    def check_counters(self, config, counters, requests, runtime):
        # SID-GR keys its CUDA graphs by prompt length and captures them only at
        # startup; a capture during measured requests would land in their latency.
        captures = counters.get("prefill.graph_captures")
        if captures:
            raise RuntimeError(
                f"SID-GR captured {captures} prefill CUDA graphs during measured "
                "requests"
            )

    def graph_usage(self, rounds):
        counts = sum_counters(rounds)
        requests = sum(measured_requests(result) for result in rounds)
        enabled = rounds[0]["runtime"]["cuda_graphs"]
        parts = [
            (
                f"{phase}: {counts[f'{phase}.graph_replays'] / requests:.2f} replays/request"
                if enabled[phase]
                else f"{phase}: disabled"
            )
            for phase in ("prefill", "decode")
        ]
        if enabled["decode"]:
            parts[-1] += ", " + self.eager_decode(counts)
        return "; ".join(parts)

    def eager_decode(self, counts):
        """What ran decode without a graph although decode graphs are on."""
        return f"{counts['decode.eager_fallbacks']} eager fallbacks"


class GRInferenceInternalAdapter(SidGrAdapter):
    """gr-inference-internal from its perf_precision_fix branch on. Its server
    keeps SID-GR's classes, so it runs under the same launcher, but serves HTTP
    through ASGI and takes its settings as serve_qwen3_gr_http.py options as
    well as variables."""

    name = "gr-inference-internal"
    recipe = (
        "Dockerfile of the source, with the base image's entrypoint (images/build.sh)"
    )
    title = "gr-inference-internal"
    source_label = "a gr-inference-internal checkout with its submodule"
    entry = "scripts/serve_qwen3_gr_http.sh"
    fingerprint = (
        "pyproject.toml",
        "scripts",
        "src",
        "tools",
        "third_party/gr-decode-attention",
    )
    sections = ("server_settings",)
    # Its ASGI server awaits _handle_sglang_generate_async for each request.
    features = frozenset({"asgi"})
    managed_env = SidGrAdapter.managed_env | {
        "GR_CACHE_ROOT",
        "GR_MAX_OUTPUT_TOKENS",
        "GR_REC_VOCAB_FILE",
    }
    # Options the script sets from the variables above; a solution must not.
    managed_settings = frozenset(
        """host port model_dir model revision context_len decode_steps beam_width
        max_batch_size beam_kv_pool_capacity context_kv_pool_capacity
        decode_cuda_graph_batch_buckets rec_vocab_file""".split()
    )
    # Its catalog names semantic-ID codes, which it maps to <sid_N> tokens; Qwen3
    # has none.
    catalog_supported = False
    # It orders the final beams in 2e-5-wide score bands, by token path within one.
    score_order_tolerance = 2e-5

    def validate(self, solution, scenario):
        super().validate(solution, scenario)
        managed = sorted(self.managed_settings & solution.server_settings.keys())
        if managed:
            fail(f"server_settings must not set {managed}: the runner sets them")

    def server(self, config, port, profile):
        env = self.environment(config, profile)
        env.update(self.shape(config, port))
        env["GR_DECODE_ATTEN_ROOT"] = "/source/third_party/gr-decode-attention"
        # The script makes its cache directories there; /source is read-only.
        env["GR_CACHE_ROOT"] = "/cache/gr-inference"
        options = server_options(config.solution.server_settings)
        return ["bash", INTERNAL_SCRIPT, *options], env

    def check_counters(self, config, counters, requests, runtime):
        # Its CUDA graphs are captured at startup; a capture during measured
        # requests would land in their latency.
        for phase in ("prefill", "decode"):
            captures = counters.get(f"{phase}.graph_captures")
            if captures:
                raise RuntimeError(
                    f"{self.title} captured {captures} {phase} CUDA graphs during "
                    "measured requests"
                )
        # It builds its beam-selection CUDA extension at first use and selects
        # beams in torch if the build fails: a broken image, not the engine.
        selection = runtime["beam_selection"]
        if selection is None:
            raise RuntimeError(f"{self.title} has no beam-selection CUDA extension")
        if selection["failed"]:
            raise RuntimeError(
                f"{self.title}'s beam-selection CUDA extension did not build: "
                f"{selection['error']}"
            )

    def eager_decode(self, counts):
        # Decode model steps run without a graph, and mixed-step batches it
        # reran as one batch per step.
        return (
            f"{counts['decode.eager_model_steps']} eager steps, "
            f"{counts['decode.mixed_step_fallbacks']} mixed-step fallbacks"
        )


def server_options(settings):
    """serve_qwen3_gr_http.py options: true is --name, false --no-name, a list is
    comma-separated and anything else is --name value."""
    argv = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is False:
            argv.append("--no-" + flag[2:])
        elif isinstance(value, list):
            argv += [flag, ",".join(map(str, value))]
        else:
            argv += [flag, str(value)]
    return argv
