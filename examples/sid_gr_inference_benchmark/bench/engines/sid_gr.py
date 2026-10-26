# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SID-GR: configured through its environment, served by its own HTTP server,
which runs in the launcher sid_gr_server.py."""

from bench.common import call, fail
from bench.engines.adapter import Adapter, measured_requests, sum_counters

SCRIPT = "/source/examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
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
                "sid-gr builds its model in bfloat16 and has no dtype setting; "
                f"the scenario asks for model.dtype {scenario.model.dtype}"
            )

    def server(self, config, port, profile):
        workload = config.workload
        # Every request in flight shares one batch; one deployment serves every
        # concurrency, so it is sized for the largest.
        batch = max(workload.levels)
        env = self.environment(config, profile)
        env.update(
            {
                "PYTHON_BIN": LAUNCHER,
                "MODEL_DIR": "/model",
                "GR_DECODE_ATTEN_ROOT": "/source/corelib/gr_decode_atten",
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
        )
        if config.catalog_items is not None:
            env["GR_CATALOG_JSONL"] = self.catalog_source
        return ["bash", SCRIPT], env

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
            parts[-1] += f", {counts['decode.eager_fallbacks']} eager fallbacks"
        return "; ".join(parts)
