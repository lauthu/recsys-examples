"""Pinned vLLM-GR legacy CUSTOM beam search, without catalog constraints."""

import inspect
from pathlib import Path

from entrypoints import beam_common
from harness.validation import Unsupported

DEFAULTS = {
    "prefix_cache": True,
    "gpu_memory_utilization": 0.70,
    "enforce_eager": False,
    "beam_graph_enabled": True,
    "enable_chunked_prefill": False,
}


def check_support(case, solution):
    values = beam_common.options(case, solution, DEFAULTS)
    if not 2 <= case["beam_width"] <= 256:
        raise Unsupported(
            "Pinned vLLM-GR CUSTOM beam search supports widths 2 through 256"
        )
    if not values["prefix_cache"]:
        raise Unsupported(
            "vLLM-GR CUSTOM beam search requires prefix_cache to retain prompt KV"
        )
    if values["enforce_eager"] and values["beam_graph_enabled"]:
        raise Unsupported("Disable beam_graph_enabled when selecting enforce_eager")
    if case["batch_size"] != 1:
        raise Unsupported(
            "Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only"
        )


class Adapter:
    def __init__(self, request):
        check_support(request["case"], request["solution"])
        self.case = request["case"]
        values = beam_common.options(self.case, request["solution"], DEFAULTS)
        beam_common.source_path(request["solution"])
        self.version = beam_common.require_version("vllm", "0.22.1")
        import vllm_gr
        from vllm_gr.entrypoints.gr import GRLLM
        from vllm_gr.sampling_params import BeamSearchParams

        if "output_mode" not in inspect.signature(GRLLM.beam_search).parameters:
            raise Unsupported("vLLM-GR must provide the compact beam-search output API")
        self.module = vllm_gr.__file__
        c = self.case
        width, steps = c["beam_width"], c["generation"]["output_tokens"]
        # CUSTOM packs the shared prefix plus one token per active beam into
        # one internal request. Reserve that capacity and align it to the
        # pinned runtime's 128-token prefill graph buckets.
        capacity = ((c["context_length"] + max(width, steps) + 127) // 128) * 128
        self.engine_options = {
            "model": request["model"]["path"],
            "dtype": "bfloat16",
            "quantization": None,
            "kv_cache_dtype": "auto",
            "tensor_parallel_size": 1,
            "seed": c["inputs"].get("seed", 42),
            "enable_prefix_caching": values["prefix_cache"],
            "enable_chunked_prefill": values["enable_chunked_prefill"],
            "gpu_memory_utilization": values["gpu_memory_utilization"],
            "enforce_eager": values["enforce_eager"],
            "max_model_len": capacity,
            "max_num_seqs": c["batch_size"] * width,
            "max_num_batched_tokens": max(
                capacity, c["batch_size"] * c["context_length"], c["batch_size"] * width
            ),
            "max_logprobs": width,
            "vllm_gr_config": {
                "schema_version": 1,
                "attention_backend": "CUSTOM",
                "beam": {
                    "execution_mode": "legacy",
                    "max_width": width,
                    "max_decode_steps": steps,
                    "graph_enabled": values["beam_graph_enabled"],
                    "worker_decision": True,
                    "engine_driven": True,
                },
                "constraint_table": {"enabled": False, "max_top_k": width},
            },
        }
        self.sampling_params = BeamSearchParams(
            beam_width=width,
            max_tokens=steps,
            ignore_eos=True,
            temperature=1.0,
            length_penalty=0.0,
            begin_token=None,
            end_token=None,
            catalog_strategy=None,
        )
        self.engine = GRLLM(**self.engine_options)

    def generate(self, workload):
        output = self.engine.beam_search(
            [{"prompt_token_ids": list(row["input_ids"])} for row in workload],
            self.sampling_params,
            use_tqdm=False,
            output_mode="compact",
        )
        return beam_common.normalize(
            output,
            workload,
            sequences="sequences",
            tokens="token_ids",
            score="cumulative_logprob",
        )

    def synchronize(self):
        # beam_search blocks until its worker returns eagerly materialized CPU outputs.
        pass

    def close(self):
        self.engine.shutdown()

    def provenance(self):
        from harness.artifacts import source_info

        return {
            "module": self.module,
            "source": source_info(Path(self.module).parent),
            "vllm_version": self.version,
            "engine_options": self.engine_options,
            "api": "GRLLM.beam_search(output_mode=compact)",
        }
