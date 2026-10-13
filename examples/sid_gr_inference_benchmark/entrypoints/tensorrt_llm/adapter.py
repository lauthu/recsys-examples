"""TensorRT-LLM 1.2.1 PyTorch backend with native fixed-width beam search."""

from pathlib import Path

from entrypoints import beam_common
from harness.validation import Unsupported

DEFAULTS = {
    "prefix_cache": True,
    "free_gpu_memory_fraction": 0.70,
    "enable_chunked_prefill": False,
    "enable_cuda_graph": True,
    "disable_overlap_scheduler": True,
}


def check_support(case, solution):
    beam_common.options(case, solution, DEFAULTS)
    if not 2 <= case["beam_width"] <= 1024:
        raise Unsupported("TensorRT-LLM beam search requires width 2 through 1024")
    if case["batch_size"] != 1:
        raise Unsupported(
            "TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores "
            "supports batch_size=1 only"
        )


class Adapter:
    def __init__(self, request):
        check_support(request["case"], request["solution"])
        self.case = request["case"]
        values = beam_common.options(self.case, request["solution"], DEFAULTS)
        # TensorRT-LLM's Python package and compiled bindings must be installed
        # together. A checkout is recorded as provenance, not put on sys.path.
        self.version = beam_common.require_version("tensorrt_llm", "1.2.1")
        import tensorrt_llm
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

        self.module = tensorrt_llm.__file__
        c = self.case
        batch, width = c["batch_size"], c["beam_width"]
        steps = c["generation"]["output_tokens"]
        self.engine_options = {
            "model": request["model"]["path"],
            "backend": "pytorch",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "skip_tokenizer_init": True,
            "max_batch_size": batch,
            "max_beam_width": width,
            "max_seq_len": c["context_length"] + steps,
            "max_num_tokens": max(batch * c["context_length"], batch * width),
            "enable_chunked_prefill": values["enable_chunked_prefill"],
            "disable_overlap_scheduler": values["disable_overlap_scheduler"],
            "kv_cache_config": {
                "dtype": "auto",
                "enable_block_reuse": values["prefix_cache"],
                "free_gpu_memory_fraction": values["free_gpu_memory_fraction"],
            },
            # PyTorch derives weight quantization from the verified checkpoint;
            # quant_config is a TensorRT-engine-only constructor argument.
            "force_dynamic_quantization": False,
            "cuda_graph_config": {"batch_sizes": list(range(1, batch + 1))}
            if values["enable_cuda_graph"]
            else None,
        }
        self.sampling_params = SamplingParams(
            n=width,
            best_of=width,
            use_beam_search=True,
            max_tokens=steps,
            ignore_eos=True,
            end_id=-1,
            pad_id=0,
            temperature=1.0,
            length_penalty=0.0,
            early_stopping=0,
            logprobs=1,
            detokenize=False,
            add_special_tokens=False,
            exclude_input_from_output=True,
        )
        options = dict(self.engine_options)
        options["kv_cache_config"] = KvCacheConfig(**options["kv_cache_config"])
        if options["cuda_graph_config"] is not None:
            options["cuda_graph_config"] = CudaGraphConfig(
                **options["cuda_graph_config"]
            )
        try:
            self.engine = LLM(**options)
        except RuntimeError as exc:
            # The executor wraps its worker's capacity assertion. Preserve the
            # exact limit instead of shortening prompts or reducing beam width.
            cause = exc.__cause__
            if (
                cause is not None
                and "exceeds the KV cache manager's maximum supported length"
                in str(cause)
            ):
                raise Unsupported(f"TensorRT-LLM KV capacity: {cause}") from exc
            raise

    def generate(self, workload):
        output = self.engine.generate(
            [list(row["input_ids"]) for row in workload],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        return beam_common.normalize(
            output,
            workload,
            sequences="outputs",
            tokens="token_ids",
            score="cumulative_logprob",
        )

    def synchronize(self):
        # LLM.generate waits on all request futures, including their CPU outputs.
        pass

    def close(self):
        self.engine.shutdown()

    def provenance(self):
        from harness.artifacts import source_info

        return {
            "module": self.module,
            "source": source_info(Path(self.module).parent),
            "version": self.version,
            "engine_options": self.engine_options,
            "api": "LLM.generate(use_beam_search=True)",
        }
