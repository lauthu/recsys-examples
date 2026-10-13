"""FlashRec in-process batched beam search with BF16 and a full vocabulary."""

from dataclasses import asdict
from pathlib import Path

from entrypoints import beam_common
from harness.validation import Unsupported

DEFAULTS = {
    "prefix_cache": True,
    "mem_fraction_static": 0.70,
    "enable_cuda_graph": True,
    "enable_fused_expand": True,
    "enable_graph_expand": True,
    "enable_prefill_batch": True,
    "flashinfer_variant": "fa2",
    "host_worker_threads": 4,
}


def check_support(case, solution):
    values = beam_common.options(case, solution, DEFAULTS)
    if values["flashinfer_variant"] not in {"fa2", "fa3"}:
        raise Unsupported("flashinfer_variant must be fa2 or fa3")


class Adapter:
    def __init__(self, request):
        check_support(request["case"], request["solution"])
        self.case = request["case"]
        values = beam_common.options(self.case, request["solution"], DEFAULTS)
        beam_common.source_path(request["solution"], "python")
        import flashrec
        import torch
        from flashrec import BeamRecConfig, BeamRecEngine
        from flashrec.core import BeamRequest

        self.torch, self.request_type = torch, BeamRequest
        self.invocation = 0
        self.module = flashrec.__file__
        c = self.case
        batch, width = c["batch_size"], c["beam_width"]
        prefix_cache = values.pop("prefix_cache")
        config = BeamRecConfig(
            model_path=request["model"]["path"],
            compute_dtype="bfloat16",
            kv_cache_dtype="bfloat16",
            quantization=None,
            sid_vocab_file=None,
            sid_token_range=None,
            sid_codebook_sizes=None,
            sid_boundary_tokens=None,
            enable_restricted_lm_head=False,
            enable_per_codebook_lm_head=False,
            enable_fused_rms_fp8=False,
            enable_fused_silu_fp8=False,
            enable_radix=prefix_cache,
            enable_warmup=False,
            warmup_iters=0,
            beam_width=width,
            max_tokens=c["generation"]["output_tokens"],
            length_penalty=0.0,
            gpu_id=0,
            max_seq_len=c["context_length"] + c["generation"]["output_tokens"],
            # The pool reserves row zero and holds prefill rows alongside beam
            # rows. One extra request's capacity covers these at batch size 8.
            max_running_requests=batch + 1,
            max_batch_requests=batch,
            target_batch_requests=batch,
            batch_slots=batch * width,
            cuda_graph_max_bs=batch * width,
            cuda_graph_capture_sizes=[width * n for n in range(1, batch + 1)],
            **values,
        )
        self.engine = BeamRecEngine(config)
        try:
            runner = self.engine.runner
            if (
                runner.compute_dtype != torch.bfloat16
                or runner.kv_pool.dtype != torch.bfloat16
            ):
                raise Unsupported(
                    "FlashRec did not initialize BF16 compute and KV cache"
                )
            if (
                self.engine.boundary_ids is not None
                or config.sid_vocab_file is not None
            ):
                raise Unsupported(
                    "FlashRec unexpectedly enabled SID constraints or boundary tokens"
                )
            self.engine.ensure_cuda_graph()
            self.engine_options = asdict(config)
        except BaseException:
            self.close()
            raise

    def generate(self, workload):
        c = self.case
        # FlashRec caches decode wave state by request IDs. Each submission needs
        # fresh internal IDs, including identical warmup and timed workloads.
        self.invocation += 1
        requests = [
            self.request_type(
                rid=f"{self.invocation}:{row['request_id']}",
                input_ids=list(row["input_ids"]),
                beam_width=c["beam_width"],
                max_new_tokens=c["generation"]["output_tokens"],
                ignore_eos=True,
                length_penalty=0.0,
                temperature=0.0,
            )
            for row in workload
        ]
        output = self.engine.generate_many(requests)
        return beam_common.normalize(
            output,
            workload,
            sequences="sequences",
            tokens="tokens",
            score="cum_logprob",
        )

    def synchronize(self):
        self.engine.flush_host()
        self.torch.cuda.synchronize()

    def close(self):
        self.engine.host_pool.shutdown()
        self.torch.cuda.synchronize()
        self.engine = None

    def provenance(self):
        from harness.artifacts import source_info

        return {
            "module": self.module,
            "source": source_info(Path(self.module).parent),
            "engine_options": self.engine_options,
            "api": "BeamRecEngine.generate_many",
        }
