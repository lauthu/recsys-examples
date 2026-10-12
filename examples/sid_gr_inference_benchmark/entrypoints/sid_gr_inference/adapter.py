"""SID-GR execution; reuse model loading, not the old measurement loop."""

import sys
from pathlib import Path
from types import SimpleNamespace

from harness.validation import Unsupported, supported


def check_support(case, solution):
    supported(case)
    if case["generation"]["output_tokens"] < 2:
        raise Unsupported(
            "SID-GR currently requires at least one post-prefill decode step"
        )
    if set(solution["engine"]) - {"prefix_cache"}:
        raise Unsupported(
            "SID-GR accepts prefix_cache; use GR_INFERENCE environment flags for kernels"
        )
    if type(solution["engine"].get("prefix_cache", False)) is not bool:
        raise Unsupported("prefix_cache must be a boolean")


class Adapter:
    def __init__(self, request):
        import torch

        self.torch, self.case = torch, request["case"]
        self.prefix_cache = request["solution"]["engine"].get("prefix_cache", False)
        source = Path(request["solution"]["source"]["path"])
        sys.path[:0] = [str(source / "src"), str(source / "tools")]
        import run_qwen3_real_weight_serving as gr

        self.gr = gr
        c = self.case
        self.args = SimpleNamespace(
            model_dir=request["model"]["path"],
            context_len=c["context_length"],
            decode_steps=c["generation"]["output_tokens"] - 1,
            beam_width=c["beam_width"],
            device="cuda",
            batched_decode=True,
            continuous=True,
            decode_backend="real",
            beam_kv_pool_capacity=c["batch_size"],
            context_kv_pool_capacity=c["batch_size"],
        )
        backend = gr.make_decode_backend(self.args, "cuda").ensure_available()
        model, config, device = gr.load_model(self.args, torch)
        self.engine = gr.GRServingEngine(
            model=model,
            decode_engine=gr.GRDecodeEngine(
                attention=gr.GRDecodeAttention(backend=backend),
                fixed_beam_width=c["beam_width"],
            ),
            config=gr.GRServingConfig(
                max_decode_steps=self.args.decode_steps,
                max_beam_width=c["beam_width"],
                enable_batched_decode=True,
                return_beam_details=False,
                beam_score_mode="logprob",
            ),
        )
        self.beam_pool = gr._make_beam_kv_pool(self.args, torch, config, device)
        self.context_pool = gr._make_context_kv_pool(self.args, torch, config, device)

    def generate(self, workload):
        gr, torch, c = self.gr, self.torch, self.case
        scheduler = gr.GRContinuousScheduler(
            policy=gr.GRContinuousBatchingPolicy(
                max_prefill_batch_size=c["batch_size"],
                max_decode_batch_size=c["batch_size"],
                max_running_requests=c["batch_size"],
            )
        )
        executor = gr.GRContinuousServingExecutor(
            engine=self.engine,
            scheduler=scheduler,
            synchronize=torch.cuda.synchronize,
            sync_timing=False,
            beam_kv_pool=self.beam_pool,
            context_kv_pool=self.context_pool,
            prefill_cache_enabled=self.prefix_cache,
        )
        with torch.no_grad():
            for row in workload:
                executor.submit(
                    gr.GRServingRequest(
                        request_id=row["request_id"],
                        input_ids=torch.tensor(
                            row["input_ids"], device="cuda", dtype=torch.long
                        ).unsqueeze(0),
                        max_decode_steps=self.args.decode_steps,
                        beam_width=c["beam_width"],
                    )
                )
            responses = {r.request_id: r for r in executor.run_until_empty()}
            outputs = []
            for row in workload:
                response = responses[row["request_id"]]
                if response.metadata.get("failed"):
                    meta = response.metadata
                    raise RuntimeError(
                        f"SID-GR {meta.get('stop_reason')}: {meta.get('error_type')}: {meta.get('error_message')}"
                    )
                beams = gr.normalized_beam_results_from_metadata(
                    response.metadata, max_new_tokens=c["generation"]["output_tokens"]
                )
                if not beams:
                    raise RuntimeError(f"SID-GR returned no beams: {response.metadata}")
                outputs.append(
                    {
                        "request_id": row["request_id"],
                        "beams": [
                            {
                                "token_ids": [int(t) for t in b["output_ids"]],
                                "score": float(b["meta_info"]["sequence_score"]),
                            }
                            for b in beams
                        ],
                    }
                )
        return outputs

    def synchronize(self):
        self.torch.cuda.synchronize()

    def close(self):
        self.engine = self.beam_pool = self.context_pool = None

    def provenance(self):
        from gr_inference.gr_kernels.attention.existing_kernel_backend import (
            ExistingGRDecodeAttentionBackend,
        )

        from harness.artifacts import source_info

        return {
            "module": self.gr.__file__,
            "decode_kernel": source_info(
                ExistingGRDecodeAttentionBackend().kernel_root
            ),
        }
