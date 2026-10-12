"""SID-GR execution; reuse model loading, not the old measurement loop."""

import sys
from pathlib import Path
from types import SimpleNamespace

from validation import Unsupported, supported


def check_support(case, solution):
    supported(case)
    if case["generation"]["output_tokens"] < 2:
        raise Unsupported("SID-GR currently requires at least one post-prefill decode step")
    if solution.get("engine"):
        raise Unsupported("SID-GR engine options are fixed; use explicit GR_INFERENCE environment flags")
    fused = str(solution.get("env", {}).get("GR_INFERENCE_FUSED_BEAM_LOGSUMEXP", "0")).lower()
    if fused in {"1", "true", "yes", "on"}:
        source = Path(solution["source"]["path"])
        if not (source / "src/gr_inference/gr_runtime/fused_beam_logsumexp.py").is_file():
            raise Unsupported("Selected SID-GR checkout does not include the optional fused beam normalizer")


class Adapter:
    def __init__(self, request):
        import torch
        self.torch, self.case = torch, request["case"]
        source = Path(request["solution"]["source"]["path"])
        sys.path[:0] = [str(source / "src"), str(source / "tools")]
        import run_qwen3_real_weight_serving as gr
        self.gr = gr
        c = self.case
        self.args = SimpleNamespace(model_dir=request["model"]["path"], context_len=c["context_length"],
            decode_steps=c["generation"]["output_tokens"] - 1, beam_width=c["beam_width"],
            device="cuda", batched_decode=True, continuous=True, decode_backend="real",
            beam_kv_pool_capacity=c["batch_size"], context_kv_pool_capacity=c["batch_size"])
        backend = gr.make_decode_backend(self.args, "cuda").ensure_available()
        model, config, device = gr.load_model(self.args, torch)
        self.engine = gr.GRServingEngine(model=model,
            decode_engine=gr.GRDecodeEngine(attention=gr.GRDecodeAttention(backend=backend), fixed_beam_width=c["beam_width"]),
            config=gr.GRServingConfig(max_decode_steps=self.args.decode_steps, max_beam_width=c["beam_width"],
                                     enable_batched_decode=True, return_beam_details=False, beam_score_mode="logprob"))
        self.beam_pool = gr._make_beam_kv_pool(self.args, torch, config, device)
        self.context_pool = gr._make_context_kv_pool(self.args, torch, config, device)

    def generate(self, workload):
        gr, torch, c = self.gr, self.torch, self.case
        scheduler = gr.GRContinuousScheduler(policy=gr.GRContinuousBatchingPolicy(
            max_prefill_batch_size=c["batch_size"], max_decode_batch_size=c["batch_size"], max_running_requests=c["batch_size"]))
        executor = gr.GRContinuousServingExecutor(engine=self.engine, scheduler=scheduler,
            synchronize=torch.cuda.synchronize, sync_timing=False,
            beam_kv_pool=self.beam_pool, context_kv_pool=self.context_pool, prefill_cache_enabled=False)
        with torch.no_grad():
            for row in workload:
                executor.submit(gr.GRServingRequest(request_id=row["request_id"],
                    input_ids=torch.tensor(row["input_ids"], device="cuda", dtype=torch.long).unsqueeze(0),
                    max_decode_steps=self.args.decode_steps, beam_width=c["beam_width"]))
            responses = {r.request_id: r for r in executor.run_until_empty()}
            outputs = []
            for row in workload:
                response = responses[row["request_id"]]
                if response.metadata.get("failed"):
                    meta = response.metadata
                    raise RuntimeError(f"SID-GR {meta.get('stop_reason')}: {meta.get('error_type')}: {meta.get('error_message')}")
                beams = gr.normalized_beam_results_from_metadata(response.metadata,
                    max_new_tokens=c["generation"]["output_tokens"])
                if not beams:
                    raise RuntimeError(f"SID-GR returned no beams: {response.metadata}")
                outputs.append({"request_id": row["request_id"], "beams": [
                    {"token_ids": [int(t) for t in b["output_ids"]], "score": float(b["meta_info"]["sequence_score"])} for b in beams]})
        return outputs

    def synchronize(self):
        self.torch.cuda.synchronize()

    def close(self):
        self.engine = self.beam_pool = self.context_pool = None

    def provenance(self):
        from artifacts import source_info
        from gr_inference.gr_kernels.attention.existing_kernel_backend import ExistingGRDecodeAttentionBackend
        return {"module": self.gr.__file__, "decode_kernel": source_info(ExistingGRDecodeAttentionBackend().kernel_root)}
