"""Native upstream SGLang beam search, with no multi-sampling fallback."""

import sys
from pathlib import Path

from validation import Unsupported, supported


def check_support(case, solution):
    supported(case)
    allowed = {"page_size", "disable_overlap_schedule", "mem_fraction_static", "attention_backend",
               "disable_cuda_graph", "disable_piecewise_cuda_graph", "log_level"}
    if set(solution["engine"]) - allowed:
        raise Unsupported(f"Unrecognized SGLang engine options: {sorted(set(solution['engine']) - allowed)}")
    if solution["engine"].get("page_size", 1) != 1:
        raise Unsupported("The migrated upstream beam path uses page_size=1")
    if case["beam_width"] == 1:
        raise Unsupported("SGLang width-1 sampling does not expose the same beam-score contract")


def normalize(outputs, workload, width):
    if isinstance(outputs, dict):
        outputs = [outputs]
    if len(outputs) == len(workload) and all(row.get("beam_results") or row.get("meta_info", {}).get("beam_results") for row in outputs):
        groups = [row.get("beam_results") or row["meta_info"]["beam_results"] for row in outputs]
    elif len(outputs) == len(workload) * width:
        groups = [outputs[i * width:(i + 1) * width] for i in range(len(workload))]
    else:
        raise ValueError("Unexpected native SGLang beam output format")
    result = []
    for request, beams in zip(workload, groups):
        converted = []
        for beam in beams:
            meta = beam.get("meta_info", {})
            score = beam.get("cum_logprob")
            if score is None and meta.get("sequence_score") is not None:
                score = float(meta["sequence_score"]) * len(beam["output_ids"])
            if score is None:
                raise ValueError("SGLang must return cumulative sequence_score; sampling output is unsupported")
            converted.append({"token_ids": list(beam["output_ids"]), "score": float(score)})
        result.append({"request_id": request["request_id"], "beams": converted})
    return result


class Adapter:
    def __init__(self, request):
        self.case = request["case"]
        source = request["solution"]["source"].get("path")
        if source:
            sys.path.insert(0, str(Path(source) / "python"))
        import sglang
        from sglang.srt.sampling.sampling_params import SamplingParams
        if "beam_width" not in getattr(SamplingParams, "__struct_fields__", ()):
            raise Unsupported("This SGLang version lacks the native beam_width API; use a newer environment")
        from sglang.srt.beam_search.beam_group import BeamGroup
        import inspect
        if inspect.signature(BeamGroup).parameters["length_penalty"].default != 1.0:
            raise Unsupported("SGLang beam length normalization changed; update the adapter")
        probe = object.__new__(BeamGroup)
        probe.length_penalty = 1.0
        if probe.beam_score(-6.0, 3) != -2.0:
            raise Unsupported("SGLang beam score semantics changed; update the adapter")
        self.module = sglang.__file__
        self.engine = sglang.Engine(model_path=request["model"]["path"], dtype="bfloat16",
            kv_cache_dtype="auto", disable_radix_cache=True, skip_tokenizer_init=True, tp_size=1,
            random_seed=self.case["inputs"].get("seed", 42),
            **request["solution"]["engine"])

    def generate(self, workload):
        c = self.case
        output = self.engine.generate(input_ids=[r["input_ids"] for r in workload], sampling_params={
            "beam_width": c["beam_width"], "n": c["beam_width"],
            "max_new_tokens": c["generation"]["output_tokens"], "ignore_eos": True})
        # Convert the verified native mean-logprob score to cumulative logprob.
        # Equal fixed output lengths make the internal normalization rank-equivalent.
        return normalize(output, workload, c["beam_width"])

    def synchronize(self):
        # Engine.generate waits for completed CPU outputs from its GPU workers.
        pass

    def close(self):
        self.engine.shutdown()

    def profile_start(self):
        self.engine.start_profile(activities=["CUDA_PROFILER"])

    def profile_stop(self):
        # Flush CUPTI in the GPU child before Engine.shutdown terminates it.
        import time
        self.engine.stop_profile()
        # Nsight's 100 ms periodic flush must drain before the child is killed.
        time.sleep(0.25)

    def provenance(self):
        from artifacts import source_info
        return {"module": self.module, "source": source_info(Path(self.module).parent)}
