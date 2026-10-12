"""Optional SID-GR Triton/PyTorch operator A/B, independent of engine timing."""

from pathlib import Path
import statistics
import sys
import time


def run(request):
    from validation import Unsupported
    solution = request["solution"]
    if "beam_logsumexp" not in solution.get("kernels", []):
        raise Unsupported("This adapter does not expose beam_logsumexp")
    source = Path(solution["source"]["path"]) / "src"
    if not (source / "gr_inference/gr_runtime/fused_beam_logsumexp.py").is_file():
        raise Unsupported("Selected SID-GR checkout does not include the optional fused beam normalizer")
    import torch
    import triton.testing
    from artifacts import source_info, suite_hash
    from config import ROOT
    from worker import environment
    if suite_hash(ROOT) != request["suite_sha256"]:
        raise ValueError("Suite code changed since the original e2e run")
    if source_info(request["solution"]["source"].get("path")) != request.get("source"):
        raise ValueError("Operator source changed since the original e2e run")
    if request.get("expected_environment") != environment(torch):
        raise ValueError("Operator runtime drifted from the original e2e run")
    sys.path.insert(0, str(source))
    from gr_inference.gr_runtime.fused_beam_logsumexp import fused_logsumexp
    torch.manual_seed(42)
    case = request["case"]
    shape = (case["batch_size"], case["beam_width"], request["model"]["config"]["vocab_size"])
    scores = torch.randn(shape, device="cuda", dtype=torch.float32)
    reference = torch.logsumexp(scores, -1, keepdim=True)
    records = []
    for name, fn in [("pytorch", lambda: torch.logsumexp(scores, -1, keepdim=True)), ("sid_gr_triton", lambda: fused_logsumexp(scores))]:
        out = fn()
        torch.testing.assert_close(out, reference, atol=2e-5, rtol=1e-6)
        max_error = (out - reference).abs().max().item()
        del out
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - allocated
        del out
        gpu, wall = [], []
        for _ in range(100):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            a.record()
            fn()
            b.record()
            b.synchronize()
            wall.append((time.perf_counter() - start) * 1000)
            gpu.append(a.elapsed_time(b))
        records.append({"implementation": name, "eager_gpu_ms": statistics.median(gpu),
                        "eager_wall_ms": statistics.median(wall), "cuda_graph_ms": triton.testing.do_bench_cudagraph(fn, rep=100),
                        "peak_extra_tensor_bytes": peak, "max_abs_error": max_error})
    return {"status": "success", "operation": "beam_logsumexp", "shape": list(shape), "dtype": "float32",
            "records": records, "input_reuse": "same buffers; warm-cache", "warmup_runs": 20, "repeats": 100,
            "scope": "Isolated normalization only; PyTorch is an operator reference, not a measured SGLang path"}
