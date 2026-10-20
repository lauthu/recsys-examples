# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare PyTorch and opt-in beam normalization on identical CUDA inputs.

Reports CUDA-graph GPU time, eager wall time, and peak additional tensor memory
for normalization and complete tensor-only beam selection. This is a synthetic
microbenchmark; use the real-weight serving benchmark for end-to-end latency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def measure(torch, triton_testing, fn, iterations: int, graph_ms: int) -> dict:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated() - allocated
    del result
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "eager_wall_ms_median": statistics.median(samples),
        "cuda_graph_ms": triton_testing.do_bench_cudagraph(fn, rep=graph_ms),
        "peak_extra_tensor_bytes": peak_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--graph-ms", type=int, default=100)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if (
        min(
            *args.batch_sizes,
            args.beam_width,
            args.vocab_size,
            args.iterations,
            args.graph_ms,
        )
        <= 0
    ):
        parser.error("shapes and iteration counts must be positive")
    if args.vocab_size < 4096:
        parser.error("the fused normalizer requires vocabulary size >= 4096")

    import torch
    import triton
    import triton.testing
    from gr_inference.gr_runtime.batched_beam_search import (
        _beam_logsumexp,
        select_next_topk_batched,
    )

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.manual_seed(71)
    records = []
    flag = "GR_INFERENCE_FUSED_BEAM_LOGSUMEXP"
    previous_flag = os.environ.get(flag)
    try:
        for batch in args.batch_sizes:
            logits = torch.randn(batch, args.beam_width, args.vocab_size, device="cuda")
            previous = torch.randn(batch, args.beam_width, device="cuda")
            reference = torch.logsumexp(logits, -1, keepdim=True)
            for mode, enabled in (("torch", "0"), ("fused", "1")):
                os.environ[flag] = enabled
                torch.testing.assert_close(
                    _beam_logsumexp(logits), reference, atol=2e-5, rtol=1e-6
                )
                functions = {
                    "normalization": lambda: _beam_logsumexp(logits),
                    "selection": lambda: select_next_topk_batched(
                        logits,
                        previous_scores_tensor=previous,
                        beam_width=args.beam_width,
                        score_mode="logprob",
                        materialize=False,
                        validate_finite=False,
                    ),
                }
                for operation, fn in functions.items():
                    record = {"batch": batch, "mode": mode, "operation": operation}
                    record.update(
                        measure(
                            torch, triton.testing, fn, args.iterations, args.graph_ms
                        )
                    )
                    records.append(record)
                    print(json.dumps(record), flush=True)
            del logits, previous, reference
    finally:
        if previous_flag is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = previous_flag
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "beam_width": args.beam_width,
        "vocab_size": args.vocab_size,
        "records": records,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
