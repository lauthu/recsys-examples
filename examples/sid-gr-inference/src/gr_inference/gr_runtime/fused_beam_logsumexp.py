# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in inference-only normalization for contiguous FP32 beam logits.

Reduce chunks and merge their statistics in Triton. This avoids materializing
exp(logits - max) over the whole vocabulary and launching separate PyTorch
operations for the merge. Accumulation and output remain FP32.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _partial_logsumexp(
    logits,
    maxima,
    sums,
    output,
    vocab: tl.constexpr,
    chunks: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * block + tl.arange(0, block)
    values = tl.load(
        logits + row.to(tl.int64) * vocab + offsets,
        offsets < vocab,
        other=-float("inf"),
    )
    maximum = tl.reduce(values, 0, _maximum_with_nan)
    total = tl.sum(tl.exp(values - maximum), 0)
    if chunks == 1:
        result = maximum + tl.log(total)
        result = tl.where(tl.abs(maximum) == float("inf"), maximum, result)
        tl.store(output + row, result)
    else:
        tl.store(maxima + row * chunks + chunk, maximum)
        tl.store(sums + row * chunks + chunk, total)


@triton.jit
def _merge_logsumexp(maxima, sums, output, chunks: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    chunk_maxima = tl.load(
        maxima + row * chunks + offsets, offsets < chunks, other=-float("inf")
    )
    chunk_sums = tl.load(sums + row * chunks + offsets, offsets < chunks, other=0.0)
    maximum = tl.reduce(chunk_maxima, 0, _maximum_with_nan)
    # Fully masked chunks have a NaN sum from (-inf - -inf). Their
    # contribution must be zero when another chunk contains finite logits.
    contributions = tl.where(
        chunk_maxima == -float("inf"),
        0.0,
        chunk_sums * tl.exp(chunk_maxima - maximum),
    )
    result = maximum + tl.log(tl.sum(contributions, 0))
    # Preserve torch.logsumexp's infinities and NaN propagation.
    result = tl.where(tl.abs(maximum) == float("inf"), maximum, result)
    tl.store(output + row, result)


def fused_logsumexp(scores):
    """Return [..., 1] FP32 log normalizers; caller checks inference eligibility."""
    vocab = scores.shape[-1]
    rows = scores.numel() // vocab
    chunks = triton.cdiv(vocab, 4096)
    output = torch.empty(
        (*scores.shape[:-1], 1), device=scores.device, dtype=torch.float32
    )
    if rows == 0:
        return output
    with torch.cuda.device(scores.device):
        if chunks == 1:
            maxima = sums = output  # Unused by the single-chunk specialization.
        else:
            partial = torch.empty(
                (2, rows, chunks), device=scores.device, dtype=torch.float32
            )
            maxima, sums = partial.unbind(0)
        _partial_logsumexp[(rows, chunks)](
            scores, maxima, sums, output, vocab, chunks, 4096
        )
        if chunks > 1:
            _merge_logsumexp[(rows,)](
                maxima, sums, output, chunks, triton.next_power_of_2(chunks)
            )
    return output
