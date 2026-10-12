# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def cuda_torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch


@pytest.mark.parametrize("vocab", [4096, 4097, 8193, 151936])
def test_fused_normalizer_matches_torch(cuda_torch, monkeypatch, vocab):
    torch = cuda_torch
    from gr_inference.gr_runtime.batched_beam_search import _beam_logsumexp

    monkeypatch.setenv("GR_INFERENCE_FUSED_BEAM_LOGSUMEXP", "1")
    torch.manual_seed(19)
    x = torch.randn(2, 4, vocab, device="cuda") * 7
    x[0, 0] = -torch.inf
    x[0, 1, :4096] = -torch.inf
    x[0, 2] += 1000
    x[0, 3] -= 1000
    x[1, 0, 5] = torch.inf
    x[1, 1, 5] = torch.nan
    torch.testing.assert_close(
        _beam_logsumexp(x),
        torch.logsumexp(x, -1, keepdim=True),
        atol=2e-5,
        rtol=1e-6,
        equal_nan=True,
    )


def test_masked_chunks_and_mixed_nonfinite_values(cuda_torch, monkeypatch):
    torch = cuda_torch
    from gr_inference.gr_runtime.batched_beam_search import _beam_logsumexp

    monkeypatch.setenv("GR_INFERENCE_FUSED_BEAM_LOGSUMEXP", "1")
    x = torch.full((5, 8193), -torch.inf, device="cuda")
    x[0, -1] = 12.0  # Only the final, one-element chunk is valid.
    x[1, 0] = torch.inf
    x[2, 0] = torch.inf
    x[2, -1] = torch.nan  # NaN must win over +inf in another chunk.
    x[3, -1] = torch.nan
    # Row 4 remains fully masked.
    torch.testing.assert_close(
        _beam_logsumexp(x),
        torch.logsumexp(x, -1, keepdim=True),
        equal_nan=True,
    )


def test_empty_batch(cuda_torch, monkeypatch):
    torch = cuda_torch
    from gr_inference.gr_runtime.batched_beam_search import _beam_logsumexp

    monkeypatch.setenv("GR_INFERENCE_FUSED_BEAM_LOGSUMEXP", "1")
    x = torch.empty(0, 4, 8193, device="cuda")
    torch.testing.assert_close(_beam_logsumexp(x), torch.logsumexp(x, -1, keepdim=True))


def test_beam_selection_and_graph_replay(cuda_torch, monkeypatch):
    torch = cuda_torch
    from gr_inference.gr_runtime.batched_beam_search import select_next_topk_batched

    torch.manual_seed(23)
    logits = torch.randn(2, 8, 8193, device="cuda")
    previous = torch.randn(2, 8, device="cuda")
    mask = torch.rand_like(logits) > 0.1
    kwargs = dict(
        previous_scores_tensor=previous,
        beam_width=16,
        item_mask=mask,
        score_mode="logprob",
        materialize=False,
        validate_finite=False,
    )
    monkeypatch.setenv("GR_INFERENCE_FUSED_BEAM_LOGSUMEXP", "1")
    select_next_topk_batched(logits, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = select_next_topk_batched(logits, **kwargs)
    for change in (0.0, 0.3):
        logits[:, 0, 3].add_(change)
        graph.replay()
        monkeypatch.delenv("GR_INFERENCE_FUSED_BEAM_LOGSUMEXP", raising=False)
        expected = select_next_topk_batched(logits, **kwargs)
        monkeypatch.setenv("GR_INFERENCE_FUSED_BEAM_LOGSUMEXP", "1")
        assert torch.equal(result.token_ids_tensor, expected.token_ids_tensor)
        assert torch.equal(result.parent_beams_tensor, expected.parent_beams_tensor)
        torch.testing.assert_close(
            result.scores_tensor, expected.scores_tensor, atol=2e-5, rtol=1e-6
        )


def test_cpu_and_autograd_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    from gr_inference.gr_runtime.batched_beam_search import _beam_logsumexp

    monkeypatch.setenv("GR_INFERENCE_FUSED_BEAM_LOGSUMEXP", "1")
    x = torch.randn(1, 2, 8193, requires_grad=True)
    result = _beam_logsumexp(x)
    torch.testing.assert_close(result, torch.logsumexp(x, -1, keepdim=True))
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.softmax(x.detach(), -1))


@pytest.mark.parametrize(
    "case", ["disabled", "fp16", "bf16", "strided", "grad", "small"]
)
def test_cuda_fallback(cuda_torch, monkeypatch, case):
    torch = cuda_torch
    from gr_inference.gr_runtime import fused_beam_logsumexp
    from gr_inference.gr_runtime.batched_beam_search import _beam_logsumexp

    def unexpected_kernel(_):
        pytest.fail("ineligible inputs must use PyTorch")

    monkeypatch.setattr(fused_beam_logsumexp, "fused_logsumexp", unexpected_kernel)
    monkeypatch.setenv(
        "GR_INFERENCE_FUSED_BEAM_LOGSUMEXP", "0" if case == "disabled" else "1"
    )
    x = torch.randn(2, 8193, device="cuda")
    if case in {"fp16", "bf16"}:
        x = x.to(torch.float16 if case == "fp16" else torch.bfloat16)
    elif case == "strided":
        x = x[:, ::2]
    elif case == "grad":
        x.requires_grad_()
    elif case == "small":
        x = x[:, :4095].contiguous()
    result = _beam_logsumexp(x)
    torch.testing.assert_close(result, torch.logsumexp(x, -1, keepdim=True))
    if case == "grad":
        result.sum().backward()
        torch.testing.assert_close(x.grad, torch.softmax(x.detach(), -1))
