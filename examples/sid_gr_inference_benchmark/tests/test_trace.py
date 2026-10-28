# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sqlite3

import pytest
from bench.outputs import trace

PID, THREAD, HTTP = 7, (7 << 24) | 1, (7 << 24) | 2


def make_trace(
    path, windows=((100, 200), (400, 420)), graph_level=False, unit="request"
):
    """Two requests (or engine steps); kernels launched inside phase/section ranges
    on one thread."""
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER,
            correlationId INTEGER, globalPid INTEGER, demangledName INTEGER, shortName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, correlationId INTEGER,
            globalTid INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, deviceId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER, deviceId INTEGER);
    """
    )
    names = ["gemm_kernel", "flash_fwd_kernel", "rmsnorm_kernel", "topk_kernel"]
    names += ["cudaLaunchKernel_v7000", "cudaGraphLaunch_v10000", "cuLaunchKernelEx"]
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", enumerate(names))
    nvtx = [
        (a, b, f"bench/{unit}:r{i}", None, HTTP) for i, (a, b) in enumerate(windows)
    ]
    if unit == "step":  # a scheduler iteration that chose no batch: no phase inside
        nvtx.append((302, 304, "bench/step:idle", None, HTTP))
        # Two requests in flight at once, the second past a kernel between steps
        nvtx += [(90, 210, "bench/inflight:a", None, HTTP)]
        nvtx += [(150, 430, "bench/inflight:b", None, HTTP)]
    nvtx += [
        (110, 150, "bench/phase:r0:prefill", None, THREAD),
        (112, 140, "bench/section:r0:model", None, THREAD),
        (150, 190, "bench/phase:r0:decode_1", None, THREAD),
        (400, 420, "bench/phase:r1:prefill", None, THREAD),
        (500, 501, "other/marker", None, THREAD),
    ]
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)", nvtx)
    # (launch time, correlation, kernel start, kernel end, name)
    launches = [
        (115, 1, 120, 130, 0),
        (116, 4, 125, 135, 3),
        (155, 2, 160, 170, 1),
        (195, 3, 192, 198, 2),
        (401, 6, 405, 415, 0),
        (299, 5, 300, 310, 0),
    ]
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        [(t, t + 1, c, THREAD, 6 if c == 6 else 4) for t, c, *_ in launches]
        + [(157, 158, 7, THREAD, 5)],  # a graph launch; its kernels are not traced here
    )
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, ?, ?)",
        [(a, b, c, PID << 24, n, n) for _, c, a, b, n in launches],
    )
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (175, 180, 0)")
    if graph_level:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (150, 160, 0)")
    db.commit()
    db.close()
    return path


def test_attribution_and_window_partition(tmp_path):
    result = trace.analyze(make_trace(tmp_path / "t.sqlite"))
    assert result["windows"] == 2 and result["window_ns"] == 120
    # Exclusive kernels 5+5+10+6+10, overlapping 5, memcpy alone 5, the rest idle.
    assert result["busy_ns"] == {"kernel": 36, "overlap": 5, "memory": 5, "idle": 74}
    assert result["kernels"] == 5 and result["unknown_kernels"] == 1
    assert sorted(result["work"]) == [
        ["decode_1", "unknown", "attention", 10],
        ["prefill", "model", "linear", 10],
        ["prefill", "model", "topk", 10],
        ["prefill", "unknown", "linear", 10],
        ["unknown", "unknown", "normalization", 6],
    ]
    assert result["kernel_ns"] == 46
    # Launch calls inside the request windows; the one at 299 is outside both.
    assert result["launches"] == {"kernel": 5, "graph": 1}
    # Non-kernel time by the innermost covering phase/section; it sums to busy_ns.
    assert result["gaps"] == [
        ["decode_1", "no section", "idle", 25],
        ["decode_1", "no section", "memory", 5],
        ["no phase", "no section", "idle", 14],
        ["prefill", "model", "idle", 13],
        ["prefill", "no section", "idle", 22],
    ]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"windows": ((100, 200), (150, 420))}, "overlap"),
        ({"windows": ()}, "no complete"),
        ({"graph_level": True}, "graph-level"),
    ],
)
def test_rejects_unusable_traces(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        trace.analyze(make_trace(tmp_path / "t.sqlite", **kwargs))


# Symbols from real Qwen3-0.6B profiles of SID-GR and SGLang, shortened.
FAMILY_CASES = [
    ("BatchPrefillWithPagedKVCacheKernel", "attention"),
    ("flashinfer::BatchDecodeWithPagedKVCacheKernel<2>", "attention"),
    ("kernel_cutlass_kernel_srcsm120flash_fwdFlashAttentionForwardSm120", "attention"),
    (
        "kernel_cutlass_kernel_srcflash_fwd_combineFlashAttentionForwardCombine",
        "attention_merge",
    ),
    ("flashinfer::PersistentVariableLengthMergeStatesKernel<8>", "attention_merge"),
    ("sm90_xmma_gemm_bf16", "linear"),
    (
        "cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128>",
        "linear",
    ),
    ("cublasLt::splitKreduce_kernel<32, 16, int>", "linear"),
    ("at::native::mbtopk::computeBlockDigitCounts<float>", "topk"),
    ("at::native::mbtopk::gatherTopK<float, unsigned int>", "topk"),
    ("at::native::radixSortKVInPlace<-2, -1, 32>", "sort"),
    ("cub::detail::scan_by_key::DeviceScanByKeyKernel", "sort"),
    ("at::native::cunn_SoftMaxForward<4, float>", "softmax"),
    ("fused_add_rmsnorm_kernel", "normalization"),
    ("sglang::fused_qknorm_warp<128>", "qk_norm"),
    ("sglang::fused_rope_kernel<1>", "rotary"),
    ("silu_and_mul_packed_vec_kernel", "activation"),
    ("sglang::store_kvcache_kernel<2048>", "kv_cache"),
    ("at::native::unrolled_elementwise_kernel<direct_copy_kernel_cuda>", "copy"),
    ("at::native::index_elementwise_kernel<128, 4>", "gather_scatter"),
    ("at::native::reduce_kernel<512, 1>", "reduction"),
    ("at::native::vectorized_elementwise_kernel<4, exp_kernel_cuda>", "math"),
    (
        "at::native::vectorized_elementwise_kernel<4, CUDAFunctor_add<BFloat16>>",
        "elementwise",
    ),
    ("mystery", "other"),
]


@pytest.mark.parametrize("name, expected", FAMILY_CASES)
def test_kernel_families(name, expected):
    assert trace.family(name) == expected


def test_top_kernels_are_ranked_by_time(tmp_path):
    top = trace.analyze(make_trace(tmp_path / "t.sqlite"))["top"]
    assert top[0] == ["gemm_kernel", "linear", 20, 2]
    assert [row[2] for row in top] == sorted((row[2] for row in top), reverse=True)


def test_steps_under_load_report_their_span_and_idle_stretches(tmp_path):
    """Windows are engine steps; idle time is cut at their edges and placed where
    it starts: inside a step's phase and section, or between steps."""
    result = trace.analyze(
        make_trace(tmp_path / "t.sqlite", unit="step"), window="step"
    )
    assert result["windows"] == 2 and result["window_ns"] == 120
    assert result["span_ns"] == 320  # first step's start to last step's end
    assert result["idle"] == [
        [100, 100, "between steps"],  # 200-300, before the kernel launched at 299
        [210, 90, "between steps"],  # 310-400
        [35, 25, "prefill / model"],  # 135-160
        [0, 20, "no phase / no section"],  # 100-120
        [80, 12, "decode_1 / no section"],  # 180-192
    ]
    # Each request in flight, over all of the GPU's activity: [ns, kernels
    # running (120-135, 160-170, 192-198, then 300-310 and 405-415), memory
    # operations only (175-180), no GPU activity].
    assert result["inflight"] == [[120, 31, 5, 84], [280, 36, 5, 239]]
    # Request windows report no idle stretches; profile one at a time there.
    one_at_a_time = trace.analyze(make_trace(tmp_path / "r.sqlite"))
    assert one_at_a_time["idle"] == [] and one_at_a_time["inflight"] == []
