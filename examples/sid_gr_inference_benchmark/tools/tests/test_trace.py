# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sqlite3

import pytest
from bench import trace

PID, THREAD, HTTP = 7, (7 << 24) | 1, (7 << 24) | 2


def make_trace(path, windows=((100, 200), (400, 420)), graph_level=False):
    """Two requests; kernels launched inside phase/section ranges on one thread."""
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER,
            correlationId INTEGER, globalPid INTEGER, demangledName INTEGER, shortName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, correlationId INTEGER,
            globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, deviceId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER, deviceId INTEGER);
    """
    )
    names = ["gemm_kernel", "flash_fwd_kernel", "rmsnorm_kernel", "topk_kernel"]
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", enumerate(names))
    nvtx = [
        (a, b, f"bench/request:r{i}", None, HTTP) for i, (a, b) in enumerate(windows)
    ]
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
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
        [(t, t + 1, c, THREAD) for t, c, *_ in launches],
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
    assert result["requests"] == 2 and result["window_ns"] == 120
    # Exclusive kernels 5+5+10+6+10, overlapping 5, memcpy alone 5, the rest idle.
    assert result["busy_ns"] == {"kernel": 36, "overlap": 5, "memory": 5, "idle": 74}
    assert result["kernels"] == 5 and result["unknown_kernels"] == 1
    assert sorted(result["work"]) == [
        ["decode_1", "unknown", "attention", 10],
        ["prefill", "model", "linear", 10],
        ["prefill", "model", "selection", 10],
        ["prefill", "unknown", "linear", 10],
        ["unknown", "unknown", "normalization", 6],
    ]
    assert result["kernel_ns"] == 46


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


def test_kernel_families():
    assert trace.family("BatchPrefillWithPagedKVCacheKernel") == "attention"
    assert trace.family("sm90_xmma_gemm_bf16") == "linear"
    assert trace.family("fused_add_rmsnorm_kernel") == "normalization"
    assert trace.family("mystery") == "other"
