# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU accounting per request or per engine step from an Nsight Systems SQLite
export.

A window is the server's NVTX range <prefix>/request:<rid>, or, for a capture
under load, <prefix>/step:<n> around one engine step. A kernel gets a phase (and,
separately, a section) only through launch evidence: the CUDA runtime/driver
call with the same process and correlation ID must start inside the innermost
<prefix>/phase:<unit>:<name> range on its own thread. CUDA graph nodes (captured
with --cuda-graph-trace=node) inherit their graph launch's markers. Anything
without unambiguous evidence stays "unknown".

Window time with no kernel running (memory operations only, or no GPU activity)
is labeled by time instead: the innermost phase and section range, on any
thread, that covers it, else NO_PHASE / NO_SECTION.
"""

import sqlite3
from collections import defaultdict

from bench.common import MARKER

NO_PHASE, NO_SECTION = "no phase", "no section"
BETWEEN = "between steps"  # idle time outside every step window

TOP = 12  # kernels listed in the report
IDLE = 5  # longest GPU-idle intervals listed for a capture under load

# Kernel symbol substrings (lowercase) per operation family; the first match wins.
# Only what a kernel name reveals: GEMMs of different layers (QKV, MLP, LM head)
# look alike, so they share "linear"; phase and section say where a kernel ran.
FAMILIES = [
    (line.split()[0], line.split()[1:])
    for line in """
    attention_merge  mergestate merge_state combine
    attention        attention flash_fwd flashfwd batchdecodewith batchprefillwith
    linear           gemm gemv splitkreduce split_k_reduce
    topk             topk top_k mbtopk radixfind digitcount gatherkth
    sort             sort scanbykey reducebykey unique
    softmax          softmax soft_max
    qk_norm          qknorm qk_norm
    normalization    rmsnorm rms_norm layernorm layer_norm normalization
    rotary           rope rotary
    activation       silu act_and_mul
    kv_cache         kvcache kv_cache kv_indices cache_loc req_to_token write_packed_qkv_prefill_kv write_beam_kv
    copy             copy
    gather_scatter   gather scatter index
    reduction        reduce scan
    math             exp_kernel log_kernel pow_kernel sqrt_kernel
    elementwise      elementwise functor
    """.strip().splitlines()
]


# CUDA API calls that launch GPU work, by lowercase name substring; the kernels
# inside a CUDA graph run without calls of their own.
LAUNCH_CALLS = [
    ("kernel", ("launchkernel", "launchcooperativekernel")),
    ("graph", ("graphlaunch",)),
]


def family(name):
    """Coarse operation family from the kernel symbol (not its model phase)."""
    lower = name.lower()
    for label, needles in FAMILIES:
        if any(needle in lower for needle in needles):
            return label
    return "other"


def _process(row):
    # Nsight global IDs keep the PID above a 24-bit TID; plain PIDs are small.
    value = (
        row["globalPid"] if row.get("globalPid") is not None else row.get("globalTid")
    )
    return None if value is None else value >> 24 if value >= 1 << 24 else value


def _rows(db, table, tables):
    if table not in tables:
        return []
    cursor = db.execute(f'SELECT * FROM "{table}"')
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, values)) for values in cursor]


def _attribute(kernels, apis, markers, key):
    """Set kernel[key] from the innermost enclosing marker of its launch call."""
    by_launch = defaultdict(list)
    for api in apis:
        by_launch[(_process(api), api["correlationId"])].append(api)
    by_thread = defaultdict(list)
    for marker in markers:
        by_thread[marker["globalTid"]].append(marker)
    # CUDA graph nodes share their graph launch's correlation ID, so one lookup
    # serves thousands of kernels.
    memo = {}
    for kernel in kernels:
        launch = (_process(kernel), kernel["correlationId"])
        if launch not in memo:
            labels = set()
            for api in by_launch.get(launch, []):
                enclosing = [
                    m
                    for m in by_thread[api["globalTid"]]
                    if m["start"] <= api["start"] < m["end"]
                ]
                if enclosing:
                    inner = min(enclosing, key=lambda m: m["end"] - m["start"])
                    labels.add(inner["label"])
            memo[launch] = labels.pop() if len(labels) == 1 else "unknown"
        kernel[key] = memo[launch]


def _cover(markers, a, b, default):
    """Label of the innermost marker that covers [a, b), else default."""
    covering = [m for m in markers if m["start"] <= a and b <= m["end"]]
    if not covering:
        return default
    return min(covering, key=lambda m: m["end"] - m["start"])["label"]


def _partition(start, end, kernels, memory, phases, sections):
    """Split [start, end) into kernel-only, overlap, memory-only and idle time.

    Also returns {(phase, section, state): ns} for the memory-only and idle time;
    marker edges split the window so each piece lies inside or outside a marker.
    """
    edges = defaultdict(lambda: [0, 0])
    for index, rows in enumerate((kernels, memory)):
        for row in rows:
            a, b = max(start, row["start"]), min(end, row["end"])
            if a < b:
                edges[a][index] += 1
                edges[b][index] -= 1
    for marker in phases + sections:
        for point in (marker["start"], marker["end"]):
            if start < point < end:
                edges.setdefault(point, [0, 0])  # a boundary, no count change
    totals = dict.fromkeys(("kernel", "overlap", "memory", "idle"), 0)
    gaps = defaultdict(int)
    active, previous = [0, 0], start
    for point in sorted(set(edges) | {start, end}):
        if previous < point:
            k, m = active
            state = (
                "idle"
                if not k and not m
                else (
                    "memory" if not k else ("kernel" if k == 1 and not m else "overlap")
                )
            )
            totals[state] += point - previous
            if state in ("memory", "idle"):
                phase = _cover(phases, previous, point, NO_PHASE)
                section = _cover(sections, previous, point, NO_SECTION)
                gaps[(phase, section, state)] += point - previous
        active = [active[0] + edges[point][0], active[1] + edges[point][1]]
        previous = point
    return totals, gaps


def _idle(start, end, busy):
    """[(a, b)] stretches of [start, end) that no row of `busy` overlaps."""
    stretches, cursor = [], start
    for row in sorted(busy, key=lambda r: r["start"]):
        if row["start"] > cursor:
            stretches.append((cursor, min(row["start"], end)))
        cursor = max(cursor, row["end"])
        if cursor >= end:
            break
    if cursor < end:
        stretches.append((cursor, end))
    return [(a, b) for a, b in stretches if a < b]


def _at(markers, point, default):
    """Label of the innermost marker running at `point`, else default."""
    running = [m for m in markers if m["start"] <= point < m["end"]]
    if not running:
        return default
    return min(running, key=lambda m: m["end"] - m["start"])["label"]


def longest_idle(windows, busy, phases, sections):
    """The IDLE longest stretches with no GPU activity from the first window's
    start to the last one's end, cut at window edges: [start ns from the first
    window, ns, where], with `where` the phase and section running when the
    stretch starts, or BETWEEN outside every window."""
    start, end = windows[0]["start"], windows[-1]["end"]
    edges = sorted({point for w in windows for point in (w["start"], w["end"])})
    found = []
    for a, b in _idle(start, end, busy):
        cuts = [a, *(point for point in edges if a < point < b), b]
        for c, d in zip(cuts, cuts[1:]):
            where = BETWEEN
            if any(w["start"] <= c < w["end"] for w in windows):
                phase, section = _at(phases, c, NO_PHASE), _at(sections, c, NO_SECTION)
                where = f"{phase} / {section}"
            found.append([c - start, d - c, where])
    return sorted(found, key=lambda gap: -gap[1])[:IDLE]


def analyze(path, prefix=MARKER, window="request"):
    """GPU time of each window kind ("request" or "step"), summed over windows."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        nvtx = _rows(db, "NVTX_EVENTS", tables)
        kernels = _rows(db, "CUPTI_ACTIVITY_KIND_KERNEL", tables)
        apis = _rows(db, "CUPTI_ACTIVITY_KIND_RUNTIME", tables) + _rows(
            db, "CUPTI_ACTIVITY_KIND_DRIVER", tables
        )
        memory = _rows(db, "CUPTI_ACTIVITY_KIND_MEMCPY", tables) + _rows(
            db, "CUPTI_ACTIVITY_KIND_MEMSET", tables
        )
        graph_level = _rows(db, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", tables)
    finally:
        db.close()
    ranges = defaultdict(list)
    for row in nvtx:
        text = row.get("text") or strings.get(row.get("textId"), "")
        if text.startswith(prefix + "/") and row.get("end") is not None:
            kind, _, value = text[len(prefix) + 1 :].partition(":")
            ranges[kind].append(
                {**row, "label": value.rpartition(":")[2], "rid": value}
            )
    windows = sorted(ranges[window], key=lambda w: w["start"])
    if window == "step":  # a scheduler iteration that chose no batch runs no phase
        windows = [
            w
            for w in windows
            if any(w["start"] <= p["start"] < w["end"] for p in ranges["phase"])
        ]
    if not windows:
        raise ValueError(f"trace has no complete {prefix}/{window}: NVTX ranges")
    if any(b["start"] < a["end"] for a, b in zip(windows, windows[1:])):
        raise ValueError(f"{window} windows overlap; they must run one at a time")
    span = (windows[0]["start"], windows[-1]["end"])
    in_span = [
        row
        for row in kernels + memory
        if row["start"] < span[1] and span[0] < row["end"]
    ]

    def inside(row):
        return any(row["start"] < w["end"] and w["start"] < row["end"] for w in windows)

    if any(inside(row) for row in graph_level):
        raise ValueError(
            "graph-level CUDA activity; capture with --cuda-graph-trace=node"
        )
    launches = {kind: 0 for kind, _ in LAUNCH_CALLS}
    for api in apis:
        name = strings.get(api.get("nameId"), "").lower()
        for kind, needles in LAUNCH_CALLS:
            if any(n in name for n in needles) and any(
                w["start"] <= api["start"] < w["end"] for w in windows
            ):
                launches[kind] += 1
    kernels = [k for k in kernels if inside(k)]
    devices = {k["deviceId"] for k in kernels}
    if len(devices) != 1:
        raise ValueError(
            f"expected kernels from one GPU, found devices {sorted(devices)}"
        )
    memory = [m for m in memory if m["deviceId"] in devices and inside(m)]
    _attribute(kernels, apis, ranges["phase"], "phase")
    _attribute(kernels, apis, ranges["section"], "section")

    work, gaps = defaultdict(int), defaultdict(int)
    top = defaultdict(lambda: [0, 0])  # kernel symbol -> [ns, launches]
    busy = dict.fromkeys(("kernel", "overlap", "memory", "idle"), 0)
    for each in windows:
        a, b = each["start"], each["end"]
        phases, sections = (
            [m for m in ranges[kind] if m["start"] < b and a < m["end"]]
            for kind in ("phase", "section")
        )
        in_window = [k for k in kernels if k["start"] < b and a < k["end"]]
        for kernel in in_window:
            name = strings.get(kernel.get("demangledName")) or strings.get(
                kernel.get("shortName"), "unknown"
            )
            key = (kernel["phase"], kernel["section"], family(name))
            overlap = min(b, kernel["end"]) - max(a, kernel["start"])
            work[key] += overlap
            top[name][0] += overlap
            top[name][1] += 1
        mem = [m for m in memory if m["start"] < b and a < m["end"]]
        totals, window_gaps = _partition(a, b, in_window, mem, phases, sections)
        for state, value in totals.items():
            busy[state] += value
        for key, value in window_gaps.items():
            gaps[key] += value
    idle = []
    if window == "step":
        device = [row for row in in_span if row["deviceId"] in devices]
        idle = longest_idle(windows, device, ranges["phase"], ranges["section"])
    return {
        "windows": len(windows),
        "window_ns": sum(w["end"] - w["start"] for w in windows),
        "span_ns": span[1] - span[0],
        "idle": idle,
        "busy_ns": busy,
        "kernel_ns": sum(work.values()),
        "kernels": len(kernels),
        "launches": launches,
        "unknown_kernels": sum(k["phase"] == "unknown" for k in kernels),
        "work": [[*key, value] for key, value in sorted(work.items())],
        "gaps": [[*key, value] for key, value in sorted(gaps.items())],
        "top": [
            [name, family(name), *value]
            for name, value in sorted(top.items(), key=lambda x: -x[1][0])[:TOP]
        ],
    }
