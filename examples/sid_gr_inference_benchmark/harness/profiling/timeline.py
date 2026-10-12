"""Summarize kernels within completed-request NVTX ranges, without name guesses."""

import sqlite3


def union_ns(intervals):
    total, end = 0, -1
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def summarize(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        names = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
        if not required <= names:
            return {
                "status": "unavailable",
                "reason": "Trace lacks NVTX or GPU kernel tables",
            }
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        db.row_factory = sqlite3.Row
        ranges = []
        for row in db.execute("SELECT * FROM NVTX_EVENTS"):
            row = dict(row)
            label = row.get("text") or strings.get(row.get("textId"))
            if (
                label == "sid_bench/measured"
                and row.get("end")
                and row["end"] > row["start"]
            ):
                ranges.append((row["start"], row["end"]))
        if not ranges:
            return {
                "status": "unavailable",
                "reason": "No complete measured-request ranges; startup must not be counted as inference",
            }
        if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in names:
            for start, end in db.execute(
                "SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
            ):
                if any(a <= start and end <= b for a, b in ranges):
                    return {
                        "status": "unavailable",
                        "reason": "Graph-level trace omits kernel nodes; capture with --cuda-graph-trace=node",
                    }
        totals, active = {}, []
        for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            row = dict(row)
            if not any(a <= row["start"] and row["end"] <= b for a, b in ranges):
                continue
            name = strings.get(
                row.get("demangledName"), strings.get(row.get("shortName"), "unknown")
            )
            item = totals.setdefault(name, {"name": name, "calls": 0, "gpu_ms": 0.0})
            item["calls"] += 1
            item["gpu_ms"] += (row["end"] - row["start"]) / 1e6
            active.append((row["start"], row["end"]))
    if not active:
        return {
            "status": "unavailable",
            "reason": "No GPU kernels captured inside the measured ranges",
        }
    return {
        "status": "success",
        "measured_requests": len(ranges),
        "kernel_duration_sum_ms": sum(r["gpu_ms"] for r in totals.values()),
        "gpu_active_union_ms": union_ns(active) / 1e6,
        "measured_window_ms": union_ns(ranges) / 1e6,
        "semantic_stage_attribution": "unclassified; raw kernel names only",
        "kernels": sorted(totals.values(), key=lambda r: r["gpu_ms"], reverse=True),
    }
