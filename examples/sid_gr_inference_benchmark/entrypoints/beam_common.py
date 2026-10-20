"""Framework-free checks shared by the additional beam adapters."""

import importlib.metadata
import math
import sys
from pathlib import Path

from harness.validation import Unsupported, supported


def options(case, solution, defaults):
    supported(case)
    unknown = set(solution["engine"]) - set(defaults)
    if unknown:
        raise Unsupported(f"Unrecognized {solution['name']} options: {sorted(unknown)}")
    values = {**defaults, **solution["engine"]}
    for key, default in defaults.items():
        value = values[key]
        if isinstance(default, bool) and type(value) is not bool:
            raise Unsupported(f"{key} must be a boolean")
        if type(default) is float and (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value < 1
        ):
            raise Unsupported(f"{key} must be between 0 and 1")
        if type(default) is int and (type(value) is not int or value < 0):
            raise Unsupported(f"{key} must be a nonnegative integer")
    return values


def source_path(solution, subdirectory=""):
    source = solution["source"].get("path")
    if source:
        sys.path.insert(0, str(Path(source) / subdirectory))


def require_version(distribution, expected):
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise Unsupported(
            f"Install {distribution}=={expected} in this runtime"
        ) from exc
    if version.partition("+")[0] != expected:
        raise Unsupported(
            f"This adapter requires {distribution}=={expected}; found {version}"
        )
    return version


def normalize(outputs, workload, *, sequences, tokens, score):
    """Copy every completed beam to the CPU-output contract inside the timed call."""
    outputs = list(outputs)
    if len(outputs) != len(workload):
        raise ValueError("Expected one completed beam-search result per request")
    rows = []
    for request, output in zip(workload, outputs):
        beams = []
        for sequence in getattr(output, sequences):
            value = getattr(sequence, score)
            if value is None:
                raise ValueError("Backend returned no cumulative beam score")
            beams.append(
                {"token_ids": list(getattr(sequence, tokens)), "score": float(value)}
            )
        rows.append({"request_id": request["request_id"], "beams": beams})
    return rows
