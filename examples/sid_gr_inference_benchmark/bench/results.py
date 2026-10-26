# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""results.json.gz: everything one engine's run measured and observed, saved
before any report is written. report.md, comparison.md and their charts are
rendered from these files alone, so the same results always give the same
reports, and the reports can be rendered again without the GPU, after a
template change for example:

    python3 tools/render_reports.py OUTPUT
"""

import dataclasses
import gzip
import json

from bench import compare, report
from bench.config import Scenario, Solution, parse

NAME = "results.json.gz"
FORMAT = 1  # raised when saved fields change, so old results fail loudly


def save(directory, results):
    """results holds the Scenario and Solution as dataclasses, the rest as JSON
    values. The file carries no timestamp, so the same results save the same."""
    data = {
        "format": FORMAT,
        **results,
        "scenario": dataclasses.asdict(results["scenario"]),
        "solution": dataclasses.asdict(results["solution"]),
    }
    text = json.dumps(data, separators=(",", ":"))
    with gzip.GzipFile(directory / NAME, "wb", mtime=0) as file:
        file.write(text.encode())


def load(directory):
    with gzip.open(directory / NAME, "rt") as file:
        data = json.load(file)
    found = data.pop("format", None)
    if found != FORMAT:
        raise ValueError(
            f"{directory / NAME} has format {found}, not {FORMAT}; render it with "
            "the benchmark code that wrote it"
        )
    data["scenario"] = parse(Scenario, data["scenario"], "scenario")
    data["solution"] = parse(Solution, data["solution"], "solution")
    return data


def write(directory, files):
    for name, text in files.items():
        (directory / name).write_text(text)


def write_report(directory):
    """Render report.md and its charts in directory from its results."""
    write(directory, report.render(load(directory)))


def engines(output):
    """{name: results} of the engines in output's directories."""
    paths = sorted(output.glob(f"*/{NAME}"))
    return {path.parent.name: load(path.parent) for path in paths}


def write_comparison(output, found):
    """Render comparison.md and its chart in output from the engines' results
    (engines()), in their command-line order."""
    first = next(iter(found.values()))
    names, reference = first["solutions"], first["reference"]
    missing = sorted(set(names) - found.keys())
    if missing:
        raise ValueError(f"{output} has no results of {missing}")
    write(output, compare.render([found[name] for name in names], names, reference))


def render(output):
    """Render every report under output: one engine's in output itself, or each
    engine's in its directory and the comparison across them."""
    if (output / NAME).is_file():
        write_report(output)
        return
    found = engines(output)
    if not found:
        raise ValueError(f"no {NAME} in {output} or its directories")
    for name, results in found.items():
        write(output / name, report.render(results))
    write_comparison(output, found)
