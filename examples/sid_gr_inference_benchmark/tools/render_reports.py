#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the reports of a benchmark output again from its saved results.

    python3 tools/render_reports.py OUTPUT

OUTPUT is what run_benchmark.py wrote: one solution's directory, or the
directory of several with summary.md. Rewrites every report.md, the
summary and the charts from the results.json.gz files alone, without
Docker or the GPU, as the current templates lay them out.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the benchmark's bench/
from bench.outputs import results  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="a run_benchmark.py --output")
    args = parser.parse_args(argv)
    try:
        results.render(args.output.expanduser().resolve())
    except (ValueError, OSError) as exc:
        sys.exit(f"render failed: {exc}")
    print(f"Rendered the reports in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
