#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a seeded synthetic catalog and print the scenario keys that pin it.

    python3 tools/make_catalog.py --items 1000000 --output catalog.jsonl

bench.catalog.synthesize describes the shape. The file is checked as a run
would check it, the same arguments always give the same bytes, and the printed
model.sid_codebooks and catalog.sha256 go in the scenario.
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the benchmark's bench/
from bench import catalog  # noqa: E402


def spread(counts):
    values = sorted(counts.values())
    return f"min {values[0]}, median {statistics.median(values):g}, max {values[-1]}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, required=True)
    parser.add_argument("--levels", type=int, default=3, help="tokens per item")
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--token-start", type=int, default=1000)
    parser.add_argument("--skew", type=float, default=1.0, help="Zipf exponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--beam-width",
        type=int,
        default=256,
        help="the scenario's beam width: the catalog needs that many first tokens",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    shape = dict(levels=args.levels, codebook_size=args.codebook_size)
    data = catalog.synthesize(
        args.items,
        **shape,
        token_start=args.token_start,
        skew=args.skew,
        seed=args.seed,
    )
    vocab = args.token_start + args.levels * args.codebook_size
    paths, info = catalog.parse(
        data, output_tokens=args.levels, beam_width=args.beam_width, vocab_size=vocab
    )
    args.output.write_bytes(data)
    print(f"{args.output}: {info['items']:,} items, {len(data):,} bytes")
    for depth in range(1, args.levels):
        children = Counter(path[:depth] for path in {p[: depth + 1] for p in paths})
        print(f"children per depth-{depth - 1} prefix: {spread(children)}")
    print(
        f"tokens {args.token_start} to {vocab - 1}: the model's vocabulary must reach them"
    )
    print("\nScenario keys:\n")
    print("model:\n  sid_codebooks:")
    print(
        f"    token_start: {args.token_start}\n    codebook_size: {args.codebook_size}"
    )
    print(f'catalog:\n  sha256: "{info["sha256"]}"')


if __name__ == "__main__":
    main()
