# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The catalog every solution shares: parsing and beam membership checks.

A catalog is JSONL with one item per line:
    {"item_id": "item-a", "token_ids": [1000, 2000, 3000], "metadata": {...}}
Token IDs are model token IDs. It is part of what is measured: every solution
in a run is constrained to exactly these token paths, and every returned beam is
checked against them. Each engine's native storage format is solution-specific
and lives with the engine's adapter (`Adapter.convert_catalog` in
bench/adapters.py). `synthesize` writes a seeded synthetic catalog
(tools/make_catalog.py).
"""

import hashlib
import itertools
import json
import random


def parse(data, *, output_tokens, beam_width, vocab_size):
    """Return ({token path: item_id}, identity) or raise ValueError.

    Line handling matches SID-GR's loader (universal newlines, blank lines
    skipped, optional "metadata" must be an object), so a catalog accepted here
    is accepted by every engine. Paths must have exactly output_tokens tokens,
    and at least beam_width distinct first tokens keep every beam alive.
    """
    paths = {}
    item_ids = set()
    text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"catalog line {number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"catalog line {number}: expected a JSON object")
        item_id, tokens = row.get("item_id"), row.get("token_ids")
        if (
            type(item_id) not in (str, int)
            or not str(item_id).strip()
            or item_id in item_ids
        ):
            raise ValueError(f"catalog line {number}: missing or duplicate item_id")
        if not isinstance(row.get("metadata", {}), dict):
            raise ValueError(f"catalog line {number}: metadata must be an object")
        if (
            not isinstance(tokens, list)
            or len(tokens) != output_tokens
            or any(type(t) is not int or not 0 <= t < vocab_size for t in tokens)
        ):
            raise ValueError(
                f"catalog line {number}: token_ids must be {output_tokens} "
                f"token IDs in [0, {vocab_size})"
            )
        if tuple(tokens) in paths:
            raise ValueError(f"catalog line {number}: duplicate token path")
        paths[tuple(tokens)] = item_id
        item_ids.add(item_id)
    first_tokens = len({path[0] for path in paths})
    if first_tokens < beam_width:
        raise ValueError(
            f"catalog has {first_tokens} distinct first tokens; "
            f"beam width {beam_width} needs at least that many"
        )
    return paths, {
        "sha256": hashlib.sha256(data).hexdigest(),
        "items": len(paths),
        "first_tokens": first_tokens,
    }


def synthesize(items, *, levels, codebook_size, token_start, skew, seed):
    """JSONL bytes of `items` distinct token paths of `levels` tokens each.

    The token at depth d lies in [token_start + d * codebook_size,
    token_start + (d + 1) * codebook_size), the layout a scenario's
    model.sid_codebooks describes. Each depth draws its codes from a Zipf
    distribution with exponent `skew` over a seeded shuffle of the codebook, so
    popular prefixes have many children and most combinations do not exist;
    repeated paths are drawn again. Items are sorted by path, and the same
    arguments always give the same bytes.
    """
    if items > codebook_size**levels // 2:
        raise ValueError("items must be at most half of codebook_size ** levels")
    rng = random.Random(seed)
    weights = itertools.accumulate(
        1 / rank**skew for rank in range(1, codebook_size + 1)
    )
    cumulative = list(weights)
    orders = []
    for _ in range(levels):
        codes = list(range(codebook_size))
        rng.shuffle(codes)
        orders.append(codes)
    paths = set()
    while len(paths) < items:
        need = items - len(paths)
        draws = [rng.choices(codes, cum_weights=cumulative, k=need) for codes in orders]
        paths.update(zip(*draws))
    rows = []
    for number, codes in enumerate(sorted(paths)):
        tokens = [
            token_start + d * codebook_size + code for d, code in enumerate(codes)
        ]
        row = {"item_id": f"item-{number:07d}", "token_ids": tokens}
        rows.append(json.dumps(row, separators=(",", ":")))
    return ("\n".join(rows) + "\n").encode()


def check_status(status, *, source, items):
    """The server must have loaded exactly the managed catalog file."""
    if not status or not status.get("active"):
        raise ValueError(f"server reports no active catalog: {status}")
    if status.get("source") != source or status.get("items") != items:
        raise ValueError(
            f"server catalog {status.get('source')} with {status.get('items')} items "
            f"differs from {source} with {items} items"
        )


def outside(beams, paths):
    """Number of returned beams whose token path is not a catalog item."""
    return sum(tuple(tokens) not in paths for tokens, _ in beams)
