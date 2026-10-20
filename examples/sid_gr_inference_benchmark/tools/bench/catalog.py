# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catalog parsing, native trie serialization and beam membership checks.

A catalog is JSONL with one item per line:
    {"item_id": "item-a", "token_ids": [1000, 2000, 3000], "metadata": {...}}
Token IDs are model token IDs. Conversion changes only the storage format, so
every engine constrains generation to exactly the same token paths.
"""

import array
import hashlib
import json
import pickle
import struct
import sys


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


def sglang_trie(paths):
    """Pickled token trie read by SGLang beam-search-new's prefix-constrained trie."""
    root = {"t": False, "c": {}}
    nodes = 1
    for tokens in sorted(paths):
        node = root
        for token in tokens:
            if token not in node["c"]:
                node["c"][token] = {"t": False, "c": {}}
                nodes += 1
            node = node["c"][token]
        node["t"] = True
    lengths = {len(path) for path in paths}
    return pickle.dumps(
        {
            "tree": root,
            "num_sequences": len(paths),
            "num_nodes": nodes,
            "num_edges": nodes - 1,
            "max_depth": max(lengths),
            "min_depth": min(lengths),
        },
        protocol=4,
    )


def sidtrie(paths, *, token_start, codebook_size, vocab_size):
    """trie_output_head.safetensors for the SGLang SIDTrie output head.

    Depth d of a three-token path must lie in codebook range
    [token_start + d * codebook_size, token_start + (d + 1) * codebook_size).
    The file stores range-local codes as three-level CSR int32 indices.
    """
    if token_start < 0 or codebook_size <= 0:
        raise ValueError("trie_config needs token_start >= 0 and codebook_size > 0")
    if token_start + 3 * codebook_size > vocab_size:
        raise ValueError("trie_config codebook ranges exceed the model vocabulary")
    tree = {}
    for path in paths:
        if len(path) != 3:
            raise ValueError("SIDTrie paths must have exactly three tokens")
        codes = [
            token - token_start - depth * codebook_size
            for depth, token in enumerate(path)
        ]
        if any(not 0 <= code < codebook_size for code in codes):
            raise ValueError(
                f"token path {path} is outside the trie_config codebook ranges"
            )
        tree.setdefault(codes[0], {}).setdefault(codes[1], []).append(codes[2])
    roots, level1, level2 = sorted(tree), [], []
    level1_offsets, level2_offsets = [0], [0]
    for a in roots:
        for b in sorted(tree[a]):
            level1.append(b)
            level2.extend(sorted(tree[a][b]))
            level2_offsets.append(len(level2))
        level1_offsets.append(len(level1))
    arrays = {
        "root_token_ids": roots,
        "level1_offsets": level1_offsets,
        "level1_token_ids": level1,
        "level2_offsets": level2_offsets,
        "level2_token_ids": level2,
    }
    header = {
        "__metadata__": {
            "format_version": "1",
            "token_start": str(token_start),
            "codebook_size": str(codebook_size),
            "num_codebooks": "3",
        }
    }
    payload = bytearray()
    for name, values in arrays.items():
        start = len(payload)
        packed = array.array("i", values)
        if sys.byteorder != "little":
            packed.byteswap()
        payload += packed.tobytes()
        header[name] = {
            "dtype": "I32",
            "shape": [len(values)],
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + bytes(payload)


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
