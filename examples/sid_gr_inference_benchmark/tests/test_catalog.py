# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import pickle
import struct
from types import SimpleNamespace

import pytest
from bench import catalog, engines
from bench.config import SidCodebooks

PATHS = [
    [1000 + a, 2000 + b, 3000 + c] for a in range(4) for b in range(2) for c in range(2)
]


def jsonl(rows, newline="\n"):
    return newline.join(json.dumps(row) for row in rows).encode()


def parse(data, beam_width=4, vocab_size=5000):
    return catalog.parse(
        data, output_tokens=3, beam_width=beam_width, vocab_size=vocab_size
    )


def test_parse_matches_sid_gr_line_handling():
    rows = [
        {"item_id": f"item-{i}", "token_ids": p, "metadata": {"i": i}}
        for i, p in enumerate(PATHS)
    ]
    paths, info = parse(jsonl(rows, "\r\n") + b"\r\n\n")
    assert len(paths) == info["items"] == 16
    assert info["first_tokens"] == 4
    assert paths[tuple(PATHS[0])] == "item-0"


@pytest.mark.parametrize(
    "row, message",
    [
        ({"item_id": "x", "token_ids": [1000, 2000]}, "token_ids"),
        ({"item_id": "x", "token_ids": [1000, 2000, 9000]}, "token_ids"),
        ({"item_id": "x", "token_ids": [1000, 2000, True]}, "token_ids"),
        ({"item_id": " ", "token_ids": [1009, 2000, 3000]}, "item_id"),
        ({"item_id": "item-0", "token_ids": [1009, 2000, 3000]}, "duplicate item_id"),
        ({"item_id": "y", "token_ids": PATHS[0]}, "duplicate token path"),
        (
            {"item_id": "y", "token_ids": [1009, 2000, 3000], "metadata": None},
            "metadata",
        ),
    ],
)
def test_parse_rejects_what_an_engine_would_reject(row, message):
    rows = [{"item_id": f"item-{i}", "token_ids": p} for i, p in enumerate(PATHS)] + [
        row
    ]
    with pytest.raises(ValueError, match=message):
        parse(jsonl(rows))


def test_parse_does_not_split_on_form_feed():
    rows = [{"item_id": f"item-{i}", "token_ids": p} for i, p in enumerate(PATHS)]
    with pytest.raises(ValueError, match="invalid JSON"):
        parse(jsonl(rows[:2], "\x0c"))


def test_parse_needs_beam_width_first_tokens():
    rows = [{"item_id": f"item-{i}", "token_ids": p} for i, p in enumerate(PATHS)]
    with pytest.raises(ValueError, match="distinct first tokens"):
        parse(jsonl(rows), beam_width=5)


def test_sglang_trie_contains_every_path():
    legacy = engines.get("sglang-beam-search-new")
    paths = {tuple(p): i for i, p in enumerate(PATHS)}
    data = pickle.loads(legacy.convert_catalog(paths, None, 5000))
    assert data["num_sequences"] == 16 and data["max_depth"] == data["min_depth"] == 3

    def walk(node, prefix=()):
        return ([prefix] if node["t"] else []) + [
            path
            for token, child in node["c"].items()
            for path in walk(child, prefix + (token,))
        ]

    assert sorted(walk(data["tree"])) == sorted(map(tuple, PATHS))


def test_vedantjh2_round_trip():
    model = SimpleNamespace(sid_codebooks=SidCodebooks(1000, 1000))
    paths = {tuple(p): i for i, p in enumerate(PATHS)}
    data = engines.get("vedantjh2-sglang").convert_catalog(paths, model, 5000)
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + size])
    assert size % 8 == 0 and header["__metadata__"]["token_start"] == "1000"

    def array(name):
        start, end = header[name]["data_offsets"]
        return list(
            struct.unpack(
                f"<{(end - start) // 4}i", data[8 + size + start : 8 + size + end]
            )
        )

    roots, l1_off, l1 = (
        array("root_token_ids"),
        array("level1_offsets"),
        array("level1_token_ids"),
    )
    l2_off, l2 = array("level2_offsets"), array("level2_token_ids")
    rebuilt = [
        (1000 + a, 2000 + l1[j], 3000 + l2[k])
        for i, a in enumerate(roots)
        for j in range(l1_off[i], l1_off[i + 1])
        for k in range(l2_off[j], l2_off[j + 1])
    ]
    assert sorted(rebuilt) == sorted(map(tuple, PATHS))


def test_vedantjh2_rejects_tokens_outside_codebooks():
    with pytest.raises(ValueError, match="sid_codebooks ranges"):
        engines.trie_output_head(
            {(1000, 1999, 3000): 0},
            token_start=1000,
            codebook_size=1000,
            vocab_size=5000,
        )
    with pytest.raises(ValueError, match="vocabulary"):
        engines.trie_output_head(
            {(1000, 2000, 3000): 0},
            token_start=1000,
            codebook_size=1000,
            vocab_size=3500,
        )


def test_status_and_membership():
    catalog.check_status(
        {"active": True, "source": "/work/c.jsonl", "items": 2},
        source="/work/c.jsonl",
        items=2,
    )
    for status in (
        None,
        {"active": False},
        {"active": True, "source": "/other", "items": 2},
    ):
        with pytest.raises(ValueError):
            catalog.check_status(status, source="/work/c.jsonl", items=2)
    paths = {tuple(PATHS[0]): "a"}
    assert catalog.outside([(PATHS[0], -1.0), (PATHS[1], -2.0)], paths) == 1
