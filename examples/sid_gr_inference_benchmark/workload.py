"""One serialized token workload shared by every solution in a case."""

import json
import random
from pathlib import Path


def build(case, model_config):
    vocab = int(model_config["vocab_size"])
    context, batch = case["context_length"], case["batch_size"]
    if context + case["generation"]["output_tokens"] > model_config.get(
        "max_position_embeddings", float("inf")
    ):
        raise ValueError("Requested context/output exceeds checkpoint position limit")
    if case["beam_width"] > vocab:
        raise ValueError("Initial beam width exceeds vocabulary")
    if case["inputs"]["type"] == "jsonl":
        rows = [
            json.loads(line)
            for line in Path(case["inputs"]["path"]).read_text().splitlines()
            if line.strip()
        ]
        rows = rows[:batch]
    else:
        rng = random.Random(case["inputs"].get("seed", 42))
        excluded = set()
        for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            value = model_config.get(key)
            excluded.update(
                value if isinstance(value, list) else ([] if value is None else [value])
            )
        allowed = [i for i in range(vocab) if i not in excluded]
        if not allowed:
            raise ValueError("No non-special input token IDs")
        rows = [
            {"request_id": f"request-{i}", "input_ids": rng.choices(allowed, k=context)}
            for i in range(batch)
        ]
    if len(rows) != batch or len({r["request_id"] for r in rows}) != batch:
        raise ValueError("Workload needs the requested number of distinct request IDs")
    for row in rows:
        ids = row["input_ids"]
        if len(ids) != context or any(
            type(t) is not int or not 0 <= t < vocab for t in ids
        ):
            raise ValueError(
                "Workload token length/range mismatch; input files are never silently truncated"
            )
    return [{"request_id": r["request_id"], "input_ids": r["input_ids"]} for r in rows]
