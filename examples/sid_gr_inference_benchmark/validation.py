"""Structural correctness and numerical agreement are separate results."""

import math


class Unsupported(ValueError):
    """A requested semantic or feature is not implemented by this adapter."""


def supported(case):
    model, generation = case["model"], case["generation"]
    requirements = [
        (model["dtype"] == "bfloat16", "only BF16 is currently supported"),
        (model["quantization"] == "none", "quantized cases need a dedicated adapter"),
        (model["kv_cache_dtype"] == "bfloat16", "KV cache must be BF16"),
        (generation["ignore_eos"] is True, "fixed-length runs require ignore_eos"),
        (
            generation["constraints"] == "none",
            "catalog constraints are not migrated yet",
        ),
        (
            generation["score"] == "cumulative_logprob",
            "only cumulative logprob is supported",
        ),
        (generation["length_penalty"] == 0, "only unnormalized scores are supported"),
        (generation["return_beams"] == "all", "all beams must be returned"),
        (
            case["prefix_cache"] is False,
            "prefix-cache comparisons are not migrated yet",
        ),
    ]
    for okay, reason in requirements:
        if not okay:
            raise Unsupported(reason)


def check(outputs, workload, case, vocab_size):
    if not isinstance(outputs, list) or len(outputs) != len(workload):
        raise ValueError("Wrong number of completed requests")
    for row, expected in zip(outputs, workload):
        if row["request_id"] != expected["request_id"]:
            raise ValueError("Request order/identity mismatch")
        beams = row["beams"]
        if len(beams) != case["beam_width"]:
            raise ValueError("Wrong number of beams")
        seen = set()
        for beam in beams:
            tokens = beam["token_ids"]
            if len(tokens) != case["generation"]["output_tokens"]:
                raise ValueError("Wrong output token count")
            if any(type(t) is not int or not 0 <= t < vocab_size for t in tokens):
                raise ValueError("Invalid output token ID")
            if type(beam["score"]) not in (int, float) or not math.isfinite(
                beam["score"]
            ):
                raise ValueError("Missing/non-finite beam score")
            if tuple(tokens) in seen:
                raise ValueError("Duplicate beam sequence")
            seen.add(tuple(tokens))
        if any(a["score"] < b["score"] for a, b in zip(beams, beams[1:])):
            raise ValueError("Beams must be ordered by descending score")


def compare(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("Cannot compare different request counts")
    checks = []
    for a, b in zip(reference, candidate):
        if a["request_id"] != b["request_id"]:
            raise ValueError("Cannot compare different request IDs")
        left, right = a["beams"], b["beams"]
        ai, bi = [tuple(x["token_ids"]) for x in left], [
            tuple(x["token_ids"]) for x in right
        ]
        common = set(ai) & set(bi)
        sa, sb = {tuple(x["token_ids"]): x["score"] for x in left}, {
            tuple(x["token_ids"]): x["score"] for x in right
        }
        checks.append(
            {
                "top1_exact": ai[0] == bi[0],
                "ordered_exact": ai == bi,
                "beam_overlap": len(common) / len(ai),
                "max_score_difference": max(
                    (abs(sa[t] - sb[t]) for t in common), default=None
                ),
            }
        )
    return checks
