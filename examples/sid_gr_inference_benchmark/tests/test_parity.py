"""Beam agreement distinguishes set overlap, rank agreement and score changes."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.validation import compare


def output(tokens, scores):
    return [
        {
            "request_id": "request-0",
            "beams": [
                {"token_ids": [token], "score": score}
                for token, score in zip(tokens, scores)
            ],
        }
    ]


class ParityTests(unittest.TestCase):
    def test_partial_rank_and_set_agreement(self):
        reference = output([1, 2, 3, 4], [-1, -2, -3, -4])
        candidate = output([2, 1, 3, 5], [-1.5, -2.5, -3.5, -4.5])
        row = compare(reference, candidate)[0]
        self.assertFalse(row["top1_exact"])
        self.assertFalse(row["ordered_exact"])
        self.assertEqual(row["ordered_match_fraction"], 0.25)
        self.assertEqual(row["beam_overlap"], 0.75)
        self.assertEqual(row["max_score_difference"], 1.5)

    def test_matching_tokens_can_have_different_scores(self):
        reference = output([1, 2], [-1, -2])
        candidate = output([1, 2], [-1.25, -2.25])
        row = compare(reference, candidate)[0]
        self.assertTrue(row["top1_exact"])
        self.assertTrue(row["ordered_exact"])
        self.assertEqual(row["ordered_match_fraction"], 1.0)
        self.assertEqual(row["beam_overlap"], 1.0)
        self.assertEqual(row["max_score_difference"], 0.25)


if __name__ == "__main__":
    unittest.main()
