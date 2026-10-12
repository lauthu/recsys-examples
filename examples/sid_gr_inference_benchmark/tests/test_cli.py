"""CPU checks for command-line candidate selection."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench


class CliTests(unittest.TestCase):
    def test_duplicate_solutions_are_rejected_before_launch(self):
        for command in ["plan", "e2e"]:
            with self.subTest(command=command):
                errors = io.StringIO()
                with patch.object(
                    sys,
                    "argv",
                    ["bench.py", command, "--solutions", "sglang", "sglang"],
                ), patch("bench.e2e") as launch, contextlib.redirect_stderr(errors):
                    self.assertEqual(bench.main(), 1)
                    launch.assert_not_called()
                self.assertTrue(
                    "Selected solutions must be unique" in errors.getvalue()
                )


if __name__ == "__main__":
    unittest.main()
