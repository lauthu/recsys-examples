"""CPU tests for named run configurations and their recorded environment."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.config import ROOT, load


class ConfigTests(unittest.TestCase):
    def test_named_runs_keep_output_at_suite_root(self):
        for name, baseline in [
            ("default", "sglang"),
            ("readme", "sglang_beam_search"),
        ]:
            with self.subTest(name=name):
                cfg = load(ROOT / "config/runs" / f"{name}.yml")
                self.assertEqual(cfg["baseline"], baseline)
                self.assertEqual(cfg["output_dir"], str(ROOT / "results"))
                self.assertEqual(
                    cfg["solution_configs"]["sid_gr_inference"]["source"]["path"],
                    str(
                        (
                            ROOT
                            / os.environ.get("SID_GR_SOURCE", "../sid-gr-inference")
                        ).resolve()
                    ),
                )

    def test_records_explicit_environment_and_defaults_only_when_referenced(self):
        with patch.dict(
            os.environ,
            {"BENCHMARK_GPU": "GPU-selected", "UNRELATED_SECRET": "not recorded"},
            clear=True,
        ):
            cfg = load(ROOT / "config/runs/default.yml")
        environment = cfg["environment"]
        self.assertEqual(environment["BENCHMARK_GPU"], "GPU-selected")
        self.assertEqual(environment["SID_GR_RUNTIME"], "local")
        self.assertEqual(environment["SID_GR_SOURCE"], "../sid-gr-inference")
        self.assertEqual(
            environment["QWEN3_0_6B_MODEL"], "../sid-gr-inference/models/Qwen3-0.6B"
        )
        self.assertEqual(environment["GR_DECODE_ATTEN_ROOT"], "")
        self.assertNotIn("UNRELATED_SECRET", environment)


if __name__ == "__main__":
    unittest.main()
