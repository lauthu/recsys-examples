"""CPU tests for named run configurations and their recorded environment."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.config import ROOT, load, read_yaml, scenario


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
        environment = cfg["environment"]["sid_gr_inference"]
        self.assertEqual(environment["BENCHMARK_GPU"], "GPU-selected")
        self.assertEqual(environment["SID_GR_RUNTIME"], "local")
        self.assertEqual(environment["SID_GR_SOURCE"], "../sid-gr-inference")
        self.assertEqual(environment["QWEN3_0_6B_MODEL"], "")
        self.assertEqual(
            environment["GR_DECODE_ATTEN_ROOT"], "../../corelib/gr_decode_atten"
        )
        self.assertNotIn("SID_GR_RUNTIME", cfg["environment"]["sglang"])
        for values in cfg["environment"].values():
            self.assertEqual(values["BENCHMARK_GPU"], "GPU-selected")
            self.assertEqual(values["QWEN3_0_6B_MODEL"], "")
            self.assertNotIn("UNRELATED_SECRET", values)
        self.assertEqual(cfg["run_profile"], "config/runs/default.yml")

    def test_candidates_keep_distinct_defaults_for_the_same_variable(self):
        def with_all_candidates(path, environment=None):
            value = read_yaml(path, environment)
            if Path(path).parent.name == "runs":
                value["solutions"].append("sglang_beam_search")
            return value

        with patch.dict(os.environ, {}, clear=True), patch(
            "harness.config.read_yaml", side_effect=with_all_candidates
        ):
            cfg = load(ROOT / "config/runs/default.yml")
        native = cfg["environment"]["sglang"]["SGLANG_IMAGE"]
        beam = cfg["environment"]["sglang_beam_search"]["SGLANG_IMAGE"]
        self.assertNotEqual(native, beam)
        self.assertEqual(native, cfg["solution_configs"]["sglang"]["runtime"]["image"])
        self.assertEqual(
            beam, cfg["solution_configs"]["sglang_beam_search"]["runtime"]["image"]
        )
        self.assertNotIn("SGLANG_IMAGE", cfg["environment"]["sid_gr_inference"])

    def test_models_require_immutable_hub_revisions(self):
        spec = read_yaml(ROOT / "config/scenarios/qwen3_0_6B.yml")
        for revision in ["main", "v1", "abc123", 42]:
            with self.subTest(revision=revision):
                spec["model"]["revision"] = revision
                with patch("harness.config.read_yaml", return_value=spec):
                    with self.assertRaisesRegex(ValueError, "immutable commit"):
                        scenario("qwen3_0_6B")

    def test_hardware_cpu_budget_must_be_positive_integer(self):
        for count in [0, -1, True, "10", 1.5]:
            with self.subTest(cpu_cores=count):

                def read_with_invalid_count(path, environment=None):
                    value = read_yaml(path, environment)
                    if Path(path).parent.name == "hardware":
                        value["cpu_cores"] = count
                    return value

                with patch(
                    "harness.config.read_yaml", side_effect=read_with_invalid_count
                ):
                    with self.assertRaisesRegex(ValueError, "hardware.cpu_cores"):
                        load(ROOT / "config/runs/default.yml")


if __name__ == "__main__":
    unittest.main()
