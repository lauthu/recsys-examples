"""Leaderboards compare frozen input data, independently of its local path."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.artifacts import read, write
from harness.config import ROOT, load
from harness.runner import e2e, select_cases


class InputIdentityTests(unittest.TestCase):
    def test_jsonl_revisions_stay_separate_across_partial_runs(self):
        cfg = load(ROOT / "config/runs/default.yml")
        cfg["hardware_config"]["cpu_cores"] = 1
        name = cfg["scenarios"][0]
        spec = cfg["scenario_configs"][name]
        cfg["scenario_configs"] = {name: spec}
        spec["inputs"].update(type="jsonl", context_lengths=[4], batch_sizes=[1])
        spec["generation"]["beam_widths"] = [2]
        cfg["solutions"] = [cfg["baseline"], "candidate"]
        cfg["solution_configs"]["candidate"] = copy.deepcopy(
            cfg["solution_configs"][cfg["baseline"]]
        )
        cfg["solution_configs"]["candidate"]["name"] = "candidate"
        hardware = {"name": cfg["hardware_config"]["gpu_name"], "uuid": "GPU-test"}
        for solution in cfg["solution_configs"].values():
            solution["runtime"] = {"kind": "local", "python": sys.executable}

        def launch(request, directory):
            result = {
                "status": "success",
                "samples_ms": [10, 10],
                "workload_sha256": request["workload_sha256"],
                "outputs": [
                    [
                        {
                            "request_id": "request-0",
                            "beams": [
                                {"token_ids": [3, 4, 5], "score": -1.0},
                                {"token_ids": [3, 4, 6], "score": -2.0},
                            ],
                        }
                    ]
                ],
                "environment": {"gpu": hardware["name"], "packages": {}},
            }
            write(directory / "request.json", request)
            write(directory / "result.json", result)
            return result

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = root / "inputs.jsonl"
            inputs.write_text(
                json.dumps({"request_id": "request-0", "input_ids": [3] * 4})
            )
            spec["inputs"]["path"] = str(inputs)
            spec["model"]["path"] = d
            cfg["output_dir"] = str(root / "results")
            model = {"path": d, "sha256": "model", "config": {"vocab_size": 32}}
            board = root / "leaderboards" / f"{name}.md"
            with patch("harness.runtime.gpu_info", return_value=hardware), patch(
                "harness.runner.resolve_model", return_value=model
            ), patch("harness.runner.source_info", return_value=None), patch(
                "harness.runner.launch", side_effect=launch
            ):
                first, manifest = e2e(cfg, select_cases(cfg))
                original_hash = manifest["cases"][0]["input_file_sha256"]
                self.assertEqual(board.read_text().count("\n## "), 1)

                # Moving the same data preserves comparability across runs.
                relocated = root / "moved.jsonl"
                inputs.rename(relocated)
                spec["inputs"]["path"] = str(relocated)
                cfg["solutions"] = ["candidate"]
                _, manifest = e2e(cfg, select_cases(cfg))
                self.assertEqual(
                    manifest["cases"][0]["input_file_sha256"], original_hash
                )
                self.assertEqual(board.read_text().count("\n## "), 1)

                # A candidate-only rerun on new data cannot share the old baseline.
                relocated.write_text(
                    json.dumps({"request_id": "request-0", "input_ids": [4] * 4})
                )
                latest, manifest = e2e(cfg, select_cases(cfg))
                self.assertNotEqual(
                    manifest["cases"][0]["input_file_sha256"], original_hash
                )
                sections = board.read_text().split("\n## ")[1:]
                self.assertEqual(len(sections), 2)
                section = next(s for s in sections if latest.name in s)
                self.assertNotIn(first.name, section)
                self.assertEqual(
                    read(first / "manifest.json")["cases"][0]["input_file_sha256"],
                    original_hash,
                )


if __name__ == "__main__":
    unittest.main()
