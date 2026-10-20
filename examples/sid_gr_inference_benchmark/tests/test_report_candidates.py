"""Partial reports show every configured candidate without inventing results."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.artifacts import read, write
from harness.config import ROOT, cases, load
from harness.reporting.report import render


class ReportCandidateTests(unittest.TestCase):
    def make_run(self, directory, names, finished=True):
        cfg = load(ROOT / "config/runs/default.yml")
        solution = cfg["solution_configs"]["sid_gr_inference"]
        cfg["solutions"] = names
        cfg["baseline"] = names[0]
        cfg["solution_configs"] = {
            name: {**copy.deepcopy(solution), "name": name} for name in names
        }
        case = next(cases(cfg))
        hardware = {"name": cfg["hardware_config"]["gpu_name"]}
        manifest = {
            "schema_version": 2,
            "run_id": "partial",
            "mode": "e2e",
            "status": "running",
            "config": cfg,
            "hardware": hardware,
            "scenario_jobs": {case["name"]: 2 * len(names)},
            "cases": [{"case": case, "model": {"config": {}, "sha256": "checkpoint"}}],
            "jobs": [],
        }
        if finished:
            for name, status in zip(names, ["success", "failed"]):
                job_dir = f"jobs/{name}/{case['id']}"
                manifest["jobs"].append(
                    {
                        "case_id": case["id"],
                        "solution": name,
                        "directory": job_dir,
                        "status": status,
                    }
                )
                write(
                    directory / job_dir / "request.json",
                    {"case": case, "solution": cfg["solution_configs"][name]},
                )
                write(
                    directory / job_dir / "result.json",
                    {
                        "status": status,
                        "samples_ms": [10.0, 10.0],
                        "outputs": [
                            [
                                {
                                    "request_id": "request-0",
                                    "beams": [{"token_ids": [3, 4, 5], "score": -1.0}],
                                }
                            ]
                        ],
                        "workload_sha256": "same",
                        "environment": {"gpu": hardware["name"], "packages": {}},
                    },
                )
        write(directory / "manifest.json", manifest)
        return manifest

    def test_four_candidate_partial_report_preserves_unmeasured_candidates(self):
        names = [
            "sid_gr_inference",
            "sglang",
            "sglang_beam_search",
            "sglang_beam_search_new",
        ]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "results/partial"
            manifest = self.make_run(root, names)
            data = render(root)
            report = (root / "report.md").read_text()
            overview = report.split("**Reference:**", 1)[0]
            for name in names:
                self.assertEqual(overview.count(f"| {name} |"), 1)
            self.assertIn("| 1 of 2 | in progress |", overview)
            for name in names[2:]:
                self.assertIn(
                    f"| {name} | — | — | — | — | 0 of 2 | pending |", overview
                )
            self.assertIn("2 of 8 solution–workload runs finished", overview)
            self.assertIn("1 passed; 1 failed or unsupported", overview)
            self.assertIn(
                "| Context | Batch | Beam | " + " | ".join(names) + " | Fastest |",
                report,
            )
            measurements = report.split("<summary>All measurements</summary>", 1)[
                1
            ].split("| Metric |", 1)[0]
            for name in names:
                self.assertEqual(measurements.count(f"| {name} |"), 1)
            for name in names[2:]:
                self.assertIn(f"| {name} | pending | — | — | — | — |", measurements)
            self.assertEqual([r["solution"] for r in data], names[:2])
            self.assertEqual(read(root / "summary.json"), data)
            board = (
                Path(d) / "leaderboards" / f"{next(iter(manifest['scenario_jobs']))}.md"
            ).read_text()
            for name in names[2:]:
                self.assertNotIn(name, board)

            manifest["status"] = "interrupted"
            write(root / "manifest.json", manifest)
            render(root)
            self.assertIn("| 1 of 2 | incomplete |", (root / "report.md").read_text())

    def test_report_before_first_result_has_four_pending_rows(self):
        names = ["sid", "sglang", "beam_search", "beam_search_new"]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "results/partial"
            self.make_run(root, names, finished=False)
            self.assertEqual(render(root), [])
            report = (root / "report.md").read_text()
            overview = report.split("**Reference:**", 1)[0]
            self.assertEqual(overview.count("| 0 of 2 | pending |"), 4)
            self.assertNotIn("0 of 0 cases", overview)

    def test_historical_two_candidate_report_stays_two_candidates(self):
        names = ["sid_gr_inference", "sglang_beam_search"]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "results/partial"
            manifest = self.make_run(root, names)
            manifest["scenario_jobs"] = {next(iter(manifest["scenario_jobs"])): 2}
            manifest["status"] = "completed"
            write(root / "manifest.json", manifest)
            render(root)
            report = (root / "report.md").read_text()
            overview = report.split("**Reference:**", 1)[0]
            self.assertEqual(overview.count("| 1 of 1 | completed |"), 2)
            self.assertNotIn("sglang_beam_search_new", report)
            self.assertNotIn("| sglang |", report)


if __name__ == "__main__":
    unittest.main()
