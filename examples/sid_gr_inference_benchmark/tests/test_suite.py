"""CPU tests for the shared benchmark contract, without importing frameworks."""

import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import worker
from artifacts import digest, read, source_info, suite_hash, write
from config import ROOT, cases, load, read_yaml
from entrypoints.sglang.adapter import normalize
from profiling.timeline import summarize
from runner import e2e, launch, select_cases
from sol import estimate
from validation import Unsupported, check, compare, supported
from workload import build


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.case = next(cases(load(ROOT / "benchmark.yml")))
        self.case.update(batch_size=1, context_length=4, beam_width=2)
        self.model = {"vocab_size": 32, "bos_token_id": 0, "eos_token_id": [1, 2]}
        self.workload = build(self.case, self.model)
        self.outputs = [
            {
                "request_id": "request-0",
                "beams": [
                    {"token_ids": [3, 4, 5], "score": -1.0},
                    {"token_ids": [3, 4, 6], "score": -2.0},
                ],
            }
        ]

    def test_grid_and_selection(self):
        cfg = load(ROOT / "benchmark.yml")
        self.assertEqual(len(list(cases(cfg))), 48)
        self.assertEqual(
            select_cases(cfg, "*/ctx1000_b1_w256", True)[0]["beam_width"], 256
        )
        with self.assertRaises(ValueError):
            select_cases(cfg, "missing/*")

    def test_reject_duplicate_yaml_and_unknown_run_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.yml"
            path.write_text("repeats: 3\nrepeats: 5\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_yaml(path)
            path.write_text((ROOT / "benchmark.yml").read_text() + "\nrepeets: 3\n")
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                load(path)

    def test_shared_inputs_are_reproducible(self):
        self.assertEqual(self.workload, build(self.case, self.model))
        self.assertTrue(all(x not in {0, 1, 2} for x in self.workload[0]["input_ids"]))
        altered = copy.deepcopy(self.case)
        altered["inputs"]["seed"] += 1
        self.assertNotEqual(digest(self.workload), digest(build(altered, self.model)))

    def test_jsonl_length_is_never_silently_changed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "requests.jsonl"
            path.write_text(
                json.dumps({"request_id": "request-0", "input_ids": [4]}) + "\n"
            )
            self.case["inputs"].update(type="jsonl", path=str(path))
            with self.assertRaisesRegex(ValueError, "length/range"):
                build(self.case, self.model)

    def test_reject_invalid_outputs(self):
        check(self.outputs, self.workload, self.case, 32)
        for mutation in [
            lambda b: b.update(score=float("nan")),
            lambda b: b.update(token_ids=[32, 3, 4]),
            lambda b: b.update(token_ids=[3, 4]),
            lambda b: b.update(token_ids=[3, 4, 5]),
            lambda b: b.update(score=0),
        ]:
            bad = copy.deepcopy(self.outputs)
            mutation(bad[0]["beams"][1])
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check(bad, self.workload, self.case, 32)

    def test_agreement_does_not_conflate_validity(self):
        other = copy.deepcopy(self.outputs)
        other[0]["beams"][0]["token_ids"] = [6, 5, 4]
        check(other, self.workload, self.case, 32)
        match = compare(self.outputs, other)[0]
        self.assertFalse(match["top1_exact"])
        self.assertEqual(match["beam_overlap"], 0.5)
        self.assertEqual(match["max_score_difference"], 0)

    def test_constraints_are_unsupported(self):
        self.case["generation"]["constraints"] = "catalog"
        with self.assertRaises(Unsupported):
            supported(self.case)

    def test_sglang_native_batched_output(self):
        beams = [
            {
                "output_ids": b["token_ids"],
                "meta_info": {"sequence_score": b["score"] / 3},
            }
            for b in self.outputs[0]["beams"]
        ]
        raw = [{"meta_info": {"beam_results": beams}}]
        self.assertEqual(normalize(raw, self.workload), self.outputs)
        workload = self.workload + [{**self.workload[0], "request_id": "request-1"}]
        expected = self.outputs + [{**self.outputs[0], "request_id": "request-1"}]
        self.assertEqual(normalize(raw * 2, workload), expected)

    def test_sglang_rejects_non_native_outputs(self):
        for raw in [
            [],
            {},
            [{"output_ids": [3]}],
            [{"beam_results": []}],
            [{"meta_info": {"beam_results": [{"output_ids": [3], "cum_logprob": -1}]}}],
        ]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize(raw, self.workload)

    def test_worker_times_execution_not_setup_or_validation(self):
        events = []
        adapter = SimpleNamespace(
            generate=lambda _: events.append("generate") or self.outputs,
            synchronize=lambda: events.append("sync"),
            close=lambda: events.append("close"),
            provenance=lambda: {},
            profile_start=lambda: events.append("profile_start"),
            profile_stop=lambda: events.append("profile_stop"),
        )
        module = SimpleNamespace(
            check_support=lambda *a: None,
            Adapter=lambda _: events.append("setup") or adapter,
        )
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            set_grad_enabled=lambda _: None,
            manual_seed=lambda _: None,
        )
        torch.cuda.nvtx = SimpleNamespace(
            range_push=lambda _: events.append("push"),
            range_pop=lambda: events.append("pop"),
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "workload.json"
            write(path, self.workload)
            req = {
                "suite_sha256": "same",
                "case": self.case,
                "solution": {"adapter": "fake"},
                "workload_path": str(path),
                "workload_sha256": digest(self.workload),
                "model": {"config": self.model},
                "measurement": {"warmup_runs": 1, "repeats": 2},
            }

            def clock():
                events.append("clock")
                return len(events)

            with patch.dict(sys.modules, {"torch": torch}), patch(
                "artifacts.suite_hash", return_value="same"
            ), patch("worker.environment", return_value={}), patch(
                "worker.check", side_effect=lambda *a: events.append("validate")
            ), patch(
                "worker.time.perf_counter", side_effect=clock
            ), patch(
                "worker.importlib.import_module", return_value=module
            ):
                result = worker.run(req)
                plain_events = list(events)
                events.clear()
                worker.run(req, profile=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            plain_events,
            [
                "clock",
                "setup",
                "clock",
                "generate",
                "sync",
                "validate",
                "sync",
                "clock",
                "generate",
                "sync",
                "clock",
                "validate",
                "sync",
                "clock",
                "generate",
                "sync",
                "clock",
                "validate",
                "close",
            ],
        )
        self.assertEqual(
            events,
            [
                "clock",
                "setup",
                "clock",
                "generate",
                "sync",
                "validate",
                "profile_start",
                "sync",
                "push",
                "clock",
                "generate",
                "sync",
                "clock",
                "pop",
                "validate",
                "sync",
                "push",
                "clock",
                "generate",
                "sync",
                "clock",
                "pop",
                "validate",
                "profile_stop",
                "close",
            ],
        )

    def test_missing_worker_result_and_timeout_are_failures(self):
        req = {
            "case": {"id": "test"},
            "solution": {"name": "fake"},
            "workload_sha256": "abc",
            "execution": {"timeout_seconds": 1},
        }
        for process, expected in [
            ({"returncode": 0, "timed_out": False}, "failed"),
            ({"returncode": -15, "timed_out": True}, "timed_out"),
        ]:
            with tempfile.TemporaryDirectory() as d, patch(
                "runtime.command", return_value=(["fake"], {}, None)
            ), patch("runtime.execute", return_value=process):
                self.assertEqual(launch(req, d)["status"], expected)
                self.assertEqual(read(Path(d) / "result.json")["status"], expected)

    def test_docker_uses_gpu_uuid_across_host_ordinal_remapping(self):
        with tempfile.TemporaryDirectory() as d:
            request = {
                "run_dir": d,
                "model": {"path": d},
                "execution": {"gpu": 0},
                "solution": {
                    "name": "test",
                    "source": {},
                    "runtime": {
                        "kind": "docker",
                        "python": "python3",
                        "image": "test-image",
                    },
                },
            }
            with patch(
                "runtime.gpu_info", return_value={"uuid": "GPU-assigned"}
            ) as probe:
                cmd, _, _ = runtime.command(
                    Path(d) / "request.json", Path(d) / "result.json", request
                )
            probe.assert_called_once_with(0)
            self.assertEqual(cmd[cmd.index("--gpus") + 1], "device=GPU-assigned")
            with patch("runtime.gpu_info", return_value={"uuid": None}):
                with self.assertRaisesRegex(ValueError, "Cannot resolve GPU"):
                    runtime.command(
                        Path(d) / "request.json", Path(d) / "result.json", request
                    )

    def test_untracked_nested_sources_change_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subprocess.run(["git", "init", "-q", d], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    d,
                    "-c",
                    "user.name=test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "--allow-empty",
                    "-qm",
                    "test",
                ],
                check=True,
            )
            nested = root / "nested"
            nested.mkdir()
            (nested / "kernel.py").write_text("first")
            before = source_info(nested)
            (nested / "kernel.py").write_text("second")
            self.assertNotEqual(before, source_info(nested))

    def test_suite_identity_ignores_artifacts_but_tracks_adapters(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "worker.py").write_text("# worker")
            adapter = root / "entrypoints" / "example" / "adapter.py"
            adapter.parent.mkdir(parents=True)
            adapter.write_text("# first")
            before = suite_hash(root)
            generated = root / "custom-results" / "cache" / "compiled.py"
            generated.parent.mkdir(parents=True)
            generated.write_text("# generated code")
            self.assertEqual(before, suite_hash(root))
            adapter.write_text("# changed")
            self.assertNotEqual(before, suite_hash(root))

    def test_sol_requires_sourced_matching_hardware(self):
        self.assertEqual(
            estimate(self.case, self.model, None, "GPU", 20)["status"], "unavailable"
        )
        model = {
            **self.model,
            "model_type": "qwen3",
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
        }
        hardware = {"gpu_name": "GPU", "dense_bf16_tflops": 100, "source": "fixture"}
        bound = estimate(self.case, model, hardware, "GPU", 20)
        self.assertEqual(bound["status"], "partial")
        self.assertAlmostEqual(
            bound["gap_factor"] * bound["bound_over_measured_pct"], 100
        )
        self.assertEqual(
            estimate(self.case, model, hardware, "Other", 20)["status"], "unavailable"
        )

    def test_timeline_uses_only_measured_ranges_and_union(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "capture.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript(
                    """
                    CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER);
                    CREATE TABLE StringIds(id INTEGER, value TEXT);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, demangledName INTEGER);
                    INSERT INTO StringIds VALUES (1, 'test_kernel');
                    INSERT INTO NVTX_EVENTS VALUES (1000000, 5000000, 'sid_bench/measured', NULL);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 500000, 1);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000000, 3000000, 1);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2000000, 4000000, 1);
                """
                )
            result = summarize(path)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["gpu_active_union_ms"], 3)
            self.assertEqual(result["kernels"][0]["calls"], 2)
            self.assertEqual(result["kernels"][0]["gpu_ms"], 4)
            with sqlite3.connect(path) as db:
                db.executescript(
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER, end INTEGER); INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (1000000, 3000000);"
                )
            self.assertEqual(summarize(path)["status"], "unavailable")

    def test_sid_gr_without_fused_kernel_supports_default_but_rejects_optin(self):
        from entrypoints.sid_gr_inference.adapter import check_support

        with tempfile.TemporaryDirectory() as d:
            solution = {"source": {"path": d}, "env": {}}
            check_support(self.case, solution)
            for value in ["1", "true", "yes", "on"]:
                solution["env"]["GR_INFERENCE_FUSED_BEAM_LOGSUMEXP"] = value
                with self.subTest(value=value), self.assertRaisesRegex(
                    Unsupported, "optional fused beam normalizer"
                ):
                    check_support(self.case, solution)

    def test_timeline_capture_failures_are_not_successful_results(self):
        from profiling.capture import main

        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "result.json"
            argv = [
                "capture",
                "--request",
                str(Path(d) / "request.json"),
                "--output",
                str(output),
            ]
            # Worker inference can succeed even if exporting the trace fails.
            write(output, {"status": "success", "samples_ms": [1.0]})
            with patch.object(sys, "argv", argv), patch(
                "profiling.capture.shutil.which", return_value="nsys"
            ), patch(
                "profiling.capture.subprocess.run",
                side_effect=[
                    None,
                    subprocess.CalledProcessError(1, ["nsys", "export"]),
                ],
            ):
                self.assertEqual(main(), 1)
            self.assertEqual(read(output)["status"], "failed")
            with patch.object(sys, "argv", argv), patch(
                "profiling.capture.shutil.which", return_value=None
            ):
                self.assertEqual(main(), 0)
            self.assertEqual(read(output)["status"], "unsupported")

    def test_e2e_records_unsupported_and_failed_workers_without_gpu(self):
        cfg = load(ROOT / "benchmark.yml")
        cfg["solutions"] = ["sid_gr_inference", "broken"]
        cfg["solution_configs"]["broken"] = copy.deepcopy(
            cfg["solution_configs"]["sid_gr_inference"]
        )
        cfg["solution_configs"]["broken"].update(
            name="broken", adapter="missing_benchmark_adapter"
        )
        for solution in cfg["solution_configs"].values():
            solution["runtime"] = {"kind": "local", "python": sys.executable}
        cfg["execution"]["continue_on_error"] = True
        self.case["generation"]["constraints"] = "catalog"
        with tempfile.TemporaryDirectory() as d:
            cfg["output_dir"] = d
            self.case["model"]["path"] = d
            with patch(
                "runner.model_info",
                return_value={"path": d, "sha256": "fixture", "config": self.model},
            ), patch("runtime.gpu_info", return_value={"name": None, "uuid": None}):
                directory, manifest = e2e(cfg, [self.case])
            self.assertEqual(
                [j["status"] for j in manifest["jobs"]], ["unsupported", "failed"]
            )
            self.assertEqual(manifest["status"], "completed")
            for job in manifest["jobs"]:
                job_dir = directory / job["directory"]
                self.assertTrue((job_dir / "request.json").is_file())
                self.assertTrue((job_dir / "worker.log").is_file())
                self.assertEqual(read(job_dir / "result.json")["status"], job["status"])
            self.assertEqual(
                [r["status"] for r in read(directory / "summary.json")],
                ["unsupported", "failed"],
            )
            self.assertTrue((directory / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
