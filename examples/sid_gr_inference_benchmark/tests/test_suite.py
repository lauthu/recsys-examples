"""CPU tests for the shared benchmark contract, without importing frameworks."""

import copy
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entrypoints.sglang.adapter import normalize
from harness import runtime, worker
from harness.artifacts import digest, read, source_info, suite_hash, write
from harness.config import ROOT, cases, load, read_yaml
from harness.runner import e2e, launch, select_cases
from profiling.timeline import summarize
from reporting.sol import estimate
from scenarios.validation import Unsupported, check, compare, supported
from scenarios.workload import build


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.case = next(cases(load(ROOT / "run.template.yml")))
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
        cfg = load(ROOT / "run.template.yml")
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
            path.write_text((ROOT / "run.template.yml").read_text() + "\nrepeets: 3\n")
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
                "harness.artifacts.suite_hash", return_value="same"
            ), patch("harness.worker.environment", return_value={}), patch(
                "harness.worker.check", side_effect=lambda *a: events.append("validate")
            ), patch(
                "harness.worker.time.perf_counter", side_effect=clock
            ), patch(
                "harness.worker.importlib.import_module", return_value=module
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
                "harness.runtime.command", return_value=(["fake"], {}, None)
            ), patch("harness.runtime.execute", return_value=process):
                self.assertEqual(launch(req, d)["status"], expected)
                self.assertEqual(read(Path(d) / "result.json")["status"], expected)

    def test_gpu_uuid_is_used_for_docker_and_local_despite_inherited_remapping(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "3,1"}
        ):
            request = {
                "run_dir": d,
                "model": {"path": d},
                "execution": {"gpu": "GPU-assigned"},
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
            cmd, _, _ = runtime.command(
                Path(d) / "request.json", Path(d) / "result.json", request
            )
            self.assertEqual(cmd[cmd.index("--gpus") + 1], "device=GPU-assigned")
            self.assertEqual(
                cmd[cmd.index("--cpuset-cpus") + 1],
                ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))),
            )
            self.assertTrue("CUDA_VISIBLE_DEVICES=GPU-assigned" in cmd)
            request["solution"]["runtime"]["kind"] = "local"
            cmd, env, _ = runtime.command(
                Path(d) / "request.json", Path(d) / "result.json", request
            )
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-assigned")
            self.assertTrue(env["USER"])
            self.assertEqual(
                env["TVM_FFI_CACHE_DIR"], str(Path(d) / "cache/test/tvm_ffi")
            )
            self.assertEqual(cmd[1:3], ["-m", "harness.worker"])
        for gpu in [0, "0", "", "3,1"]:
            with self.subTest(gpu=gpu), self.assertRaisesRegex(
                ValueError, "full GPU UUID"
            ):
                runtime.gpu_info(gpu)
        with patch(
            "harness.runtime.subprocess.check_output",
            return_value="GPU name, GPU-assigned, 123, 96000",
        ) as probe:
            self.assertEqual(runtime.gpu_info("GPU-assigned")["uuid"], "GPU-assigned")
            self.assertTrue("--id=GPU-assigned" in probe.call_args.args[0])

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
            estimate(
                self.case,
                self.model,
                {"gpu_name": "GPU", "source": "fixture"},
                "GPU",
                20,
            )["status"],
            "unavailable",
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

    def test_sglang_missing_score_api_is_unsupported(self):
        from entrypoints.sglang.adapter import Adapter

        class BeamGroupWithoutPenalty:
            def __init__(self, width):
                pass

        modules = {
            "sglang": SimpleNamespace(),
            "sglang.srt.sampling.sampling_params": SimpleNamespace(
                SamplingParams=SimpleNamespace(__struct_fields__=("beam_width",))
            ),
            "sglang.srt.beam_search.beam_group": SimpleNamespace(
                BeamGroup=BeamGroupWithoutPenalty
            ),
        }
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(
            Unsupported, "length normalization"
        ):
            Adapter({"case": self.case, "solution": {"source": {}}})

    def test_scenario_owns_measurement_and_smoke_is_an_explicit_override(self):
        cfg = load(ROOT / "run.template.yml")
        self.assertNotIn("measurement", cfg)
        self.assertEqual(self.case["measurement"], {"warmup_runs": 5, "repeats": 30})
        smoke = select_cases(cfg, smoke=True)[0]
        self.assertEqual(smoke["measurement"], {"warmup_runs": 2, "repeats": 3})
        self.assertEqual(next(cases(cfg))["measurement"]["repeats"], 30)

    def test_hardware_is_required_and_sku_mismatch_prevents_launch(self):
        template = (ROOT / "run.template.yml").read_text()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run.yml"
            path.write_text(
                template.replace(
                    "hardware: rtx_pro_6000_blackwell_server", "hardware: null"
                )
            )
            with self.assertRaisesRegex(ValueError, "hardware must name a profile"):
                load(path)
            path.write_text(
                template.replace("hardware: rtx_pro_6000_blackwell_server", "")
            )
            with self.assertRaisesRegex(ValueError, "missing keys.*hardware"):
                load(path)
            cfg = load(ROOT / "run.template.yml")
            cfg["output_dir"] = d
            with patch(
                "harness.runtime.gpu_info", return_value={"name": "Other GPU"}
            ), patch("harness.runner.launch") as launch_worker:
                with self.assertRaisesRegex(
                    ValueError, "does not match hardware profile"
                ):
                    e2e(cfg, [self.case])
                launch_worker.assert_not_called()

    def test_scenario_leaderboards_track_latest_results_without_rolling_back(self):
        from reporting.report import render

        cfg = load(ROOT / "run.template.yml")
        baseline = cfg["baseline"]
        changed = copy.deepcopy(self.outputs)
        changed[0]["beams"][1]["token_ids"] = [8, 9, 10]
        other_case = copy.deepcopy(self.case)
        other_case.update(name="other_scenario", id="other_scenario/test")
        hardware = {"name": cfg["hardware_config"]["gpu_name"], "uuid": "GPU-assigned"}
        cfg["scenario_configs"]["other_scenario"] = other_case
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run-one"
            manifest = {
                "schema_version": 2,
                "run_id": "run-one",
                "mode": "e2e",
                "status": "completed",
                "config": cfg,
                "hardware": hardware,
                "scenario_jobs": {self.case["name"]: 3, other_case["name"]: 1},
                "cases": [
                    {"case": c, "model": {"config": self.model, "sha256": "checkpoint"}}
                    for c in [self.case, other_case]
                ],
                "jobs": [],
            }
            for case, name, outputs, status in [
                (self.case, baseline, self.outputs, "success"),
                (self.case, "candidate", changed, "success"),
                (self.case, "broken", [], "failed"),
                (other_case, baseline, self.outputs, "success"),
            ]:
                directory = f"jobs/{name}/{case['id']}"
                manifest["jobs"].append(
                    {
                        "case_id": case["id"],
                        "solution": name,
                        "directory": directory,
                        "status": status,
                    }
                )
                write(root / directory / "request.json", {})
                write(
                    root / directory / "result.json",
                    {
                        "status": status,
                        "samples_ms": [10, 10],
                        "outputs": [outputs],
                        "workload_sha256": "same",
                        "environment": {"gpu": hardware["name"], "packages": {}},
                    },
                )
            partial = {**manifest, "status": "running", "jobs": manifest["jobs"][:1]}
            write(root / "manifest.json", partial)
            render(root)
            board = (
                root.parent / "leaderboards" / f"{self.case['name']}.md"
            ).read_text()
            self.assertTrue("sid_gr_inference" in board)
            self.assertTrue("running" in board)
            write(root / "manifest.json", manifest)
            records = render(root)
            candidate = next(r for r in records if r["solution"] == "candidate")
            self.assertEqual(candidate["median_ms"], 10)
            self.assertEqual(candidate["requests_per_second"], 100)
            self.assertEqual(candidate["parity"], "different")
            self.assertEqual(candidate["agreement"]["min_overlap"], 0.5)
            self.assertEqual(records[0]["parity"], "exact")
            board = (
                root.parent / "leaderboards" / f"{self.case['name']}.md"
            ).read_text()
            self.assertTrue("run-one" in board)
            self.assertTrue("candidate" in board)
            self.assertTrue("failed" in board)
            self.assertNotIn("other_scenario", board)
            self.assertTrue(
                (root.parent / "leaderboards" / "other_scenario.md").is_file()
            )
            board_path = root.parent / "leaderboards" / f"{self.case['name']}.md"

            # A partial rerun replaces only its completed row, including failure.
            import shutil

            newer = root.parent / "run-two"
            shutil.copytree(root, newer)
            rerun = copy.deepcopy(manifest)
            rerun.update(run_id="run-two", jobs=manifest["jobs"][1:2])
            write(newer / "manifest.json", rerun)
            write(
                newer / rerun["jobs"][0]["directory"] / "result.json",
                {"status": "failed"},
            )
            render(newer)
            board = board_path.read_text()
            self.assertTrue("run-two" in board)
            self.assertTrue("| candidate | failed |" in board)
            self.assertTrue("| sid_gr_inference | success |" in board)
            render(root)
            self.assertEqual(board, board_path.read_text())

            # Smoke executions cannot replace measured rankings.
            smoke = root.parent / "run-three"
            shutil.copytree(root, smoke)
            write(
                smoke / "manifest.json",
                {**manifest, "run_id": "run-three", "mode": "smoke"},
            )
            render(smoke)
            self.assertEqual(board, board_path.read_text())

            # CPU changes and scenario/checkpoint revisions get separate sections.
            third = root.parent / "run-z"
            shutil.copytree(root, third)
            different = copy.deepcopy(manifest)
            different.update(run_id="run-z")
            different["hardware"]["cpu"] = {
                "model": "Other CPU",
                "sockets": 1,
                "physical_cores": 8,
                "logical_cpus": 16,
                "affinity": list(range(16)),
            }
            write(third / "manifest.json", different)
            render(third)
            self.assertEqual(board_path.read_text().count("\n## "), 2)
            different["hardware"] = hardware
            different["cases"][0]["model"]["sha256"] = "new-checkpoint"
            write(third / "manifest.json", different)
            render(third)
            self.assertEqual(board_path.read_text().count("\n## "), 2)

    def test_cpu_topology_and_process_affinity_are_recorded(self):
        fields = {
            "Model name:": "Test CPU",
            "Socket(s):": "2",
            "Core(s) per socket:": "4",
            "CPU(s):": "16",
        }
        probe = json.dumps(
            {"lscpu": [{"field": k, "data": v} for k, v in fields.items()]}
        )
        with patch(
            "harness.runtime.subprocess.check_output", return_value=probe
        ), patch("harness.runtime.os.sched_getaffinity", return_value={2, 3}):
            cpu = runtime.cpu_info()
        self.assertEqual(
            cpu,
            {
                "model": "Test CPU",
                "sockets": 2,
                "physical_cores": 8,
                "logical_cpus": 16,
                "affinity": [2, 3],
            },
        )

    def test_readme_sglang_candidate_uses_global_beam_search_not_sampling(self):
        from entrypoints.sglang.beam_search import Adapter

        engine = Mock()
        factory = Mock(return_value=engine)
        modules = {
            "sglang": SimpleNamespace(Engine=factory, __file__="sglang/__init__.py"),
            "sglang.srt.server_args": SimpleNamespace(
                ServerArgs=SimpleNamespace(enable_beam_search=False)
            ),
        }
        beams = [
            {
                "output_ids": b["token_ids"],
                "meta_info": {"sequence_score": b["score"] / 3},
            }
            for b in self.outputs[0]["beams"]
        ]
        engine.generate.return_value = [{"meta_info": {"beam_results": beams}}]
        with patch.dict(sys.modules, modules):
            adapter = Adapter(
                {
                    "case": self.case,
                    "model": {"path": "model"},
                    "solution": {"source": {}, "engine": {}},
                }
            )
            self.assertEqual(adapter.generate(self.workload), self.outputs)
        self.assertTrue(factory.call_args.kwargs["enable_beam_search"])
        self.assertEqual(
            engine.generate.call_args.kwargs["sampling_params"],
            {
                "n": 2,
                "max_new_tokens": 3,
                "ignore_eos": True,
            },
        )
        cfg = load(ROOT / "run.readme.yml")
        self.assertEqual(len(select_cases(cfg, "qwen3_1_7B/*_w256")), 8)
        self.assertEqual(
            cfg["solution_configs"]["sglang_beam_search"]["source"]["expected_commit"],
            "af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d",
        )

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
        cfg = load(ROOT / "run.template.yml")
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
        cfg["execution"]["gpu"] = "GPU-assigned"
        self.case["generation"]["constraints"] = "catalog"
        with tempfile.TemporaryDirectory() as d:
            cfg["output_dir"] = d
            self.case["model"]["path"] = d
            with patch(
                "harness.runner.model_info",
                return_value={"path": d, "sha256": "fixture", "config": self.model},
            ), patch(
                "harness.runtime.gpu_info",
                return_value={
                    "name": cfg["hardware_config"]["gpu_name"],
                    "uuid": "GPU-assigned",
                },
            ):
                directory, manifest = e2e(cfg, [self.case])
                original = (directory / "report.md").read_bytes()
                second, _ = e2e(cfg, [self.case])
                self.assertNotEqual(directory, second)
                self.assertEqual((directory / "report.md").read_bytes(), original)
                self.assertTrue(
                    (
                        second.parent / "leaderboards" / f"{self.case['name']}.md"
                    ).is_file()
                )
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
            board = (
                directory.parent / "leaderboards" / f"{self.case['name']}.md"
            ).read_text()
            self.assertTrue(second.name in board)
            self.assertTrue("unsupported" in board)
            self.assertTrue("failed" in board)


if __name__ == "__main__":
    unittest.main()
