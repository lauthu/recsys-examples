"""CPU contracts for native beam calls, exact token inputs and precision guards."""

import copy
import importlib.metadata
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entrypoints import beam_common
from entrypoints.flash_rec import adapter as flash_rec
from entrypoints.tensorrt_llm import adapter as tensorrt_llm
from entrypoints.vllm_gr import adapter as vllm_gr
from harness.config import ROOT, cases, load
from harness.validation import Unsupported, check


class AdditionalAdaptersTests(unittest.TestCase):
    def setUp(self):
        self.case = next(cases(load(ROOT / "config/runs/default.yml")))
        self.case.update(beam_width=2, batch_size=2)
        self.workload = [
            {"request_id": "second", "input_ids": [9, 8, 7]},
            {"request_id": "first", "input_ids": [6, 5, 4]},
        ]
        self.expected = [
            {
                "request_id": row["request_id"],
                "beams": [
                    {"token_ids": [1, 2, 3], "score": -1.25},
                    {"token_ids": [1, 2, 4], "score": -2.5},
                ],
            }
            for row in self.workload
        ]

    def request(self, name, engine=None):
        return {
            "case": self.case,
            "model": {"path": "/model"},
            "solution": {
                "name": name,
                "source": {},
                "engine": engine or {},
            },
        }

    def raw(self, sequence_key, token_key, score_key):
        return [
            NS(
                **{
                    sequence_key: [
                        NS(
                            **{
                                token_key: tuple(beam["token_ids"]),
                                score_key: beam["score"],
                            }
                        )
                        for beam in row["beams"]
                    ]
                }
            )
            for row in self.expected
        ]

    def test_modules_and_seven_engine_plan_do_not_import_frameworks(self):
        code = """
import sys
from entrypoints.vllm_gr import adapter
from entrypoints.tensorrt_llm import adapter
from entrypoints.flash_rec import adapter
from harness.config import ROOT, cases, load
cfg = load(ROOT / 'config/runs/seven_engine.yml')
assert len(cfg['solutions']) == 7
assert len(list(cases(cfg))) == 48
assert not {'torch', 'vllm', 'vllm_gr', 'tensorrt_llm', 'flashrec'} & set(sys.modules)
"""
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

    def test_common_semantics_cannot_be_overridden_by_engine_options(self):
        for name, module in [
            ("vllm_gr", vllm_gr),
            ("tensorrt_llm", tensorrt_llm),
            ("flash_rec", flash_rec),
        ]:
            for key, value in [
                ("dtype", "float16"),
                ("quantization", "fp8"),
                ("catalog_path", "/catalog"),
                ("temperature", 0.5),
            ]:
                with self.subTest(name=name, key=key), self.assertRaises(Unsupported):
                    module.check_support(
                        self.case, self.request(name, {key: value})["solution"]
                    )
            for section, key, value in [
                ("model", "kv_cache_dtype", "fp8"),
                ("model", "dtype", "float16"),
                ("generation", "constraints", "catalog"),
                ("generation", "ignore_eos", False),
            ]:
                case = copy.deepcopy(self.case)
                case[section][key] = value
                with self.subTest(name=name, key=key), self.assertRaises(Unsupported):
                    module.check_support(case, self.request(name)["solution"])

    def test_invalid_options_and_backend_limits_are_explicit(self):
        for name, module in [
            ("vllm_gr", vllm_gr),
            ("tensorrt_llm", tensorrt_llm),
            ("flash_rec", flash_rec),
        ]:
            with self.subTest(name=name), self.assertRaisesRegex(
                Unsupported, "boolean"
            ):
                module.check_support(
                    self.case, self.request(name, {"prefix_cache": "false"})["solution"]
                )
        with self.assertRaisesRegex(Unsupported, "prompt KV"):
            vllm_gr.check_support(
                self.case, self.request("vllm_gr", {"prefix_cache": False})["solution"]
            )
        self.case["beam_width"] = 512
        with self.assertRaisesRegex(Unsupported, "256"):
            vllm_gr.check_support(self.case, self.request("vllm_gr")["solution"])
        with patch(
            "importlib.metadata.version", return_value="0.30.0"
        ), self.assertRaisesRegex(Unsupported, "0.22.1"):
            beam_common.require_version("vllm", "0.22.1")
        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ), self.assertRaisesRegex(Unsupported, "Install"):
            beam_common.require_version("vllm", "0.22.1")

    def test_batched_beams_are_rejected_when_native_scores_are_unavailable(self):
        for name, module in [("vllm_gr", vllm_gr), ("tensorrt_llm", tensorrt_llm)]:
            with self.subTest(name=name), self.assertRaisesRegex(
                Unsupported, "batch_size=1"
            ):
                module.check_support(self.case, self.request(name)["solution"])

    def test_vllm_gr_uses_compact_beams_without_prompt_or_boundary_tokens(self):
        self.case["batch_size"] = 1
        self.workload = self.workload[:1]
        self.expected = self.expected[:1]
        raw = self.raw("sequences", "token_ids", "cumulative_logprob")
        factory = Mock()
        factory.beam_search = lambda self, prompts, params, *, output_mode: None
        engine = factory.return_value
        engine.beam_search.return_value = raw
        modules = {
            "vllm_gr": NS(__file__="/test/vllm_gr/__init__.py"),
            "vllm_gr.entrypoints.gr": NS(GRLLM=factory),
            "vllm_gr.sampling_params": NS(BeamSearchParams=NS),
        }
        with patch.dict(sys.modules, modules), patch.object(
            beam_common, "require_version", return_value="0.22.1"
        ):
            adapter = vllm_gr.Adapter(self.request("vllm_gr"))
            output = adapter.generate(self.workload)
        opts = factory.call_args.kwargs
        self.assertEqual(opts["dtype"], "bfloat16")
        self.assertIsNone(opts["quantization"])
        self.assertGreaterEqual(opts["max_num_batched_tokens"], opts["max_model_len"])
        self.assertGreaterEqual(
            opts["max_model_len"], self.case["context_length"] + self.case["beam_width"]
        )
        self.assertEqual(opts["max_model_len"] % 128, 0)
        self.assertEqual(opts["vllm_gr_config"]["beam"]["execution_mode"], "legacy")
        self.assertFalse(opts["vllm_gr_config"]["constraint_table"]["enabled"])
        self.assertEqual(opts["vllm_gr_config"]["constraint_table"]["max_top_k"], 2)
        prompts, params = engine.beam_search.call_args.args
        self.assertEqual(
            prompts, [{"prompt_token_ids": r["input_ids"]} for r in self.workload]
        )
        self.assertEqual(engine.beam_search.call_args.kwargs["output_mode"], "compact")
        self.assertTrue(params.ignore_eos)
        self.assertEqual(
            (params.beam_width, params.max_tokens, params.length_penalty), (2, 3, 0)
        )
        self.assertIsNone(params.begin_token)
        self.assertIsNone(params.end_token)
        self.assertEqual(output, self.expected)
        adapter.close()
        engine.shutdown.assert_called_once()

    def test_tensorrt_uses_native_beams_and_requests_cumulative_scores(self):
        self.case["batch_size"] = 1
        self.workload = self.workload[:1]
        self.expected = self.expected[:1]
        factory = Mock()
        factory.return_value.generate.return_value = self.raw(
            "outputs", "token_ids", "cumulative_logprob"
        )
        modules = {
            "tensorrt_llm": NS(
                __file__="/test/tensorrt_llm/__init__.py",
                LLM=factory,
                SamplingParams=NS,
            ),
            "tensorrt_llm.llmapi": NS(
                CudaGraphConfig=NS, KvCacheConfig=NS, QuantConfig=NS
            ),
        }
        with patch.dict(sys.modules, modules), patch.object(
            beam_common, "require_version", return_value="1.2.1"
        ):
            adapter = tensorrt_llm.Adapter(self.request("tensorrt_llm"))
            output = adapter.generate(self.workload)
        opts = factory.call_args.kwargs
        self.assertEqual(
            (opts["backend"], opts["dtype"], opts["max_beam_width"]),
            ("pytorch", "bfloat16", 2),
        )
        self.assertNotIn("quant_config", opts)
        self.assertFalse(opts["force_dynamic_quantization"])
        self.assertEqual(opts["kv_cache_config"].dtype, "auto")
        self.assertEqual(
            factory.return_value.generate.call_args.args[0],
            [r["input_ids"] for r in self.workload],
        )
        params = adapter.sampling_params
        self.assertTrue(params.use_beam_search)
        self.assertEqual((params.n, params.best_of, params.logprobs), (2, 2, 1))
        self.assertTrue(params.ignore_eos)
        self.assertFalse(params.detokenize)
        self.assertTrue(params.exclude_input_from_output)
        self.assertEqual(output, self.expected)

    def test_tensorrt_preserves_native_capacity_failures(self):
        self.case["batch_size"] = 1
        error = RuntimeError("Executor worker returned error")
        error.__cause__ = AssertionError(
            "The max KV cache length of input sequences (5002) exceeds the KV cache manager's maximum supported length (2272)."
        )
        factory = Mock(side_effect=error)
        modules = {
            "tensorrt_llm": NS(
                __file__="/test/tensorrt_llm/__init__.py",
                LLM=factory,
                SamplingParams=NS,
            ),
            "tensorrt_llm.llmapi": NS(CudaGraphConfig=NS, KvCacheConfig=NS),
        }
        with patch.dict(sys.modules, modules), patch.object(
            beam_common, "require_version", return_value="1.2.1"
        ):
            with self.assertRaisesRegex(
                Unsupported, "maximum supported length \\(2272\\)"
            ):
                tensorrt_llm.Adapter(self.request("tensorrt_llm"))
            factory.side_effect = RuntimeError("unrelated worker failure")
            with self.assertRaisesRegex(RuntimeError, "unrelated worker failure"):
                tensorrt_llm.Adapter(self.request("tensorrt_llm"))

    def test_flashrec_disables_fp8_and_catalog_defaults(self):
        engine = Mock()
        engine.runner.compute_dtype = engine.runner.kv_pool.dtype = "BF16"
        engine.boundary_ids = None
        engine.generate_many.return_value = self.raw(
            "sequences", "tokens", "cum_logprob"
        )
        factory = Mock(return_value=engine)
        torch = NS(bfloat16="BF16", cuda=NS(synchronize=Mock()))
        modules = {
            "torch": torch,
            "flashrec": NS(
                __file__="/test/flashrec/__init__.py",
                BeamRecConfig=NS,
                BeamRecEngine=factory,
            ),
            "flashrec.core": NS(BeamRequest=NS),
        }
        with patch.dict(sys.modules, modules), patch.object(
            flash_rec, "asdict", side_effect=vars
        ):
            # Upstream's pool reserves row zero, then temporarily holds both
            # prefill rows and all expanded beams. Exercise the batch-8 boundary.
            large = copy.deepcopy(self.request("flash_rec"))
            large["case"].update(batch_size=8, beam_width=128)
            flash_rec.Adapter(large)
            large_config = factory.call_args.args[0]
            capacity = max(large_config.max_running_requests * 128 + 8, 1024) - 1
            self.assertGreaterEqual(capacity, 8 * (128 + 1))
            self.assertEqual(large_config.max_batch_requests, 8)
            adapter = flash_rec.Adapter(self.request("flash_rec"))
            output = adapter.generate(self.workload)
        config = factory.call_args.args[0]
        self.assertIsNone(config.quantization)
        self.assertIsNone(config.sid_vocab_file)
        self.assertIsNone(config.sid_boundary_tokens)
        self.assertEqual(config.kv_cache_dtype, "bfloat16")
        self.assertFalse(config.enable_restricted_lm_head)
        self.assertFalse(config.enable_per_codebook_lm_head)
        self.assertFalse(config.enable_fused_rms_fp8)
        self.assertFalse(config.enable_fused_silu_fp8)
        self.assertFalse(config.enable_warmup)
        requests = engine.generate_many.call_args.args[0]
        self.assertEqual(
            [r.input_ids for r in requests], [r["input_ids"] for r in self.workload]
        )
        self.assertEqual(len({r.rid for r in requests}), len(requests))
        repeated_output = adapter.generate(self.workload)
        repeated_requests = engine.generate_many.call_args.args[0]
        self.assertFalse({r.rid for r in requests} & {r.rid for r in repeated_requests})
        self.assertEqual(repeated_output, self.expected)
        self.assertTrue(
            all(
                r.ignore_eos and r.temperature == 0 and r.length_penalty == 0
                for r in requests
            )
        )
        self.assertEqual(output, self.expected)
        adapter.synchronize()
        engine.flush_host.assert_called_once()
        adapter.close()
        engine.host_pool.shutdown.assert_called_once()

    def test_output_conversion_does_not_hide_missing_or_invalid_beams(self):
        kwargs = dict(
            sequences="outputs", tokens="token_ids", score="cumulative_logprob"
        )
        raw = self.raw("outputs", "token_ids", "cumulative_logprob")
        with self.assertRaisesRegex(ValueError, "one completed"):
            beam_common.normalize(raw[:1], self.workload, **kwargs)
        raw[0].outputs[0].cumulative_logprob = None
        with self.assertRaisesRegex(ValueError, "no cumulative"):
            beam_common.normalize(raw, self.workload, **kwargs)
        for mutate in [
            lambda beams: beams.pop(),
            lambda beams: setattr(beams[0], "token_ids", [9, 8, 7, 1, 2, 3]),
            lambda beams: setattr(beams[1], "token_ids", beams[0].token_ids),
            lambda beams: setattr(beams[0], "cumulative_logprob", float("nan")),
            lambda beams: beams.reverse(),
        ]:
            raw = self.raw("outputs", "token_ids", "cumulative_logprob")
            mutate(raw[0].outputs)
            output = beam_common.normalize(raw, self.workload, **kwargs)
            with self.assertRaises(ValueError):
                check(output, self.workload, self.case, 100)


if __name__ == "__main__":
    unittest.main()
