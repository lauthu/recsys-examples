"""The updated branch must request beam search, never ordinary n-way sampling."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entrypoints.sglang_beam_search_new import adapter as beam_search_new
from harness.config import ROOT, cases, load
from harness.validation import Unsupported


class UpdatedBeamTests(unittest.TestCase):
    def setUp(self):
        self.case = next(cases(load(ROOT / "config/runs/default.yml")))
        self.solution = {
            "source": {},
            "engine": {
                "prefix_cache": True,
                "enable_beam_cascade_attention": True,
                "decode_attention_backend": "flashinfer",
            },
        }

    def test_request_explicitly_activates_beam_search(self):
        engine = Mock()
        modules = {
            "sglang": SimpleNamespace(
                __file__="/test/sglang/__init__.py", Engine=engine
            ),
            "sglang.srt.server_args": SimpleNamespace(
                ServerArgs=SimpleNamespace(enable_beam_search=False)
            ),
            "sglang.srt.sampling.sampling_params": SimpleNamespace(
                SamplingParams=SimpleNamespace(__struct_fields__=("use_beam_search",))
            ),
        }
        request = {
            "case": self.case,
            "model": {"path": "/model"},
            "solution": self.solution,
        }
        with patch.dict(sys.modules, modules):
            adapter = beam_search_new.Adapter(request)
        self.assertIs(engine.call_args.kwargs["enable_beam_search"], True)
        self.assertIs(engine.call_args.kwargs["enable_beam_cascade_attention"], True)
        self.assertIs(engine.call_args.kwargs["disable_radix_cache"], False)
        self.assertEqual(
            engine.call_args.kwargs["decode_attention_backend"], "flashinfer"
        )
        self.assertIs(adapter.sampling_params["use_beam_search"], True)
        self.assertEqual(adapter.sampling_params["n"], self.case["beam_width"])
        self.assertEqual(
            adapter.sampling_params["max_new_tokens"],
            self.case["generation"]["output_tokens"],
        )
        self.assertIs(adapter.sampling_params["ignore_eos"], True)

    def test_missing_new_request_api_is_unsupported(self):
        adapter = object.__new__(beam_search_new.Adapter)
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.sampling.sampling_params": SimpleNamespace(
                    SamplingParams=SimpleNamespace(__struct_fields__=())
                )
            },
        ), self.assertRaisesRegex(Unsupported, "pinned"):
            adapter.beam_api()

    def test_cascade_requires_flashinfer_but_can_be_disabled(self):
        beam_search_new.check_support(self.case, self.solution)
        solution = copy.deepcopy(self.solution)
        solution["engine"]["decode_attention_backend"] = "triton"
        with self.assertRaisesRegex(Unsupported, "requires"):
            beam_search_new.check_support(self.case, solution)
        solution["engine"]["enable_beam_cascade_attention"] = False
        beam_search_new.check_support(self.case, solution)
        solution["engine"]["enable_beam_cascade_attention"] = "false"
        with self.assertRaisesRegex(Unsupported, "boolean"):
            beam_search_new.check_support(self.case, solution)


if __name__ == "__main__":
    unittest.main()
