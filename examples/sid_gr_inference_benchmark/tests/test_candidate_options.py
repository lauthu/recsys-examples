"""Candidate optimizations do not change the common generation contract."""

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entrypoints.sglang import adapter as sglang
from entrypoints.sid_gr_inference import adapter as sid_gr
from harness.config import ROOT, cases, load
from harness.validation import Unsupported


class CandidateOptionsTests(unittest.TestCase):
    def setUp(self):
        self.case = next(cases(load(ROOT / "config/runs/default.yml")))

    def test_prefix_cache_is_independent_for_each_candidate(self):
        for adapter in (sglang, sid_gr):
            for enabled in (True, False):
                with self.subTest(adapter=adapter.__name__, enabled=enabled):
                    adapter.check_support(
                        self.case, {"engine": {"prefix_cache": enabled}}
                    )
            with self.assertRaisesRegex(Unsupported, "must be a boolean"):
                adapter.check_support(self.case, {"engine": {"prefix_cache": "false"}})

    def test_sglang_passes_cache_setting_to_engine(self):
        module = SimpleNamespace(__file__="/test/sglang/__init__.py", Engine=Mock())
        for enabled in (True, False):
            request = {
                "case": self.case,
                "model": {"path": "/model"},
                "solution": {
                    "source": {},
                    "engine": {"prefix_cache": enabled, "page_size": 1},
                },
            }
            with patch.dict(sys.modules, {"sglang": module}), patch.object(
                sglang.Adapter, "beam_api", return_value=({}, {})
            ):
                sglang.Adapter(request)
            options = module.Engine.call_args.kwargs
            self.assertEqual(options["disable_radix_cache"], not enabled)
            self.assertEqual(options["page_size"], 1)
            self.assertNotIn("prefix_cache", options)
            self.assertEqual(request["solution"]["engine"]["prefix_cache"], enabled)

    def test_sid_gr_passes_cache_setting_to_executor(self):
        gr = Mock()
        gr.GRContinuousServingExecutor.return_value.run_until_empty.return_value = []
        adapter = object.__new__(sid_gr.Adapter)
        adapter.gr, adapter.case, adapter.engine = gr, self.case, object()
        adapter.beam_pool = adapter.context_pool = None
        adapter.torch = SimpleNamespace(
            no_grad=nullcontext, cuda=SimpleNamespace(synchronize=Mock())
        )
        for enabled in (True, False):
            adapter.prefix_cache = enabled
            self.assertEqual(adapter.generate([]), [])
            self.assertEqual(
                gr.GRContinuousServingExecutor.call_args.kwargs[
                    "prefill_cache_enabled"
                ],
                enabled,
            )


if __name__ == "__main__":
    unittest.main()
