"""Public upstream SGLang using its native beam_width API."""

from entrypoints import sglang_common
from harness.validation import Unsupported


def check_support(case, solution):
    sglang_common.check_support(case, solution)


class Adapter(sglang_common.BaseAdapter):
    def beam_api(self):
        from sglang.srt.sampling.sampling_params import SamplingParams

        if "beam_width" not in getattr(SamplingParams, "__struct_fields__", ()):
            raise Unsupported(
                "This SGLang version lacks the native beam_width API; use a newer environment"
            )
        import inspect

        from sglang.srt.beam_search.beam_group import BeamGroup

        penalty = inspect.signature(BeamGroup).parameters.get("length_penalty")
        if penalty is None or penalty.default != 1.0:
            raise Unsupported(
                "SGLang beam length normalization changed; update the adapter"
            )
        probe = object.__new__(BeamGroup)
        probe.length_penalty = 1.0
        if probe.beam_score(-6.0, 3) != -2.0:
            raise Unsupported("SGLang beam score semantics changed; update the adapter")
        return {}, {"beam_width": self.case["beam_width"]}
