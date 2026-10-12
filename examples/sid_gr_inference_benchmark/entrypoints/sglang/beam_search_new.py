"""Pinned feature/beam_search_update_0801 with explicit per-request beam search."""

from entrypoints.sglang import adapter, beam_search
from harness.validation import Unsupported


def check_support(case, solution):
    adapter.check_support(
        case,
        solution,
        extra_options=("enable_beam_cascade_attention", "decode_attention_backend"),
    )
    options = solution["engine"]
    cascade = options.get("enable_beam_cascade_attention", False)
    if type(cascade) is not bool:
        raise Unsupported("enable_beam_cascade_attention must be a boolean")
    if (
        cascade
        and options.get("decode_attention_backend", options.get("attention_backend"))
        != "flashinfer"
    ):
        raise Unsupported("Beam cascade requires decode_attention_backend=flashinfer")


class Adapter(beam_search.Adapter):
    def beam_api(self):
        from sglang.srt.sampling.sampling_params import SamplingParams

        if "use_beam_search" not in getattr(SamplingParams, "__struct_fields__", ()):
            raise Unsupported("Use the pinned feature/beam_search_update_0801 checkout")
        engine_options, beam_params = super().beam_api()
        # This branch needs the server switch AND the request flag; n alone samples.
        # Its pinned sequence_score remains mean generated-token logprob.
        return engine_options, {**beam_params, "use_beam_search": True}
