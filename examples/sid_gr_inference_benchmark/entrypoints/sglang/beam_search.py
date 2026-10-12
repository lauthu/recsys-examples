"""The pinned feature/beam_search baseline named in the SID-GR README."""

from entrypoints.sglang import adapter
from harness.validation import Unsupported


def check_support(case, solution):
    adapter.check_support(case, solution)


class Adapter(adapter.Adapter):
    def beam_api(self):
        from sglang.srt.server_args import ServerArgs

        if not hasattr(ServerArgs, "enable_beam_search"):
            raise Unsupported("Use the pinned feature/beam_search checkout")
        # This pinned branch enables beam search globally and uses n as its width.
        # Its native sequence_score has the same mean-logprob normalization.
        return {"enable_beam_search": True}, {}
