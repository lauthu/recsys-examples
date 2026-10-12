# Engine entrypoints

Each solution has its own `adapter.py` and independent runtime configuration.

| Solution | Entrypoint | Beam-search API | Runtime |
| --- | --- | --- | --- |
| SID-GR | [sid_gr_inference/adapter.py](sid_gr_inference/adapter.py) | SID-GR executor | Shares the old beam branch's image |
| Public SGLang | [sglang/adapter.py](sglang/adapter.py) | Request `beam_width` | Pinned public image |
| Old SGLang beam branch | [sglang_beam_search/adapter.py](sglang_beam_search/adapter.py) | Server `enable_beam_search`; request `n` | [Dockerfile](sglang_beam_search/Dockerfile) |
| New SGLang beam branch | [sglang_beam_search_new/adapter.py](sglang_beam_search_new/adapter.py) | Server `enable_beam_search`; request `n` and `use_beam_search` | [Dockerfile](sglang_beam_search_new/Dockerfile) |

[sglang_common.py](sglang_common.py) shares engine setup, output normalization,
profiling and cleanup. Each SGLang adapter owns its API checks and beam settings;
none inherits from another solution. The new branch also validates its optional
FlashInfer cascade setting.

The [solution configs](../config/solutions) select adapters, source commits,
images and engine options. The worker calls `check_support`, then `Adapter` to
generate all beams with cumulative log-probability scores. Incompatible APIs are
reported as unsupported; beam search never falls back to sampling.
