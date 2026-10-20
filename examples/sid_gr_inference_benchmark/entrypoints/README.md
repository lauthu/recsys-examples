# Engine entrypoints

Each solution has its own `adapter.py` and independent runtime configuration.

| Solution | Entrypoint | Beam-search API | Runtime |
| --- | --- | --- | --- |
| SID-GR | [sid_gr_inference/adapter.py](sid_gr_inference/adapter.py) | SID-GR executor | Shares the old beam branch's image |
| Public SGLang | [sglang/adapter.py](sglang/adapter.py) | Request `beam_width` | Pinned public image |
| Old SGLang beam branch | [sglang_beam_search/adapter.py](sglang_beam_search/adapter.py) | Server `enable_beam_search`; request `n` | [Dockerfile](sglang_beam_search/Dockerfile) |
| New SGLang beam branch | [sglang_beam_search_new/adapter.py](sglang_beam_search_new/adapter.py) | Server `enable_beam_search`; request `n` and `use_beam_search` | [Dockerfile](sglang_beam_search_new/Dockerfile) |
| vLLM-GR | [vllm_gr/adapter.py](vllm_gr/adapter.py) | `GRLLM.beam_search(output_mode="compact")`, legacy CUSTOM backend | [Dockerfile](vllm_gr/Dockerfile), vLLM 0.22.1 |
| TensorRT-LLM | [tensorrt_llm/adapter.py](tensorrt_llm/adapter.py) | `LLM.generate`, `SamplingParams(use_beam_search=True)` | Pinned 1.2.1 image, PyTorch backend |
| FlashRec | [flash_rec/adapter.py](flash_rec/adapter.py) | `BeamRecEngine.generate_many` | [Dockerfile](flash_rec/Dockerfile) |

[sglang_common.py](sglang_common.py) shares engine setup, output normalization,
profiling and cleanup. Each SGLang adapter owns its API checks and beam settings;
none inherits from another solution. The new branch also validates its optional
FlashInfer cascade setting.

The [solution configs](../config/solutions) select adapters, source commits,
images and engine options. The worker calls `check_support`, then `Adapter` to
generate all beams with cumulative log-probability scores. Incompatible APIs are
reported as unsupported; beam search never falls back to sampling.

[beam_common.py](beam_common.py) validates the additional adapters' scenario and
engine settings without importing GPU frameworks. It copies native completed beams
to the output contract without trimming, sorting, deduplicating or repairing them;
the worker validates every warmup and measured output. Setup and graph capture
occur before timing; output materialization and synchronization are timed.
