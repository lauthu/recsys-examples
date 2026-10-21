# SID-GR and SGLang HTTP beam-search benchmark

Measures beam-search latency of an HTTP inference server at concurrency 1 and
profiles where its GPU time goes. Three servers are supported:

| `engine` | Server | Beam-search API |
| --- | --- | --- |
| `sid-gr` | [SID-GR](../../sid-gr-inference) | `n` beams |
| `sglang-beam-search-new` | SGLang [cswuyg/sglang@df191d6](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891) | `use_beam_search` |
| `sglang-sidtrie` | SGLang [vedantjh2/sglang@646a7f5](https://github.com/vedantjh2/sglang/tree/646a7f5ce6f94be84237c50cd41bb2e7beca7788) | native `beam_width` with a trie output head |

A run writes three files:

- `report.md`: latency of every run and pooled, GPU kernel time by phase and section, and the setup observed at runtime.
- `sunburst.svg`: GPU kernel time per request, phase › section › kernel family; embedded in the report.
- `profile.nsys-rep`: the Nsight Systems trace of the profiled requests.

## Run

Requirements: Linux, Python 3.10+, Docker with the NVIDIA runtime, a bfloat16
Qwen3 checkpoint, the engine source, and a local image that has the engine's
Python dependencies and `nsys`. The source is mounted read-only, not installed,
so one image per dependency set is enough: for example `lmsysorg/sglang` with
the versions pinned in the SGLang checkout's `python/pyproject.toml`, or in
`examples/sid-gr-inference/pyproject.toml` for SID-GR.

```bash
cd examples/sid_gr_inference_benchmark

# SID-GR from this repository
python3 tools/run_benchmark.py --config tools/benchmark.json \
  --source ../.. --model /path/to/Qwen3-0.6B --image IMAGE --output logs/sid_gr

# SGLang from its checkout
python3 tools/run_benchmark.py --config tools/benchmark_sglang.json \
  --source /path/to/sglang --model /path/to/Qwen3-0.6B --image IMAGE --output logs/sglang
```

`--dry-run` validates the inputs and prints the `docker run` commands. The output
directory must be new or empty. JIT caches go to `.cache/<image>` next to the
output directory (or `--cache-dir`) and are reused by later runs with the same
image. If a run fails, `OUTPUT/.work` keeps the container and server logs.

## What is measured

- **Workload:** seeded random token IDs without the model's special tokens, so every engine gets the same requests for the same model and seed.
- **Latency:** `runs.clean` separate server processes, each sent `runs.warmups` unmeasured and then `runs.requests` measured requests, one at a time over a new loopback connection. The client times each request from JSON encoding to the parsed response.
- **Correctness:** every returned request must have `beam_width` distinct beams of `output_tokens` valid token IDs with finite scores in descending order. Scores are cumulative log-probabilities; SGLang's mean log-probability is multiplied by the beam length.
- **Profile:** one more server process runs under `nsys` and captures `runs.profile_requests` requests after the warmups. Kernels are attributed to a phase (`prefill`, `decode_1`, ...) and section through the NVTX range around their launch call. Kernel time adds up kernel durations; the server window splits each request's handler time into kernel, overlap, memory-only and no-GPU-activity time.
- **Runtime facts:** attention backends, package versions and CUDA graph use are read from the running server, not inferred from settings.

The runner refuses to start while another benchmark container is running, and it
fails if the source, tools or model files change during the runs.

## Configuration

| Key | Meaning |
| --- | --- |
| `engine` | `sid-gr`, `sglang-beam-search-new` or `sglang-sidtrie` |
| `workload` | `context_length`, `beam_width`, `output_tokens` and `seed` |
| `runs` | `clean` processes × `requests`, `warmups` per process, `profile_requests` |
| `cpus` | Disjoint `server` and `client` CPU lists; the container is limited to both |
| `gpu` | Device index or GPU UUID |
| `env` | String environment variables for the server, passed verbatim |
| `server_settings` | SGLang `ServerArgs`; the runner sets the model path, host, port and catalog |
| `trie_config` | `sglang-sidtrie` only: `token_start` and `codebook_size` of the three SID codebooks |

The runner derives the request shape (`GR_CONTEXT_LEN`, `GR_BEAM_WIDTH`,
`GR_DECODE_STEPS`, batch size 1) and the profiler settings; `env` must not set
them. The report copies `env` and `server_settings` verbatim.

## Catalog filtering

`--catalog items.jsonl` constrains every beam to a catalog item:

```jsonl
{"item_id": "item-a", "token_ids": [1000, 2000, 3000]}
{"item_id": "item-b", "token_ids": [1001, 2000, 3001], "metadata": {"genre": "drama"}}
```

Token IDs are model token IDs. Item IDs and token paths must be unique, every
path has exactly `output_tokens` tokens, and the catalog needs at least
`beam_width` distinct first tokens so every beam survives each step. Each engine
gets the same paths in its native form: SID-GR loads the JSONL, SGLang
beam-search-new a generated token-trie pickle, and SIDTrie a generated
`trie_output_head.safetensors` placed next to links to the original checkpoint.
For SIDTrie, the token at depth `d` must lie in
`[token_start + d * codebook_size, token_start + (d + 1) * codebook_size)`.

The run fails unless the server reports the catalog as loaded and every returned
beam is a catalog item. SID-GR normalizes scores over the allowed tokens; both
SGLang engines keep full-vocabulary normalization, so scores differ across
engines even when the beams match.

## Tests

```bash
python3 -m pytest tools/tests
```

The tests use temporary files and a fake HTTP server; they need no GPU, Docker or model.
