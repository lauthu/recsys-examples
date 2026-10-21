# SID-GR inference benchmark

Measures beam-search latency of HTTP inference servers at concurrency 1 and
shows where each engine's GPU time goes. Every engine gets the same seeded
requests, the same checks and the same three outputs.

| `engine` | Server | Beam-search API |
| --- | --- | --- |
| `sid-gr` | [SID-GR](../sid-gr-inference) | `n` beams |
| `sglang` | Public [SGLang](https://github.com/sgl-project/sglang) v0.5.22 or newer | native `beam_width`; no catalog support |
| `sglang-beam-search-new` | SGLang [cswuyg/sglang@df191d6](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891) | `use_beam_search` |
| `sglang-sidtrie` | SGLang [vedantjh2/sglang@646a7f5](https://github.com/vedantjh2/sglang/tree/646a7f5ce6f94be84237c50cd41bb2e7beca7788) | native `beam_width` with a trie output head |

## Outputs

Each engine's directory receives:

- `report.md`: latency of every run and pooled, GPU kernel time by phase and section, the longest-running kernels, and the setup observed at runtime.
- `sunburst.svg`: GPU kernel time per request, phase › section › kernel family. The report embeds it as an image; open the file in a browser to hover for details and click a slice to zoom in (the center or Esc zooms out).
- `profile.nsys-rep`: the Nsight Systems trace of the profiled requests.

When several engines run in one invocation, `comparison.md` in the output
directory puts them side by side: latency, kernel time, GPU idle share and
output agreement with the first engine.

```text
OUTPUT/comparison.md             (several engines only)
OUTPUT/<engine>/report.md        (one engine: these files go directly in OUTPUT)
OUTPUT/<engine>/sunburst.svg
OUTPUT/<engine>/profile.nsys-rep
```

## Layout

```text
run_benchmark.py       host CLI: validates, starts containers, writes the outputs
configs/
  benchmark.json       what is measured: workload, run counts, CPUs, GPU
  <engine>.json        how one engine is launched: environment, server settings
bench/
  engines.py           one class per engine: everything engine-specific
  config.py            configuration, checkpoint and workload checks
  client.py            runs inside the container: server, requests, checks
  server_*.py          the servers with benchmark endpoints and NVTX markers
  catalog.py           item catalog parsing and native trie formats
  trace.py sunburst.py report.py compare.py    outputs
tests/                 CPU-only tests
```

## Run

Requirements: Linux, Python 3.10+, Docker with the NVIDIA runtime, a bfloat16
Qwen3 checkpoint, the engine source and a local image that has the engine's
Python dependencies and `nsys`. The source is mounted read-only, not installed,
so one image per dependency set is enough: for example `lmsysorg/sglang` with
the versions pinned in the SGLang checkout's `python/pyproject.toml`, or the
dependencies in `examples/sid-gr-inference/pyproject.toml` for SID-GR.

```bash
cd examples/sid_gr_inference_benchmark

# One engine: SID-GR from this repository
python3 run_benchmark.py --model /path/to/Qwen3-0.6B --output logs/sid_gr \
  --engine configs/sid_gr.json ../.. IMAGE

# Compare engines; the first is the reference
python3 run_benchmark.py --model /path/to/Qwen3-0.6B --output logs/compare \
  --engine configs/sid_gr.json ../.. SID_GR_IMAGE \
  --engine configs/sglang_beam_search_new.json /path/to/sglang SGLANG_IMAGE
```

`--dry-run` validates the inputs and prints the `docker run` commands. The output
directory must be new or empty. JIT caches go to `.cache/<image>` next to the
output directory (or `--cache-dir`) and are reused by later runs with the same
image. If a run fails, `.work` keeps the container and server logs.
Results are comparable only when every engine runs on the same machine, which
the runner does one after another.

## What is measured

- **Workload:** seeded random token IDs without the model's special tokens, so every engine gets the same requests for the same model and seed.
- **Latency:** `runs.clean` separate server processes, each sent `runs.warmups` unmeasured and then `runs.requests` measured requests, one at a time over a new loopback connection. The client times each request from JSON encoding to the parsed response.
- **Correctness:** every returned request must have `beam_width` distinct beams of `output_tokens` valid token IDs with finite scores in descending order. Scores are cumulative log-probabilities; SGLang's mean log-probability is multiplied by the beam length.
- **Profile:** one more server process runs under `nsys` and captures `runs.profile_requests` requests after the warmups. Kernels are attributed to a phase (`prefill`, `decode_1`, ...) and section through the NVTX range around their launch call. Each kernel also gets a family from its name (attention, attention_merge, linear, topk, sort, softmax, normalization, qk_norm, rotary, activation, kv_cache, copy, gather_scatter, reduction, math, elementwise); names cannot tell the GEMMs of different layers apart, so they share `linear`. Kernel time adds up kernel durations; the server window splits each request's handler time into kernel, overlap, memory-only and no-GPU-activity time.
- **Runtime facts:** attention backends, package versions and CUDA graph use are read from the running server, not inferred from settings.
- **Output agreement** (`comparison.md`): beams of the first clean run, request by request, against the reference engine: top-1 match, share of shared beams, rank match and the largest score difference.

The runner refuses to start while another benchmark container is running, and it
fails if the source, benchmark code or model files change during the runs.

## Configuration

`configs/benchmark.json`, shared by every engine in a run:

| Key | Meaning |
| --- | --- |
| `workload` | `context_length`, `beam_width`, `output_tokens` and `seed` |
| `runs` | `clean` processes × `requests`, `warmups` per process, `profile_requests` |
| `cpus` | Disjoint `server` and `client` CPU lists; the container is limited to both |
| `gpu` | Device index or GPU UUID |

`configs/<engine>.json`, one per engine:

| Key | Meaning |
| --- | --- |
| `engine` | `sid-gr`, `sglang`, `sglang-beam-search-new` or `sglang-sidtrie` |
| `env` | String environment variables for the server, passed verbatim |
| `server_settings` | SGLang `ServerArgs`; the runner sets the model path, host, port and catalog. `cuda_graph_bs_decode` and `cuda_graph_bs_prefill` must match the workload's beam width and context length |
| `trie_config` | `sglang-sidtrie` only: `token_start` and `codebook_size` of the three SID codebooks |

The runner derives the request shape (`GR_CONTEXT_LEN`, `GR_BEAM_WIDTH`,
`GR_DECODE_STEPS`, batch size 1) and the profiler settings; `env` must not set
them. The report copies `env` and `server_settings` verbatim. To vary an engine
setting, copy its file; the file name names the engine's directory in the output.

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
beam is a catalog item. Public SGLang has no catalog support, so `--catalog` rejects it. SID-GR normalizes scores over the allowed tokens; both
SGLang engines keep full-vocabulary normalization, so scores differ across
engines even when the beams match.

## Adding an engine

An engine is one subclass of `Engine` in [bench/engines.py](bench/engines.py),
listed in `ENGINES`, plus a config file. The class declares

- the source file the checkout must contain, where it is mounted, and the config keys and checks;
- the server command, readiness endpoint and profiler start/stop calls;
- the request payload and how to read beams and scores from the response, as cumulative log-probabilities;
- the runtime facts to check and report (graph use, the code path taken).

The engine needs an HTTP server with a beam-search request, a way to start and
stop `cudaProfilerApi` captures, and NVTX ranges named `bench/request:<rid>`,
`bench/phase:<rid>:<phase>` and `bench/section:<rid>:<name>` around the request,
its phases and its sections, as in [server_sglang.py](bench/server_sglang.py).
Without phase and section ranges the profile still reports kernel time, but
attributes it to `unknown`. Engines without an HTTP server, such as vLLM-GR,
TensorRT-LLM and FlashRec, first need a thin server around their beam API.

## Tests

```bash
python3 -m pytest tests
```

The tests use temporary files and a fake HTTP server; they need no GPU, Docker or model.
