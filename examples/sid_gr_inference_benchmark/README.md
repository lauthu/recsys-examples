# SID-GR inference benchmark

Measures the beam-search latency and throughput of HTTP inference servers and
shows where each engine's time goes. Every engine gets the same seeded
requests, the same checks and the same report.

| `engine` | Server | Beam-search API |
| --- | --- | --- |
| `sid-gr` | [SID-GR](../sid-gr-inference) | `n` beams |
| `gr-inference-internal` | [cb521/gr-inference-internal](https://github.com/cb521/gr-inference-internal), the repository SID-GR was imported from, at its `perf_precision_fix` branch ([facff3d](https://github.com/cb521/gr-inference-internal/tree/facff3db9c4d94ba5a5f1676169e318898cf82ba)) | `n` beams; no catalog support |
| `sglang` | Public [SGLang](https://github.com/sgl-project/sglang) v0.5.22 or newer | native `beam_width`; no catalog support |
| `sglang-beam-search-new` | SGLang [cswuyg/sglang@df191d6](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891) | `use_beam_search` |
| `vedantjh2-sglang` | SGLang [vedantjh2/sglang@646a7f5](https://github.com/vedantjh2/sglang/tree/646a7f5ce6f94be84237c50cd41bb2e7beca7788) | native `beam_width`; with a catalog, a trie output head |

## Quick start

Requirements: Linux; Python 3.10+ with PyYAML and Jinja2; Docker with the NVIDIA
runtime; the checkpoint the scenario pins; each engine's source at its pinned
commit; and a local image per engine, built from [images/](images/README.md).

```bash
cd examples/sid_gr_inference_benchmark
images/build.sh sglang                       # once per engine

python3 run_benchmark.py --scenario configs/scenarios/qwen3_0_6b.yaml \
  --model /path/to/Qwen3-0.6B --output logs/sglang \
  --server-cpus 0-9 --client-cpus 10 --gpu 0 \
  --solution configs/solutions/sglang.yaml /path/to/sglang sid-gr-bench:sglang
```

- Repeat `--solution CONFIG SOURCE IMAGE` to compare engines in one invocation;
  they run one after another on the same GPU.
- `--reference NAME`: the engine the others are compared with (default: public
  SGLang if it runs, else the first solution).
- `--dry-run`: check the inputs and print what the benchmark adds to each
  solution and the `docker run` commands, without running them.
- `--allow-dirty`: run a source that differs from its pin; the report says so.
- `python3 tools/render_reports.py OUTPUT`: render the reports again from the
  saved results, without Docker or the GPU (after a template change, for example).

## Outputs

```text
OUTPUT/summary.md                      several engines: all of them side by side
OUTPUT/load.svg                        several engines, several concurrency values
OUTPUT/lengths.svg                     several engines, prompts of several lengths
OUTPUT/<engine>/report.md              one engine (with one engine, its files are in OUTPUT)
OUTPUT/<engine>/results.json.gz
OUTPUT/<engine>/load.svg               several concurrency values
OUTPUT/<engine>/breakdown.svg
OUTPUT/<engine>/sunburst.svg
OUTPUT/<engine>/profile.nsys-rep
OUTPUT/<engine>/load_profile.nsys-rep  several concurrency values
```

- `report.md`: how the engine was measured; throughput and latency at each
  concurrency; where a request's time goes; the engine's steps; the GPU
  profiles and how they were captured; the setup; and how to reproduce it.
- `summary.md`: the engines side by side, with the change in mean latency
  against the reference, latency by prompt length when lengths vary, kernel
  time per request, output agreement and versions.
- `results.json.gz`: everything measured and observed. The reports are rendered
  from it alone, so the same results always give the same reports.
- `load.svg`, `breakdown.svg`, `lengths.svg`: throughput and latency against
  concurrency, where a request's time goes, and latency against prompt length. `sunburst.svg`: GPU time per request by phase,
  section and kernel family; open it in a browser to zoom.
- `*.nsys-rep`: the Nsight Systems traces of the GPU profiles.

## How it measures

1. **Rounds.** The runner measures each engine in `runs.rounds` rounds. Each
   round starts a new container and server and measures every concurrency in
   turn, starting at a different one each time. The results pool all rounds,
   so they include the variation between server starts.
2. **Load.** At concurrency *k*, *k* clients each send a request and, as soon
   as its reply arrives, their next one (a closed loop). Each client first
   sends `runs.warmups` unmeasured requests, so the server runs full before
   measuring, then the clients send `runs.requests` measured ones; no request
   is sent twice.
3. **Latency and throughput.** Latency is the client's time from sending a
   request to parsing its reply, over a new loopback HTTP connection.
   Throughput is measured requests per second, from the first send to the last
   reply in each round.
4. **Statistics.** Mean, p50, p95 and p99, with 95% confidence intervals from a
   bootstrap that resamples the rounds and then the requests within each
   (2,000 resamples). Every engine gets the same draws, so its change against
   the reference is paired request by request.
5. **Where the time goes.** The launcher notes when the server's request handler
   starts and returns and when each engine step runs, on the client's clock.
   Each request's latency splits into queue, prefill, decode and HTTP and
   response, and the steps give batch sizes and the share of time the engine
   was busy.
6. **GPU profiles.** One more server process runs under Nsight Systems and
   captures `runs.profile_requests` requests one at a time, so GPU time is
   attributed to each request's phases and sections. With several concurrency
   values, another one captures `runs.profile_concurrency` (by default the
   largest) under load, per engine step, except with SGLang's overlap
   scheduler, whose steps a trace cannot tell apart.
   Captures hold only the measured requests (`cudaProfilerApi`); tracing slows
   the server, so no latency comes from them.
7. **Checks.** Every reply must hold `beam_width` distinct beams of
   `output_tokens` valid tokens with finite scores in descending order
   (gr-inference-internal orders beams within 2e-5 of each other by token path,
   so it may return them up to 2e-5 out of order); with a catalog, every beam
   must be a catalog item. The server must run the mounted
   source, hold its weights in the scenario's dtype and take the requested code
   paths. Any failed check fails the run.
8. **Output agreement** (`summary.md`): each engine's beams in its first round
   against the reference's, request by request. Agreement shows consistency,
   not correctness: reduced precision can flip near-tied beams.

## What the benchmark patches

The benchmark runs each engine's own HTTP server from the mounted source, with
only the hooks its launcher lists in `PATCHES`
([sglang_server.py](bench/engines/sglang_server.py),
[sid_gr_server.py](bench/engines/sid_gr_server.py)). A hook calls the original
with the same arguments and returns its result; it only counts calls, notes
times or pushes NVTX ranges, and never synchronizes the GPU. A missing target
stops the server, and `server.log` lists every patch applied.

| What | For | SGLang | SID-GR and gr-inference-internal |
| --- | --- | --- | --- |
| Routes | readiness, runtime facts, configuration, profiler | adds `/pilot/ready`, `/pilot/runtime`, `/pilot/config` | adds `/pilot/runtime`, `/pilot/config`, `/pilot/profile/start` and `stop` |
| Runtime facts | packages, attention backends, weights' dtype, overlap scheduler | `Scheduler.get_internal_state` | read in `/pilot/runtime` |
| Request times | queue, prefill, decode, response | `/generate` | `_handle_sglang_generate`, and `_handle_sglang_generate_async` where the server is ASGI |
| Step times | batch sizes, engine busy | `get_next_batch_to_run`, `run_batch`, `process_batch_result`, `ScheduleBatch.copy` | `_run_prefill`, `_run_decode_batch` |
| Counters | CUDA graph use, code paths taken | `ModelRunner.forward`, `BeamGroup.finalize`, `shared_context_attention` | read from the engine's own status and gr-inference-internal's beam-selection extension |
| NVTX ranges, GPU profiles only | phases, sections, steps | input preparation, forward, beam selection, beam KV, results | prefill, decode steps, SID-GR's profile sections |

## Configuration

Each value lives in one place:

| Input | Says | Where |
| --- | --- | --- |
| Scenario | what is measured, the same for every engine: model, workload, rounds, optional catalog | `configs/scenarios/*.yaml` |
| Solution | how one engine is configured: source pin, environment, server settings | `configs/solutions/*.yaml` |
| Command line | where it runs: checkpoint, catalog file, CPUs, GPU | `run_benchmark.py` options |

The engine's adapter derives the rest from these, such as CUDA graph sizes,
request slots, ports and paths, so a solution must not set them and any
solution runs with any scenario on any machine. All three load into typed
classes ([bench/config.py](bench/config.py)): unknown keys, wrong types and
out-of-range values fail before any container starts. For each solution, the
runner writes them as one `RunConfig`, `/work/run_config.json`, which is all
the container reads.

### Scenario

```yaml
model:                        # Hub id, revision and file SHA-256s; dtype while serving
workload:
  context_length: 2048        # prompt tokens; the longest with min_context_length
  min_context_length: 256     # optional: lengths drawn uniformly up to context_length
  beam_width: 256
  output_tokens: 3            # one from prefill, the rest from decode steps
  concurrency: [1, 2, 4, 8]   # requests in flight: one value, or several in turn
  seed: 42                    # seeds the token IDs and lengths
runs:
  rounds: 5                   # each starts a new server and measures every concurrency
  requests: 500               # measured requests per concurrency in each round
  warmups: 10                 # unmeasured requests per client before them
  profile_requests: 3         # requests each GPU profile captures
  profile_concurrency: 4      # optional: the concurrency profiled under load; default the largest
catalog:                      # optional: see Catalog filtering
  sha256: <SHA-256 of the catalog JSONL>
```

- One deployment per engine serves every concurrency, sized for the largest:
  CUDA graph sizes, request slots, and the batch and pools of SID-GR and
  gr-inference-internal.
- With `min_context_length`, SGLang captures prefill graphs in 1.5× steps,
  since it replays a graph only below 2× padding. SID-GR captures graphs only
  for `context_length`, at startup, so other lengths run without them, as the
  report's CUDA graph row shows.
- `model.sid_codebooks` (`token_start`, `codebook_size`): where semantic-ID
  tokens sit in the vocabulary, for vedantjh2-sglang's trie output head.

### Solution

| Key | Meaning |
| --- | --- |
| `engine` | `sid-gr`, `gr-inference-internal`, `sglang`, `sglang-beam-search-new` or `vedantjh2-sglang` |
| `source` | `url` and full `commit` of the engine source; optional `branch`, which names a from-source image |
| `env` | environment variables for the server, as strings; `OMP_NUM_THREADS` and `MKL_NUM_THREADS` default to the number of server CPUs |
| `server_settings` | SGLang engines: `ServerArgs`, passed verbatim. The runner sets the model path, host, port, dtype, seed, CUDA graph sizes, request slots and beam-search-new's catalog. gr-inference-internal: `tools/serve_qwen3_gr_http.py` options, passed after the ones its script sets from the runner's variables (`true` is `--name`, `false` `--no-name`) |

Values a run depends on, such as `tp_size: 1` or `page_size: 1`, stay in the
solution with a comment saying why. To vary a setting, copy the file: its name
names the engine's output directory.

## Catalog filtering

A scenario with `catalog.sha256` constrains every beam to a catalog item; pass
the file with `--catalog items.jsonl`:

```jsonl
{"item_id": "item-a", "token_ids": [1000, 2000, 3000]}
{"item_id": "item-b", "token_ids": [1001, 2000, 3001], "metadata": {"genre": "drama"}}
```

- Token IDs are model token IDs. Item IDs and paths are unique, every path has
  `output_tokens` tokens, and there are at least `beam_width` distinct first
  tokens, so every beam survives each step.
- Each engine gets the same paths in its native form: SID-GR the JSONL,
  beam-search-new a token-trie pickle, vedantjh2-sglang a
  `trie_output_head.safetensors` next to links to the checkpoint. Public SGLang
  has no catalog support, and neither has gr-inference-internal here: its
  catalog lists semantic-ID codes, which it maps to `<sid_N>` tokens that Qwen3
  lacks.
- vedantjh2-sglang's trie output head takes exactly three codebooks from
  `model.sid_codebooks`, so `output_tokens` must be 3; without a catalog it
  serves the checkpoint as is.
- Engines normalize scores differently: SID-GR over the allowed tokens,
  beam-search-new over the vocabulary, vedantjh2-sglang over the vocabulary or,
  with `SGLANG_BEAM_TRIE_TOPK_LOGPROB: "true"`, the active codebook. Scores
  therefore differ even when the beams match.
- [tools/make_catalog.py](tools/make_catalog.py) writes a seeded synthetic
  catalog in the `model.sid_codebooks` layout, with Zipf fan-out, and prints
  the scenario keys for it:

```bash
python3 tools/make_catalog.py --items 1000000 --codebook-size 1024 --output catalog-1m.jsonl
```

## Reproducibility

The runner refuses to start when a pin does not hold:

| Pin | Where | Checked |
| --- | --- | --- |
| Engine source | `source.url` and `source.commit` in the solution | The checkout contains the commit, and the paths the engine reads (`python/` for SGLang; `examples/sid-gr-inference` and `corelib/gr_decode_atten` for SID-GR; `pyproject.toml`, `scripts`, `src`, `tools` and the `third_party/gr-decode-attention` submodule for gr-inference-internal) match it with no uncommitted changes |
| Checkpoint | `model.id`, `model.revision` and `model.sha256` in the scenario | SHA-256 of `config.json` and every weight and index file |
| Catalog | `catalog.sha256` in the scenario | SHA-256 of the `--catalog` file |
| Image | a recipe in [images/](images/README.md); dev branches built from source and tagged with repository, branch and commit | Recorded, not enforced: a local build gets a new ID each time |

It also refuses to start while another benchmark container runs, and fails if
the source, benchmark code or model files change during the runs. Each report
ends with what reproduces it: commits, image identity, file hashes, the Nsight
Systems version and the command.

## Layout

```text
run_benchmark.py                  the CLI: checks the inputs, runs the rounds and profiles, writes the outputs
configs/                          scenarios/ (what is measured) and solutions/ (how each engine is configured)
images/                           image recipes and build.sh
bench/
  config.py                       Scenario, Solution, Resources and RunConfig; loading and checks
  provenance.py                   source, checkpoint and image pins
  client.py                       in the container: starts the server, sends and times requests, checks beams
  catalog.py                      the catalog: parsing, membership checks, synthesis
  common.py                       helpers shared by the host and the container
  engines/                        one adapter and one launcher per engine family
    adapter.py                    Adapter: what each engine implements
    sglang.py sid_gr.py           the adapters: configure, start, call and check the engine
    sglang_server.py sid_gr_server.py   the launchers: the engine's own server plus PATCHES
    patching.py                   how launchers declare, apply and log patches
  outputs/                        everything written after the runs
    results.py                    results.json.gz: save, load, render
    trace.py                      Nsight Systems trace summaries
    report.py summary.py          the numbers of report.md and summary.md
    chart.py sunburst.py          the charts
    template.py templates/        the reports' words and tables, the charts' styles
tools/                            make_catalog.py, render_reports.py
tests/                            CPU-only tests
```

## Adding an engine

1. An adapter: a subclass of `Adapter` (bench/engines/adapter.py), listed in
   `ENGINES` in bench/engines/\_\_init\_\_.py. It holds no settings and declares
   the source checkout and image recipe, the solution checks, the server
   command and the settings it sets itself, the profiler calls, the request
   payload, how to read beams and scores, catalog support, and the runtime
   facts to check and report.
2. A launcher that runs the engine's own HTTP server with `PATCHES` for request
   and step times, the `/pilot/*` routes, and, for the GPU profiles, NVTX
   ranges `bench/request:<rid>`, `bench/phase:<rid>:<phase>` and
   `bench/section:<rid>:<name>`. Without phase and section ranges, kernel time
   is still reported, as `unknown`.
3. A solution file with its source pin and settings, and an image recipe in
   images/.

Engines without an HTTP server, such as vLLM-GR, TensorRT-LLM and FlashRec,
first need a thin server around their beam-search API.

## Tests

```bash
python3 -m pytest tests
```

The tests use temporary files and a fake HTTP server; they need no GPU, Docker
or model.
