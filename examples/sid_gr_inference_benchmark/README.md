# SID-GR inference benchmarks

Compare SID-GR and native SGLang beam search on identical token inputs. The suite
owns workloads, timing, validation and reports; adapters call the existing engines.

## Quick start

Run commands from this directory. The controller needs Python 3.10+ and PyYAML.
Each engine runs in a prepared Python environment or Docker image.

```bash
python3 -m pip install -r requirements.txt
export QWEN3_0_6B_MODEL=/absolute/path/to/Qwen3-0.6B
export GR_DECODE_ATTEN_ROOT=/absolute/path/to/gr_decode_atten
export SID_GR_PYTHON=/absolute/path/to/sid-gr-env/bin/python
export SGLANG_PYTHON=/absolute/path/to/sglang-env/bin/python

python3 bench.py plan --smoke
python3 bench.py e2e --smoke
```

SID-GR defaults to `../sid-gr-inference` and needs that project's dependencies
and external CuTe attention kernel. SGLang needs the native
`SamplingParams.beam_width` API and `meta_info.beam_results` responses with
length-normalized `sequence_score`. Set `SID_GR_SOURCE` or `SGLANG_SOURCE` to use
another checkout; an empty SGLang source uses the installed package.

For Docker, set these instead of the Python executable settings:

```bash
export SID_GR_RUNTIME=docker SGLANG_RUNTIME=docker
export SID_GR_IMAGE=your-prepared-gr-image
export SGLANG_IMAGE=your-prepared-sglang-image
python3 bench.py e2e --smoke
```

Images must exist locally; each run resolves tags to immutable image IDs.
The selected GPU is passed to Docker by UUID because host and session indices
can differ. Sources and models are mounted read-only; artifacts and compilation
caches are writable. Extra dependency paths go in `runtime.mounts`.

The suite does not download models or install frameworks. The optional SID-GR
[Dockerfile](entrypoints/sid_gr_inference/Dockerfile) pins CUTLASS DSL 4.5.1 and
Quack 0.4.1 for the existing GR kernel. The optional
[profiling Dockerfile](entrypoints/Dockerfile.profile) adds Nsight Systems from
`PROFILER_IMAGE` to `BASE_IMAGE`; set `NSYS_ROOT` to its installation directory.

## Configuration and execution

| File | Contents |
| --- | --- |
| `scenarios/*.yml` | Checkpoint, inputs, context/batch/beam grids and generation semantics |
| `solutions/*.yml` | Adapter, source, runtime and engine settings |
| `benchmark.yml` | Selected scenarios/solutions, baseline, repetitions, GPU and output directory |

`${NAME:-default}` expands environment variables. Paths in scenario and solution
files resolve from this directory; `output_dir` resolves beside the run config.
Copy `benchmark.yml` to an ignored `benchmark.local.yml` for local selections.

```bash
python3 bench.py plan --config benchmark.local.yml
python3 bench.py e2e --case 'qwen3_0_6B/*' --repeats 30
python3 bench.py e2e --smoke --case qwen3_0_6B/ctx1000_b1_w256
python3 bench.py e2e --smoke --solutions sglang
```

The full grid includes Qwen3-1.7B; set `QWEN3_1_7B_MODEL` before running it.
`--smoke` selects the first matching case, with two warmups and three measurements.
Initial compilation and graph capture can take several minutes. Caches are shared
across a solution's cases within the run. Solutions run serially on one GPU;
their order rotates between cases.

Synthetic inputs are seeded vocabulary IDs excluding BOS/EOS/PAD. For real
requests, use `inputs.type: jsonl` and `inputs.path`, with rows such as
`{"request_id": "request-0", "input_ids": [12, 34, 56]}`. The first batch-size rows
must have exactly the requested context length. One serialized input file is
shared across solutions; the suite does not tokenize, pad or truncate it.

Both adapters support BF16 weights/KV, no quantization or prefix reuse, fixed
output length, all beams, no catalog constraints and cumulative log-probabilities.
Unsupported semantics fail explicitly. Three output tokens mean prefill selection
plus two SID-GR decode steps. SGLang's mean log-probability is multiplied by output
length; equal fixed lengths preserve its ranking.

SID-GR works without the fused normalizer. A checkout with that optional
optimization can enable `GR_INFERENCE_FUSED_BEAM_LOGSUMEXP=1` (default `0`).
Enabling it on a checkout without the implementation is unsupported. To compare
two variants, copy the solution YAML, give it a new name and change its `env`.
Kernel microbenchmarks belong with the corresponding optimization; this suite
measures complete inference batches and GPU timelines.

## Measurement and reports

The timer starts with CPU token IDs and ends with completed CPU beam IDs and
scores, including submission, transfers, scheduling, output conversion and
synchronization. Setup, warmup, validation and serialization are outside the timer.
Every measured repeat is validated for request identity, beam counts, token
lengths/ranges, unique sequences, finite scores and descending score order.

Reports show median/IQR batch latency, requests/s, and beam agreement against the
baseline. Numerical disagreement stays visible separately from structural validity.
Synthetic inputs do not establish recommendation quality; smoke runs do not
establish performance rankings.

Each run creates `results/<run-id>/`:

```text
manifest.json       # Effective configuration, source/model/input identities
workloads/          # Exact shared input IDs
jobs/               # Requests, commands, logs, raw latency samples and all beams
cache/              # Per-solution compilation caches
summary.json         # Derived comparison data
report.md           # Human-readable comparison
diagnostics/        # Separate GPU timeline captures
```

```bash
python3 bench.py report --run results/<run-id>
python3 bench.py report --run results/<run-id> --leaderboard leaderboard.md
```

The leaderboard contains one run. Sources, dirty-tree hashes, package versions,
checkpoint hashes and image IDs are recorded. `source.expected_commit` requires
a clean pinned checkout. Failed and unsupported jobs remain in the report.

An optional `hardware` entry supplies the exact GPU name, a **dense BF16** TFLOP/s
ceiling and its source. The Qwen3 estimator divides matrix FLOPs by that ceiling.
This is a partial compute floor: memory traffic, reductions, selection and host
costs are omitted. It is not an attainable speedup or GPU-utilization estimate.
Missing or mismatched hardware data is reported as unavailable. GPU memory is not
measured across engine process trees.

## GPU timelines

```bash
python3 bench.py diagnose --run results/<run-id> --case qwen3_0_6B/ctx1000_b1_w64
```

Diagnosis reuses the original inputs/settings and checks source, checkpoint,
suite, GPU/driver and runtime identity. Rerun e2e after changing an implementation.
`nsys` must be available in each solution runtime; otherwise its capture is marked
unsupported. Only kernels within `sid_bench/measured` NVTX ranges enter the summary.
Kernel-duration sums and GPU-active time unions are reported separately. Native
traces remain available for inspection; profiled timings never enter e2e results.

## Code structure

`bench.py` selects a command. `runner.py` expands cases and launches isolated
workers through `runtime.py`. `worker.py` owns timing and validation; adapters own
framework calls. `report.py` derives reports from saved results. `diagnose.py`
replays a case under Nsight Systems.

To add an engine, implement `check_support(case, solution)` and an `Adapter(request)`
with `generate(workload)`, `synchronize()`, `close()` and `provenance()`. Output rows
have `request_id` and `beams: [{token_ids: [...], score: float}]`. Engines with GPU
subprocesses can add `profile_start()`/`profile_stop()` hooks. Add a solution YAML
only when the adapter exists and select it in the run config.

```bash
python3 -m unittest discover -s tests
```
