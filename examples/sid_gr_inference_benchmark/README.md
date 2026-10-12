# SID-GR inference benchmarks

Compare SID-GR and native SGLang beam search on identical token inputs. End-to-end
runs report batch latency, request throughput and output parity. Separate profiling
runs show each candidate's GPU kernels without changing the headline measurements.

## Scenario, candidate, hardware and run

| Definition | Owns | Changes when |
| --- | --- | --- |
| `scenarios/*.yml` | Model/precision, input recipe and seed, context/batch/beam grid, generation semantics, warmups and repetitions | The workload or measurement protocol changes |
| `solutions/*.yml` | Candidate adapter, source checkout, runtime and engine settings | The inference implementation or its configuration changes |
| `hardware/*.yml` | Expected GPU SKU and specification source; optional sourced dense BF16 ceiling | The target SKU changes |
| `run.template.yml` | Selected scenarios/candidates, reference candidate, hardware profile, device UUID and execution policy | You choose what to execute and where |
| `results/<run-id>/` | Frozen definitions, actual source/runtime/device identities, exact inputs and results | Every invocation creates a fresh run |

A scenario is reusable across candidates, GPUs and runs. Its definition is the
shared contract; cases expand its context/batch/beam grid. A run is one execution
of selected cases on one GPU. Model bytes, materialized inputs and actual source
versions are hashed in that run, so moving checkpoints and changed checkouts stay
visible even if their configured names are unchanged.

Copy the run template and reuse that file for successive runs. Each invocation
gets a new run ID. It never overwrites an earlier execution or combines different
GPUs into one ranking. To change the protocol, edit/copy a scenario. `--smoke` is
an explicitly labeled exception: the first selected case, two warmups and three
measurements. Its effective protocol is recorded, and its results are not a
performance ranking.

## Quick start

Run commands from this directory. The controller needs Python 3.10+ and PyYAML;
each candidate runs in a prepared Python environment or Docker image.

```bash
python3 -m pip install -r requirements.txt
cp run.template.yml run.local.yml
export QWEN3_0_6B_MODEL=/absolute/path/to/Qwen3-0.6B
export GR_DECODE_ATTEN_ROOT=/absolute/path/to/gr_decode_atten
export SID_GR_PYTHON=/absolute/path/to/sid-gr-env/bin/python
export SGLANG_PYTHON=/absolute/path/to/sglang-env/bin/python

nvidia-smi -L
export BENCHMARK_GPU=GPU-...  # Copy the selected full UUID from the output above.
# Select the matching hardware profile in run.local.yml.
python3 bench.py plan --config run.local.yml --smoke
python3 bench.py e2e --config run.local.yml --smoke
```

Hardware is required. The runner queries the selected UUID, verifies its GPU name
against the profile, and records UUID, driver and memory before running workers.
Both local CUDA and Docker use that UUID, independent of inherited device-index
remapping. Numeric ordinals are intentionally unsupported. Add a profile with the
exact `nvidia-smi` GPU name and a specification `source` for another SKU.

SID-GR defaults to `../sid-gr-inference` and uses its standard batched serving and
CUDA-graph path. SGLang requires native `SamplingParams.beam_width` and
`meta_info.beam_results` with mean-log-probability `sequence_score`. Its selected
no-overlap configuration was faster in the saved
[overlap comparison](../sid-gr-inference/benchmark_artifacts/sglang_latest/summary.json);
this is evidence for that configuration, not a guarantee across GPUs and workloads.
Set `SID_GR_SOURCE` or `SGLANG_SOURCE` to select a checkout. An empty SGLang source
uses the installed package. Source commits, dirty changes and installed versions
are captured; `source.expected_commit` can require a clean pinned checkout.

For Docker, set these instead of the Python executable settings:

```bash
export SID_GR_RUNTIME=docker SGLANG_RUNTIME=docker
export SID_GR_IMAGE=your-prepared-gr-image
export SGLANG_IMAGE=your-prepared-sglang-image
python3 bench.py e2e --config run.local.yml --smoke
```

Images must exist locally; each run resolves tags to immutable image IDs. Source
and model directories are mounted read-only; artifacts and compilation caches are
writable. Extra dependencies go in `runtime.mounts`. The suite does not download
models or install frameworks. The optional SID-GR
[Dockerfile](entrypoints/sid_gr_inference/Dockerfile) pins CUTLASS DSL 4.5.1 and
Quack 0.4.1 for the existing external GR kernel. The optional
[profiling Dockerfile](entrypoints/Dockerfile.profile) adds Nsight Systems from
`PROFILER_IMAGE` to `BASE_IMAGE`; set `NSYS_ROOT` to its installation directory.

`${NAME:-default}` expands environment variables. Paths in scenario and candidate
files resolve from this directory; `output_dir` resolves beside the run template.
The full grid includes Qwen3-1.7B; set `QWEN3_1_7B_MODEL` before running it.

```bash
python3 bench.py e2e --config run.local.yml --case 'qwen3_0_6B/*'
python3 bench.py e2e --config run.local.yml --smoke --solutions sid_gr_inference
```

## Measurement and parity

Synthetic inputs are seeded vocabulary IDs excluding BOS/EOS/PAD. For real
requests, use `inputs.type: jsonl` and `inputs.path`, with rows such as
`{"request_id": "request-0", "input_ids": [12, 34, 56]}`. The first batch-size rows
must have exactly the requested context length. One serialized input file is
shared across candidates; the suite does not tokenize, pad or truncate it.

Both adapters support BF16 weights/KV, no quantization or prefix reuse, fixed
output length, all beams, no catalog constraints and cumulative log-probabilities.
Unsupported semantics fail explicitly. Three output tokens mean prefill selection
plus two SID-GR decode steps. SGLang's mean log-probability is multiplied by output
length; equal fixed lengths preserve its ranking.

The timer starts with CPU token IDs and ends with completed CPU beam IDs and
scores, including submission, transfers, scheduling, output conversion and
synchronization. Setup, warmup, validation and serialization are outside the timer.
Every repeat is checked for request identity, beam counts, token lengths/ranges,
unique sequences, finite scores and descending score order. Candidates run serially,
rotating their order between cases. Compilation caches are shared within each run.

- **Latency:** median and IQR of completed-batch wall time.
- **Throughput:** completed requests/s = batch size / median latency. This is
  offline batch throughput, not a sustained online load test.
- **Parity:** ordered token/score equality against the reference, with top-1
  agreement, minimum beam overlap and maximum shared-beam score difference.
  `exact`, `different` and `unavailable` remain separate from structural validity.
  The reference's own row checks repeat consistency. No tolerance is silently
  assumed, and disagreement does not hide a fast candidate.

## Run artifacts and scenario leaderboards

```text
results/<run-id>/
  manifest.json                 # Frozen definitions, hardware and run status
  workloads/                    # Exact shared input IDs
  jobs/                         # Requests, commands, logs, samples and beam outputs
  cache/                        # Per-candidate compilation caches
  summary.json                  # Latency, throughput, parity and compute estimates
  report.md                     # Whole-run report
  leaderboards/<scenario>.md    # This scenario's current status within this run
  diagnostics/                  # Separate GPU timeline captures
```

Leaderboards refresh after each completed job and at run completion/failure.
Each is bound to one run, one scenario and one GPU, with progress counts and links
to the frozen run. Failed and unsupported jobs stay visible. Regenerate the report
and all scenario leaderboards from saved results with:

```bash
python3 bench.py report --run results/<run-id>
```

Hardware profiles may supply a documented **dense BF16** TFLOP/s ceiling. The
Qwen3 estimator divides matrix FLOPs by that ceiling; memory traffic, reductions,
selection and host costs are omitted. Hardware identity is always required, but
a compute floor remains unavailable without a sourced dense ceiling. It is not
an attainable speedup or GPU-utilization estimate. GPU memory is not measured
across engine process trees. Synthetic inputs do not establish recommendation quality.

## GPU kernel profiling

```bash
python3 bench.py diagnose --run results/<run-id> --case qwen3_0_6B/ctx1000_b1_w64
```

Diagnosis reuses original inputs/settings and checks source, checkpoint, suite,
GPU/driver and runtime identity. Rerun e2e after changing an implementation.
`nsys` must exist in each candidate runtime; otherwise its capture is unsupported.
Per-candidate kernel tables and native traces support kernel-to-kernel inspection.
Only kernels within `sid_bench/measured` NVTX ranges enter summaries. Duration sums
and GPU-active time unions are separate; names alone do not establish equivalent
operations across engines. Profiled timings never enter e2e results.

## Code structure

- `bench.py`: command-line entry point.
- `harness/`: configuration, artifacts, orchestration, runtimes and isolated workers.
- `scenarios/`: deterministic scenario definitions, input construction and validation.
- `entrypoints/`: framework-specific adapters and optional image recipes.
- `reporting/`: latency/throughput/parity reports, scenario leaderboards and compute floor.
- `profiling/`: replay, Nsight capture and timeline summaries.
- `solutions/`, `hardware/`: candidate configurations and GPU profiles.

To add an engine, implement `check_support(case, solution)` and an `Adapter(request)`
with `generate(workload)`, `synchronize()`, `close()` and `provenance()`. Output rows
have `request_id` and `beams: [{token_ids: [...], score: float}]`. Engines with GPU
subprocesses can add `profile_start()`/`profile_stop()` hooks. Add a candidate YAML
when its adapter exists and select it in the run template.

```bash
python3 -m unittest discover -s tests
```
