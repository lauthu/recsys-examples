# SID-GR inference benchmarks

A shared benchmark for fixed-length semantic-ID beam search. Workloads and
measurement live here; inference implementations stay in their own projects.

| Mode | Purpose | Output |
| --- | --- | --- |
| `e2e` | Compare completed offline batches on identical token inputs | Latency samples, beam agreement, speed-of-light estimate |
| `diagnose` | Investigate one case from an existing e2e run | Optional GPU timeline, operator A/B and hardware counters |

The migrated adapters support SID-GR and SGLang native beam search. TensorRT-LLM,
FlashRec and vLLM-GR remain explicitly disabled until their adapters are implemented.

## Quick start

Run commands from this directory. The controller needs Python 3.10+ and PyYAML;
each solution runs in its own Python environment or Docker image.

```bash
python3 -m pip install -r requirements.txt
export QWEN3_0_6B_MODEL=/absolute/path/to/Qwen3-0.6B
export GR_DECODE_ATTEN_ROOT=/absolute/path/to/gr_decode_atten

# Use existing, prepared solution environments.
export SID_GR_PYTHON=/absolute/path/to/sid-gr-env/bin/python
export SGLANG_PYTHON=/absolute/path/to/sglang-env/bin/python

python3 bench.py plan --smoke
python3 bench.py e2e --smoke
```

SID-GR reuses `../sid-gr-inference` and its external CuTe attention kernel. Its
environment needs the original project's dependencies, including Torch, Triton,
safetensors, the prefill backend, CUTLASS DSL, Quack and TVM FFI. SGLang needs a
version with the native `SamplingParams.beam_width` API. Set `SID_GR_SOURCE` or
`SGLANG_SOURCE` to use a different checkout; an empty SGLang source uses the
installed package. No runner downloads models or changes framework installations.

For Docker, replace the Python environment settings with:

```bash
export SID_GR_RUNTIME=docker SGLANG_RUNTIME=docker
export SID_GR_IMAGE=your-prepared-gr-image
export SGLANG_IMAGE=your-prepared-sglang-image
python3 bench.py e2e --smoke
```

Images must already exist locally. Tags are resolved to immutable image IDs for
each run. The selected GPU is resolved to its UUID before launching Docker, so
session and Docker-host GPU indices can differ. Source and model directories are
mounted read-only; job artifacts and caches are writable. Custom dependency mounts
belong in `runtime.mounts`.
`entrypoints/Dockerfile` is an optional repair recipe for a SGLang base with CUDA
13.0 runtime headers but a newer nvcc/CRT. It requires an explicit `BASE_IMAGE`;
it is not a general SGLang installer. `Dockerfile.profile` optionally adds Nsight
tools from a `PROFILER_IMAGE`; pass its installation directories as `NSYS_ROOT`
and `NCU_ROOT`. Choose a profiler version supporting the installed CUDA driver.
`entrypoints/sid_gr_inference/Dockerfile` pins CUTLASS DSL 4.5.1 and Quack 0.4.1
for the existing external GR kernel, which uses APIs removed in DSL 4.6.
Keep these dependencies separate from SGLang's environment.

## Configuration

| File | Owns |
| --- | --- |
| `scenarios/*.yml` | Checkpoint, precision, token inputs, context/batch/beam grids, generation semantics |
| `solutions/*.yml` | Adapter, source, isolated runtime, framework options and supported operator probes |
| `benchmark.yml` | Selected scenarios/solutions, baseline, repetitions, GPU, diagnosis defaults and optional hardware ceiling |

`${NAME:-default}` values resolve from the environment. Relative scenario/solution
paths resolve from this directory; `output_dir` resolves beside the run config.
Copy `benchmark.yml` to an ignored `benchmark.local.yml` for local selections.

```bash
python3 bench.py plan --config benchmark.local.yml
python3 bench.py e2e --case 'qwen3_0_6B/*' --repeats 30
python3 bench.py e2e --smoke --case qwen3_0_6B/ctx1000_b1_w256
python3 bench.py e2e --smoke --solutions sglang
```

The full default grid includes Qwen3-1.7B; set `QWEN3_1_7B_MODEL` before running
it, or select the 0.6B cases. `--smoke` takes the first selected case with two
warmups and three measurements. These are functional checks, not stable rankings.
Initial framework compilation and graph capture can take several minutes. Each
solution keeps a writable compilation cache within the run, shared across cases
and diagnosis. A full run needs more disk space than the small JSON reports alone.

Synthetic inputs are seeded vocabulary IDs, excluding BOS/EOS/PAD values listed
in the model config. For representative requests, use `inputs.type: jsonl`
and `inputs.path` with rows such as
`{"request_id": "request-0", "input_ids": [12, 34, 56]}`. Each selected row must
have exactly the requested context length. No tokenizer, padding or truncation
changes the workload. One serialized input file is shared across solutions.

## Measurement and correctness

The e2e timer starts with CPU token IDs and ends with completed CPU beam IDs and
scores. It includes submission, device transfers, scheduling, output conversion
and synchronization. Model loading, compilation during warmup, validation and
report generation are outside the timer. Every measured repeat is validated.
Median and IQR describe offline batch latency; requests/s is batch size divided
by median latency. This is not an online load generator or a tail-latency SLA.

Both adapters use BF16 weights/KV, no quantization, no prefix reuse, fixed output
length, all beams, no catalog constraints and cumulative log-probabilities.
Three output tokens mean prefill selection plus two SID-GR decode steps.
SGLang's verified native score is mean log-probability; the adapter multiplies
by output length. Fixed equal lengths preserve beam ranking. Unsupported API
versions or semantics produce an explicit status, never a sampling fallback.

The suite works with the unmodified SID-GR implementation. For a SID-GR checkout
that includes the optional fused normalization optimization,
`GR_INFERENCE_FUSED_BEAM_LOGSUMEXP=1` enables it (default `0`). Enabling it against
a checkout without the kernel produces an unsupported result. For a comparison
within one run, copy the SID-GR solution YAML, give it a new name and set that
flag in `env`; both variants can use the same adapter.

Structural validation checks request identity, beam counts, token lengths/ranges,
unique sequences, finite scores and descending score order. Reports separately
show top-1 agreement, beam overlap and shared-beam score differences against the
baseline. Numerical disagreement remains visible; it does not silently remove a
fast result or claim model quality equivalence. There is no external accuracy
oracle or configurable accuracy gate in this first version.

## Reports and speed of light

Every run creates an ignored `results/<run-id>/` directory containing:

```text
manifest.json       # Effective configuration, source/model/input identities
workloads/          # Exact shared input IDs
jobs/               # Requests, commands, logs, raw latency samples and all beams
cache/              # Per-solution compilation caches, reused during diagnosis
summary.json        # Derived comparison data
report.md           # Filled reports/template.md
diagnostics/        # Separate optional captures and diagnosis reports
```

The report includes a brief speed-of-light section. Configure `hardware` with
the exact GPU name, a **dense BF16** TFLOP/s ceiling and its source. The initial
estimator counts Qwen3 projection/MLP/LM-head and causal-attention matrix FLOPs,
then divides by that ceiling. It reports this optimistic compute floor,
floor/measured ratio and gap. Memory traffic, reductions, selection and host
costs are omitted, so this is a partial lower bound, not an attainable speedup
or measured GPU utilization. Missing or mismatched hardware data is explicitly
unavailable. GPU memory reporting likewise stays unavailable until there is a
comparable collector covering both engines' process trees.

```bash
python3 bench.py report --run results/<run-id>
python3 bench.py report --run results/<run-id> --leaderboard leaderboard.md
```

The leaderboard contains one compatible run; raw runs remain intact. Historical
benchmark numbers use different timing boundaries and are not imported as new
measurements. `source.ref` records intended tracking; actual commits, dirty-tree
hashes, package versions, checkpoint hashes and Docker image IDs are recorded
separately. `source.expected_commit` can require a clean pinned checkout.

## Optional diagnosis

```bash
python3 bench.py diagnose --run results/<run-id> --case qwen3_0_6B/ctx1000_b1_w64
python3 bench.py diagnose --run results/<run-id> --case qwen3_0_6B/ctx1000_b1_w64 \
  --no-timeline --kernels beam_logsumexp
python3 bench.py diagnose --run results/<run-id> --case qwen3_0_6B/ctx1000_b1_w64 \
  --kernels beam_logsumexp --counters beam_logsumexp
```

Timeline capture needs `nsys` inside the solution runtime; counters need `ncu`
and GPU counter access. Missing tools/capabilities are recorded as unsupported.
Diagnosis checks for source, checkpoint, suite, GPU/driver and runtime drift and
uses the original inputs/settings. Rerun e2e after changing the implementation.
Profiler timings never replace headline e2e results.

The timeline summarizes exact kernel names within `sid_bench/measured` NVTX
ranges and their GPU-active time union. It does not infer prefill/decode stages
from ambiguous names. When the selected SID-GR checkout includes the optional
fused normalizer, the first operator probe compares its
`beam_logsumexp` with PyTorch on the same shape/dtype, reporting error, eager and
CUDA-graph GPU time and extra live tensor memory. Buffers are reused with warm
caches. This reference is not presented as SGLang's internal operator. Counter
capture is separate from the operator timing comparison. A checkout without the
optional kernel records this probe as unsupported; e2e and timeline modes remain
available.

## Adding and refreshing solutions

Add one `entrypoints/<name>/adapter.py` with `check_support(case, solution)` and
an `Adapter(request)` exposing `generate(workload)`, `synchronize()`, `close()`
and `provenance()`. `generate` returns ordered request rows with
`beams: [{token_ids: [...], score: float}]`. Keep API conversion here; the shared
worker owns timing and validation. Add a solution YAML and select it in the run
config. Optional operator probes are separate capabilities.
Adapters with GPU subprocesses can supply `profile_start()` / `profile_stop()`
to start and flush profiler collection in those processes. The worker invokes
these hooks outside measured intervals; SGLang uses its native CUDA profiler API.

`python3 bench.py updates` checks configured upstream Git refs without upgrading
anything. Prepare/pin the desired environment, run a smoke case, then run the
full grid and regenerate the report. Scheduling and automatic upgrades are not
part of this initial suite.

```bash
python3 -m unittest discover -s tests
```
