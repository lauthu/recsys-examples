# SID-GR inference benchmarks

Compare latency, throughput and beam parity on identical token inputs.
Use separate Nsight captures to inspect GPU kernels.

## Commands

Run from this directory. Use `python3 bench.py COMMAND --help` for options.

| Command | Inputs | Work and output |
| --- | --- | --- |
| `plan` | `--config run.local.yml` | List cases, candidates and measurement counts. No GPU work. |
| `e2e` | `--config run.local.yml` | Run candidates serially. Save inputs, results and reports in a new run directory. |
| `report` | `--run results/RUN_ID` | Rebuild the run report, saved diagnosis reports and scenario leaderboards. No GPU work. |
| `diagnose` | `--run results/RUN_ID --case CASE_ID` | Profile successful candidates for one saved case. Save kernel tables and Nsight traces. |

`plan` and `e2e` accept `--case 'qwen3_0_6B/*'` and `--solutions NAME ...`.
`--smoke` selects one case with two warmups and three measurements.
Smoke and profiled results never enter scenario rankings.

## Quick start

The controller needs Python 3.10+ and PyYAML. Prepare model files and candidate
runtimes first; the suite does not download models or install frameworks.

```bash
python3 -m pip install -r requirements.txt
cp run.template.yml run.local.yml
export QWEN3_0_6B_MODEL=/absolute/path/to/Qwen3-0.6B
export GR_DECODE_ATTEN_ROOT=/absolute/path/to/gr_decode_atten
export SID_GR_PYTHON=/absolute/path/to/sid-gr-env/bin/python
export SGLANG_PYTHON=/absolute/path/to/sglang-env/bin/python
nvidia-smi -L
export BENCHMARK_GPU=GPU-...  # Use the full UUID from nvidia-smi.
# Select your GPU's hardware profile in run.local.yml.
python3 bench.py plan --config run.local.yml --smoke
python3 bench.py e2e --config run.local.yml --smoke
```

### Runtime settings

| Candidate setting | Meaning |
| --- | --- |
| `runtime.kind: local` | Start `runtime.python` as a subprocess. Default. |
| `runtime.kind: docker` | Start `runtime.python` inside `runtime.image`. |
| `runtime.image` | Prepared local image. Tags are frozen to image IDs. |
| `runtime.mounts` | Extra directories mounted read-only at the same paths in Docker. |
| `env` | Environment variables forwarded to the worker. |
| `source.path` | Engine checkout. Empty SGLang path uses the installed package. |
| `source.expected_commit` | Require a clean checkout at this commit. |

`${NAME:-default}` reads the controller's environment when loading YAML.
For example, exported `GR_DECODE_ATTEN_ROOT` fills the SID-GR candidate's `env`
and `runtime.mounts`. Local workers receive it in their process environment;
Docker workers receive `--env GR_DECODE_ATTEN_ROOT=...` and the directory mount.

For Docker, set:

```bash
export SID_GR_RUNTIME=docker SGLANG_RUNTIME=docker
export SID_GR_IMAGE=your-prepared-gr-image SGLANG_IMAGE=your-prepared-sglang-image
export SID_GR_PYTHON=python3 SGLANG_PYTHON=python3
```

Both runtimes use the selected GPU UUID and controller CPU affinity.
The manifest records GPU identity, driver, CPU model, topology and affinity.
Use the optional [SID-GR image recipe](entrypoints/sid_gr_inference/Dockerfile)
for the external GR kernel's CUTLASS DSL and Quack dependencies.

## Configuration and measurements

| File | Defines |
| --- | --- |
| `scenarios/*.yml` | Model, inputs, context/batch/beam grid, generation rules and repeat counts. |
| `solutions/*.yml` | Adapter, source checkout, runtime and engine settings. |
| `hardware/*.yml` | Required GPU SKU; optional sourced dense BF16 compute ceiling. |
| `run.template.yml` | Selected scenarios, candidates, reference, GPU UUID and execution policy. |

A scenario defines reusable cases. Each `e2e` invocation creates a new run.
Paths in scenario/candidate files resolve from this directory.
`output_dir` resolves beside the run template.

| Measurement | Definition |
| --- | --- |
| Latency | Median completed-batch wall time, with IQR. |
| Throughput | Batch size / median latency. Offline requests/s. |
| Timing scope | CPU token IDs through completed CPU beam IDs/scores, including synchronization. |
| Excluded work | Setup, warmup, validation and serialization. |
| Structural checks | Request IDs, beam counts, lengths, unique sequences, finite scores and score order. |
| Parity | Exact ordered beams/scores; top-1 agreement, minimum overlap and shared-beam score difference. |
| Compute floor | Partial matrix-compute estimate. Not GPU utilization or a promised speedup. |

Adapters require BF16 weights/KV, fixed-length output, all beams, no quantization,
no prefix reuse and no catalog constraints. Unsupported settings are explicit.
Synthetic inputs use a fixed seed and exclude BOS/EOS/PAD. For real inputs, set
`inputs.type: jsonl` and `inputs.path`; rows contain `request_id` and `input_ids`.
Inputs must match the case's batch and context lengths; no padding or truncation.

## Published samples

| Sample | Scope | Report | Data |
| --- | --- | --- | --- |
| README comparison | Qwen3-1.7B, 8 cases × 2 engines, 5 warmups / 30 repeats | [Run](results/20261012T043440525028Z-9b035a/report.md) · [Leaderboard](results/leaderboards/qwen3_1_7B.md) | [Archive](results/20261012T043440525028Z-9b035a/raw_artifacts.tar.gz) |
| GPU diagnosis | Qwen3-0.6B SID-GR smoke; 1 profiled repeat | [Run](results/20261012T035001Z-153317/report.md) · [Diagnosis](results/20261012T035001Z-153317/diagnostics/20261012T035247Z-f176c8/diagnosis.md) | [Archive](results/20261012T035001Z-153317/raw_artifacts.tar.gz) |

Each sample commits Markdown reports and one archive. The archive contains frozen
metadata, inputs, requests, commands, logs and results; diagnosis samples include
native traces and SQLite exports. Compilation caches are omitted.
Extract an archive before rebuilding reports:

```bash
benchmark_run=results/20261012T043440525028Z-9b035a
# For the diagnosis sample, use results/20261012T035001Z-153317 instead.
tar -xzf "$benchmark_run/raw_artifacts.tar.gz" -C "$benchmark_run"
python3 bench.py report --run "$benchmark_run"
```

| Comparison finding | Result |
| --- | --- |
| Structural validity | All 16 jobs passed. |
| Exact cross-engine parity | Different in all 8 cases. |
| Top-1 | Mismatch at context 5000 / batch 4; cause not isolated. |
| Hardware | RTX PRO 6000 Blackwell; original SID-GR README used H100. |
| Interpretation | Workflow validation, not reproduction of H100 latencies or recommendation quality. |

Leaderboards keep the latest non-smoke result per case/candidate. Partial runs
replace only executed rows; failures replace older successes. Hardware and
scenario/checkpoint revisions get separate groups. Parity uses each row's source
run. See [harness internals](harness/README.md) for the artifact layout.

## Reproduce the SID-GR README comparison

`sglang_beam_search` pins the README's `feature/beam_search` source commit.
The upstream `sglang` candidate requires native `SamplingParams.beam_width`.

```bash
export QWEN3_1_7B_MODEL=/absolute/path/to/Qwen3-1.7B
export SGLANG_BEAM_SOURCE=/absolute/path/to/sglang_beam_search
# Check out source.expected_commit from solutions/sglang_beam_search.yml.
docker build -f entrypoints/sglang/Dockerfile.beam_search \
  -t sid-gr-benchmark:readme "$SGLANG_BEAM_SOURCE"
export SID_GR_RUNTIME=docker SGLANG_RUNTIME=docker
export SID_GR_IMAGE=sid-gr-benchmark:readme SGLANG_IMAGE=sid-gr-benchmark:readme
export SID_GR_PYTHON=python3 SGLANG_PYTHON=python3
# Set GR_DECODE_ATTEN_ROOT, BENCHMARK_GPU and the hardware profile as above.
python3 bench.py e2e --config run.readme.yml --case 'qwen3_1_7B/*_w256'
```

For profiling, install `nsys` in each runtime **before** the source `e2e` run.
The optional [profiling image recipe](entrypoints/Dockerfile.profile) can add it.
`diagnose` checks that source, checkpoint, suite, hardware and runtime are unchanged.

```bash
python3 bench.py diagnose --run results/RUN_ID --case qwen3_1_7B/ctx1000_b1_w256
```

## Development

See [harness workflow and modules](harness/README.md) for implementation details.
Run CPU checks with `python3 -m unittest discover -s tests`.
