# SID-GR inference benchmarks

Compare latency, throughput and beam parity on identical token inputs.
Use separate Nsight captures to inspect GPU kernels.

## Commands

Run from this directory. Use `python3 bench.py COMMAND --help` for options.

| Command | Inputs | Work and output |
| --- | --- | --- |
| `plan` | `--config config/runs/default.yml` | List cases, candidates and measurement counts. No GPU work. |
| `e2e` | `--config config/runs/default.yml` | Run candidates serially. Save inputs, results and reports in a new run directory. |
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
export QWEN3_0_6B_MODEL=/absolute/path/to/Qwen3-0.6B
export GR_DECODE_ATTEN_ROOT=/absolute/path/to/gr_decode_atten
export SID_GR_PYTHON=/absolute/path/to/sid-gr-env/bin/python
export SGLANG_PYTHON=/absolute/path/to/sglang-env/bin/python
nvidia-smi -L
export BENCHMARK_GPU=GPU-...  # Use the full UUID from nvidia-smi.
# The default hardware profile is RTX PRO 6000 Blackwell Server Edition.
python3 bench.py plan --config config/runs/default.yml --smoke
python3 bench.py e2e --config config/runs/default.yml --smoke
```

Use the checked-in config directly, or copy `config/runs/default.yml` to
`config/runs/local.yml` to change scenarios, candidates, hardware or execution
policy. This optional, Git-ignored copy is your local run configuration;
pass its path with `--config`. Select a matching `config/hardware/` profile
before running on a different GPU.

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
These are all environment overrides in the supplied configurations; paths
resolve from this directory. Defaults apply when the variable is unset.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `BENCHMARK_GPU` | Empty | Required full GPU UUID for execution. |
| `QWEN3_0_6B_MODEL` | `../sid-gr-inference/models/Qwen3-0.6B` | Local 0.6B checkpoint. |
| `QWEN3_1_7B_MODEL` | `../sid-gr-inference/models/Qwen3-1.7B` | Local 1.7B checkpoint. |
| `SID_GR_SOURCE` | `../sid-gr-inference` | SID-GR checkout. |
| `SGLANG_SOURCE` | Empty | Upstream SGLang checkout; empty uses the installed package. |
| `SGLANG_BEAM_SOURCE` | `../sglang_beam_search` | Pinned README baseline checkout. |
| `SID_GR_RUNTIME`, `SGLANG_RUNTIME` | `local` | Select `local` or `docker`. |
| `SID_GR_PYTHON`, `SGLANG_PYTHON` | `python3` | Python executable inside the chosen runtime. |
| `SID_GR_IMAGE`, `SGLANG_IMAGE` | Empty | Image name or ID; required for Docker. |
| `GR_DECODE_ATTEN_ROOT` | Empty | External GR kernel checkout. Forwarded to workers and mounted in Docker. |

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
| `config/scenarios/*.yml` | Model, inputs, context/batch/beam grid, generation rules and repeat counts. |
| `config/solutions/*.yml` | Adapter, source checkout, runtime and engine settings. |
| `config/hardware/*.yml` | Required GPU SKU; optional sourced dense BF16 compute ceiling. |
| `config/runs/default.yml` | Selected scenarios, candidates, SGLang reference, GPU UUID and execution policy. |
| `config/runs/readme.yml` | Qwen3-1.7B comparison against the pinned SGLang beam-search branch. |

A scenario defines reusable cases. Each `e2e` invocation creates a new run.
Paths in scenario/candidate files resolve from this directory.
`output_dir` resolves beside the run config (`../../results` in supplied configs).

| Measurement | Definition |
| --- | --- |
| Latency | Median completed-batch wall time; IQR is the 75th minus 25th percentile, in ms. |
| Δ latency % | `100 × (candidate median / SGLang median − 1)`; negative means faster. |
| Throughput | Batch size / median latency. Offline requests/s. |
| Timing scope | CPU token IDs through completed CPU beam IDs/scores, including synchronization. |
| Excluded work | Setup, warmup, validation and serialization. |
| Structural checks | Request IDs, beam counts, lengths, unique sequences, finite scores and score order. |
| Compute floor | Partial matrix-compute estimate. Not GPU utilization or a promised speedup. |

Adapters require BF16 weights/KV, fixed-length output, all beams, no quantization,
no prefix reuse and no catalog constraints. Unsupported settings are explicit.
Synthetic inputs use a fixed seed and exclude BOS/EOS/PAD. For real inputs, set
`inputs.type: jsonl` and `inputs.path`; rows contain `request_id` and `input_ids`.
JSONL uses the first `batch_size` rows. Each input must match the case's context
length; no token padding or truncation.

### Output parity

For each case, compare every request in every measured repeat with that request
in the reference candidate's first measured repeat from the same run. Inputs
must have identical checksums. Reference rows therefore also check repeat
consistency. Failed jobs or a missing reference leave parity unavailable.

| Metric | Meaning |
| --- | --- |
| Exact | All beam sequences, their order and cumulative scores match exactly; no tolerance. |
| Top-1 % | Percentage of request/repeat pairs with identical rank-1 token IDs. |
| Mean / min overlap % | Average / minimum fraction of beam sequences shared with the reference, ignoring rank. |
| Rank match % | Average fraction of beams with identical token IDs at the same rank. |
| Max score Δ | Largest absolute cumulative-score difference among shared sequences. |

These cover the existing [offline accuracy benchmark](../sid-gr-inference/scripts/run_offline_accuracy_benchmark.sh)'s
top-1, mean overlap and ordered-match metrics; structural checks enforce its
output-length and token-budget requirements. SGLang's verified mean-logprob
scores are multiplied by fixed output length to match SID-GR cumulative scores.
`success` means execution and structural validation passed; it does not imply
cross-engine parity. Mismatches remain visible beside performance results.

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
replace only executed rows; failures replace older successes. Hardware, checkpoint,
scenario and JSONL input revisions get separate groups. Parity uses each row's source
run. See [harness internals](harness/README.md) for the artifact layout.

## Reproduce the SID-GR README comparison

`sglang_beam_search` pins the README's `feature/beam_search` source commit.
The upstream `sglang` candidate requires native `SamplingParams.beam_width`.

```bash
export QWEN3_1_7B_MODEL=/absolute/path/to/Qwen3-1.7B
export SGLANG_BEAM_SOURCE=/absolute/path/to/sglang_beam_search
# Check out source.expected_commit from config/solutions/sglang_beam_search.yml.
docker build -f entrypoints/sglang/Dockerfile.beam_search \
  -t sid-gr-benchmark:readme "$SGLANG_BEAM_SOURCE"
export SID_GR_RUNTIME=docker SGLANG_RUNTIME=docker
export SID_GR_IMAGE=sid-gr-benchmark:readme SGLANG_IMAGE=sid-gr-benchmark:readme
export SID_GR_PYTHON=python3 SGLANG_PYTHON=python3
# Set GR_DECODE_ATTEN_ROOT, BENCHMARK_GPU and the hardware profile as above.
python3 bench.py e2e --config config/runs/readme.yml --case 'qwen3_1_7B/*_w256'
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
