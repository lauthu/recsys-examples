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

The controller needs Python 3.10+ and the small dependencies in `requirements.txt`.
Prepare candidate runtimes first. `e2e` downloads the scenario's pinned Hugging Face
checkpoint when needed; `plan` and `report` do not download models or import GPU frameworks.

```bash
python3 -m pip install -r requirements.txt
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
| `engine.prefix_cache` | Candidate-owned prefix reuse; supplied candidates enable it. |

`${NAME:-default}` reads the controller's environment when loading YAML.
These are all environment overrides in the supplied configurations; paths
resolve from this directory. Defaults apply when the variable is unset.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `BENCHMARK_GPU` | Empty | Required full GPU UUID for execution. |
| `QWEN3_0_6B_MODEL`, `QWEN3_1_7B_MODEL` | Empty | Optional local copy, verified against the scenario's pinned Hugging Face revision. |
| `SID_GR_SOURCE` | `../sid-gr-inference` | SID-GR checkout. |
| `SGLANG_SOURCE` | Empty | Upstream SGLang checkout; empty uses the installed package. |
| `SGLANG_BEAM_SOURCE` | `../sglang_beam_search` | Pinned README baseline checkout. |
| `SID_GR_RUNTIME`, `SGLANG_RUNTIME` | `local` | Select `local` or `docker`. |
| `SID_GR_PYTHON`, `SGLANG_PYTHON` | `python3` | Python executable inside the chosen runtime. |
| `SID_GR_IMAGE` | `sid-gr-benchmark:beam-af4e3f4-cu130` | Local image built from the [shared recipe](entrypoints/sglang/Dockerfile.beam_search). |
| `SGLANG_IMAGE` | [Native candidate: pinned public digest](config/solutions/sglang.yml); beam candidate: `sid-gr-benchmark:beam-af4e3f4-cu130` | Runtime image for the selected SGLang candidate. |
| `GR_DECODE_ATTEN_ROOT` | `../../corelib/gr_decode_atten` | [Tracked GR kernel](../../corelib/gr_decode_atten); forwarded to workers and mounted in Docker. |

For Docker, set:

```bash
export SID_GR_RUNTIME=docker SGLANG_RUNTIME=docker
export SID_GR_PYTHON=python3 SGLANG_PYTHON=python3
```

Build or pull the selected image before execution; the suite freezes its local image
ID and never pulls implicitly. The native SGLang candidate uses the public image;
the pinned beam-search comparison uses the shared recipe below for both engines.

The supplied hardware profile assigns **10 physical CPU cores**, one allowed logical
CPU per core. Local workers use `taskset`; Docker uses the same CPU set. The manifest
records the assigned affinity, host topology, GPU UUID and driver. Fewer available
cores or a different GPU SKU fail before execution.

The scenario's model ID and immutable revision identify the checkpoint:

| Scenario | Hugging Face snapshot |
| --- | --- |
| `qwen3_0_6B` | [Qwen/Qwen3-0.6B @ c1899de](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) |
| `qwen3_1_7B` | [Qwen/Qwen3-1.7B @ 70d244c](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) |

`e2e` verifies local overrides against Hub file hashes too; Hub metadata must be
reachable. Reports retain checkpoint hashes and source repository/commit links.

## Configuration and measurements

| File | Defines |
| --- | --- |
| `config/scenarios/*.yml` | Model, inputs, context/batch/beam grid, generation rules and repeat counts. |
| `config/solutions/*.yml` | Adapter, source checkout, runtime and engine settings. |
| `config/hardware/*.yml` | Required GPU SKU and physical CPU core count; optional dense BF16 compute ceiling. |
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

Adapters require BF16 weights/KV, fixed-length output, all beams, no quantization
and no catalog constraints. Unsupported settings are explicit.
Warmup and measured repeats reuse identical inputs. With `engine.prefix_cache: true`,
measurements include warm prefix reuse; set it to `false` in each candidate YAML
for a comparison without prefix reuse.

The [SID-GR candidate](config/solutions/sid_gr_inference.yml) records its optimization
defaults explicitly; edit that YAML to change them:

| Setting | Default |
| --- | --- |
| Prefill backend | `GR_INFERENCE_PREFILL_BACKEND=auto` |
| Prefill/decode CUDA graphs; flattened linear | Corresponding `GR_INFERENCE_DISABLE_*=0` |
| TRT-LLM kernel JIT | `GR_INFERENCE_GR_TRTLLM_KERNELS_JIT=1` |
| Fused QK norm/RoPE and packed GEMM | `GR_INFERENCE_TRTLLM_QK_NORM_ROPE_PHASE=all`, `GR_INFERENCE_TRTLLM_PACKED_GEMM_SCOPE=all` |

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

## Published example

Qwen3-1.7B: 8 cases × 2 engines, with 5 warmups and 30 measured repeats.
Nsight compares both engines at context 1000 / batch 1 / beam 256.

| Example | Reports | Raw data |
| --- | --- | --- |
| README comparison + GPU diagnosis | [Run](results/20261012T074633875336Z-dcc97c/report.md) · [Diagnosis](results/20261012T074633875336Z-dcc97c/report.md#gpu-diagnosis) · [Leaderboard](leaderboards/qwen3_1_7B.md) | [Archive](results/20261012T074633875336Z-dcc97c/raw_artifacts.tar.gz) |

One archive contains frozen metadata, inputs, requests, logs, results and native
Nsight/SQLite traces. Compilation caches are omitted. Extract it before rebuilding:

```bash
benchmark_run=results/20261012T074633875336Z-dcc97c
tar -xzf "$benchmark_run/raw_artifacts.tar.gz" -C "$benchmark_run"
python3 bench.py report --run "$benchmark_run"
```

The report shows output mismatches alongside performance. Measurements use RTX
PRO 6000 Blackwell; the original README's H100 timings are not directly comparable.

Suite-root `leaderboards/` keeps the latest non-smoke result per case/candidate. Partial runs
replace only executed rows; failures replace older successes. Hardware, checkpoint,
scenario and JSONL input revisions get separate groups. Parity uses each row's source
run. See [harness internals](harness/README.md) for the artifact layout.

## Reproduce the SID-GR README comparison

`sglang_beam_search` pins the README's `feature/beam_search` source commit.
The upstream `sglang` candidate requires native `SamplingParams.beam_width`.

```bash
export SGLANG_BEAM_SOURCE=/absolute/path/to/sglang_beam_search
git clone https://github.com/cswuyg/sglang.git "$SGLANG_BEAM_SOURCE"
git -C "$SGLANG_BEAM_SOURCE" checkout af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d
docker build -f entrypoints/sglang/Dockerfile.beam_search \
  -t sid-gr-benchmark:beam-af4e3f4-cu130 "$SGLANG_BEAM_SOURCE"
export SID_GR_RUNTIME=docker SGLANG_RUNTIME=docker
export SID_GR_PYTHON=python3 SGLANG_PYTHON=python3
# Set BENCHMARK_GPU and the hardware profile as above.
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
