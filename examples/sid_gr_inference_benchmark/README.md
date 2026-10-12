# SID-GR inference benchmarks

Compare latency, throughput and beam parity on identical token inputs.
Use separate Nsight captures to inspect GPU kernels.

Four independent [entrypoints](entrypoints/README.md) cover SID-GR and the three
SGLang solutions. Each workload has one measurement row per selected candidate.

[Leaderboards](leaderboards/README.md) · [Published example](#published-example)

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

The default compares Qwen3-0.6B and Qwen3-1.7B across all four candidates:
48 workloads and 192 jobs, with five warmups and 30 measured repeats per job.

## Quick start

The controller needs Python 3.10+ and the small dependencies in `requirements.txt`.
Prepare candidate runtimes first. `e2e` downloads the scenario's pinned Hugging Face
checkpoint when needed; `plan` and `report` do not download models or import GPU frameworks.
For the two beam branches, use the pinned source checkouts listed under
[Compare all four candidates](#compare-all-four-candidates).

```bash
python3 -m pip install -r requirements.txt
export SID_GR_RUNTIME=local SGLANG_RUNTIME=local
export SGLANG_BEAM_RUNTIME=local SGLANG_BEAM_NEW_RUNTIME=local
export SID_GR_PYTHON=/absolute/path/to/sid-gr-env/bin/python
export SGLANG_PYTHON=/absolute/path/to/sglang-env/bin/python
export SGLANG_BEAM_PYTHON=/absolute/path/to/sglang-beam-env/bin/python
export SGLANG_BEAM_NEW_PYTHON=/absolute/path/to/sglang-beam-new-env/bin/python
export SGLANG_BEAM_SOURCE=/absolute/path/to/sglang_beam_search
export SGLANG_BEAM_NEW_SOURCE=/absolute/path/to/sglang_beam_search_new
nvidia-smi -L
export BENCHMARK_GPU=GPU-...  # Use the full UUID from nvidia-smi.
# The default hardware profile is RTX PRO 6000 Blackwell Server Edition.
python3 bench.py plan --config config/runs/default.yml --smoke
python3 bench.py e2e --config config/runs/default.yml --smoke
```

Remove `--smoke` to run the full matrix for both models.

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
| `runtime.build_recipe` | Optional Dockerfile link shown beside the recorded image identity. |
| `runtime.mounts` | Extra directories mounted read-only at the same paths in Docker. |
| `env` | Environment variables forwarded to the worker. |
| `source.path` | Engine checkout. Empty SGLang path uses the installed package. |
| `source.expected_commit` | Require a clean checkout at this commit. |
| `engine.prefix_cache` | Candidate-owned prefix reuse; supplied candidates enable it. |

`${NAME:-default}` reads the controller's environment when loading YAML.
Paths resolve from this directory; defaults apply when a variable is unset.
Each candidate has independent source and runtime settings.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BENCHMARK_GPU` | Empty | Required full GPU UUID. |
| `QWEN3_0_6B_MODEL`, `QWEN3_1_7B_MODEL` | Empty | Optional local copy, verified against the pinned Hugging Face revision. |
| `GR_DECODE_ATTEN_ROOT` | `../../corelib/gr_decode_atten` | [Tracked GR kernel](../../corelib/gr_decode_atten), used only by SID-GR. |

For each prefix below, `_RUNTIME` defaults to `local`, `_PYTHON` to `python3`, and
`_IMAGE` to the listed image. For example, `SGLANG_BEAM_NEW_RUNTIME=docker` selects
Docker for the new beam branch without changing the other candidates.

| Candidate | Variable prefix | `_SOURCE` default | `_IMAGE` default |
| --- | --- | --- | --- |
| `sid_gr_inference` | `SID_GR` | `../sid-gr-inference` | `sid-gr-benchmark:beam-af4e3f4-cu130` |
| `sglang` | `SGLANG` | Empty: installed package | [Pinned public digest](config/solutions/sglang.yml) |
| `sglang_beam_search` | `SGLANG_BEAM` | `../sglang_beam_search` | `sid-gr-benchmark:beam-af4e3f4-cu130` |
| `sglang_beam_search_new` | `SGLANG_BEAM_NEW` | `../sglang_beam_search_new` | `sid-gr-benchmark:beam-df191d6-cu130` |

Build or pull images before execution; the suite freezes local image IDs and never
pulls implicitly. SID-GR shares the old branch's image. The new branch has its own
[recipe](entrypoints/sglang_beam_search_new/Dockerfile) with matching dependencies.

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
| `config/runs/default.yml` | Both models, all four candidates, public SGLang reference, GPU UUID and execution policy. |
| `config/runs/readme.yml` | Qwen3-1.7B comparison against the pinned old SGLang beam branch. |
| `config/runs/comparison.yml` | Equivalent to the default; retained for existing comparison commands. |

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

Qwen3-0.6B and Qwen3-1.7B: 24 workloads per model × four engines. All 192 jobs
passed execution and structural checks, with 5 warmups and 30 measured repeats
per job (5,760 timing samples). All candidates use 10 CPU cores and enable prefix
caching. The saved `comparison.yml` profile is equivalent to the current default.

Representative median batch latency at **context 1000 / batch 1 / beam 256**;
lower is better. The linked reports contain all 48 workloads.

| Candidate | Qwen3-0.6B ms | Qwen3-1.7B ms |
| --- | ---: | ---: |
| `sid_gr_inference` | 10.920 | 15.868 |
| `sglang` | 34.160 | 46.678 |
| `sglang_beam_search` | 27.903 | 34.228 |
| `sglang_beam_search_new` | 25.589 | 31.601 |

| Reports | Raw data |
| --- | --- |
| [Full run](results/20261012T113832092061Z-c0ad33/report.md) · [0.6B leaderboard](leaderboards/qwen3_0_6B.md) · [1.7B leaderboard](leaderboards/qwen3_1_7B.md) · [GPU diagnosis](results/20261012T113832092061Z-c0ad33/report.md#gpu-diagnosis) | [Archive](results/20261012T113832092061Z-c0ad33/raw_artifacts.tar.gz) |

Top-1 outputs agree for the representative workload above. Other workloads
have top-1 mismatches; all parity differences remain visible in the reports.
Nsight compares all four engines on Qwen3-1.7B at context 1000 / batch 1 / beam 256.

One archive contains frozen metadata, inputs, requests, logs, results and native
Nsight/SQLite traces. Compilation caches are omitted. Extract it before rebuilding:

```bash
benchmark_run=results/20261012T113832092061Z-c0ad33
tar -xzf "$benchmark_run/raw_artifacts.tar.gz" -C "$benchmark_run"
python3 bench.py report --run "$benchmark_run"
```

Measurements use RTX PRO 6000 Blackwell; the original README's H100 timings
are not directly comparable.

Suite-root `leaderboards/` keeps the latest non-smoke result per case/candidate. Partial runs
replace only executed rows; failures replace older successes. Hardware, checkpoint,
scenario and JSONL input revisions get separate groups. Parity uses each row's source
run. See [harness internals](harness/README.md) for the artifact layout.

## Compare all four candidates

| SGLang candidate | Pinned source | Beam implementation |
| --- | --- | --- |
| Public/native | [Upstream e6072935](https://github.com/sgl-project/sglang/tree/e6072935bfa1036d4e20e8ed042ce1ff09872f57) in the public image | Native `beam_width` API |
| Old branch | [feature/beam_search @ af4e3f42](https://github.com/cswuyg/sglang/tree/af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d) | Server beam mode; `n` sets width |
| New branch | [feature/beam_search_update_0801 @ df191d67](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891) | Request `use_beam_search`; FlashInfer cascade enabled |

The new candidate sets `enable_beam_cascade_attention: true` and
`decode_attention_backend: flashinfer`. Each version uses its matching dependencies;
performance differences include runtime changes as well as beam implementation changes.

```bash
export SGLANG_BEAM_SOURCE=/absolute/path/to/sglang_beam_search
export SGLANG_BEAM_NEW_SOURCE=/absolute/path/to/sglang_beam_search_new
git clone https://github.com/cswuyg/sglang.git "$SGLANG_BEAM_SOURCE"
git -C "$SGLANG_BEAM_SOURCE" checkout af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d
git clone https://github.com/cswuyg/sglang.git "$SGLANG_BEAM_NEW_SOURCE"
git -C "$SGLANG_BEAM_NEW_SOURCE" checkout df191d672f1955d273bf56fe21684b60dcf6a891
docker pull lmsysorg/sglang@sha256:b5c488694d74c13587af6057b90220498c4869571c8e1e3c43dd6f43ae721306
docker build -f entrypoints/sglang_beam_search/Dockerfile \
  -t sid-gr-benchmark:beam-af4e3f4-cu130 "$SGLANG_BEAM_SOURCE"
docker build -f entrypoints/sglang_beam_search_new/Dockerfile \
  -t sid-gr-benchmark:beam-df191d6-cu130 "$SGLANG_BEAM_NEW_SOURCE"
export SID_GR_RUNTIME=docker SGLANG_RUNTIME=docker
export SGLANG_BEAM_RUNTIME=docker SGLANG_BEAM_NEW_RUNTIME=docker
export SID_GR_PYTHON=python3 SGLANG_PYTHON=python3
export SGLANG_BEAM_PYTHON=python3 SGLANG_BEAM_NEW_PYTHON=python3
# Set BENCHMARK_GPU and the hardware profile as above.
python3 bench.py plan --config config/runs/default.yml
python3 bench.py e2e --config config/runs/default.yml
```

This profile defines **48 cases × 4 candidates = 192 jobs**: two models, context
lengths 1000/5000, batches 1/2/4/8, and beam widths 64/128/256. Each job uses five
warmups and 30 measured repeats. Add `--smoke` for one case, or select cases and
candidates with `--case` and `--solutions`.

For the original two-candidate README matrix instead:

```bash
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
