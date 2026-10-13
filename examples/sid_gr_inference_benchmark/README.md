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

## Quick start

The controller needs Python 3.10+ and `requirements.txt`.
Prepare each candidate's runtime using its [solution YAML](#configuration).
Docker setup for the supplied example is [below](#compare-all-four-candidates).
`plan` and `report` need no GPU frameworks; `e2e` downloads pinned checkpoints when needed.

```bash
python3 -m pip install -r requirements.txt
nvidia-smi --query-gpu=name,uuid --format=csv
export BENCHMARK_GPU=GPU-...  # Select one device using its full UUID.
python3 bench.py plan --config config/runs/default.yml --smoke
python3 bench.py e2e --config config/runs/default.yml --smoke
```

The hardware YAML defines the required GPU type. The UUID selects the actual device,
even when CUDA or Docker remaps device numbers. Remove `--smoke` for the full run.

## Configuration

| File | Defines |
| --- | --- |
| [Scenarios](config/scenarios/) | Model snapshot, inputs, context/batch/beam grid, generation rules and repeat counts. |
| [Solutions](config/solutions/) | Adapter, source, runtime, backend environment variables and engine settings. |
| [Hardware](config/hardware/) | Required GPU type and CPU core allocation; optional dense BF16 compute ceiling. |
| [Default run](config/runs/default.yml) | Scenarios, candidates, baseline, hardware profile, GPU selection and execution policy. |
| [Four-engine run](config/runs/four_engine.yml) | Explicit four-engine profile, currently equivalent to the default. |
| [SID-GR README workload grid](config/runs/readme.yml) | Qwen3-1.7B with SID-GR and the old SGLang beam branch. |

A scenario defines reusable cases. Each `e2e` invocation creates a new run.
Copy `config/runs/default.yml` to the Git-ignored `config/runs/local.yml` for a custom
run; pass it with `--config`. Select a matching hardware profile for your machine.

Paths in scenario and solution files resolve from this directory.
`output_dir` resolves beside the run config. `${NAME:-default}` reads the controller's
environment; defaults apply when a variable is unset.

### Runtime settings

| Solution field | Meaning |
| --- | --- |
| `runtime.kind: local` | Run `runtime.python` in an existing local environment. |
| `runtime.kind: docker` | Run `runtime.python` inside a prepared `runtime.image`. |
| `runtime.image` | Local image tag or digest; frozen to an image ID before execution. |
| `runtime.build_recipe` | Optional Dockerfile link recorded with the image identity. |
| `runtime.mounts` | Extra directories mounted read-only at the same paths in Docker. |
| `source.path` | Engine checkout; an empty SGLang path uses the installed package. |
| `source.expected_commit` | Require a clean checkout at this commit. |
| `env` | Backend environment variables passed to the worker. |

Each solution YAML lists its environment overrides, defaults and engine options:
[SID-GR](config/solutions/sid_gr_inference.yml), [public SGLang](config/solutions/sglang.yml),
[old beam branch](config/solutions/sglang_beam_search.yml),
[new beam branch](config/solutions/sglang_beam_search_new.yml).
Build or pull images before execution; the suite never pulls implicitly.

Workers receive the CPU allocation from the hardware profile. Local workers use
`taskset`; Docker uses the same CPU set. The manifest records CPU topology,
affinity, GPU identity and driver. A mismatched GPU type or insufficient cores fails
before execution.

Model IDs, immutable revisions and optional local-path overrides live in the
[scenario YAMLs](config/scenarios/). `e2e` verifies local copies against Hub file hashes;
Hub metadata must be reachable. Reports retain checkpoint hashes and source identities.

## Measurements

| Measurement | Definition |
| --- | --- |
| Latency | Median completed-batch wall time; IQR is the 75th minus 25th percentile, in ms. |
| Δ latency % | `100 × (candidate median / baseline median − 1)`; negative means faster. The run YAML selects `baseline`. |
| Throughput | Batch size / median latency. Offline requests/s. |
| Timing scope | CPU token IDs through completed CPU beam IDs/scores, including synchronization. |
| Excluded work | Setup, warmup, validation and serialization. |
| Structural checks | Request IDs, beam counts, lengths, unique sequences, finite scores and score order. |
| Compute floor | Partial matrix-compute estimate. Not GPU utilization or a promised speedup. |

Adapters require BF16 weights/KV, fixed-length output, all beams, no quantization
and no catalog constraints. Unsupported settings are explicit.
Warmup and measured repeats reuse identical inputs. Each solution controls its
own optimizations; see its YAML for the configured behavior.

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

Top-1, mean overlap and rank match use the definitions from the existing
[offline accuracy benchmark](../sid-gr-inference/scripts/run_offline_accuracy_benchmark.sh).
Structural checks enforce its output-length and token-budget requirements.
SGLang's verified mean-logprob scores are multiplied by fixed output length to
match SID-GR cumulative scores. `success` means execution and structural validation
passed; cross-engine mismatches remain visible beside performance results.

## Published example

The saved run uses the [four-engine profile](config/runs/four_engine.yml), equivalent
to the current default. Frozen metadata retains its original name, `comparison.yml`.

| Setting | Recorded value / configuration |
| --- | --- |
| Models and grid | [Qwen3-0.6B](config/scenarios/qwen3_0_6B.yml) and [Qwen3-1.7B](config/scenarios/qwen3_1_7B.yml); contexts 1000/5000, batches 1/2/4/8, beams 64/128/256 |
| Hardware | [RTX PRO 6000 Blackwell Server Edition](config/hardware/rtx_pro_6000_blackwell_server.yml), 10 physical CPU cores |
| Measurement count | 48 workloads × four engines; five warmups and 30 timed repeats per job |
| Result | 192 successful jobs, 5,760 timing samples |
| Reference | Public SGLang |
| Candidate settings | All four solution YAMLs enable prefix caching for this run |

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

This run does not reproduce the original SID-GR README's H100 latency numbers.
That protocol disables prefix reuse, uses different input tokens and repeat counts,
and has narrower timing boundaries. The suite also times SID-GR input transfer and
both adapters' output conversion; the headline parity definitions are shared.

One archive contains frozen metadata, inputs, requests, logs, results and native
Nsight/SQLite traces. Model weights, runtime images and compilation caches are omitted.
Saved reports can be regenerated; identical timings or bit-for-bit outputs on a fresh
run are not guaranteed. Extract the archive before rebuilding:

```bash
benchmark_run=results/20261012T113832092061Z-c0ad33
tar -xzf "$benchmark_run/raw_artifacts.tar.gz" -C "$benchmark_run"
python3 bench.py report --run "$benchmark_run"
```

Suite-root `leaderboards/` keeps the latest non-smoke result per case/candidate. Partial runs
replace only executed rows; failures replace older successes. Hardware, checkpoint,
scenario and JSONL input revisions get separate groups. Parity uses each row's source
run. See [harness internals](harness/README.md) for the artifact layout.

### Compare all four candidates

The [solution YAMLs](config/solutions/) define source/runtime settings, backend
environment variables and engine options. These commands prepare Docker runtimes
for the example; each branch uses its matching dependencies.

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
# Select BENCHMARK_GPU and the matching hardware profile as above.
python3 bench.py plan --config config/runs/four_engine.yml
python3 bench.py e2e --config config/runs/four_engine.yml
```

For the original README's model/workload grid with two candidates:

```bash
python3 bench.py e2e --config config/runs/readme.yml --case 'qwen3_1_7B/*_w256'
```

This selects the workload grid only; it still uses this suite's inputs, timing
and configured solution settings.

For profiling, install `nsys` in each runtime **before** the source `e2e` run.
The optional [profiling image recipe](entrypoints/Dockerfile.profile) can add it.
`diagnose` checks that source, checkpoint, suite, hardware and runtime are unchanged.

```bash
python3 bench.py diagnose --run results/RUN_ID --case qwen3_1_7B/ctx1000_b1_w256
```

## Development

See [harness workflow and modules](harness/README.md) for implementation details.
Run CPU checks with `python3 -m unittest discover -s tests`.
