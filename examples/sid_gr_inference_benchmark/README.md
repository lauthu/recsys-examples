# SID-GR inference benchmarks

Compare latency, throughput and beam parity on identical token inputs.
Use separate Nsight captures to inspect GPU kernels.

Seven independent [entrypoints](entrypoints/README.md) cover SID-GR, three
SGLang solutions, vLLM-GR, TensorRT-LLM and FlashRec. Each workload has one
measurement row per selected candidate. The default profile retains four engines.

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
| [Seven-engine run](config/runs/seven_engine.yml) | Adds vLLM-GR, TensorRT-LLM and FlashRec; 48 cases / 336 jobs. |
| [SID-GR versus new solutions](config/runs/sid_gr_new_solutions.yml) | SID-GR, vLLM-GR, TensorRT-LLM and FlashRec; four-engine workload settings, 48 cases / 192 jobs, SID-GR reference. |
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

The [seven-engine profile](config/runs/seven_engine.yml) also provides runnable
[vLLM-GR](config/solutions/vllm_gr.yml), [TensorRT-LLM](config/solutions/tensorrt_llm.yml)
and [FlashRec](config/solutions/flash_rec.yml) adapters. They use native beam search,
pass token IDs directly and return every beam with its cumulative log-probability.
Their source, runtime and engine settings are independent; see setup below.

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

### Prepare vLLM-GR, TensorRT-LLM and FlashRec

Run these commands from the benchmark directory. The vLLM-GR checkout requires
access to the private `vllm-project/je` repository. If the checkouts already exist,
use them at the pinned commits; do not clone over them.

```bash
export VLLM_GR_SOURCE=/absolute/path/to/je
export FLASH_REC_SOURCE=/absolute/path/to/FlashRec
git clone --branch vllm-gr https://github.com/vllm-project/je.git "$VLLM_GR_SOURCE"
git -C "$VLLM_GR_SOURCE" checkout 4c7cd23e93c34d855b8d7aa2b43598f0c37bed1d
git clone https://github.com/sohu-mptc/FlashRec.git "$FLASH_REC_SOURCE"
git -C "$FLASH_REC_SOURCE" checkout 1089682a087ae5493f3b357ed1ec501cb684717b
docker build -f entrypoints/vllm_gr/Dockerfile \
  -t sid-gr-benchmark:vllm-gr-4c7cd23 "$VLLM_GR_SOURCE"
docker build -f entrypoints/flash_rec/Dockerfile \
  -t sid-gr-benchmark:flashrec-1089682 "$FLASH_REC_SOURCE"
docker pull nvcr.io/nvidia/tensorrt-llm/release@sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9
export VLLM_GR_RUNTIME=docker TENSORRT_LLM_RUNTIME=docker FLASH_REC_RUNTIME=docker
# Select BENCHMARK_GPU and the matching hardware profile as above.
python3 bench.py plan --config config/runs/seven_engine.yml \
  --solutions vllm_gr tensorrt_llm flash_rec --smoke
python3 bench.py e2e --config config/runs/seven_engine.yml \
  --solutions vllm_gr tensorrt_llm flash_rec --smoke
```

Remove `--smoke` to run the scenario grid. Prepare the four original runtimes and
omit `--solutions` to compare all seven, with public SGLang as reference. A run
without its reference still validates outputs and records timings; parity is
unavailable for that run.

vLLM-GR uses the pinned legacy `CUSTOM` beam API and vLLM 0.22.1. Prefix caching is
required to preserve prompt KV; beam widths above 256 and batches above one are
unsupported. Its shared beam KV pool holds only one active session. Its image
includes a UUID-to-NVML device mapping fix for vLLM 0.22.1, needed by the harness's
GPU selection. Local installations need the same fix shown in its Dockerfile.

TensorRT-LLM uses release 1.2.1's **PyTorch backend** with native beam search.
The matching Python package and compiled bindings must be installed together;
`TENSORRT_LLM_SOURCE` only records an optional checkout, without overriding the
installed package. Its YAML supplies library paths that the release image normally
sets in shell startup; override `TENSORRT_LLM_LD_LIBRARY_PATH` for local environments.
This release supports the required cumulative beam scores only at batch size one.
Long contexts with wide beams can also exceed its KV capacity on the selected GPU;
the adapter reports that native limit without shortening the workload.

FlashRec uses the pinned `sohu-mptc/FlashRec` source. The adapter explicitly selects
BF16 compute and KV cache and disables FP8, SID catalogs, boundary tokens and
restricted LM heads. Fresh internal request IDs keep repeated benchmark calls
independent while the output retains the workload's original request IDs.
The pool reserves enough slots for both prefill requests and expanded beams.

The seven-engine grid retains unsupported rows in its reports. vLLM-GR and
TensorRT-LLM each support 12 batch-one cases in the supplied grid, subject to runtime
capacity; their 36 larger-batch cases are explicitly marked unsupported. They do
not silently serialize a requested batch or omit its scores.

To compare SID-GR with these three candidates using the four-engine run's model,
input, generation and measurement settings, prepare SID-GR's runtime above and run:

```bash
export SID_GR_RUNTIME=docker
export VLLM_GR_RUNTIME=docker TENSORRT_LLM_RUNTIME=docker FLASH_REC_RUNTIME=docker
python3 bench.py e2e --config config/runs/sid_gr_new_solutions.yml
```

This profile uses SID-GR as the reference, with five warmups and 30 timed repeats
per supported job. The report records the actual hardware and runtime identities.

### Saved comparison with SID-GR

[Full report](results/20261013T063653115757Z-b34f36/report.md) ·
[Raw archive](results/20261013T063653115757Z-b34f36/raw_artifacts.tar.gz) ·
[Protocol audit](results/20261013T063653115757Z-b34f36/protocol_audit.json)

The completed run contains 192 jobs: 116 successful and 76 unsupported, with no
failed or timed-out jobs. Its 3,480 timed samples use the same 48 token workloads,
checkpoint hashes, BF16 precision, generation rules, five warmups and 30 timed
repeats as the published four-engine run. All four candidates enable prefix
caching. This comparison uses the same RTX PRO 6000 Blackwell Server GPU model
and ten physical CPU cores, but a different GPU device and CPU host: Xeon Platinum
8480+ here versus Xeon Gold 6444Y in the earlier run. The GPU drivers are also
different: 595.58.03 here versus 615.71.09 earlier. SID-GR was measured again
alongside the new candidates. The published leaderboards include the two archived
comparisons; local regeneration also includes other non-smoke runs in `results/`.

Representative median batch latency at **context 1000 / batch 1 / beam 256**;
lower is better. Successful workload counts cover the full 48-case grid.

| Candidate | Successful workloads | Qwen3-0.6B ms | Qwen3-1.7B ms |
| --- | ---: | ---: | ---: |
| `sid_gr_inference` | 48 / 48 | 10.299 | 15.081 |
| `vllm_gr` | 12 / 48 | 25.835 | 32.917 |
| `tensorrt_llm` | 8 / 48 | 125.313 | 129.292 |
| `flash_rec` | 48 / 48 | 35.094 | 43.354 |

SID-GR has the lowest median in all 48 workloads. Both vLLM-GR and TensorRT-LLM
record 36 unsupported larger-batch cases. TensorRT-LLM also exceeds its KV capacity
in four context-5000 cases with beams 128/256 at the configured 0.70 memory
fraction. These limits remain explicit in the report; the workloads were not
shortened. The CLI exits with status 1 when any row is unsupported, even when the
run manifest is `completed`.

All successful jobs passed structural output checks. Against SID-GR, top-1
mismatches occur in one vLLM-GR case, one TensorRT-LLM case and six FlashRec cases;
the report retains the complete beam and score comparisons. The archive includes
every request, raw sample, output, log and input, plus the frozen suite source and
protocol audit. This run contains unprofiled end-to-end measurements.

## Development

See [harness workflow and modules](harness/README.md) for implementation details.
Run CPU checks with `python3 -m unittest discover -s tests`.
