# SID-GR benchmark

Run `20261012T112352414091Z-d89b64` · [Run profile](../../config/runs/comparison.yml)

[Latency](#latency-comparison) · [Parity](#output-parity) · [GPU diagnosis](#gpu-diagnosis) · [Environment](#run-details) · [Artifacts](#artifacts)

| Run summary | Meaning |
| --- | --- |
| Status | completed |
| Workload | 1 context / batch / beam combinations |
| Solutions compared | sid_gr_inference, sglang, sglang_beam_search, sglang_beam_search_new |
| Progress | 4 of 4 solution–workload runs finished |
| Execution checks | 4 passed; 0 failed or unsupported |

Each solution runs every workload combination, with the warmups and repeats listed below.

Parity counts below include finished workload cases only; the reference row checks repeat consistency.

| Solution | Exact beam/score match | Different beams or scores | Top-1 mismatches | Not compared | Finished runs | Status |
| --- | --- | --- | --- | --- | --- | --- |
| sid_gr_inference | 0 of 1 cases | 1 of 1 cases | 0 cases | 0 cases | 1 of 1 | completed |
| sglang | 1 of 1 cases | 0 of 1 cases | 0 cases | 0 cases | 1 of 1 | completed |
| sglang_beam_search | 0 of 1 cases | 1 of 1 cases | 0 cases | 0 cases | 1 of 1 | completed |
| sglang_beam_search_new | 0 of 1 cases | 1 of 1 cases | 0 cases | 0 cases | 1 of 1 | completed |

**Reference:** `sglang` · Offline batches on one GPU

Hardware profile: [rtx_pro_6000_blackwell_server](../../config/hardware/rtx_pro_6000_blackwell_server.yml).

## Latency comparison

Median batch **ms**; lower is better. **Bold** = lowest recorded median. Pending = no finished result. — = unavailable.

### qwen3_1_7B

| Context | Batch | Beam | sid_gr_inference | sglang | sglang_beam_search | sglang_beam_search_new | Fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 256 | **15.879** | 47.932 | 34.142 | 31.793 | **sid_gr_inference** |



<details>
<summary>All measurements</summary>

### qwen3_1_7B

| Case | Candidate | Status | Median ms | IQR ms | Δ latency % | Requests/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ctx1000_b1_w256 | sid_gr_inference | success | 15.879 | 0.034 | -66.872 | 62.976 |
| ctx1000_b1_w256 | sglang | success | 47.932 | 0.207 | 0.000 | 20.863 |
| ctx1000_b1_w256 | sglang_beam_search | success | 34.142 | 0.207 | -28.770 | 29.290 |
| ctx1000_b1_w256 | sglang_beam_search_new | success | 31.793 | 0.249 | -33.671 | 31.453 |

| Metric | Meaning |
| --- | --- |
| Δ latency % | Relative to reference; negative = faster |
| IQR ms | 75th − 25th percentile latency; spread of the middle 50% of repeats |
| Requests/s | Batch size / median batch latency |
| Bold median | Lowest observed value; no significance or parity claim |

</details>

## Output parity

<details>
<summary>All parity checks</summary>

### qwen3_1_7B

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w256 | sglang_beam_search | different | 100.000 | 94.922 | 94.922 | 7.422 | 0.2 |
| ctx1000_b1_w256 | sglang_beam_search_new | different | 100.000 | 96.094 | 96.094 | 9.375 | 0.38 |
| ctx1000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 10.938 | 0.191 |

Compare every request and repeat with the first same-run reference output on identical token inputs.

| Check | Meaning |
| --- | --- |
| Success | Structural checks passed |
| Exact | Ordered beams and scores match the first same-run reference output |
| Top-1 % | Requests with matching leading sequence / all compared requests |
| Mean overlap % | Mean shared-beam fraction, ignoring rank |
| Min overlap % | Lowest shared-beam fraction across requests and repeats |
| Rank match % | Mean fraction of beam sequences matching at the same rank |
| Max score Δ | Largest score difference among shared sequences |
| Token lengths / output budget | Every beam must contain exactly the requested token count; checked before success |
| Reference row | Repeat consistency |

</details>

## GPU diagnosis

| Saved GPU diagnosis |
| --- |
| [20261012T113125Z-f8effe](diagnostics/20261012T113125Z-f8effe/diagnosis.md) |

## Run details

<details>
<summary>Configuration environment</summary>

| Configuration variable | Solution(s) | Value used | Reproduce from |
| --- | --- | --- | --- |
| `BENCHMARK_GPU` | sid_gr_inference, sglang, sglang_beam_search, sglang_beam_search_new | `GPU-7330c4aa-0e83-7249-af71-60feb195d4eb` | — |
| `GR_DECODE_ATTEN_ROOT` | sid_gr_inference | `/home/scratch.xinyliu_gpu/recsys-examples/corelib/gr_decode_atten` | [10744f6421cf](https://github.com/lauthu/recsys-examples/tree/10744f6421cfc42b82badfc0cafefe7574269777/corelib/gr_decode_atten) |
| `GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_DISABLE_FLATTEN_LINEAR` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_DISABLE_PREFILL_CUDA_GRAPH` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_GR_TRTLLM_KERNELS_JIT` | sid_gr_inference | `1` | — |
| `GR_INFERENCE_PREFILL_BACKEND` | sid_gr_inference | `auto` | — |
| `GR_INFERENCE_TRTLLM_PACKED_GEMM_SCOPE` | sid_gr_inference | `all` | — |
| `GR_INFERENCE_TRTLLM_QK_NORM_ROPE_PHASE` | sid_gr_inference | `all` | — |
| `QWEN3_0_6B_MODEL` | sid_gr_inference, sglang, sglang_beam_search, sglang_beam_search_new | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/repo/models/Qwen3-0.6B` | — |
| `QWEN3_1_7B_MODEL` | sid_gr_inference, sglang, sglang_beam_search, sglang_beam_search_new | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/repo/models/Qwen3-1.7B` | [Qwen/Qwen3-1.7B@70d244cc86cc](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) |
| `SGLANG_BEAM_IMAGE` | sglang_beam_search | `sid-gr-benchmark:beam-af4e3f4-cu130` | `sha256:a2482e6986f5c84f7f85c47c89b9f5b87f621b62941aea6cfc4b776c7d2e2503` · [Build recipe](../../entrypoints/sglang_beam_search/Dockerfile) |
| `SGLANG_BEAM_NEW_IMAGE` | sglang_beam_search_new | `sid-gr-benchmark:beam-df191d6-cu130` | `sha256:a86148a0611614d3ee6943f24ba1447a417b4b6a67d0cf970610ee7a3bb9c955` · [Build recipe](../../entrypoints/sglang_beam_search_new/Dockerfile) |
| `SGLANG_BEAM_NEW_PYTHON` | sglang_beam_search_new | `python3` | — |
| `SGLANG_BEAM_NEW_RUNTIME` | sglang_beam_search_new | `docker` | — |
| `SGLANG_BEAM_NEW_SOURCE` | sglang_beam_search_new | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/sglang_beam_search_new` | [df191d672f19](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891) |
| `SGLANG_BEAM_PYTHON` | sglang_beam_search | `python3` | — |
| `SGLANG_BEAM_RUNTIME` | sglang_beam_search | `docker` | — |
| `SGLANG_BEAM_SOURCE` | sglang_beam_search | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/sglang` | [af4e3f42a5a1](https://github.com/cswuyg/sglang/tree/af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d) |
| `SGLANG_IMAGE` | sglang | `lmsysorg/sglang@sha256:b5c488694d74c13587af6057b90220498c4869571c8e1e3c43dd6f43ae721306` | `sha256:a5a221f4fe3a7c4ca826cfcb30e06c869ff561d50f02f048236b58d3f4a5d273` · `lmsysorg/sglang@sha256:b5c488694d74c13587af6057b90220498c4869571c8e1e3c43dd6f43ae721306` |
| `SGLANG_PYTHON` | sglang | `python3` | — |
| `SGLANG_RUNTIME` | sglang | `docker` | — |
| `SGLANG_SOURCE` | sglang | (unset) | — |
| `SID_GR_IMAGE` | sid_gr_inference | `sid-gr-benchmark:beam-af4e3f4-cu130` | `sha256:a2482e6986f5c84f7f85c47c89b9f5b87f621b62941aea6cfc4b776c7d2e2503` · [Build recipe](../../entrypoints/sglang_beam_search/Dockerfile) |
| `SID_GR_PYTHON` | sid_gr_inference | `python3` | — |
| `SID_GR_RUNTIME` | sid_gr_inference | `docker` | — |
| `SID_GR_SOURCE` | sid_gr_inference | `/home/scratch.xinyliu_gpu/recsys-examples/examples/sid-gr-inference` | [10744f6421cf](https://github.com/lauthu/recsys-examples/tree/10744f6421cfc42b82badfc0cafefe7574269777/examples/sid-gr-inference) |

| Solution | Engine option | Value used |
| --- | --- | --- |
| sid_gr_inference | `prefix_cache` | `true` |
| sglang | `disable_cuda_graph` | `false` |
| sglang | `disable_overlap_schedule` | `true` |
| sglang | `log_level` | `warning` |
| sglang | `mem_fraction_static` | `0.7` |
| sglang | `page_size` | `1` |
| sglang | `prefix_cache` | `true` |
| sglang_beam_search | `disable_cuda_graph` | `false` |
| sglang_beam_search | `disable_overlap_schedule` | `true` |
| sglang_beam_search | `disable_piecewise_cuda_graph` | `false` |
| sglang_beam_search | `log_level` | `warning` |
| sglang_beam_search | `mem_fraction_static` | `0.7` |
| sglang_beam_search | `page_size` | `1` |
| sglang_beam_search | `prefix_cache` | `true` |
| sglang_beam_search_new | `decode_attention_backend` | `flashinfer` |
| sglang_beam_search_new | `disable_cuda_graph` | `false` |
| sglang_beam_search_new | `disable_overlap_schedule` | `true` |
| sglang_beam_search_new | `enable_beam_cascade_attention` | `true` |
| sglang_beam_search_new | `log_level` | `warning` |
| sglang_beam_search_new | `mem_fraction_static` | `0.7` |
| sglang_beam_search_new | `page_size` | `1` |
| sglang_beam_search_new | `prefix_cache` | `true` |

</details>

<details>
<summary>Protocol and source versions</summary>

| Scenario | Warmups | Measurements |
| --- | --- | --- |
| qwen3_1_7B | 5 | 30 |

| Solution | Component | Reproducible source / image ID |
| --- | --- | --- |
| All | Model | [Qwen/Qwen3-1.7B@70d244cc86cc](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) |
| sid_gr_inference | Engine source | [10744f6421cf](https://github.com/lauthu/recsys-examples/tree/10744f6421cfc42b82badfc0cafefe7574269777/examples/sid-gr-inference) |
| sid_gr_inference | GR decode kernels | [10744f6421cf](https://github.com/lauthu/recsys-examples/tree/10744f6421cfc42b82badfc0cafefe7574269777/corelib/gr_decode_atten) |
| sid_gr_inference | Container image | `sha256:a2482e6986f5c84f7f85c47c89b9f5b87f621b62941aea6cfc4b776c7d2e2503` · [Build recipe](../../entrypoints/sglang_beam_search/Dockerfile) |
| sglang | Engine source | `ecf2caebbda0a9182994008407e14caeb835fe5c6a0549d7ec2371ad2cd922c2` |
| sglang | Container image | `sha256:a5a221f4fe3a7c4ca826cfcb30e06c869ff561d50f02f048236b58d3f4a5d273` · `lmsysorg/sglang@sha256:b5c488694d74c13587af6057b90220498c4869571c8e1e3c43dd6f43ae721306` |
| sglang_beam_search | Engine source | [af4e3f42a5a1](https://github.com/cswuyg/sglang/tree/af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d) |
| sglang_beam_search | Container image | `sha256:a2482e6986f5c84f7f85c47c89b9f5b87f621b62941aea6cfc4b776c7d2e2503` · [Build recipe](../../entrypoints/sglang_beam_search/Dockerfile) |
| sglang_beam_search_new | Engine source | [df191d672f19](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891) |
| sglang_beam_search_new | Container image | `sha256:a86148a0611614d3ee6943f24ba1447a417b4b6a67d0cf970610ee7a3bb9c955` · [Build recipe](../../entrypoints/sglang_beam_search_new/Dockerfile) |

Docker rows identify the actual image; build recipes link the pinned public base and branch dependencies.

| Solution | Source commit/tree | Dirty | Key packages |
| --- | --- | --- | --- |
| sid_gr_inference | [10744f6421cf](https://github.com/lauthu/recsys-examples/tree/10744f6421cfc42b82badfc0cafefe7574269777/examples/sid-gr-inference) | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |
| sglang | `ecf2caebbda0a9182994008407e14caeb835fe5c6a0549d7ec2371ad2cd922c2` | False | sglang 0.0.0.dev1+ge6072935b, torch 2.14.1+cu130, triton 3.8.0 |
| sglang_beam_search | [af4e3f42a5a1](https://github.com/cswuyg/sglang/tree/af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d) | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |
| sglang_beam_search_new | [df191d672f19](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891) | False | sglang 0.0.0.dev1+ge6072935b, torch 2.11.0+cu130, triton 3.6.0 |

</details>

<details>
<summary>Optional compute-floor estimate</summary>

| Estimate | Reason |
| --- | --- |
| Compute floor unavailable | No sourced dense BF16 ceiling in the selected hardware profile |

This estimate needs a sourced dense BF16 ceiling. GPU diagnosis above uses measured Nsight traces independently.

</details>

<details>
<summary>Measurement scope and limitations</summary>

| Item | Scope |
| --- | --- |
| Timer | CPU token IDs through completed CPU beam IDs/scores |
| Included | Submission, transfers, execution, conversion and synchronization |
| Excluded | Setup, warmup, validation and serialization |
| Execution | Serial candidates; order rotates by case; clocks unlocked |
| GPU memory usage | Not measured across engine process trees |
| Compute floor | Matrix math only; omits memory traffic and host work |
| Synthetic inputs | No recommendation-quality claim |
| Historical timings | Different timing boundary; excluded from rankings |

</details>

## Artifacts

[Run archive](raw_artifacts.tar.gz)

```bash
tar -xzf results/20261012T112352414091Z-d89b64/raw_artifacts.tar.gz -C results/20261012T112352414091Z-d89b64
python3 bench.py report --run results/20261012T112352414091Z-d89b64
```

| Saved data | Location inside the run |
| --- | --- |
| Frozen configuration and hardware | `manifest.json` |
| Derived measurements | `summary.json` |
| Exact inputs | `workloads/` |
| Requests, commands, logs, samples and beams | `jobs/<candidate>/<scenario>/<case>/` |

**Scenario leaderboards**

- [qwen3_1_7B](../../leaderboards/qwen3_1_7B.md)
