# SID-GR benchmark

Run `20261012T085705933753Z-b3fc71` · [Run profile](../../config/runs/readme.yml)

[Latency](#latency-comparison) · [Parity](#output-parity) · [GPU diagnosis](#gpu-diagnosis) · [Environment](#run-details) · [Artifacts](#artifacts)

| Run summary | Meaning |
| --- | --- |
| Status | completed |
| Workload | 8 context / batch / beam combinations |
| Solutions compared | sid_gr_inference, sglang_beam_search |
| Progress | 16 of 16 solution–workload runs finished |
| Execution checks | 16 passed; 0 failed or unsupported |

Each solution runs every workload combination, with the warmups and repeats listed below.

Counts below are workload cases; the reference row checks repeat consistency.

| Solution | Exact beam/score match | Different beams or scores | Top-1 mismatches | Not compared |
| --- | --- | --- | --- | --- |
| sid_gr_inference | 0 of 8 cases | 8 of 8 cases | 0 cases | 0 cases |
| sglang_beam_search | 8 of 8 cases | 0 of 8 cases | 0 cases | 0 cases |

**Reference:** `sglang_beam_search` · Offline batches on one GPU

Hardware profile: [rtx_pro_6000_blackwell_server](../../config/hardware/rtx_pro_6000_blackwell_server.yml).

## Latency comparison

Median batch **ms**; lower is better. **Bold** = lowest recorded median. — = no result.

### qwen3_1_7B

| Context | Batch | Beam | sglang_beam_search | sid_gr_inference | Fastest |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 256 | 33.706 | **15.253** | **sid_gr_inference** |
| 1000 | 2 | 256 | 53.025 | **27.037** | **sid_gr_inference** |
| 1000 | 4 | 256 | 88.240 | **47.586** | **sid_gr_inference** |
| 1000 | 8 | 256 | 160.210 | **86.245** | **sid_gr_inference** |
| 5000 | 1 | 256 | 76.638 | **18.175** | **sid_gr_inference** |
| 5000 | 2 | 256 | 118.639 | **31.977** | **sid_gr_inference** |
| 5000 | 4 | 256 | 214.106 | **57.007** | **sid_gr_inference** |
| 5000 | 8 | 256 | 414.206 | **109.061** | **sid_gr_inference** |



<details>
<summary>All measurements</summary>

### qwen3_1_7B

| Case | Candidate | Status | Median ms | IQR ms | Δ latency % | Requests/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ctx1000_b1_w256 | sglang_beam_search | success | 33.706 | 0.619 | 0.000 | 29.668 |
| ctx1000_b1_w256 | sid_gr_inference | success | 15.253 | 0.153 | -54.747 | 65.560 |
| ctx1000_b2_w256 | sglang_beam_search | success | 53.025 | 1.527 | 0.000 | 37.718 |
| ctx1000_b2_w256 | sid_gr_inference | success | 27.037 | 0.211 | -49.011 | 73.974 |
| ctx1000_b4_w256 | sglang_beam_search | success | 88.240 | 1.274 | 0.000 | 45.331 |
| ctx1000_b4_w256 | sid_gr_inference | success | 47.586 | 0.428 | -46.072 | 84.059 |
| ctx1000_b8_w256 | sglang_beam_search | success | 160.210 | 2.292 | 0.000 | 49.934 |
| ctx1000_b8_w256 | sid_gr_inference | success | 86.245 | 0.469 | -46.167 | 92.759 |
| ctx5000_b1_w256 | sglang_beam_search | success | 76.638 | 0.727 | 0.000 | 13.048 |
| ctx5000_b1_w256 | sid_gr_inference | success | 18.175 | 0.259 | -76.285 | 55.021 |
| ctx5000_b2_w256 | sglang_beam_search | success | 118.639 | 0.542 | 0.000 | 16.858 |
| ctx5000_b2_w256 | sid_gr_inference | success | 31.977 | 0.211 | -73.047 | 62.545 |
| ctx5000_b4_w256 | sglang_beam_search | success | 214.106 | 1.186 | 0.000 | 18.682 |
| ctx5000_b4_w256 | sid_gr_inference | success | 57.007 | 0.206 | -73.375 | 70.167 |
| ctx5000_b8_w256 | sglang_beam_search | success | 414.206 | 3.186 | 0.000 | 19.314 |
| ctx5000_b8_w256 | sid_gr_inference | success | 109.061 | 0.739 | -73.670 | 73.354 |

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
| ctx1000_b1_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 9.375 | 0.193 |
| ctx1000_b2_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w256 | sid_gr_inference | different | 100.000 | 96.680 | 96.484 | 6.055 | 0.258 |
| ctx1000_b4_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w256 | sid_gr_inference | different | 100.000 | 95.312 | 93.359 | 7.910 | 0.244 |
| ctx1000_b8_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w256 | sid_gr_inference | different | 100.000 | 95.654 | 94.531 | 9.277 | 0.409 |
| ctx5000_b1_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | sid_gr_inference | different | 100.000 | 95.703 | 95.703 | 6.250 | 0.41 |
| ctx5000_b2_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w256 | sid_gr_inference | different | 100.000 | 94.531 | 93.359 | 6.250 | 0.342 |
| ctx5000_b4_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w256 | sid_gr_inference | different | 100.000 | 95.898 | 94.922 | 9.473 | 0.252 |
| ctx5000_b8_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w256 | sid_gr_inference | different | 100.000 | 96.143 | 94.922 | 9.814 | 0.241 |

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
| [20261012T091357Z-5b4d91](diagnostics/20261012T091357Z-5b4d91/diagnosis.md) |

## Run details

<details>
<summary>Configuration environment</summary>

| Configuration variable | Solution(s) | Value used | Reproduce from |
| --- | --- | --- | --- |
| `BENCHMARK_GPU` | sid_gr_inference, sglang_beam_search | `GPU-5e870363-33d3-3eaa-fbb5-70964cf8c8c7` | — |
| `GR_DECODE_ATTEN_ROOT` | sid_gr_inference | `/home/scratch.xinyliu_gpu/recsys-examples/corelib/gr_decode_atten` | [f02c6afea719](https://github.com/lauthu/recsys-examples/tree/f02c6afea7193a57f58425ebe36b879e6d004603/corelib/gr_decode_atten) |
| `GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_DISABLE_FLATTEN_LINEAR` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_DISABLE_PREFILL_CUDA_GRAPH` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_GR_TRTLLM_KERNELS_JIT` | sid_gr_inference | `1` | — |
| `GR_INFERENCE_PREFILL_BACKEND` | sid_gr_inference | `auto` | — |
| `GR_INFERENCE_TRTLLM_PACKED_GEMM_SCOPE` | sid_gr_inference | `all` | — |
| `GR_INFERENCE_TRTLLM_QK_NORM_ROPE_PHASE` | sid_gr_inference | `all` | — |
| `QWEN3_1_7B_MODEL` | sid_gr_inference, sglang_beam_search | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/repo/models/Qwen3-1.7B` | [Qwen/Qwen3-1.7B@70d244cc86cc](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) |
| `SGLANG_BEAM_SOURCE` | sglang_beam_search | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/sglang` | [af4e3f42a5a1](https://github.com/cswuyg/sglang/tree/af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d) |
| `SGLANG_IMAGE` | sglang_beam_search | `sid-gr-benchmark:beam-af4e3f4-cu130` | `sha256:e95aae5f9b3fdeab685d7045b559bd59a1bf971195f6a943a33dc6160adfb78a` |
| `SGLANG_PYTHON` | sglang_beam_search | `python3` | — |
| `SGLANG_RUNTIME` | sglang_beam_search | `docker` | — |
| `SID_GR_IMAGE` | sid_gr_inference | `sid-gr-benchmark:beam-af4e3f4-cu130` | `sha256:e95aae5f9b3fdeab685d7045b559bd59a1bf971195f6a943a33dc6160adfb78a` |
| `SID_GR_PYTHON` | sid_gr_inference | `python3` | — |
| `SID_GR_RUNTIME` | sid_gr_inference | `docker` | — |
| `SID_GR_SOURCE` | sid_gr_inference | `../sid-gr-inference` | — |

| Solution | Engine option | Value used |
| --- | --- | --- |
| sid_gr_inference | `prefix_cache` | `true` |
| sglang_beam_search | `disable_cuda_graph` | `false` |
| sglang_beam_search | `disable_overlap_schedule` | `true` |
| sglang_beam_search | `disable_piecewise_cuda_graph` | `false` |
| sglang_beam_search | `log_level` | `warning` |
| sglang_beam_search | `mem_fraction_static` | `0.7` |
| sglang_beam_search | `page_size` | `1` |
| sglang_beam_search | `prefix_cache` | `true` |

</details>

<details>
<summary>Protocol and source versions</summary>

| Scenario | Warmups | Measurements |
| --- | --- | --- |
| qwen3_1_7B | 5 | 30 |

| Solution | Component | Reproducible source / image ID |
| --- | --- | --- |
| All | Model | [Qwen/Qwen3-1.7B@70d244cc86cc](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) |
| sid_gr_inference | Engine source | [f02c6afea719](https://github.com/lauthu/recsys-examples/tree/f02c6afea7193a57f58425ebe36b879e6d004603/examples/sid-gr-inference) |
| sid_gr_inference | GR decode kernels | [f02c6afea719](https://github.com/lauthu/recsys-examples/tree/f02c6afea7193a57f58425ebe36b879e6d004603/corelib/gr_decode_atten) |
| sid_gr_inference | Container image | `sha256:e95aae5f9b3fdeab685d7045b559bd59a1bf971195f6a943a33dc6160adfb78a` |
| sglang_beam_search | Engine source | [af4e3f42a5a1](https://github.com/cswuyg/sglang/tree/af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d) |
| sglang_beam_search | Container image | `sha256:e95aae5f9b3fdeab685d7045b559bd59a1bf971195f6a943a33dc6160adfb78a` |

SID-GR / pinned SGLang: [Docker build recipe](../../entrypoints/sglang/Dockerfile.beam_search), including the public base image and beam-search source commit. Recorded image IDs above identify Docker runs.

| Solution | Source commit/tree | Dirty | Key packages |
| --- | --- | --- | --- |
| sid_gr_inference | [f02c6afea719](https://github.com/lauthu/recsys-examples/tree/f02c6afea7193a57f58425ebe36b879e6d004603/examples/sid-gr-inference) | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |
| sglang_beam_search | [af4e3f42a5a1](https://github.com/cswuyg/sglang/tree/af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d) | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |

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
tar -xzf results/20261012T085705933753Z-b3fc71/raw_artifacts.tar.gz -C results/20261012T085705933753Z-b3fc71
python3 bench.py report --run results/20261012T085705933753Z-b3fc71
```

| Saved data | Location inside the run |
| --- | --- |
| Frozen configuration and hardware | `manifest.json` |
| Derived measurements | `summary.json` |
| Exact inputs | `workloads/` |
| Requests, commands, logs, samples and beams | `jobs/<candidate>/<scenario>/<case>/` |

**Scenario leaderboards**

- [qwen3_1_7B](../../leaderboards/qwen3_1_7B.md)
