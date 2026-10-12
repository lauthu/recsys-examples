# SID-GR benchmark

Run `20261012T074633875336Z-dcc97c`

[Latency](#latency-comparison) · [Parity](#output-parity) · [GPU diagnosis](#gpu-diagnosis) · [Environment](#run-details) · [Artifacts](#artifacts)

| Run status | Cases | Candidates | Completed evaluations | Passed | Failed / unsupported | Mode |
| --- | --- | --- | --- | --- | --- | --- |
| completed | 8 | 2 | 16 of 16 | 16 | 0 | e2e |

One evaluation = one candidate on one case, including its warmups and timed repeats.

| Exact | Different | Top-1 mismatches | Unavailable |
| ---: | ---: | ---: | ---: |
| 8 | 8 | 1 | 0 |

| Top-1 mismatch case | Candidate |
| --- | --- |
| qwen3_1_7B/ctx5000_b4_w256 | sid_gr_inference |

**Reference:** `sglang_beam_search` · Offline batches on one GPU

| Hardware | Recorded configuration |
| --- | --- |
| GPU | [NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) |
| GPU memory / driver | 97887.0 MiB / 595.58.03 |
| CPU | Intel(R) Xeon(R) Platinum 8480+ |
| CPU topology / affinity | 2 sockets · 112 physical cores · 224 logical CPUs · 28 allowed CPUs |

Hardware profile: [rtx_pro_6000_blackwell_server](../../config/hardware/rtx_pro_6000_blackwell_server.yml).

## Latency comparison

Median batch **ms**; lower is better. **Bold** = lowest recorded median. — = no result.

### qwen3_1_7B

| Context | Batch | Beam | sglang_beam_search | sid_gr_inference |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 1 | 256 | 41.914 | **27.635** |
| 1000 | 2 | 256 | 68.078 | **48.180** |
| 1000 | 4 | 256 | 118.159 | **85.068** |
| 1000 | 8 | 256 | 220.812 | **157.756** |
| 5000 | 1 | 256 | 125.303 | **74.566** |
| 5000 | 2 | 256 | 214.662 | **140.066** |
| 5000 | 4 | 256 | 419.827 | **275.755** |
| 5000 | 8 | 256 | 837.465 | **546.940** |



<details>
<summary>All measurements</summary>

### qwen3_1_7B

| Case | Candidate | Status | Median ms | IQR ms | Δ latency % | Requests/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ctx1000_b1_w256 | sglang_beam_search | success | 41.914 | 0.322 | 0.000 | 23.859 |
| ctx1000_b1_w256 | sid_gr_inference | success | 27.635 | 0.051 | -34.066 | 36.185 |
| ctx1000_b2_w256 | sglang_beam_search | success | 68.078 | 0.214 | 0.000 | 29.378 |
| ctx1000_b2_w256 | sid_gr_inference | success | 48.180 | 0.130 | -29.229 | 41.511 |
| ctx1000_b4_w256 | sglang_beam_search | success | 118.159 | 0.807 | 0.000 | 33.853 |
| ctx1000_b4_w256 | sid_gr_inference | success | 85.068 | 0.121 | -28.005 | 47.021 |
| ctx1000_b8_w256 | sglang_beam_search | success | 220.812 | 1.118 | 0.000 | 36.230 |
| ctx1000_b8_w256 | sid_gr_inference | success | 157.756 | 0.283 | -28.556 | 50.711 |
| ctx5000_b1_w256 | sglang_beam_search | success | 125.303 | 0.233 | 0.000 | 7.981 |
| ctx5000_b1_w256 | sid_gr_inference | success | 74.566 | 0.070 | -40.492 | 13.411 |
| ctx5000_b2_w256 | sglang_beam_search | success | 214.662 | 1.046 | 0.000 | 9.317 |
| ctx5000_b2_w256 | sid_gr_inference | success | 140.066 | 0.323 | -34.750 | 14.279 |
| ctx5000_b4_w256 | sglang_beam_search | success | 419.827 | 1.156 | 0.000 | 9.528 |
| ctx5000_b4_w256 | sid_gr_inference | success | 275.755 | 0.236 | -34.317 | 14.506 |
| ctx5000_b8_w256 | sglang_beam_search | success | 837.465 | 5.178 | 0.000 | 9.553 |
| ctx5000_b8_w256 | sid_gr_inference | success | 546.940 | 1.290 | -34.691 | 14.627 |

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
| ctx1000_b1_w256 | sid_gr_inference | different | 100.000 | 95.312 | 95.312 | 12.109 | 0.218 |
| ctx1000_b2_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w256 | sid_gr_inference | different | 100.000 | 96.484 | 95.312 | 6.055 | 0.213 |
| ctx1000_b4_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w256 | sid_gr_inference | different | 100.000 | 96.094 | 94.141 | 8.594 | 0.246 |
| ctx1000_b8_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w256 | sid_gr_inference | different | 100.000 | 95.605 | 94.141 | 9.424 | 0.391 |
| ctx5000_b1_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | sid_gr_inference | different | 100.000 | 96.875 | 96.875 | 5.859 | 0.39 |
| ctx5000_b2_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w256 | sid_gr_inference | different | 100.000 | 95.117 | 93.750 | 8.984 | 0.334 |
| ctx5000_b4_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w256 | sid_gr_inference | different | 75.000 | 95.801 | 94.922 | 8.691 | 0.274 |
| ctx5000_b8_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w256 | sid_gr_inference | different | 100.000 | 95.947 | 94.922 | 8.545 | 0.322 |

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
| [20261012T080351Z-729d27](diagnostics/20261012T080351Z-729d27/diagnosis.md) |

## Run details

<details>
<summary>Configuration environment</summary>

| Configuration variable | Value used (including defaults) |
| --- | --- |
| `BENCHMARK_GPU` | `GPU-5e870363-33d3-3eaa-fbb5-70964cf8c8c7` |
| `GR_DECODE_ATTEN_ROOT` | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/gr_decode_atten` |
| `QWEN3_1_7B_MODEL` | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/repo/models/Qwen3-1.7B` |
| `SGLANG_BEAM_SOURCE` | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/sglang` |
| `SGLANG_IMAGE` | `sid-gr-benchmark:readme-20261011` |
| `SGLANG_PYTHON` | `python3` |
| `SGLANG_RUNTIME` | `docker` |
| `SID_GR_IMAGE` | `sid-gr-benchmark:readme-20261011` |
| `SID_GR_PYTHON` | `python3` |
| `SID_GR_RUNTIME` | `docker` |
| `SID_GR_SOURCE` | `../sid-gr-inference` |

</details>

<details>
<summary>Protocol and source versions</summary>

| Scenario | Warmups | Measurements |
| --- | --- | --- |
| qwen3_1_7B | 5 | 30 |

| Solution | Source commit/tree | Dirty | Key packages |
| --- | --- | --- | --- |
| sid_gr_inference | 28a83abdf6b85378 | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |
| sglang_beam_search | af4e3f42a5a1a79f | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |

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
tar -xzf results/20261012T074633875336Z-dcc97c/raw_artifacts.tar.gz -C results/20261012T074633875336Z-dcc97c
python3 bench.py report --run results/20261012T074633875336Z-dcc97c
```

| Saved data | Location inside the run |
| --- | --- |
| Frozen configuration and hardware | `manifest.json` |
| Derived measurements | `summary.json` |
| Exact inputs | `workloads/` |
| Requests, commands, logs, samples and beams | `jobs/<candidate>/<scenario>/<case>/` |

**Scenario leaderboards**

- [qwen3_1_7B](../leaderboards/qwen3_1_7B.md)
