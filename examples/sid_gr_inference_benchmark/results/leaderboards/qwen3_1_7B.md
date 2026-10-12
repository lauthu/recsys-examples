# qwen3_1_7B leaderboard

Latest non-smoke result per case/candidate. Groups separate hardware and scenario revisions.

Median batch **ms**; lower is better. **Bold** = lowest recorded median. — = no result.

## NVIDIA RTX PRO 6000 Blackwell Server Edition

Scenario/checkpoint revision: `c12dfcf838df`.

**CPU:** Intel(R) Xeon(R) Platinum 8480+ · **Driver:** 595.58.03

| Exact | Different | Top-1 mismatches | Unavailable |
| ---: | ---: | ---: | ---: |
| 8 | 8 | 1 | 0 |

| Top-1 mismatch case | Candidate |
| --- | --- |
| qwen3_1_7B/ctx5000_b4_w256 | sid_gr_inference |

<details>
<summary>Hardware configuration</summary>

| Hardware | Recorded configuration |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory / driver | 97887.0 MiB / 595.58.03 |
| CPU | Intel(R) Xeon(R) Platinum 8480+ |
| CPU topology / affinity | 2 sockets · 112 physical cores · 224 logical CPUs · 28 allowed CPUs |

</details>

### Latency comparison

| Context | Batch | Beam | sid_gr_inference | sglang_beam_search |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 1 | 256 | **27.635** | 41.914 |
| 1000 | 2 | 256 | **48.180** | 68.078 |
| 1000 | 4 | 256 | **85.068** | 118.159 |
| 1000 | 8 | 256 | **157.756** | 220.812 |
| 5000 | 1 | 256 | **74.566** | 125.303 |
| 5000 | 2 | 256 | **140.066** | 214.662 |
| 5000 | 4 | 256 | **275.755** | 419.827 |
| 5000 | 8 | 256 | **546.940** | 837.465 |

<details>
<summary>Output parity, including reference repeat consistency</summary>

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w256 | sid_gr_inference | different | 100.000 | 95.312 | 95.312 | 12.109 | 0.218 |
| ctx1000_b1_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w256 | sid_gr_inference | different | 100.000 | 96.484 | 95.312 | 6.055 | 0.213 |
| ctx1000_b2_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w256 | sid_gr_inference | different | 100.000 | 96.094 | 94.141 | 8.594 | 0.246 |
| ctx1000_b4_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w256 | sid_gr_inference | different | 100.000 | 95.605 | 94.141 | 9.424 | 0.391 |
| ctx1000_b8_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | sid_gr_inference | different | 100.000 | 96.875 | 96.875 | 5.859 | 0.39 |
| ctx5000_b1_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w256 | sid_gr_inference | different | 100.000 | 95.117 | 93.750 | 8.984 | 0.334 |
| ctx5000_b2_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w256 | sid_gr_inference | different | 75.000 | 95.801 | 94.922 | 8.691 | 0.274 |
| ctx5000_b4_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w256 | sid_gr_inference | different | 100.000 | 95.947 | 94.922 | 8.545 | 0.322 |
| ctx5000_b8_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |

</details>

<details>
<summary>Throughput, status and source runs</summary>

| Case | Candidate | Status | Requests/s | Reference | Source run |
| --- | --- | --- | --- | --- | --- |
| ctx1000_b1_w256 | sid_gr_inference | success | 36.185 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx1000_b1_w256 | sglang_beam_search | success | 23.859 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx1000_b2_w256 | sid_gr_inference | success | 41.511 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx1000_b2_w256 | sglang_beam_search | success | 29.378 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx1000_b4_w256 | sid_gr_inference | success | 47.021 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx1000_b4_w256 | sglang_beam_search | success | 33.853 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx1000_b8_w256 | sid_gr_inference | success | 50.711 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx1000_b8_w256 | sglang_beam_search | success | 36.230 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx5000_b1_w256 | sid_gr_inference | success | 13.411 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx5000_b1_w256 | sglang_beam_search | success | 7.981 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx5000_b2_w256 | sid_gr_inference | success | 14.279 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx5000_b2_w256 | sglang_beam_search | success | 9.317 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx5000_b4_w256 | sid_gr_inference | success | 14.506 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx5000_b4_w256 | sglang_beam_search | success | 9.528 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx5000_b8_w256 | sid_gr_inference | success | 14.627 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |
| ctx5000_b8_w256 | sglang_beam_search | success | 9.553 | sglang_beam_search | [20261012T074633875336Z-dcc97c](../20261012T074633875336Z-dcc97c/report.md) (completed) |

</details>

<details>
<summary>Ranking rules</summary>

| Rule | Behavior |
| --- | --- |
| Latest result | Each finished job replaces its case/candidate row |
| Partial run | Unexecuted rows keep their last result |
| Failed / unsupported | Replaces earlier success |
| Grouping | Scenario, checkpoint, GPU, CPU and driver |
| Source runs | Recorded per row; latency cells may come from different runs |
| Parity | Same-run reference only; never recomputed across runs |
| Reference parity | Repeat consistency |
| Requests/s | Batch size / median batch latency |
| Bold median | No significance or output-equivalence claim |
| Excluded | Smoke and diagnosis runs |

</details>
