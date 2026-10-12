# qwen3_1_7B leaderboard

Compare each solution's latest latency on the same workload and hardware.

Median batch **ms**; lower is better. The **Fastest** column names the winner for each workload.

Parity counts are workload cases; each reference row checks repeat consistency.

## [NVIDIA RTX PRO 6000 Blackwell Server Edition](../config/hardware/rtx_pro_6000_blackwell_server.yml)

**CPU:** Intel(R) Xeon(R) Platinum 8480+ · **Worker cores:** 10 · **Driver:** 595.58.03

| Solution | Exact beam/score match | Different beams or scores | Top-1 mismatches | Not compared |
| --- | --- | --- | --- | --- |
| sid_gr_inference | 0 of 8 cases | 8 of 8 cases | 0 cases | 0 cases |
| sglang_beam_search | 8 of 8 cases | 0 of 8 cases | 0 cases | 0 cases |



### Latency comparison

| Context | Batch | Beam | sid_gr_inference | sglang_beam_search | Fastest |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 256 | **15.253** | 33.706 | **sid_gr_inference** |
| 1000 | 2 | 256 | **27.037** | 53.025 | **sid_gr_inference** |
| 1000 | 4 | 256 | **47.586** | 88.240 | **sid_gr_inference** |
| 1000 | 8 | 256 | **86.245** | 160.210 | **sid_gr_inference** |
| 5000 | 1 | 256 | **18.175** | 76.638 | **sid_gr_inference** |
| 5000 | 2 | 256 | **31.977** | 118.639 | **sid_gr_inference** |
| 5000 | 4 | 256 | **57.007** | 214.106 | **sid_gr_inference** |
| 5000 | 8 | 256 | **109.061** | 414.206 | **sid_gr_inference** |

<details>
<summary>Output parity, including reference repeat consistency</summary>

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 9.375 | 0.193 |
| ctx1000_b1_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w256 | sid_gr_inference | different | 100.000 | 96.680 | 96.484 | 6.055 | 0.258 |
| ctx1000_b2_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w256 | sid_gr_inference | different | 100.000 | 95.312 | 93.359 | 7.910 | 0.244 |
| ctx1000_b4_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w256 | sid_gr_inference | different | 100.000 | 95.654 | 94.531 | 9.277 | 0.409 |
| ctx1000_b8_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | sid_gr_inference | different | 100.000 | 95.703 | 95.703 | 6.250 | 0.41 |
| ctx5000_b1_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w256 | sid_gr_inference | different | 100.000 | 94.531 | 93.359 | 6.250 | 0.342 |
| ctx5000_b2_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w256 | sid_gr_inference | different | 100.000 | 95.898 | 94.922 | 9.473 | 0.252 |
| ctx5000_b4_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w256 | sid_gr_inference | different | 100.000 | 96.143 | 94.922 | 9.814 | 0.241 |
| ctx5000_b8_w256 | sglang_beam_search | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |

</details>

<details>
<summary>Throughput, status and source runs</summary>

| Case | Candidate | Status | Requests/s | Reference | Source run |
| --- | --- | --- | --- | --- | --- |
| ctx1000_b1_w256 | sid_gr_inference | success | 65.560 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx1000_b1_w256 | sglang_beam_search | success | 29.668 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx1000_b2_w256 | sid_gr_inference | success | 73.974 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx1000_b2_w256 | sglang_beam_search | success | 37.718 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx1000_b4_w256 | sid_gr_inference | success | 84.059 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx1000_b4_w256 | sglang_beam_search | success | 45.331 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx1000_b8_w256 | sid_gr_inference | success | 92.759 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx1000_b8_w256 | sglang_beam_search | success | 49.934 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx5000_b1_w256 | sid_gr_inference | success | 55.021 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx5000_b1_w256 | sglang_beam_search | success | 13.048 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx5000_b2_w256 | sid_gr_inference | success | 62.545 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx5000_b2_w256 | sglang_beam_search | success | 16.858 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx5000_b4_w256 | sid_gr_inference | success | 70.167 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx5000_b4_w256 | sglang_beam_search | success | 18.682 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx5000_b8_w256 | sid_gr_inference | success | 73.354 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |
| ctx5000_b8_w256 | sglang_beam_search | success | 19.314 | sglang_beam_search | [20261012T085705933753Z-b3fc71](../results/20261012T085705933753Z-b3fc71/report.md) (completed) |

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
| Excluded | Short validation runs (`--smoke`) and profiled timings |

</details>

Generated from the [leaderboard template](../harness/reporting/templates/leaderboard.md).
