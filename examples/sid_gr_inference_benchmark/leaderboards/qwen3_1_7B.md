# qwen3_1_7B leaderboard

Compare each solution's latest latency on the same workload and hardware.

Median batch **ms**; lower is better. The **Fastest** column names the winner for each workload.

Parity counts are workload cases; each reference row checks repeat consistency.

## [NVIDIA RTX PRO 6000 Blackwell Server Edition](../config/hardware/rtx_pro_6000_blackwell_server.yml)

**CPU:** Intel(R) Xeon(R) Gold 6444Y · **Worker cores:** 10 · **Driver:** 615.71.09

| Solution | Exact beam/score match | Different beams or scores | Top-1 mismatches | Not compared |
| --- | --- | --- | --- | --- |
| sid_gr_inference | 0 of 1 cases | 1 of 1 cases | 0 cases | 0 cases |
| sglang_beam_search_new | 0 of 1 cases | 1 of 1 cases | 0 cases | 0 cases |
| sglang_beam_search | 0 of 1 cases | 1 of 1 cases | 0 cases | 0 cases |
| sglang | 1 of 1 cases | 0 of 1 cases | 0 cases | 0 cases |



### Latency comparison

| Context | Batch | Beam | sid_gr_inference | sglang_beam_search_new | sglang_beam_search | sglang | Fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 256 | **15.879** | 31.793 | 34.142 | 47.932 | **sid_gr_inference** |

<details>
<summary>Output parity, including reference repeat consistency</summary>

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 10.938 | 0.191 |
| ctx1000_b1_w256 | sglang_beam_search_new | different | 100.000 | 96.094 | 96.094 | 9.375 | 0.38 |
| ctx1000_b1_w256 | sglang_beam_search | different | 100.000 | 94.922 | 94.922 | 7.422 | 0.2 |
| ctx1000_b1_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |

</details>

<details>
<summary>Throughput, status and source runs</summary>

| Case | Candidate | Status | Requests/s | Reference | Source run |
| --- | --- | --- | --- | --- | --- |
| ctx1000_b1_w256 | sid_gr_inference | success | 62.976 | sglang | [20261012T112352414091Z-d89b64](../results/20261012T112352414091Z-d89b64/report.md) (completed) |
| ctx1000_b1_w256 | sglang_beam_search_new | success | 31.453 | sglang | [20261012T112352414091Z-d89b64](../results/20261012T112352414091Z-d89b64/report.md) (completed) |
| ctx1000_b1_w256 | sglang_beam_search | success | 29.290 | sglang | [20261012T112352414091Z-d89b64](../results/20261012T112352414091Z-d89b64/report.md) (completed) |
| ctx1000_b1_w256 | sglang | success | 20.863 | sglang | [20261012T112352414091Z-d89b64](../results/20261012T112352414091Z-d89b64/report.md) (completed) |

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
