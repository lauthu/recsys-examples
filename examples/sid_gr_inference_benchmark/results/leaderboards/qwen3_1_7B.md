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
| qwen3_1_7B/ctx5000_b4_w256 | sglang_beam_search |

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
| 1000 | 1 | 256 | **27.635** | 41.869 |
| 1000 | 2 | 256 | **48.414** | 69.253 |
| 1000 | 4 | 256 | **85.085** | 118.139 |
| 1000 | 8 | 256 | **158.833** | 226.693 |
| 5000 | 1 | 256 | **74.757** | 126.296 |
| 5000 | 2 | 256 | **139.439** | 213.576 |
| 5000 | 4 | 256 | **275.301** | 416.612 |
| 5000 | 8 | 256 | **546.016** | 829.489 |

<details>
<summary>Output parity, including reference repeat consistency</summary>

| Case | Candidate | Parity | Top-1 | Overlap % | Max score Δ |
| --- | --- | --- | --- | ---: | ---: |
| ctx1000_b1_w256 | sid_gr_inference | exact | Match | 100.000 | 0 |
| ctx1000_b1_w256 | sglang_beam_search | different | Match | 95.312 | 0.218 |
| ctx1000_b2_w256 | sid_gr_inference | exact | Match | 100.000 | 0 |
| ctx1000_b2_w256 | sglang_beam_search | different | Match | 95.312 | 0.213 |
| ctx1000_b4_w256 | sid_gr_inference | exact | Match | 100.000 | 0 |
| ctx1000_b4_w256 | sglang_beam_search | different | Match | 94.141 | 0.246 |
| ctx1000_b8_w256 | sid_gr_inference | exact | Match | 100.000 | 0 |
| ctx1000_b8_w256 | sglang_beam_search | different | Match | 94.141 | 0.391 |
| ctx5000_b1_w256 | sid_gr_inference | exact | Match | 100.000 | 0 |
| ctx5000_b1_w256 | sglang_beam_search | different | Match | 96.875 | 0.39 |
| ctx5000_b2_w256 | sid_gr_inference | exact | Match | 100.000 | 0 |
| ctx5000_b2_w256 | sglang_beam_search | different | Match | 93.750 | 0.334 |
| ctx5000_b4_w256 | sid_gr_inference | exact | Match | 100.000 | 0 |
| ctx5000_b4_w256 | sglang_beam_search | different | **Mismatch** | 94.922 | 0.274 |
| ctx5000_b8_w256 | sid_gr_inference | exact | Match | 100.000 | 0 |
| ctx5000_b8_w256 | sglang_beam_search | different | Match | 94.922 | 0.322 |

</details>

<details>
<summary>Throughput, status and source runs</summary>

| Case | Candidate | Status | Requests/s | Reference | Source run |
| --- | --- | --- | --- | --- | --- |
| ctx1000_b1_w256 | sid_gr_inference | success | 36.186 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx1000_b1_w256 | sglang_beam_search | success | 23.884 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx1000_b2_w256 | sid_gr_inference | success | 41.310 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx1000_b2_w256 | sglang_beam_search | success | 28.880 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx1000_b4_w256 | sid_gr_inference | success | 47.012 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx1000_b4_w256 | sglang_beam_search | success | 33.859 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx1000_b8_w256 | sid_gr_inference | success | 50.367 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx1000_b8_w256 | sglang_beam_search | success | 35.290 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx5000_b1_w256 | sid_gr_inference | success | 13.377 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx5000_b1_w256 | sglang_beam_search | success | 7.918 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx5000_b2_w256 | sid_gr_inference | success | 14.343 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx5000_b2_w256 | sglang_beam_search | success | 9.364 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx5000_b4_w256 | sid_gr_inference | success | 14.530 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx5000_b4_w256 | sglang_beam_search | success | 9.601 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx5000_b8_w256 | sid_gr_inference | success | 14.652 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| ctx5000_b8_w256 | sglang_beam_search | success | 9.644 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |

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
