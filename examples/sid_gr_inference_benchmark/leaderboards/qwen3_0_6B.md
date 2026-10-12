# qwen3_0_6B leaderboard

Compare each solution's latest latency on the same workload and hardware.

Median batch **ms**; lower is better. The **Fastest** column names the winner for each workload.

Parity counts are workload cases; each reference row checks repeat consistency.

## [NVIDIA RTX PRO 6000 Blackwell Server Edition](../config/hardware/rtx_pro_6000_blackwell_server.yml)

**CPU:** Intel(R) Xeon(R) Gold 6444Y · **Worker cores:** 10 · **Driver:** 615.71.09

| Solution | Exact beam/score match | Different beams or scores | Top-1 mismatches | Not compared |
| --- | --- | --- | --- | --- |
| sid_gr_inference | 0 of 24 cases | 24 of 24 cases | 6 cases | 0 cases |
| sglang_beam_search | 0 of 24 cases | 24 of 24 cases | 6 cases | 0 cases |
| sglang_beam_search_new | 0 of 24 cases | 24 of 24 cases | 9 cases | 0 cases |
| sglang | 24 of 24 cases | 0 of 24 cases | 0 cases | 0 cases |

| Top-1 mismatch case | Candidate |
| --- | --- |
| qwen3_0_6B/ctx1000_b4_w64 | sid_gr_inference |
| qwen3_0_6B/ctx1000_b4_w64 | sglang_beam_search_new |
| qwen3_0_6B/ctx1000_b4_w64 | sglang_beam_search |
| qwen3_0_6B/ctx1000_b8_w128 | sid_gr_inference |
| qwen3_0_6B/ctx1000_b8_w128 | sglang_beam_search_new |
| qwen3_0_6B/ctx1000_b8_w128 | sglang_beam_search |
| qwen3_0_6B/ctx1000_b8_w256 | sid_gr_inference |
| qwen3_0_6B/ctx1000_b8_w256 | sglang_beam_search_new |
| qwen3_0_6B/ctx1000_b8_w256 | sglang_beam_search |
| qwen3_0_6B/ctx1000_b8_w64 | sglang_beam_search_new |
| qwen3_0_6B/ctx1000_b8_w64 | sglang_beam_search |
| qwen3_0_6B/ctx5000_b1_w128 | sid_gr_inference |
| qwen3_0_6B/ctx5000_b1_w256 | sglang_beam_search_new |
| qwen3_0_6B/ctx5000_b1_w256 | sglang_beam_search |
| qwen3_0_6B/ctx5000_b1_w64 | sglang_beam_search_new |
| qwen3_0_6B/ctx5000_b2_w128 | sglang_beam_search_new |
| qwen3_0_6B/ctx5000_b4_w256 | sid_gr_inference |
| qwen3_0_6B/ctx5000_b4_w256 | sglang_beam_search_new |
| qwen3_0_6B/ctx5000_b4_w256 | sglang_beam_search |
| qwen3_0_6B/ctx5000_b4_w64 | sglang_beam_search_new |
| qwen3_0_6B/ctx5000_b8_w128 | sid_gr_inference |



### Latency comparison

| Context | Batch | Beam | sid_gr_inference | sglang_beam_search | sglang_beam_search_new | sglang | Fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 64 | **6.320** | 13.917 | 18.870 | 20.791 | **sid_gr_inference** |
| 1000 | 1 | 128 | **7.671** | 17.817 | 21.425 | 24.021 | **sid_gr_inference** |
| 1000 | 1 | 256 | **10.920** | 27.903 | 25.589 | 34.160 | **sid_gr_inference** |
| 1000 | 2 | 64 | **8.025** | 18.683 | 17.601 | 29.559 | **sid_gr_inference** |
| 1000 | 2 | 128 | **11.549** | 29.042 | 23.287 | 39.241 | **sid_gr_inference** |
| 1000 | 2 | 256 | **18.840** | 42.875 | 35.286 | 54.396 | **sid_gr_inference** |
| 1000 | 4 | 64 | **12.986** | 30.386 | 24.373 | 49.166 | **sid_gr_inference** |
| 1000 | 4 | 128 | **19.884** | 44.430 | 36.159 | 62.828 | **sid_gr_inference** |
| 1000 | 4 | 256 | **33.387** | 71.961 | 44.584 | 90.864 | **sid_gr_inference** |
| 1000 | 8 | 64 | **22.473** | 47.407 | 39.101 | 82.703 | **sid_gr_inference** |
| 1000 | 8 | 128 | **35.531** | 75.846 | 47.787 | 105.278 | **sid_gr_inference** |
| 1000 | 8 | 256 | **61.387** | 131.188 | 74.608 | 165.659 | **sid_gr_inference** |
| 5000 | 1 | 64 | **8.033** | 24.363 | 24.880 | 54.483 | **sid_gr_inference** |
| 5000 | 1 | 128 | **9.376** | 37.834 | 28.361 | 67.239 | **sid_gr_inference** |
| 5000 | 1 | 256 | **13.624** | 77.487 | 39.151 | 105.749 | **sid_gr_inference** |
| 5000 | 2 | 64 | **11.646** | 39.424 | 24.833 | 100.390 | **sid_gr_inference** |
| 5000 | 2 | 128 | **15.311** | 79.175 | 35.949 | 140.191 | **sid_gr_inference** |
| 5000 | 2 | 256 | **24.242** | 106.738 | 38.917 | 169.892 | **sid_gr_inference** |
| 5000 | 4 | 64 | **20.367** | 82.388 | 36.409 | 190.018 | **sid_gr_inference** |
| 5000 | 4 | 128 | **27.344** | 110.512 | 41.499 | 229.231 | **sid_gr_inference** |
| 5000 | 4 | 256 | **43.530** | 199.626 | 54.833 | 318.185 | **sid_gr_inference** |
| 5000 | 8 | 64 | **37.224** | 119.639 | 46.500 | 361.243 | **sid_gr_inference** |
| 5000 | 8 | 128 | **50.232** | 206.812 | 59.747 | 447.399 | **sid_gr_inference** |
| 5000 | 8 | 256 | **86.033** | 393.593 | 93.893 | 637.220 | **sid_gr_inference** |

<details>
<summary>Output parity, including reference repeat consistency</summary>

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sid_gr_inference | different | 100.000 | 95.312 | 95.312 | 10.156 | 0.266 |
| ctx1000_b1_w128 | sglang_beam_search | different | 100.000 | 91.406 | 91.406 | 10.156 | 0.33 |
| ctx1000_b1_w128 | sglang_beam_search_new | different | 100.000 | 93.750 | 93.750 | 13.281 | 0.345 |
| ctx1000_b1_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 12.500 | 0.234 |
| ctx1000_b1_w256 | sglang_beam_search_new | different | 100.000 | 94.531 | 94.531 | 7.031 | 0.43 |
| ctx1000_b1_w256 | sglang_beam_search | different | 100.000 | 94.141 | 94.141 | 10.547 | 0.299 |
| ctx1000_b1_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w64 | sid_gr_inference | different | 100.000 | 96.875 | 96.875 | 23.438 | 0.282 |
| ctx1000_b1_w64 | sglang_beam_search | different | 100.000 | 89.062 | 89.062 | 23.438 | 0.301 |
| ctx1000_b1_w64 | sglang_beam_search_new | different | 100.000 | 92.188 | 92.188 | 23.438 | 0.158 |
| ctx1000_b1_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w128 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 10.547 | 0.296 |
| ctx1000_b2_w128 | sglang_beam_search_new | different | 100.000 | 96.094 | 96.094 | 10.938 | 0.237 |
| ctx1000_b2_w128 | sglang_beam_search | different | 100.000 | 95.312 | 93.750 | 13.281 | 0.323 |
| ctx1000_b2_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w256 | sid_gr_inference | different | 100.000 | 96.289 | 95.703 | 5.664 | 0.362 |
| ctx1000_b2_w256 | sglang_beam_search_new | different | 100.000 | 97.266 | 96.875 | 9.766 | 0.268 |
| ctx1000_b2_w256 | sglang_beam_search | different | 100.000 | 97.266 | 96.094 | 7.031 | 0.375 |
| ctx1000_b2_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w64 | sid_gr_inference | different | 100.000 | 92.188 | 90.625 | 19.531 | 0.26 |
| ctx1000_b2_w64 | sglang_beam_search_new | different | 100.000 | 91.406 | 87.500 | 23.438 | 0.237 |
| ctx1000_b2_w64 | sglang_beam_search | different | 100.000 | 95.312 | 93.750 | 25.000 | 0.33 |
| ctx1000_b2_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w128 | sid_gr_inference | different | 100.000 | 96.680 | 95.312 | 12.109 | 0.334 |
| ctx1000_b4_w128 | sglang_beam_search_new | different | 100.000 | 93.750 | 90.625 | 9.570 | 0.366 |
| ctx1000_b4_w128 | sglang_beam_search | different | 100.000 | 94.531 | 89.062 | 13.086 | 0.254 |
| ctx1000_b4_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w256 | sid_gr_inference | different | 100.000 | 95.215 | 94.141 | 7.715 | 0.358 |
| ctx1000_b4_w256 | sglang_beam_search_new | different | 100.000 | 94.531 | 89.844 | 6.934 | 0.403 |
| ctx1000_b4_w256 | sglang_beam_search | different | 100.000 | 95.703 | 94.141 | 8.105 | 0.275 |
| ctx1000_b4_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w64 | sid_gr_inference | different | 75.000 | 94.531 | 92.188 | 22.656 | 0.36 |
| ctx1000_b4_w64 | sglang_beam_search_new | different | 75.000 | 93.359 | 89.062 | 12.109 | 0.39 |
| ctx1000_b4_w64 | sglang_beam_search | different | 75.000 | 96.094 | 93.750 | 17.578 | 0.334 |
| ctx1000_b4_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w128 | sid_gr_inference | different | 87.500 | 95.215 | 92.188 | 13.184 | 0.333 |
| ctx1000_b8_w128 | sglang_beam_search_new | different | 87.500 | 95.605 | 93.750 | 13.574 | 0.416 |
| ctx1000_b8_w128 | sglang_beam_search | different | 87.500 | 95.410 | 89.062 | 10.059 | 0.512 |
| ctx1000_b8_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w256 | sid_gr_inference | different | 87.500 | 95.361 | 91.406 | 6.934 | 0.374 |
| ctx1000_b8_w256 | sglang_beam_search_new | different | 75.000 | 95.410 | 92.969 | 8.057 | 0.369 |
| ctx1000_b8_w256 | sglang_beam_search | different | 75.000 | 94.678 | 88.672 | 6.738 | 0.41 |
| ctx1000_b8_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w64 | sid_gr_inference | different | 100.000 | 94.336 | 92.188 | 19.727 | 0.528 |
| ctx1000_b8_w64 | sglang_beam_search_new | different | 75.000 | 94.141 | 90.625 | 20.898 | 0.372 |
| ctx1000_b8_w64 | sglang_beam_search | different | 75.000 | 94.336 | 89.062 | 20.312 | 0.342 |
| ctx1000_b8_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w128 | sid_gr_inference | different | 0.000 | 96.875 | 96.875 | 16.406 | 0.529 |
| ctx5000_b1_w128 | sglang_beam_search_new | different | 100.000 | 96.875 | 96.875 | 11.719 | 0.444 |
| ctx5000_b1_w128 | sglang_beam_search | different | 100.000 | 96.875 | 96.875 | 17.188 | 0.303 |
| ctx5000_b1_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 6.641 | 0.207 |
| ctx5000_b1_w256 | sglang_beam_search_new | different | 0.000 | 95.703 | 95.703 | 8.984 | 0.289 |
| ctx5000_b1_w256 | sglang_beam_search | different | 0.000 | 96.484 | 96.484 | 9.766 | 0.185 |
| ctx5000_b1_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w64 | sid_gr_inference | different | 100.000 | 95.312 | 95.312 | 26.562 | 0.252 |
| ctx5000_b1_w64 | sglang_beam_search | different | 100.000 | 92.188 | 92.188 | 14.062 | 0.202 |
| ctx5000_b1_w64 | sglang_beam_search_new | different | 0.000 | 95.312 | 95.312 | 28.125 | 0.222 |
| ctx5000_b1_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w128 | sid_gr_inference | different | 100.000 | 96.094 | 95.312 | 11.719 | 0.23 |
| ctx5000_b2_w128 | sglang_beam_search_new | different | 50.000 | 98.438 | 98.438 | 16.406 | 0.237 |
| ctx5000_b2_w128 | sglang_beam_search | different | 100.000 | 95.703 | 94.531 | 14.062 | 0.238 |
| ctx5000_b2_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w256 | sid_gr_inference | different | 100.000 | 96.289 | 95.703 | 7.812 | 0.253 |
| ctx5000_b2_w256 | sglang_beam_search_new | different | 100.000 | 95.703 | 94.141 | 7.227 | 0.258 |
| ctx5000_b2_w256 | sglang_beam_search | different | 100.000 | 94.727 | 93.359 | 9.375 | 0.335 |
| ctx5000_b2_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w64 | sid_gr_inference | different | 100.000 | 93.750 | 92.188 | 21.875 | 0.266 |
| ctx5000_b2_w64 | sglang_beam_search_new | different | 100.000 | 96.094 | 95.312 | 32.031 | 0.334 |
| ctx5000_b2_w64 | sglang_beam_search | different | 100.000 | 93.750 | 93.750 | 19.531 | 0.34 |
| ctx5000_b2_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w128 | sid_gr_inference | different | 100.000 | 95.508 | 94.531 | 12.109 | 0.545 |
| ctx5000_b4_w128 | sglang_beam_search_new | different | 100.000 | 95.703 | 94.531 | 15.625 | 0.525 |
| ctx5000_b4_w128 | sglang_beam_search | different | 100.000 | 95.117 | 91.406 | 14.062 | 0.436 |
| ctx5000_b4_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w256 | sid_gr_inference | different | 75.000 | 95.605 | 94.531 | 8.594 | 0.317 |
| ctx5000_b4_w256 | sglang_beam_search_new | different | 75.000 | 95.215 | 93.359 | 7.910 | 0.454 |
| ctx5000_b4_w256 | sglang_beam_search | different | 75.000 | 95.020 | 93.359 | 6.836 | 0.375 |
| ctx5000_b4_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w64 | sid_gr_inference | different | 100.000 | 94.531 | 92.188 | 21.875 | 0.359 |
| ctx5000_b4_w64 | sglang_beam_search_new | different | 75.000 | 94.141 | 92.188 | 20.703 | 0.35 |
| ctx5000_b4_w64 | sglang_beam_search | different | 100.000 | 94.141 | 92.188 | 19.141 | 0.272 |
| ctx5000_b4_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w128 | sid_gr_inference | different | 87.500 | 94.531 | 91.406 | 10.645 | 0.48 |
| ctx5000_b8_w128 | sglang_beam_search_new | different | 100.000 | 94.922 | 92.188 | 13.574 | 0.42 |
| ctx5000_b8_w128 | sglang_beam_search | different | 100.000 | 94.434 | 90.625 | 15.234 | 0.345 |
| ctx5000_b8_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w256 | sid_gr_inference | different | 100.000 | 95.654 | 94.141 | 8.057 | 0.385 |
| ctx5000_b8_w256 | sglang_beam_search_new | different | 100.000 | 95.264 | 94.141 | 6.885 | 0.648 |
| ctx5000_b8_w256 | sglang_beam_search | different | 100.000 | 95.850 | 94.922 | 8.252 | 0.468 |
| ctx5000_b8_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w64 | sid_gr_inference | different | 100.000 | 95.312 | 92.188 | 23.242 | 0.529 |
| ctx5000_b8_w64 | sglang_beam_search_new | different | 100.000 | 94.727 | 92.188 | 24.414 | 0.558 |
| ctx5000_b8_w64 | sglang_beam_search | different | 100.000 | 94.727 | 93.750 | 23.242 | 0.485 |
| ctx5000_b8_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |

</details>

<details>
<summary>Throughput, status and source runs</summary>

| Case | Candidate | Status | Requests/s | Reference | Source run |
| --- | --- | --- | --- | --- | --- |
| ctx1000_b1_w128 | sid_gr_inference | success | 130.362 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w128 | sglang_beam_search | success | 56.125 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w128 | sglang_beam_search_new | success | 46.675 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w128 | sglang | success | 41.630 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w256 | sid_gr_inference | success | 91.576 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w256 | sglang_beam_search_new | success | 39.079 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w256 | sglang_beam_search | success | 35.839 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w256 | sglang | success | 29.274 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w64 | sid_gr_inference | success | 158.228 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w64 | sglang_beam_search | success | 71.856 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w64 | sglang_beam_search_new | success | 52.993 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w64 | sglang | success | 48.097 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w128 | sid_gr_inference | success | 173.168 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w128 | sglang_beam_search_new | success | 85.885 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w128 | sglang_beam_search | success | 68.867 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w128 | sglang | success | 50.967 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w256 | sid_gr_inference | success | 106.160 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w256 | sglang_beam_search_new | success | 56.680 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w256 | sglang_beam_search | success | 46.647 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w256 | sglang | success | 36.767 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w64 | sid_gr_inference | success | 249.211 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w64 | sglang_beam_search_new | success | 113.632 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w64 | sglang_beam_search | success | 107.050 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w64 | sglang | success | 67.662 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w128 | sid_gr_inference | success | 201.168 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w128 | sglang_beam_search_new | success | 110.623 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w128 | sglang_beam_search | success | 90.030 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w128 | sglang | success | 63.666 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w256 | sid_gr_inference | success | 119.806 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w256 | sglang_beam_search_new | success | 89.718 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w256 | sglang_beam_search | success | 55.585 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w256 | sglang | success | 44.022 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w64 | sid_gr_inference | success | 308.017 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w64 | sglang_beam_search_new | success | 164.117 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w64 | sglang_beam_search | success | 131.639 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w64 | sglang | success | 81.357 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w128 | sid_gr_inference | success | 225.156 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w128 | sglang_beam_search_new | success | 167.411 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w128 | sglang_beam_search | success | 105.478 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w128 | sglang | success | 75.989 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w256 | sid_gr_inference | success | 130.321 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w256 | sglang_beam_search_new | success | 107.227 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w256 | sglang_beam_search | success | 60.981 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w256 | sglang | success | 48.292 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w64 | sid_gr_inference | success | 355.979 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w64 | sglang_beam_search_new | success | 204.596 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w64 | sglang_beam_search | success | 168.751 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w64 | sglang | success | 96.732 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w128 | sid_gr_inference | success | 106.657 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w128 | sglang_beam_search_new | success | 35.259 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w128 | sglang_beam_search | success | 26.432 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w128 | sglang | success | 14.872 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w256 | sid_gr_inference | success | 73.401 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w256 | sglang_beam_search_new | success | 25.542 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w256 | sglang_beam_search | success | 12.905 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w256 | sglang | success | 9.456 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w64 | sid_gr_inference | success | 124.481 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w64 | sglang_beam_search | success | 41.045 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w64 | sglang_beam_search_new | success | 40.193 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w64 | sglang | success | 18.355 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w128 | sid_gr_inference | success | 130.629 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w128 | sglang_beam_search_new | success | 55.634 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w128 | sglang_beam_search | success | 25.260 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w128 | sglang | success | 14.266 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w256 | sid_gr_inference | success | 82.500 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w256 | sglang_beam_search_new | success | 51.391 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w256 | sglang_beam_search | success | 18.738 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w256 | sglang | success | 11.772 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w64 | sid_gr_inference | success | 171.729 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w64 | sglang_beam_search_new | success | 80.537 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w64 | sglang_beam_search | success | 50.730 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w64 | sglang | success | 19.922 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w128 | sid_gr_inference | success | 146.284 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w128 | sglang_beam_search_new | success | 96.389 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w128 | sglang_beam_search | success | 36.195 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w128 | sglang | success | 17.450 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w256 | sid_gr_inference | success | 91.891 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w256 | sglang_beam_search_new | success | 72.949 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w256 | sglang_beam_search | success | 20.037 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w256 | sglang | success | 12.571 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w64 | sid_gr_inference | success | 196.393 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w64 | sglang_beam_search_new | success | 109.864 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w64 | sglang_beam_search | success | 48.551 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w64 | sglang | success | 21.051 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w128 | sid_gr_inference | success | 159.260 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w128 | sglang_beam_search_new | success | 133.898 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w128 | sglang_beam_search | success | 38.682 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w128 | sglang | success | 17.881 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w256 | sid_gr_inference | success | 92.988 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w256 | sglang_beam_search_new | success | 85.203 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w256 | sglang_beam_search | success | 20.326 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w256 | sglang | success | 12.555 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w64 | sid_gr_inference | success | 214.915 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w64 | sglang_beam_search_new | success | 172.044 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w64 | sglang_beam_search | success | 66.868 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w64 | sglang | success | 22.146 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |

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
