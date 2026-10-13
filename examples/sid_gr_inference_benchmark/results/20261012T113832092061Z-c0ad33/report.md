# SID-GR benchmark

Run `20261012T113832092061Z-c0ad33` · [Run profile](../../config/runs/four_engine.yml)

[Latency](#latency-comparison) · [Parity](#output-parity) · [GPU diagnosis](#gpu-diagnosis) · [Environment](#run-details) · [Artifacts](#artifacts)

| Run summary | Meaning |
| --- | --- |
| Status | completed |
| Workload | 48 context / batch / beam combinations |
| Solutions compared | sid_gr_inference, sglang, sglang_beam_search, sglang_beam_search_new |
| Progress | 192 of 192 solution–workload runs finished |
| Execution checks | 192 passed; 0 failed or unsupported |

Each solution runs every workload combination, with the warmups and repeats listed below.

Parity counts below include finished workload cases only; the reference row checks repeat consistency.

| Solution | Exact beam/score match | Different beams or scores | Top-1 mismatches | Not compared | Finished runs | Status |
| --- | --- | --- | --- | --- | --- | --- |
| sid_gr_inference | 0 of 48 cases | 48 of 48 cases | 13 cases | 0 cases | 48 of 48 | completed |
| sglang | 48 of 48 cases | 0 of 48 cases | 0 cases | 0 cases | 48 of 48 | completed |
| sglang_beam_search | 0 of 48 cases | 48 of 48 cases | 13 cases | 0 cases | 48 of 48 | completed |
| sglang_beam_search_new | 0 of 48 cases | 48 of 48 cases | 16 cases | 0 cases | 48 of 48 | completed |

| Top-1 mismatch case | Candidate |
| --- | --- |
| qwen3_0_6B/ctx1000_b4_w64 | sglang_beam_search |
| qwen3_0_6B/ctx1000_b4_w64 | sglang_beam_search_new |
| qwen3_0_6B/ctx1000_b4_w64 | sid_gr_inference |
| qwen3_0_6B/ctx1000_b8_w64 | sglang_beam_search |
| qwen3_0_6B/ctx1000_b8_w64 | sglang_beam_search_new |
| qwen3_0_6B/ctx1000_b8_w128 | sglang_beam_search |
| qwen3_0_6B/ctx1000_b8_w128 | sglang_beam_search_new |
| qwen3_0_6B/ctx1000_b8_w128 | sid_gr_inference |
| qwen3_0_6B/ctx1000_b8_w256 | sglang_beam_search_new |
| qwen3_0_6B/ctx1000_b8_w256 | sid_gr_inference |
| qwen3_0_6B/ctx1000_b8_w256 | sglang_beam_search |
| qwen3_0_6B/ctx5000_b1_w64 | sglang_beam_search_new |
| qwen3_0_6B/ctx5000_b1_w128 | sid_gr_inference |
| qwen3_0_6B/ctx5000_b1_w256 | sglang_beam_search |
| qwen3_0_6B/ctx5000_b1_w256 | sglang_beam_search_new |
| qwen3_0_6B/ctx5000_b2_w128 | sglang_beam_search_new |
| qwen3_0_6B/ctx5000_b4_w64 | sglang_beam_search_new |
| qwen3_0_6B/ctx5000_b4_w256 | sid_gr_inference |
| qwen3_0_6B/ctx5000_b4_w256 | sglang_beam_search |
| qwen3_0_6B/ctx5000_b4_w256 | sglang_beam_search_new |
| qwen3_0_6B/ctx5000_b8_w128 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b1_w64 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b1_w64 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b1_w128 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b1_w128 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b2_w64 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b2_w64 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b2_w128 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b2_w256 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b2_w256 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b2_w256 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b4_w128 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b4_w128 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b4_w256 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b4_w256 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b4_w256 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b8_w64 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b8_w64 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b8_w64 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b8_w128 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b8_w128 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b8_w128 | sid_gr_inference |

**Reference:** `sglang` · Offline batches on one GPU

Hardware profile: [rtx_pro_6000_blackwell_server](../../config/hardware/rtx_pro_6000_blackwell_server.yml).

## Latency comparison

Median batch **ms**; lower is better. **Bold** = lowest recorded median. Pending = no finished result. — = unavailable.

### qwen3_0_6B

| Context | Batch | Beam | sid_gr_inference | sglang | sglang_beam_search | sglang_beam_search_new | Fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 64 | **6.320** | 20.791 | 13.917 | 18.870 | **sid_gr_inference** |
| 1000 | 1 | 128 | **7.671** | 24.021 | 17.817 | 21.425 | **sid_gr_inference** |
| 1000 | 1 | 256 | **10.920** | 34.160 | 27.903 | 25.589 | **sid_gr_inference** |
| 1000 | 2 | 64 | **8.025** | 29.559 | 18.683 | 17.601 | **sid_gr_inference** |
| 1000 | 2 | 128 | **11.549** | 39.241 | 29.042 | 23.287 | **sid_gr_inference** |
| 1000 | 2 | 256 | **18.840** | 54.396 | 42.875 | 35.286 | **sid_gr_inference** |
| 1000 | 4 | 64 | **12.986** | 49.166 | 30.386 | 24.373 | **sid_gr_inference** |
| 1000 | 4 | 128 | **19.884** | 62.828 | 44.430 | 36.159 | **sid_gr_inference** |
| 1000 | 4 | 256 | **33.387** | 90.864 | 71.961 | 44.584 | **sid_gr_inference** |
| 1000 | 8 | 64 | **22.473** | 82.703 | 47.407 | 39.101 | **sid_gr_inference** |
| 1000 | 8 | 128 | **35.531** | 105.278 | 75.846 | 47.787 | **sid_gr_inference** |
| 1000 | 8 | 256 | **61.387** | 165.659 | 131.188 | 74.608 | **sid_gr_inference** |
| 5000 | 1 | 64 | **8.033** | 54.483 | 24.363 | 24.880 | **sid_gr_inference** |
| 5000 | 1 | 128 | **9.376** | 67.239 | 37.834 | 28.361 | **sid_gr_inference** |
| 5000 | 1 | 256 | **13.624** | 105.749 | 77.487 | 39.151 | **sid_gr_inference** |
| 5000 | 2 | 64 | **11.646** | 100.390 | 39.424 | 24.833 | **sid_gr_inference** |
| 5000 | 2 | 128 | **15.311** | 140.191 | 79.175 | 35.949 | **sid_gr_inference** |
| 5000 | 2 | 256 | **24.242** | 169.892 | 106.738 | 38.917 | **sid_gr_inference** |
| 5000 | 4 | 64 | **20.367** | 190.018 | 82.388 | 36.409 | **sid_gr_inference** |
| 5000 | 4 | 128 | **27.344** | 229.231 | 110.512 | 41.499 | **sid_gr_inference** |
| 5000 | 4 | 256 | **43.530** | 318.185 | 199.626 | 54.833 | **sid_gr_inference** |
| 5000 | 8 | 64 | **37.224** | 361.243 | 119.639 | 46.500 | **sid_gr_inference** |
| 5000 | 8 | 128 | **50.232** | 447.399 | 206.812 | 59.747 | **sid_gr_inference** |
| 5000 | 8 | 256 | **86.033** | 637.220 | 393.593 | 93.893 | **sid_gr_inference** |

### qwen3_1_7B

| Context | Batch | Beam | sid_gr_inference | sglang | sglang_beam_search | sglang_beam_search_new | Fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 64 | **9.682** | 31.425 | 18.255 | 23.063 | **sid_gr_inference** |
| 1000 | 1 | 128 | **11.090** | 34.607 | 21.990 | 24.534 | **sid_gr_inference** |
| 1000 | 1 | 256 | **15.868** | 46.678 | 34.228 | 31.601 | **sid_gr_inference** |
| 1000 | 2 | 64 | **11.382** | 45.733 | 23.220 | 21.694 | **sid_gr_inference** |
| 1000 | 2 | 128 | **16.459** | 56.902 | 35.183 | 28.216 | **sid_gr_inference** |
| 1000 | 2 | 256 | **28.105** | 73.205 | 53.195 | 39.290 | **sid_gr_inference** |
| 1000 | 4 | 64 | **17.817** | 74.804 | 37.061 | 29.677 | **sid_gr_inference** |
| 1000 | 4 | 128 | **29.217** | 89.991 | 55.465 | 40.427 | **sid_gr_inference** |
| 1000 | 4 | 256 | **49.799** | 123.284 | 85.638 | 60.272 | **sid_gr_inference** |
| 1000 | 8 | 64 | **31.808** | 129.553 | 58.644 | 43.731 | **sid_gr_inference** |
| 1000 | 8 | 128 | **52.064** | 160.896 | 90.123 | 63.774 | **sid_gr_inference** |
| 1000 | 8 | 256 | **88.593** | 230.797 | 155.543 | 102.508 | **sid_gr_inference** |
| 5000 | 1 | 64 | **11.346** | 82.708 | 29.101 | 28.674 | **sid_gr_inference** |
| 5000 | 1 | 128 | **12.881** | 94.604 | 41.748 | 32.419 | **sid_gr_inference** |
| 5000 | 1 | 256 | **18.476** | 130.143 | 83.531 | 44.315 | **sid_gr_inference** |
| 5000 | 2 | 64 | **15.161** | 156.851 | 44.654 | 29.683 | **sid_gr_inference** |
| 5000 | 2 | 128 | **20.143** | 191.279 | 79.591 | 41.549 | **sid_gr_inference** |
| 5000 | 2 | 256 | **33.430** | 234.368 | 117.011 | 50.243 | **sid_gr_inference** |
| 5000 | 4 | 64 | **25.283** | 305.269 | 89.309 | 43.326 | **sid_gr_inference** |
| 5000 | 4 | 128 | **36.623** | 347.044 | 120.496 | 51.847 | **sid_gr_inference** |
| 5000 | 4 | 256 | **59.507** | 443.267 | 216.492 | 70.795 | **sid_gr_inference** |
| 5000 | 8 | 64 | **46.440** | 589.925 | 127.440 | 57.388 | **sid_gr_inference** |
| 5000 | 8 | 128 | **66.478** | 683.185 | 224.088 | 77.105 | **sid_gr_inference** |
| 5000 | 8 | 256 | **110.685** | 886.314 | 422.067 | 119.025 | **sid_gr_inference** |



<details>
<summary>All measurements</summary>

### qwen3_0_6B

| Case | Candidate | Status | Median ms | IQR ms | Δ latency % | Requests/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sid_gr_inference | success | 7.671 | 0.020 | -68.066 | 130.362 |
| ctx1000_b1_w128 | sglang | success | 24.021 | 0.125 | 0.000 | 41.630 |
| ctx1000_b1_w128 | sglang_beam_search | success | 17.817 | 0.095 | -25.826 | 56.125 |
| ctx1000_b1_w128 | sglang_beam_search_new | success | 21.425 | 0.088 | -10.810 | 46.675 |
| ctx1000_b1_w256 | sid_gr_inference | success | 10.920 | 0.031 | -68.033 | 91.576 |
| ctx1000_b1_w256 | sglang | success | 34.160 | 0.313 | 0.000 | 29.274 |
| ctx1000_b1_w256 | sglang_beam_search | success | 27.903 | 0.195 | -18.318 | 35.839 |
| ctx1000_b1_w256 | sglang_beam_search_new | success | 25.589 | 0.163 | -25.090 | 39.079 |
| ctx1000_b1_w64 | sid_gr_inference | success | 6.320 | 0.030 | -69.603 | 158.228 |
| ctx1000_b1_w64 | sglang | success | 20.791 | 0.155 | 0.000 | 48.097 |
| ctx1000_b1_w64 | sglang_beam_search | success | 13.917 | 0.077 | -33.065 | 71.856 |
| ctx1000_b1_w64 | sglang_beam_search_new | success | 18.870 | 0.055 | -9.238 | 52.993 |
| ctx1000_b2_w128 | sid_gr_inference | success | 11.549 | 0.352 | -70.568 | 173.168 |
| ctx1000_b2_w128 | sglang | success | 39.241 | 0.243 | 0.000 | 50.967 |
| ctx1000_b2_w128 | sglang_beam_search | success | 29.042 | 0.130 | -25.992 | 68.867 |
| ctx1000_b2_w128 | sglang_beam_search_new | success | 23.287 | 0.083 | -40.657 | 85.885 |
| ctx1000_b2_w256 | sid_gr_inference | success | 18.840 | 0.097 | -65.366 | 106.160 |
| ctx1000_b2_w256 | sglang | success | 54.396 | 0.454 | 0.000 | 36.767 |
| ctx1000_b2_w256 | sglang_beam_search | success | 42.875 | 0.244 | -21.180 | 46.647 |
| ctx1000_b2_w256 | sglang_beam_search_new | success | 35.286 | 0.295 | -35.131 | 56.680 |
| ctx1000_b2_w64 | sid_gr_inference | success | 8.025 | 0.036 | -72.850 | 249.211 |
| ctx1000_b2_w64 | sglang | success | 29.559 | 0.090 | 0.000 | 67.662 |
| ctx1000_b2_w64 | sglang_beam_search | success | 18.683 | 0.047 | -36.794 | 107.050 |
| ctx1000_b2_w64 | sglang_beam_search_new | success | 17.601 | 0.050 | -40.455 | 113.632 |
| ctx1000_b4_w128 | sid_gr_inference | success | 19.884 | 0.134 | -68.352 | 201.168 |
| ctx1000_b4_w128 | sglang | success | 62.828 | 0.460 | 0.000 | 63.666 |
| ctx1000_b4_w128 | sglang_beam_search | success | 44.430 | 0.132 | -29.284 | 90.030 |
| ctx1000_b4_w128 | sglang_beam_search_new | success | 36.159 | 0.155 | -42.448 | 110.623 |
| ctx1000_b4_w256 | sid_gr_inference | success | 33.387 | 0.108 | -63.256 | 119.806 |
| ctx1000_b4_w256 | sglang | success | 90.864 | 2.429 | 0.000 | 44.022 |
| ctx1000_b4_w256 | sglang_beam_search | success | 71.961 | 2.021 | -20.803 | 55.585 |
| ctx1000_b4_w256 | sglang_beam_search_new | success | 44.584 | 0.391 | -50.933 | 89.718 |
| ctx1000_b4_w64 | sid_gr_inference | success | 12.986 | 0.049 | -73.587 | 308.017 |
| ctx1000_b4_w64 | sglang | success | 49.166 | 0.250 | 0.000 | 81.357 |
| ctx1000_b4_w64 | sglang_beam_search | success | 30.386 | 0.139 | -38.197 | 131.639 |
| ctx1000_b4_w64 | sglang_beam_search_new | success | 24.373 | 0.137 | -50.428 | 164.117 |
| ctx1000_b8_w128 | sid_gr_inference | success | 35.531 | 0.110 | -66.250 | 225.156 |
| ctx1000_b8_w128 | sglang | success | 105.278 | 3.940 | 0.000 | 75.989 |
| ctx1000_b8_w128 | sglang_beam_search | success | 75.846 | 1.567 | -27.957 | 105.478 |
| ctx1000_b8_w128 | sglang_beam_search_new | success | 47.787 | 0.280 | -54.609 | 167.411 |
| ctx1000_b8_w256 | sid_gr_inference | success | 61.387 | 0.187 | -62.944 | 130.321 |
| ctx1000_b8_w256 | sglang | success | 165.659 | 4.933 | 0.000 | 48.292 |
| ctx1000_b8_w256 | sglang_beam_search | success | 131.188 | 2.312 | -20.809 | 60.981 |
| ctx1000_b8_w256 | sglang_beam_search_new | success | 74.608 | 0.520 | -54.963 | 107.227 |
| ctx1000_b8_w64 | sid_gr_inference | success | 22.473 | 0.135 | -72.827 | 355.979 |
| ctx1000_b8_w64 | sglang | success | 82.703 | 0.605 | 0.000 | 96.732 |
| ctx1000_b8_w64 | sglang_beam_search | success | 47.407 | 0.146 | -42.678 | 168.751 |
| ctx1000_b8_w64 | sglang_beam_search_new | success | 39.101 | 0.221 | -52.721 | 204.596 |
| ctx5000_b1_w128 | sid_gr_inference | success | 9.376 | 0.034 | -86.056 | 106.657 |
| ctx5000_b1_w128 | sglang | success | 67.239 | 1.847 | 0.000 | 14.872 |
| ctx5000_b1_w128 | sglang_beam_search | success | 37.834 | 0.158 | -43.733 | 26.432 |
| ctx5000_b1_w128 | sglang_beam_search_new | success | 28.361 | 0.134 | -57.820 | 35.259 |
| ctx5000_b1_w256 | sid_gr_inference | success | 13.624 | 0.054 | -87.117 | 73.401 |
| ctx5000_b1_w256 | sglang | success | 105.749 | 6.660 | 0.000 | 9.456 |
| ctx5000_b1_w256 | sglang_beam_search | success | 77.487 | 0.139 | -26.726 | 12.905 |
| ctx5000_b1_w256 | sglang_beam_search_new | success | 39.151 | 0.194 | -62.978 | 25.542 |
| ctx5000_b1_w64 | sid_gr_inference | success | 8.033 | 0.062 | -85.255 | 124.481 |
| ctx5000_b1_w64 | sglang | success | 54.483 | 0.746 | 0.000 | 18.355 |
| ctx5000_b1_w64 | sglang_beam_search | success | 24.363 | 0.196 | -55.282 | 41.045 |
| ctx5000_b1_w64 | sglang_beam_search_new | success | 24.880 | 0.162 | -54.334 | 40.193 |
| ctx5000_b2_w128 | sid_gr_inference | success | 15.311 | 0.059 | -89.079 | 130.629 |
| ctx5000_b2_w128 | sglang | success | 140.191 | 11.856 | 0.000 | 14.266 |
| ctx5000_b2_w128 | sglang_beam_search | success | 79.175 | 0.095 | -43.523 | 25.260 |
| ctx5000_b2_w128 | sglang_beam_search_new | success | 35.949 | 0.131 | -74.357 | 55.634 |
| ctx5000_b2_w256 | sid_gr_inference | success | 24.242 | 0.111 | -85.731 | 82.500 |
| ctx5000_b2_w256 | sglang | success | 169.892 | 2.163 | 0.000 | 11.772 |
| ctx5000_b2_w256 | sglang_beam_search | success | 106.738 | 1.996 | -37.173 | 18.738 |
| ctx5000_b2_w256 | sglang_beam_search_new | success | 38.917 | 0.225 | -77.093 | 51.391 |
| ctx5000_b2_w64 | sid_gr_inference | success | 11.646 | 0.041 | -88.399 | 171.729 |
| ctx5000_b2_w64 | sglang | success | 100.390 | 3.349 | 0.000 | 19.922 |
| ctx5000_b2_w64 | sglang_beam_search | success | 39.424 | 0.106 | -60.729 | 50.730 |
| ctx5000_b2_w64 | sglang_beam_search_new | success | 24.833 | 0.095 | -75.263 | 80.537 |
| ctx5000_b4_w128 | sid_gr_inference | success | 27.344 | 0.129 | -88.071 | 146.284 |
| ctx5000_b4_w128 | sglang | success | 229.231 | 0.741 | 0.000 | 17.450 |
| ctx5000_b4_w128 | sglang_beam_search | success | 110.512 | 2.722 | -51.790 | 36.195 |
| ctx5000_b4_w128 | sglang_beam_search_new | success | 41.499 | 0.218 | -81.897 | 96.389 |
| ctx5000_b4_w256 | sid_gr_inference | success | 43.530 | 0.104 | -86.319 | 91.891 |
| ctx5000_b4_w256 | sglang | success | 318.185 | 0.950 | 0.000 | 12.571 |
| ctx5000_b4_w256 | sglang_beam_search | success | 199.626 | 0.558 | -37.261 | 20.037 |
| ctx5000_b4_w256 | sglang_beam_search_new | success | 54.833 | 0.585 | -82.767 | 72.949 |
| ctx5000_b4_w64 | sid_gr_inference | success | 20.367 | 0.057 | -89.281 | 196.393 |
| ctx5000_b4_w64 | sglang | success | 190.018 | 8.290 | 0.000 | 21.051 |
| ctx5000_b4_w64 | sglang_beam_search | success | 82.388 | 0.133 | -56.642 | 48.551 |
| ctx5000_b4_w64 | sglang_beam_search_new | success | 36.409 | 0.115 | -80.839 | 109.864 |
| ctx5000_b8_w128 | sid_gr_inference | success | 50.232 | 0.067 | -88.772 | 159.260 |
| ctx5000_b8_w128 | sglang | success | 447.399 | 1.245 | 0.000 | 17.881 |
| ctx5000_b8_w128 | sglang_beam_search | success | 206.812 | 1.861 | -53.775 | 38.682 |
| ctx5000_b8_w128 | sglang_beam_search_new | success | 59.747 | 0.316 | -86.646 | 133.898 |
| ctx5000_b8_w256 | sid_gr_inference | success | 86.033 | 0.206 | -86.499 | 92.988 |
| ctx5000_b8_w256 | sglang | success | 637.220 | 2.744 | 0.000 | 12.555 |
| ctx5000_b8_w256 | sglang_beam_search | success | 393.593 | 1.562 | -38.233 | 20.326 |
| ctx5000_b8_w256 | sglang_beam_search_new | success | 93.893 | 0.579 | -85.265 | 85.203 |
| ctx5000_b8_w64 | sid_gr_inference | success | 37.224 | 0.122 | -89.696 | 214.915 |
| ctx5000_b8_w64 | sglang | success | 361.243 | 0.833 | 0.000 | 22.146 |
| ctx5000_b8_w64 | sglang_beam_search | success | 119.639 | 1.895 | -66.881 | 66.868 |
| ctx5000_b8_w64 | sglang_beam_search_new | success | 46.500 | 0.215 | -87.128 | 172.044 |

### qwen3_1_7B

| Case | Candidate | Status | Median ms | IQR ms | Δ latency % | Requests/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sid_gr_inference | success | 11.090 | 0.020 | -67.953 | 90.168 |
| ctx1000_b1_w128 | sglang | success | 34.607 | 0.121 | 0.000 | 28.896 |
| ctx1000_b1_w128 | sglang_beam_search | success | 21.990 | 0.048 | -36.457 | 45.475 |
| ctx1000_b1_w128 | sglang_beam_search_new | success | 24.534 | 0.074 | -29.107 | 40.760 |
| ctx1000_b1_w256 | sid_gr_inference | success | 15.868 | 0.023 | -66.006 | 63.021 |
| ctx1000_b1_w256 | sglang | success | 46.678 | 0.178 | 0.000 | 21.424 |
| ctx1000_b1_w256 | sglang_beam_search | success | 34.228 | 0.127 | -26.672 | 29.216 |
| ctx1000_b1_w256 | sglang_beam_search_new | success | 31.601 | 0.215 | -32.299 | 31.644 |
| ctx1000_b1_w64 | sid_gr_inference | success | 9.682 | 0.041 | -69.190 | 103.284 |
| ctx1000_b1_w64 | sglang | success | 31.425 | 0.125 | 0.000 | 31.822 |
| ctx1000_b1_w64 | sglang_beam_search | success | 18.255 | 0.112 | -41.909 | 54.779 |
| ctx1000_b1_w64 | sglang_beam_search_new | success | 23.063 | 0.098 | -26.609 | 43.359 |
| ctx1000_b2_w128 | sid_gr_inference | success | 16.459 | 0.038 | -71.075 | 121.516 |
| ctx1000_b2_w128 | sglang | success | 56.902 | 0.083 | 0.000 | 35.148 |
| ctx1000_b2_w128 | sglang_beam_search | success | 35.183 | 0.169 | -38.169 | 56.846 |
| ctx1000_b2_w128 | sglang_beam_search_new | success | 28.216 | 0.136 | -50.413 | 70.881 |
| ctx1000_b2_w256 | sid_gr_inference | success | 28.105 | 0.054 | -61.607 | 71.162 |
| ctx1000_b2_w256 | sglang | success | 73.205 | 3.608 | 0.000 | 27.321 |
| ctx1000_b2_w256 | sglang_beam_search | success | 53.195 | 0.475 | -27.334 | 37.597 |
| ctx1000_b2_w256 | sglang_beam_search_new | success | 39.290 | 0.160 | -46.328 | 50.904 |
| ctx1000_b2_w64 | sid_gr_inference | success | 11.382 | 0.034 | -75.111 | 175.711 |
| ctx1000_b2_w64 | sglang | success | 45.733 | 0.389 | 0.000 | 43.732 |
| ctx1000_b2_w64 | sglang_beam_search | success | 23.220 | 0.051 | -49.226 | 86.131 |
| ctx1000_b2_w64 | sglang_beam_search_new | success | 21.694 | 0.096 | -52.564 | 92.193 |
| ctx1000_b4_w128 | sid_gr_inference | success | 29.217 | 0.097 | -67.533 | 136.907 |
| ctx1000_b4_w128 | sglang | success | 89.991 | 2.057 | 0.000 | 44.449 |
| ctx1000_b4_w128 | sglang_beam_search | success | 55.465 | 0.411 | -38.365 | 72.117 |
| ctx1000_b4_w128 | sglang_beam_search_new | success | 40.427 | 0.140 | -55.077 | 98.945 |
| ctx1000_b4_w256 | sid_gr_inference | success | 49.799 | 0.090 | -59.606 | 80.323 |
| ctx1000_b4_w256 | sglang | success | 123.284 | 1.463 | 0.000 | 32.445 |
| ctx1000_b4_w256 | sglang_beam_search | success | 85.638 | 1.538 | -30.536 | 46.708 |
| ctx1000_b4_w256 | sglang_beam_search_new | success | 60.272 | 1.228 | -51.112 | 66.366 |
| ctx1000_b4_w64 | sid_gr_inference | success | 17.817 | 0.058 | -76.182 | 224.502 |
| ctx1000_b4_w64 | sglang | success | 74.804 | 1.804 | 0.000 | 53.473 |
| ctx1000_b4_w64 | sglang_beam_search | success | 37.061 | 0.142 | -50.456 | 107.931 |
| ctx1000_b4_w64 | sglang_beam_search_new | success | 29.677 | 0.113 | -60.327 | 134.784 |
| ctx1000_b8_w128 | sid_gr_inference | success | 52.064 | 0.116 | -67.641 | 153.657 |
| ctx1000_b8_w128 | sglang | success | 160.896 | 0.585 | 0.000 | 49.722 |
| ctx1000_b8_w128 | sglang_beam_search | success | 90.123 | 1.722 | -43.987 | 88.767 |
| ctx1000_b8_w128 | sglang_beam_search_new | success | 63.774 | 0.305 | -60.364 | 125.444 |
| ctx1000_b8_w256 | sid_gr_inference | success | 88.593 | 1.237 | -61.614 | 90.301 |
| ctx1000_b8_w256 | sglang | success | 230.797 | 1.596 | 0.000 | 34.662 |
| ctx1000_b8_w256 | sglang_beam_search | success | 155.543 | 1.063 | -32.606 | 51.433 |
| ctx1000_b8_w256 | sglang_beam_search_new | success | 102.508 | 1.180 | -55.585 | 78.043 |
| ctx1000_b8_w64 | sid_gr_inference | success | 31.808 | 0.106 | -75.448 | 251.506 |
| ctx1000_b8_w64 | sglang | success | 129.553 | 1.206 | 0.000 | 61.751 |
| ctx1000_b8_w64 | sglang_beam_search | success | 58.644 | 0.320 | -54.734 | 136.417 |
| ctx1000_b8_w64 | sglang_beam_search_new | success | 43.731 | 0.152 | -66.245 | 182.936 |
| ctx5000_b1_w128 | sid_gr_inference | success | 12.881 | 0.053 | -86.384 | 77.632 |
| ctx5000_b1_w128 | sglang | success | 94.604 | 0.255 | 0.000 | 10.570 |
| ctx5000_b1_w128 | sglang_beam_search | success | 41.748 | 1.577 | -55.871 | 23.953 |
| ctx5000_b1_w128 | sglang_beam_search_new | success | 32.419 | 0.257 | -65.732 | 30.846 |
| ctx5000_b1_w256 | sid_gr_inference | success | 18.476 | 0.046 | -85.803 | 54.123 |
| ctx5000_b1_w256 | sglang | success | 130.143 | 0.850 | 0.000 | 7.684 |
| ctx5000_b1_w256 | sglang_beam_search | success | 83.531 | 6.981 | -35.816 | 11.972 |
| ctx5000_b1_w256 | sglang_beam_search_new | success | 44.315 | 0.208 | -65.949 | 22.566 |
| ctx5000_b1_w64 | sid_gr_inference | success | 11.346 | 0.040 | -86.282 | 88.140 |
| ctx5000_b1_w64 | sglang | success | 82.708 | 1.205 | 0.000 | 12.091 |
| ctx5000_b1_w64 | sglang_beam_search | success | 29.101 | 0.185 | -64.814 | 34.363 |
| ctx5000_b1_w64 | sglang_beam_search_new | success | 28.674 | 0.095 | -65.331 | 34.875 |
| ctx5000_b2_w128 | sid_gr_inference | success | 20.143 | 0.051 | -89.469 | 99.292 |
| ctx5000_b2_w128 | sglang | success | 191.279 | 0.474 | 0.000 | 10.456 |
| ctx5000_b2_w128 | sglang_beam_search | success | 79.591 | 6.382 | -58.390 | 25.128 |
| ctx5000_b2_w128 | sglang_beam_search_new | success | 41.549 | 0.130 | -78.278 | 48.135 |
| ctx5000_b2_w256 | sid_gr_inference | success | 33.430 | 0.119 | -85.736 | 59.827 |
| ctx5000_b2_w256 | sglang | success | 234.368 | 0.401 | 0.000 | 8.534 |
| ctx5000_b2_w256 | sglang_beam_search | success | 117.011 | 0.408 | -50.074 | 17.092 |
| ctx5000_b2_w256 | sglang_beam_search_new | success | 50.243 | 0.146 | -78.562 | 39.806 |
| ctx5000_b2_w64 | sid_gr_inference | success | 15.161 | 0.043 | -90.334 | 131.915 |
| ctx5000_b2_w64 | sglang | success | 156.851 | 0.658 | 0.000 | 12.751 |
| ctx5000_b2_w64 | sglang_beam_search | success | 44.654 | 0.968 | -71.531 | 44.789 |
| ctx5000_b2_w64 | sglang_beam_search_new | success | 29.683 | 0.091 | -81.076 | 67.378 |
| ctx5000_b4_w128 | sid_gr_inference | success | 36.623 | 0.125 | -89.447 | 109.220 |
| ctx5000_b4_w128 | sglang | success | 347.044 | 0.479 | 0.000 | 11.526 |
| ctx5000_b4_w128 | sglang_beam_search | success | 120.496 | 0.156 | -65.279 | 33.196 |
| ctx5000_b4_w128 | sglang_beam_search_new | success | 51.847 | 0.231 | -85.060 | 77.150 |
| ctx5000_b4_w256 | sid_gr_inference | success | 59.507 | 0.570 | -86.575 | 67.219 |
| ctx5000_b4_w256 | sglang | success | 443.267 | 1.261 | 0.000 | 9.024 |
| ctx5000_b4_w256 | sglang_beam_search | success | 216.492 | 1.233 | -51.160 | 18.476 |
| ctx5000_b4_w256 | sglang_beam_search_new | success | 70.795 | 2.286 | -84.029 | 56.501 |
| ctx5000_b4_w64 | sid_gr_inference | success | 25.283 | 0.080 | -91.718 | 158.211 |
| ctx5000_b4_w64 | sglang | success | 305.269 | 0.675 | 0.000 | 13.103 |
| ctx5000_b4_w64 | sglang_beam_search | success | 89.309 | 1.825 | -70.744 | 44.788 |
| ctx5000_b4_w64 | sglang_beam_search_new | success | 43.326 | 0.177 | -85.807 | 92.324 |
| ctx5000_b8_w128 | sid_gr_inference | success | 66.478 | 0.737 | -90.269 | 120.340 |
| ctx5000_b8_w128 | sglang | success | 683.185 | 1.993 | 0.000 | 11.710 |
| ctx5000_b8_w128 | sglang_beam_search | success | 224.088 | 1.501 | -67.200 | 35.700 |
| ctx5000_b8_w128 | sglang_beam_search_new | success | 77.105 | 1.011 | -88.714 | 103.755 |
| ctx5000_b8_w256 | sid_gr_inference | success | 110.685 | 2.686 | -87.512 | 72.277 |
| ctx5000_b8_w256 | sglang | success | 886.314 | 2.969 | 0.000 | 9.026 |
| ctx5000_b8_w256 | sglang_beam_search | success | 422.067 | 2.061 | -52.380 | 18.954 |
| ctx5000_b8_w256 | sglang_beam_search_new | success | 119.025 | 1.052 | -86.571 | 67.213 |
| ctx5000_b8_w64 | sid_gr_inference | success | 46.440 | 0.146 | -92.128 | 172.265 |
| ctx5000_b8_w64 | sglang | success | 589.925 | 0.868 | 0.000 | 13.561 |
| ctx5000_b8_w64 | sglang_beam_search | success | 127.440 | 0.279 | -78.397 | 62.775 |
| ctx5000_b8_w64 | sglang_beam_search_new | success | 57.388 | 0.155 | -90.272 | 139.402 |

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

### qwen3_0_6B

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w128 | sglang_beam_search | different | 100.000 | 91.406 | 91.406 | 10.156 | 0.33 |
| ctx1000_b1_w128 | sglang_beam_search_new | different | 100.000 | 93.750 | 93.750 | 13.281 | 0.345 |
| ctx1000_b1_w128 | sid_gr_inference | different | 100.000 | 95.312 | 95.312 | 10.156 | 0.266 |
| ctx1000_b1_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w256 | sglang_beam_search | different | 100.000 | 94.141 | 94.141 | 10.547 | 0.299 |
| ctx1000_b1_w256 | sglang_beam_search_new | different | 100.000 | 94.531 | 94.531 | 7.031 | 0.43 |
| ctx1000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 12.500 | 0.234 |
| ctx1000_b1_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w64 | sglang_beam_search | different | 100.000 | 89.062 | 89.062 | 23.438 | 0.301 |
| ctx1000_b1_w64 | sglang_beam_search_new | different | 100.000 | 92.188 | 92.188 | 23.438 | 0.158 |
| ctx1000_b1_w64 | sid_gr_inference | different | 100.000 | 96.875 | 96.875 | 23.438 | 0.282 |
| ctx1000_b2_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w128 | sglang_beam_search | different | 100.000 | 95.312 | 93.750 | 13.281 | 0.323 |
| ctx1000_b2_w128 | sglang_beam_search_new | different | 100.000 | 96.094 | 96.094 | 10.938 | 0.237 |
| ctx1000_b2_w128 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 10.547 | 0.296 |
| ctx1000_b2_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w256 | sglang_beam_search | different | 100.000 | 97.266 | 96.094 | 7.031 | 0.375 |
| ctx1000_b2_w256 | sglang_beam_search_new | different | 100.000 | 97.266 | 96.875 | 9.766 | 0.268 |
| ctx1000_b2_w256 | sid_gr_inference | different | 100.000 | 96.289 | 95.703 | 5.664 | 0.362 |
| ctx1000_b2_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w64 | sglang_beam_search | different | 100.000 | 95.312 | 93.750 | 25.000 | 0.33 |
| ctx1000_b2_w64 | sglang_beam_search_new | different | 100.000 | 91.406 | 87.500 | 23.438 | 0.237 |
| ctx1000_b2_w64 | sid_gr_inference | different | 100.000 | 92.188 | 90.625 | 19.531 | 0.26 |
| ctx1000_b4_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w128 | sglang_beam_search | different | 100.000 | 94.531 | 89.062 | 13.086 | 0.254 |
| ctx1000_b4_w128 | sglang_beam_search_new | different | 100.000 | 93.750 | 90.625 | 9.570 | 0.366 |
| ctx1000_b4_w128 | sid_gr_inference | different | 100.000 | 96.680 | 95.312 | 12.109 | 0.334 |
| ctx1000_b4_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w256 | sglang_beam_search | different | 100.000 | 95.703 | 94.141 | 8.105 | 0.275 |
| ctx1000_b4_w256 | sglang_beam_search_new | different | 100.000 | 94.531 | 89.844 | 6.934 | 0.403 |
| ctx1000_b4_w256 | sid_gr_inference | different | 100.000 | 95.215 | 94.141 | 7.715 | 0.358 |
| ctx1000_b4_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w64 | sglang_beam_search | different | 75.000 | 96.094 | 93.750 | 17.578 | 0.334 |
| ctx1000_b4_w64 | sglang_beam_search_new | different | 75.000 | 93.359 | 89.062 | 12.109 | 0.39 |
| ctx1000_b4_w64 | sid_gr_inference | different | 75.000 | 94.531 | 92.188 | 22.656 | 0.36 |
| ctx1000_b8_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w128 | sglang_beam_search | different | 87.500 | 95.410 | 89.062 | 10.059 | 0.512 |
| ctx1000_b8_w128 | sglang_beam_search_new | different | 87.500 | 95.605 | 93.750 | 13.574 | 0.416 |
| ctx1000_b8_w128 | sid_gr_inference | different | 87.500 | 95.215 | 92.188 | 13.184 | 0.333 |
| ctx1000_b8_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w256 | sglang_beam_search | different | 75.000 | 94.678 | 88.672 | 6.738 | 0.41 |
| ctx1000_b8_w256 | sglang_beam_search_new | different | 75.000 | 95.410 | 92.969 | 8.057 | 0.369 |
| ctx1000_b8_w256 | sid_gr_inference | different | 87.500 | 95.361 | 91.406 | 6.934 | 0.374 |
| ctx1000_b8_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w64 | sglang_beam_search | different | 75.000 | 94.336 | 89.062 | 20.312 | 0.342 |
| ctx1000_b8_w64 | sglang_beam_search_new | different | 75.000 | 94.141 | 90.625 | 20.898 | 0.372 |
| ctx1000_b8_w64 | sid_gr_inference | different | 100.000 | 94.336 | 92.188 | 19.727 | 0.528 |
| ctx5000_b1_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w128 | sglang_beam_search | different | 100.000 | 96.875 | 96.875 | 17.188 | 0.303 |
| ctx5000_b1_w128 | sglang_beam_search_new | different | 100.000 | 96.875 | 96.875 | 11.719 | 0.444 |
| ctx5000_b1_w128 | sid_gr_inference | different | 0.000 | 96.875 | 96.875 | 16.406 | 0.529 |
| ctx5000_b1_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | sglang_beam_search | different | 0.000 | 96.484 | 96.484 | 9.766 | 0.185 |
| ctx5000_b1_w256 | sglang_beam_search_new | different | 0.000 | 95.703 | 95.703 | 8.984 | 0.289 |
| ctx5000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 6.641 | 0.207 |
| ctx5000_b1_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w64 | sglang_beam_search | different | 100.000 | 92.188 | 92.188 | 14.062 | 0.202 |
| ctx5000_b1_w64 | sglang_beam_search_new | different | 0.000 | 95.312 | 95.312 | 28.125 | 0.222 |
| ctx5000_b1_w64 | sid_gr_inference | different | 100.000 | 95.312 | 95.312 | 26.562 | 0.252 |
| ctx5000_b2_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w128 | sglang_beam_search | different | 100.000 | 95.703 | 94.531 | 14.062 | 0.238 |
| ctx5000_b2_w128 | sglang_beam_search_new | different | 50.000 | 98.438 | 98.438 | 16.406 | 0.237 |
| ctx5000_b2_w128 | sid_gr_inference | different | 100.000 | 96.094 | 95.312 | 11.719 | 0.23 |
| ctx5000_b2_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w256 | sglang_beam_search | different | 100.000 | 94.727 | 93.359 | 9.375 | 0.335 |
| ctx5000_b2_w256 | sglang_beam_search_new | different | 100.000 | 95.703 | 94.141 | 7.227 | 0.258 |
| ctx5000_b2_w256 | sid_gr_inference | different | 100.000 | 96.289 | 95.703 | 7.812 | 0.253 |
| ctx5000_b2_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w64 | sglang_beam_search | different | 100.000 | 93.750 | 93.750 | 19.531 | 0.34 |
| ctx5000_b2_w64 | sglang_beam_search_new | different | 100.000 | 96.094 | 95.312 | 32.031 | 0.334 |
| ctx5000_b2_w64 | sid_gr_inference | different | 100.000 | 93.750 | 92.188 | 21.875 | 0.266 |
| ctx5000_b4_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w128 | sglang_beam_search | different | 100.000 | 95.117 | 91.406 | 14.062 | 0.436 |
| ctx5000_b4_w128 | sglang_beam_search_new | different | 100.000 | 95.703 | 94.531 | 15.625 | 0.525 |
| ctx5000_b4_w128 | sid_gr_inference | different | 100.000 | 95.508 | 94.531 | 12.109 | 0.545 |
| ctx5000_b4_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w256 | sglang_beam_search | different | 75.000 | 95.020 | 93.359 | 6.836 | 0.375 |
| ctx5000_b4_w256 | sglang_beam_search_new | different | 75.000 | 95.215 | 93.359 | 7.910 | 0.454 |
| ctx5000_b4_w256 | sid_gr_inference | different | 75.000 | 95.605 | 94.531 | 8.594 | 0.317 |
| ctx5000_b4_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w64 | sglang_beam_search | different | 100.000 | 94.141 | 92.188 | 19.141 | 0.272 |
| ctx5000_b4_w64 | sglang_beam_search_new | different | 75.000 | 94.141 | 92.188 | 20.703 | 0.35 |
| ctx5000_b4_w64 | sid_gr_inference | different | 100.000 | 94.531 | 92.188 | 21.875 | 0.359 |
| ctx5000_b8_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w128 | sglang_beam_search | different | 100.000 | 94.434 | 90.625 | 15.234 | 0.345 |
| ctx5000_b8_w128 | sglang_beam_search_new | different | 100.000 | 94.922 | 92.188 | 13.574 | 0.42 |
| ctx5000_b8_w128 | sid_gr_inference | different | 87.500 | 94.531 | 91.406 | 10.645 | 0.48 |
| ctx5000_b8_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w256 | sglang_beam_search | different | 100.000 | 95.850 | 94.922 | 8.252 | 0.468 |
| ctx5000_b8_w256 | sglang_beam_search_new | different | 100.000 | 95.264 | 94.141 | 6.885 | 0.648 |
| ctx5000_b8_w256 | sid_gr_inference | different | 100.000 | 95.654 | 94.141 | 8.057 | 0.385 |
| ctx5000_b8_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w64 | sglang_beam_search | different | 100.000 | 94.727 | 93.750 | 23.242 | 0.485 |
| ctx5000_b8_w64 | sglang_beam_search_new | different | 100.000 | 94.727 | 92.188 | 24.414 | 0.558 |
| ctx5000_b8_w64 | sid_gr_inference | different | 100.000 | 95.312 | 92.188 | 23.242 | 0.529 |

### qwen3_1_7B

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w128 | sglang_beam_search | different | 100.000 | 95.312 | 95.312 | 13.281 | 0.146 |
| ctx1000_b1_w128 | sglang_beam_search_new | different | 100.000 | 96.875 | 96.875 | 11.719 | 0.172 |
| ctx1000_b1_w128 | sid_gr_inference | different | 100.000 | 96.875 | 96.875 | 14.062 | 0.272 |
| ctx1000_b1_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w256 | sglang_beam_search | different | 100.000 | 94.922 | 94.922 | 7.422 | 0.2 |
| ctx1000_b1_w256 | sglang_beam_search_new | different | 100.000 | 96.094 | 96.094 | 9.375 | 0.38 |
| ctx1000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 10.938 | 0.191 |
| ctx1000_b1_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w64 | sglang_beam_search | different | 100.000 | 95.312 | 95.312 | 32.812 | 0.187 |
| ctx1000_b1_w64 | sglang_beam_search_new | different | 100.000 | 93.750 | 93.750 | 26.562 | 0.207 |
| ctx1000_b1_w64 | sid_gr_inference | different | 100.000 | 95.312 | 95.312 | 21.875 | 0.19 |
| ctx1000_b2_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w128 | sglang_beam_search | different | 100.000 | 92.578 | 91.406 | 12.500 | 0.304 |
| ctx1000_b2_w128 | sglang_beam_search_new | different | 100.000 | 94.531 | 93.750 | 8.594 | 0.175 |
| ctx1000_b2_w128 | sid_gr_inference | different | 100.000 | 95.703 | 95.312 | 10.938 | 0.269 |
| ctx1000_b2_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w256 | sglang_beam_search | different | 100.000 | 97.266 | 96.484 | 9.570 | 0.266 |
| ctx1000_b2_w256 | sglang_beam_search_new | different | 100.000 | 96.094 | 95.312 | 8.398 | 0.266 |
| ctx1000_b2_w256 | sid_gr_inference | different | 100.000 | 95.898 | 94.922 | 7.812 | 0.24 |
| ctx1000_b2_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w64 | sglang_beam_search | different | 100.000 | 95.312 | 95.312 | 33.594 | 0.196 |
| ctx1000_b2_w64 | sglang_beam_search_new | different | 100.000 | 93.750 | 92.188 | 18.750 | 0.189 |
| ctx1000_b2_w64 | sid_gr_inference | different | 100.000 | 92.969 | 92.188 | 22.656 | 0.231 |
| ctx1000_b4_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w128 | sglang_beam_search | different | 100.000 | 95.898 | 93.750 | 15.234 | 0.274 |
| ctx1000_b4_w128 | sglang_beam_search_new | different | 100.000 | 95.703 | 94.531 | 13.281 | 0.254 |
| ctx1000_b4_w128 | sid_gr_inference | different | 100.000 | 96.289 | 94.531 | 15.430 | 0.183 |
| ctx1000_b4_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w256 | sglang_beam_search | different | 100.000 | 95.410 | 92.969 | 9.668 | 0.274 |
| ctx1000_b4_w256 | sglang_beam_search_new | different | 100.000 | 95.801 | 92.188 | 6.738 | 0.277 |
| ctx1000_b4_w256 | sid_gr_inference | different | 100.000 | 95.801 | 95.312 | 8.594 | 0.218 |
| ctx1000_b4_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w64 | sglang_beam_search | different | 100.000 | 96.094 | 95.312 | 27.734 | 0.159 |
| ctx1000_b4_w64 | sglang_beam_search_new | different | 100.000 | 94.922 | 92.188 | 27.734 | 0.196 |
| ctx1000_b4_w64 | sid_gr_inference | different | 100.000 | 94.922 | 93.750 | 30.078 | 0.177 |
| ctx1000_b8_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w128 | sglang_beam_search | different | 100.000 | 96.094 | 94.531 | 15.137 | 0.334 |
| ctx1000_b8_w128 | sglang_beam_search_new | different | 100.000 | 96.191 | 94.531 | 17.090 | 0.322 |
| ctx1000_b8_w128 | sid_gr_inference | different | 100.000 | 96.094 | 94.531 | 16.992 | 0.367 |
| ctx1000_b8_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w256 | sglang_beam_search | different | 100.000 | 96.436 | 95.312 | 8.154 | 0.349 |
| ctx1000_b8_w256 | sglang_beam_search_new | different | 100.000 | 96.387 | 95.312 | 9.033 | 0.368 |
| ctx1000_b8_w256 | sid_gr_inference | different | 100.000 | 95.703 | 94.922 | 8.887 | 0.321 |
| ctx1000_b8_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w64 | sglang_beam_search | different | 100.000 | 95.508 | 89.062 | 28.320 | 0.292 |
| ctx1000_b8_w64 | sglang_beam_search_new | different | 100.000 | 95.117 | 90.625 | 28.711 | 0.222 |
| ctx1000_b8_w64 | sid_gr_inference | different | 100.000 | 96.680 | 93.750 | 30.469 | 0.236 |
| ctx5000_b1_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w128 | sglang_beam_search | different | 100.000 | 96.875 | 96.875 | 12.500 | 0.148 |
| ctx5000_b1_w128 | sglang_beam_search_new | different | 0.000 | 95.312 | 95.312 | 13.281 | 0.178 |
| ctx5000_b1_w128 | sid_gr_inference | different | 0.000 | 94.531 | 94.531 | 6.250 | 0.177 |
| ctx5000_b1_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | sglang_beam_search | different | 100.000 | 95.703 | 95.703 | 6.250 | 0.212 |
| ctx5000_b1_w256 | sglang_beam_search_new | different | 100.000 | 96.875 | 96.875 | 8.203 | 0.439 |
| ctx5000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 6.641 | 0.453 |
| ctx5000_b1_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w64 | sglang_beam_search | different | 100.000 | 95.312 | 95.312 | 21.875 | 0.21 |
| ctx5000_b1_w64 | sglang_beam_search_new | different | 0.000 | 95.312 | 95.312 | 18.750 | 0.212 |
| ctx5000_b1_w64 | sid_gr_inference | different | 0.000 | 96.875 | 96.875 | 23.438 | 0.168 |
| ctx5000_b2_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w128 | sglang_beam_search | different | 50.000 | 92.578 | 89.062 | 8.203 | 0.237 |
| ctx5000_b2_w128 | sglang_beam_search_new | different | 100.000 | 92.578 | 88.281 | 14.453 | 0.255 |
| ctx5000_b2_w128 | sid_gr_inference | different | 100.000 | 91.797 | 87.500 | 10.156 | 0.228 |
| ctx5000_b2_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w256 | sglang_beam_search | different | 50.000 | 93.945 | 92.188 | 5.273 | 0.388 |
| ctx5000_b2_w256 | sglang_beam_search_new | different | 50.000 | 96.094 | 96.094 | 8.008 | 0.251 |
| ctx5000_b2_w256 | sid_gr_inference | different | 50.000 | 95.312 | 95.312 | 5.664 | 0.331 |
| ctx5000_b2_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w64 | sglang_beam_search | different | 50.000 | 94.531 | 93.750 | 18.750 | 0.188 |
| ctx5000_b2_w64 | sglang_beam_search_new | different | 100.000 | 92.188 | 92.188 | 17.969 | 0.176 |
| ctx5000_b2_w64 | sid_gr_inference | different | 50.000 | 92.969 | 90.625 | 17.969 | 0.344 |
| ctx5000_b4_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w128 | sglang_beam_search | different | 75.000 | 96.289 | 94.531 | 15.234 | 0.212 |
| ctx5000_b4_w128 | sglang_beam_search_new | different | 75.000 | 95.703 | 93.750 | 13.086 | 0.235 |
| ctx5000_b4_w128 | sid_gr_inference | different | 100.000 | 95.508 | 92.969 | 16.797 | 0.219 |
| ctx5000_b4_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w256 | sglang_beam_search | different | 75.000 | 95.801 | 94.922 | 8.594 | 0.238 |
| ctx5000_b4_w256 | sglang_beam_search_new | different | 75.000 | 95.312 | 94.141 | 9.277 | 0.235 |
| ctx5000_b4_w256 | sid_gr_inference | different | 75.000 | 95.605 | 94.531 | 7.812 | 0.339 |
| ctx5000_b4_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w64 | sglang_beam_search | different | 100.000 | 96.094 | 95.312 | 32.031 | 0.209 |
| ctx5000_b4_w64 | sglang_beam_search_new | different | 100.000 | 95.703 | 95.312 | 27.344 | 0.218 |
| ctx5000_b4_w64 | sid_gr_inference | different | 100.000 | 94.531 | 90.625 | 23.828 | 0.204 |
| ctx5000_b8_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w128 | sglang_beam_search | different | 87.500 | 95.312 | 94.531 | 15.039 | 0.293 |
| ctx5000_b8_w128 | sglang_beam_search_new | different | 87.500 | 95.605 | 93.750 | 16.992 | 0.427 |
| ctx5000_b8_w128 | sid_gr_inference | different | 87.500 | 94.922 | 92.188 | 15.527 | 0.445 |
| ctx5000_b8_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w256 | sglang_beam_search | different | 100.000 | 96.143 | 94.922 | 9.473 | 0.259 |
| ctx5000_b8_w256 | sglang_beam_search_new | different | 100.000 | 95.947 | 94.922 | 9.180 | 0.583 |
| ctx5000_b8_w256 | sid_gr_inference | different | 100.000 | 95.850 | 93.359 | 9.229 | 0.322 |
| ctx5000_b8_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w64 | sglang_beam_search | different | 87.500 | 96.484 | 93.750 | 24.805 | 0.221 |
| ctx5000_b8_w64 | sglang_beam_search_new | different | 87.500 | 96.289 | 95.312 | 31.641 | 0.427 |
| ctx5000_b8_w64 | sid_gr_inference | different | 87.500 | 96.094 | 93.750 | 25.586 | 0.445 |

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
| [20261012T145504Z-c62e22](diagnostics/20261012T145504Z-c62e22/diagnosis.md) |

## Run details

<details>
<summary>Configuration environment</summary>

| Configuration variable | Solution(s) | Value used | Reproduce from |
| --- | --- | --- | --- |
| `BENCHMARK_GPU` | sid_gr_inference, sglang, sglang_beam_search, sglang_beam_search_new | `GPU-7330c4aa-0e83-7249-af71-60feb195d4eb` | — |
| `GR_DECODE_ATTEN_ROOT` | sid_gr_inference | `/home/scratch.xinyliu_gpu/recsys-examples/corelib/gr_decode_atten` | [7647ab3e91c1](https://github.com/lauthu/recsys-examples/tree/7647ab3e91c178d39b21f8ee511af10fe88a7fc9/corelib/gr_decode_atten) |
| `GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_DISABLE_FLATTEN_LINEAR` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_DISABLE_PREFILL_CUDA_GRAPH` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_GR_TRTLLM_KERNELS_JIT` | sid_gr_inference | `1` | — |
| `GR_INFERENCE_PREFILL_BACKEND` | sid_gr_inference | `auto` | — |
| `GR_INFERENCE_TRTLLM_PACKED_GEMM_SCOPE` | sid_gr_inference | `all` | — |
| `GR_INFERENCE_TRTLLM_QK_NORM_ROPE_PHASE` | sid_gr_inference | `all` | — |
| `QWEN3_0_6B_MODEL` | sid_gr_inference, sglang, sglang_beam_search, sglang_beam_search_new | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/repo/models/Qwen3-0.6B` | [Qwen/Qwen3-0.6B@c1899de289a0](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) |
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
| `SID_GR_SOURCE` | sid_gr_inference | `/home/scratch.xinyliu_gpu/recsys-examples/examples/sid-gr-inference` | [7647ab3e91c1](https://github.com/lauthu/recsys-examples/tree/7647ab3e91c178d39b21f8ee511af10fe88a7fc9/examples/sid-gr-inference) |

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
| qwen3_0_6B | 5 | 30 |
| qwen3_1_7B | 5 | 30 |

| Solution | Component | Reproducible source / image ID |
| --- | --- | --- |
| All | Model | [Qwen/Qwen3-0.6B@c1899de289a0](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) |
| All | Model | [Qwen/Qwen3-1.7B@70d244cc86cc](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) |
| sid_gr_inference | Engine source | [7647ab3e91c1](https://github.com/lauthu/recsys-examples/tree/7647ab3e91c178d39b21f8ee511af10fe88a7fc9/examples/sid-gr-inference) |
| sid_gr_inference | GR decode kernels | [7647ab3e91c1](https://github.com/lauthu/recsys-examples/tree/7647ab3e91c178d39b21f8ee511af10fe88a7fc9/corelib/gr_decode_atten) |
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
| sid_gr_inference | [7647ab3e91c1](https://github.com/lauthu/recsys-examples/tree/7647ab3e91c178d39b21f8ee511af10fe88a7fc9/examples/sid-gr-inference) | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |
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
tar -xzf results/20261012T113832092061Z-c0ad33/raw_artifacts.tar.gz -C results/20261012T113832092061Z-c0ad33
python3 bench.py report --run results/20261012T113832092061Z-c0ad33
```

| Saved data | Location inside the run |
| --- | --- |
| Frozen configuration and hardware | `manifest.json` |
| Derived measurements | `summary.json` |
| Exact inputs | `workloads/` |
| Requests, commands, logs, samples and beams | `jobs/<candidate>/<scenario>/<case>/` |

**Scenario leaderboards**

- [qwen3_0_6B](../../leaderboards/qwen3_0_6B.md)
- [qwen3_1_7B](../../leaderboards/qwen3_1_7B.md)
