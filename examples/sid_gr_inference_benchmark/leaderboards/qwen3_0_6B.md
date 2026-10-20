# qwen3_0_6B leaderboard

Compare each solution's latest latency on the same workload and hardware.

Median batch **ms**; lower is better. The **Fastest** column names the winner for each workload.

Parity counts are workload cases; each reference row checks repeat consistency.

## [NVIDIA RTX PRO 6000 Blackwell Server Edition](../config/hardware/rtx_pro_6000_blackwell_server.yml)

**CPU:** Intel(R) Xeon(R) Platinum 8480+ · **Worker cores:** 10 · **Driver:** 595.58.03

| Solution | Exact beam/score match | Different beams or scores | Top-1 mismatches | Not compared |
| --- | --- | --- | --- | --- |
| sid_gr_inference | 24 of 24 cases | 0 of 24 cases | 0 cases | 0 cases |
| vllm_gr | 0 of 24 cases | 6 of 24 cases | 1 cases | 18 cases |
| flash_rec | 0 of 24 cases | 24 of 24 cases | 3 cases | 0 cases |
| tensorrt_llm | 0 of 24 cases | 4 of 24 cases | 0 cases | 20 cases |

| Top-1 mismatch case | Candidate |
| --- | --- |
| qwen3_0_6B/ctx1000_b8_w128 | flash_rec |
| qwen3_0_6B/ctx5000_b1_w128 | vllm_gr |
| qwen3_0_6B/ctx5000_b2_w128 | flash_rec |
| qwen3_0_6B/ctx5000_b4_w256 | flash_rec |



### Latency comparison

| Context | Batch | Beam | sid_gr_inference | vllm_gr | flash_rec | tensorrt_llm | Fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 64 | **6.178** | 15.680 | 23.456 | 44.366 | **sid_gr_inference** |
| 1000 | 1 | 128 | **7.211** | 18.528 | 26.490 | 69.632 | **sid_gr_inference** |
| 1000 | 1 | 256 | **10.299** | 25.835 | 35.094 | 125.313 | **sid_gr_inference** |
| 1000 | 2 | 64 | **7.607** | unsupported | 27.907 | unsupported | **sid_gr_inference** |
| 1000 | 2 | 128 | **10.987** | unsupported | 37.035 | unsupported | **sid_gr_inference** |
| 1000 | 2 | 256 | **17.908** | unsupported | 50.515 | unsupported | **sid_gr_inference** |
| 1000 | 4 | 64 | **12.363** | unsupported | 46.996 | unsupported | **sid_gr_inference** |
| 1000 | 4 | 128 | **19.007** | unsupported | 59.693 | unsupported | **sid_gr_inference** |
| 1000 | 4 | 256 | **32.354** | unsupported | 88.312 | unsupported | **sid_gr_inference** |
| 1000 | 8 | 64 | **21.780** | unsupported | 79.866 | unsupported | **sid_gr_inference** |
| 1000 | 8 | 128 | **34.500** | unsupported | 107.071 | unsupported | **sid_gr_inference** |
| 1000 | 8 | 256 | **59.417** | unsupported | 167.916 | unsupported | **sid_gr_inference** |
| 5000 | 1 | 64 | **7.673** | 24.216 | 58.478 | 139.538 | **sid_gr_inference** |
| 5000 | 1 | 128 | **8.873** | 26.616 | 70.574 | unsupported | **sid_gr_inference** |
| 5000 | 1 | 256 | **12.881** | 34.142 | 103.594 | unsupported | **sid_gr_inference** |
| 5000 | 2 | 64 | **11.175** | unsupported | 110.094 | unsupported | **sid_gr_inference** |
| 5000 | 2 | 128 | **14.612** | unsupported | 142.010 | unsupported | **sid_gr_inference** |
| 5000 | 2 | 256 | **22.959** | unsupported | 180.031 | unsupported | **sid_gr_inference** |
| 5000 | 4 | 64 | **19.902** | unsupported | 233.202 | unsupported | **sid_gr_inference** |
| 5000 | 4 | 128 | **26.438** | unsupported | 271.516 | unsupported | **sid_gr_inference** |
| 5000 | 4 | 256 | **41.800** | unsupported | 360.722 | unsupported | **sid_gr_inference** |
| 5000 | 8 | 64 | **36.437** | unsupported | 473.831 | unsupported | **sid_gr_inference** |
| 5000 | 8 | 128 | **49.115** | unsupported | 562.395 | unsupported | **sid_gr_inference** |
| 5000 | 8 | 256 | **83.042** | unsupported | 752.758 | unsupported | **sid_gr_inference** |

<details>
<summary>Output parity, including reference repeat consistency</summary>

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w128 | vllm_gr | different | 100.000 | 92.969 | 92.969 | 17.969 | 0.33 |
| ctx1000_b1_w128 | flash_rec | different | 100.000 | 91.406 | 91.406 | 14.062 | 0.393 |
| ctx1000_b1_w128 | tensorrt_llm | different | 100.000 | 92.188 | 92.188 | 13.281 | 0.268 |
| ctx1000_b1_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w256 | vllm_gr | different | 100.000 | 94.141 | 94.141 | 8.984 | 0.424 |
| ctx1000_b1_w256 | flash_rec | different | 100.000 | 92.188 | 92.188 | 5.469 | 0.339 |
| ctx1000_b1_w256 | tensorrt_llm | different | 100.000 | 92.578 | 92.578 | 5.859 | 0.485 |
| ctx1000_b1_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w64 | vllm_gr | different | 100.000 | 96.875 | 96.875 | 28.125 | 0.312 |
| ctx1000_b1_w64 | flash_rec | different | 100.000 | 90.625 | 90.625 | 21.875 | 0.246 |
| ctx1000_b1_w64 | tensorrt_llm | different | 100.000 | 89.062 | 89.062 | 21.875 | 0.384 |
| ctx1000_b2_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w128 | flash_rec | different | 100.000 | 94.531 | 92.188 | 11.328 | 0.333 |
| ctx1000_b2_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b2_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b2_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w256 | flash_rec | different | 100.000 | 94.922 | 92.969 | 6.641 | 0.259 |
| ctx1000_b2_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b2_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b2_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w64 | flash_rec | different | 100.000 | 88.281 | 87.500 | 17.969 | 0.309 |
| ctx1000_b2_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b2_w64 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b4_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w128 | flash_rec | different | 100.000 | 96.289 | 95.312 | 13.672 | 0.307 |
| ctx1000_b4_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b4_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b4_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w256 | flash_rec | different | 100.000 | 94.922 | 92.578 | 9.570 | 0.348 |
| ctx1000_b4_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b4_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b4_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w64 | flash_rec | different | 100.000 | 95.312 | 92.188 | 17.969 | 0.29 |
| ctx1000_b4_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b4_w64 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b8_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w128 | flash_rec | different | 62.500 | 95.508 | 92.969 | 11.719 | 0.438 |
| ctx1000_b8_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b8_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b8_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w256 | flash_rec | different | 100.000 | 95.557 | 92.578 | 7.227 | 0.478 |
| ctx1000_b8_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b8_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b8_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w64 | flash_rec | different | 100.000 | 91.797 | 79.688 | 19.922 | 0.456 |
| ctx1000_b8_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b8_w64 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b1_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w128 | vllm_gr | different | 0.000 | 96.094 | 96.094 | 11.719 | 0.413 |
| ctx5000_b1_w128 | flash_rec | different | 100.000 | 95.312 | 95.312 | 15.625 | 0.317 |
| ctx5000_b1_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b1_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | vllm_gr | different | 100.000 | 93.750 | 93.750 | 7.422 | 0.384 |
| ctx5000_b1_w256 | flash_rec | different | 100.000 | 95.703 | 95.703 | 6.250 | 0.261 |
| ctx5000_b1_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b1_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w64 | vllm_gr | different | 100.000 | 96.875 | 96.875 | 39.062 | 0.2 |
| ctx5000_b1_w64 | flash_rec | different | 100.000 | 87.500 | 87.500 | 20.312 | 0.205 |
| ctx5000_b1_w64 | tensorrt_llm | different | 100.000 | 90.625 | 90.625 | 15.625 | 0.173 |
| ctx5000_b2_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w128 | flash_rec | different | 50.000 | 96.484 | 96.094 | 11.328 | 0.189 |
| ctx5000_b2_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b2_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b2_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w256 | flash_rec | different | 100.000 | 95.312 | 94.922 | 5.469 | 0.313 |
| ctx5000_b2_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b2_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b2_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w64 | flash_rec | different | 100.000 | 96.875 | 96.875 | 25.000 | 0.296 |
| ctx5000_b2_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b2_w64 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b4_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w128 | flash_rec | different | 100.000 | 95.117 | 92.969 | 11.523 | 0.255 |
| ctx5000_b4_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b4_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b4_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w256 | flash_rec | different | 75.000 | 95.312 | 94.531 | 7.031 | 0.371 |
| ctx5000_b4_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b4_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b4_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w64 | flash_rec | different | 100.000 | 95.703 | 92.188 | 26.172 | 0.352 |
| ctx5000_b4_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b4_w64 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b8_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w128 | flash_rec | different | 100.000 | 95.117 | 92.188 | 13.770 | 0.273 |
| ctx5000_b8_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b8_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b8_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w256 | flash_rec | different | 100.000 | 96.143 | 94.922 | 8.350 | 0.463 |
| ctx5000_b8_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b8_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b8_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w64 | flash_rec | different | 100.000 | 95.312 | 89.062 | 23.242 | 0.324 |
| ctx5000_b8_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b8_w64 | vllm_gr | unavailable | — | — | — | — | — |

</details>

<details>
<summary>Throughput, status and source runs</summary>

| Case | Candidate | Status | Requests/s | Reference | Source run |
| --- | --- | --- | --- | --- | --- |
| ctx1000_b1_w128 | sid_gr_inference | success | 138.671 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w128 | vllm_gr | success | 53.974 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w128 | flash_rec | success | 37.750 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w128 | tensorrt_llm | success | 14.361 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w256 | sid_gr_inference | success | 97.097 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w256 | vllm_gr | success | 38.707 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w256 | flash_rec | success | 28.495 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w256 | tensorrt_llm | success | 7.980 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w64 | sid_gr_inference | success | 161.874 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w64 | vllm_gr | success | 63.777 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w64 | flash_rec | success | 42.633 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w64 | tensorrt_llm | success | 22.540 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w128 | sid_gr_inference | success | 182.028 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w128 | flash_rec | success | 54.003 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w256 | sid_gr_inference | success | 111.681 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w256 | flash_rec | success | 39.592 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w64 | sid_gr_inference | success | 262.918 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w64 | flash_rec | success | 71.667 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w128 | sid_gr_inference | success | 210.448 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w128 | flash_rec | success | 67.009 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w256 | sid_gr_inference | success | 123.630 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w256 | flash_rec | success | 45.294 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w64 | sid_gr_inference | success | 323.551 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w64 | flash_rec | success | 85.113 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w128 | sid_gr_inference | success | 231.883 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w128 | flash_rec | success | 74.717 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w256 | sid_gr_inference | success | 134.641 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w256 | flash_rec | success | 47.643 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w64 | sid_gr_inference | success | 367.307 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w64 | flash_rec | success | 100.168 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w128 | sid_gr_inference | success | 112.705 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w128 | vllm_gr | success | 37.571 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w128 | flash_rec | success | 14.170 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w256 | sid_gr_inference | success | 77.633 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w256 | vllm_gr | success | 29.289 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w256 | flash_rec | success | 9.653 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w64 | sid_gr_inference | success | 130.335 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w64 | vllm_gr | success | 41.295 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w64 | flash_rec | success | 17.100 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w64 | tensorrt_llm | success | 7.166 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w128 | sid_gr_inference | success | 136.874 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w128 | flash_rec | success | 14.083 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w256 | sid_gr_inference | success | 87.114 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w256 | flash_rec | success | 11.109 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w64 | sid_gr_inference | success | 178.963 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w64 | flash_rec | success | 18.166 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w128 | sid_gr_inference | success | 151.297 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w128 | flash_rec | success | 14.732 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w256 | sid_gr_inference | success | 95.693 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w256 | flash_rec | success | 11.089 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w64 | sid_gr_inference | success | 200.987 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w64 | flash_rec | success | 17.153 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w128 | sid_gr_inference | success | 162.882 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w128 | flash_rec | success | 14.225 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w256 | sid_gr_inference | success | 96.337 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w256 | flash_rec | success | 10.628 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w64 | sid_gr_inference | success | 219.556 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w64 | flash_rec | success | 16.884 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |

</details>

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
