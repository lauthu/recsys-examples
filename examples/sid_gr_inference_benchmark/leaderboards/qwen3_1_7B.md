# qwen3_1_7B leaderboard

Compare each solution's latest latency on the same workload and hardware.

Median batch **ms**; lower is better. The **Fastest** column names the winner for each workload.

Parity counts are workload cases; each reference row checks repeat consistency.

## [NVIDIA RTX PRO 6000 Blackwell Server Edition](../config/hardware/rtx_pro_6000_blackwell_server.yml)

**CPU:** Intel(R) Xeon(R) Platinum 8480+ · **Worker cores:** 10 · **Driver:** 595.58.03

| Solution | Exact beam/score match | Different beams or scores | Top-1 mismatches | Not compared |
| --- | --- | --- | --- | --- |
| sid_gr_inference | 24 of 24 cases | 0 of 24 cases | 0 cases | 0 cases |
| vllm_gr | 0 of 24 cases | 6 of 24 cases | 0 cases | 18 cases |
| flash_rec | 0 of 24 cases | 24 of 24 cases | 3 cases | 0 cases |
| tensorrt_llm | 0 of 24 cases | 4 of 24 cases | 1 cases | 20 cases |

| Top-1 mismatch case | Candidate |
| --- | --- |
| qwen3_1_7B/ctx5000_b1_w64 | tensorrt_llm |
| qwen3_1_7B/ctx5000_b2_w128 | flash_rec |
| qwen3_1_7B/ctx5000_b2_w64 | flash_rec |
| qwen3_1_7B/ctx5000_b4_w128 | flash_rec |



### Latency comparison

| Context | Batch | Beam | sid_gr_inference | vllm_gr | flash_rec | tensorrt_llm | Fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 64 | **9.309** | 20.526 | 29.176 | 48.218 | **sid_gr_inference** |
| 1000 | 1 | 128 | **10.685** | 23.970 | 33.686 | 71.584 | **sid_gr_inference** |
| 1000 | 1 | 256 | **15.081** | 32.917 | 43.354 | 129.292 | **sid_gr_inference** |
| 1000 | 2 | 64 | **11.088** | unsupported | 42.768 | unsupported | **sid_gr_inference** |
| 1000 | 2 | 128 | **15.695** | unsupported | 53.253 | unsupported | **sid_gr_inference** |
| 1000 | 2 | 256 | **26.839** | unsupported | 71.290 | unsupported | **sid_gr_inference** |
| 1000 | 4 | 64 | **17.066** | unsupported | 75.748 | unsupported | **sid_gr_inference** |
| 1000 | 4 | 128 | **27.983** | unsupported | 92.828 | unsupported | **sid_gr_inference** |
| 1000 | 4 | 256 | **47.156** | unsupported | 128.364 | unsupported | **sid_gr_inference** |
| 1000 | 8 | 64 | **30.642** | unsupported | 134.454 | unsupported | **sid_gr_inference** |
| 1000 | 8 | 128 | **49.482** | unsupported | 167.969 | unsupported | **sid_gr_inference** |
| 1000 | 8 | 256 | **85.905** | unsupported | 240.457 | unsupported | **sid_gr_inference** |
| 5000 | 1 | 64 | **10.979** | 30.557 | 93.877 | 143.953 | **sid_gr_inference** |
| 5000 | 1 | 128 | **12.452** | 33.777 | 106.095 | unsupported | **sid_gr_inference** |
| 5000 | 1 | 256 | **18.043** | 41.814 | 140.299 | unsupported | **sid_gr_inference** |
| 5000 | 2 | 64 | **14.666** | unsupported | 179.976 | unsupported | **sid_gr_inference** |
| 5000 | 2 | 128 | **19.453** | unsupported | 213.387 | unsupported | **sid_gr_inference** |
| 5000 | 2 | 256 | **31.737** | unsupported | 256.567 | unsupported | **sid_gr_inference** |
| 5000 | 4 | 64 | **24.479** | unsupported | 389.847 | unsupported | **sid_gr_inference** |
| 5000 | 4 | 128 | **35.445** | unsupported | 432.661 | unsupported | **sid_gr_inference** |
| 5000 | 4 | 256 | **57.116** | unsupported | 529.208 | unsupported | **sid_gr_inference** |
| 5000 | 8 | 64 | **45.904** | unsupported | 759.759 | unsupported | **sid_gr_inference** |
| 5000 | 8 | 128 | **64.281** | unsupported | 856.089 | unsupported | **sid_gr_inference** |
| 5000 | 8 | 256 | **108.794** | unsupported | 1062.190 | unsupported | **sid_gr_inference** |

<details>
<summary>Output parity, including reference repeat consistency</summary>

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w128 | vllm_gr | different | 100.000 | 93.750 | 93.750 | 7.812 | 0.389 |
| ctx1000_b1_w128 | flash_rec | different | 100.000 | 93.750 | 93.750 | 17.188 | 0.246 |
| ctx1000_b1_w128 | tensorrt_llm | different | 100.000 | 94.531 | 94.531 | 14.844 | 0.176 |
| ctx1000_b1_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w256 | vllm_gr | different | 100.000 | 95.703 | 95.703 | 6.250 | 0.248 |
| ctx1000_b1_w256 | flash_rec | different | 100.000 | 93.359 | 93.359 | 7.812 | 0.236 |
| ctx1000_b1_w256 | tensorrt_llm | different | 100.000 | 93.750 | 93.750 | 8.594 | 0.461 |
| ctx1000_b1_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w64 | vllm_gr | different | 100.000 | 93.750 | 93.750 | 37.500 | 0.32 |
| ctx1000_b1_w64 | flash_rec | different | 100.000 | 92.188 | 92.188 | 28.125 | 0.238 |
| ctx1000_b1_w64 | tensorrt_llm | different | 100.000 | 93.750 | 93.750 | 26.562 | 0.183 |
| ctx1000_b2_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w128 | flash_rec | different | 100.000 | 96.484 | 95.312 | 14.453 | 0.223 |
| ctx1000_b2_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b2_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b2_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w256 | flash_rec | different | 100.000 | 96.094 | 94.922 | 9.375 | 0.261 |
| ctx1000_b2_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b2_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b2_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w64 | flash_rec | different | 100.000 | 95.312 | 93.750 | 25.000 | 0.184 |
| ctx1000_b2_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b2_w64 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b4_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w128 | flash_rec | different | 100.000 | 95.117 | 92.969 | 15.625 | 0.276 |
| ctx1000_b4_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b4_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b4_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w256 | flash_rec | different | 100.000 | 94.531 | 91.797 | 7.324 | 0.32 |
| ctx1000_b4_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b4_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b4_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w64 | flash_rec | different | 100.000 | 91.797 | 90.625 | 26.172 | 0.227 |
| ctx1000_b4_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b4_w64 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b8_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w128 | flash_rec | different | 100.000 | 95.410 | 93.750 | 16.211 | 0.349 |
| ctx1000_b8_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b8_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b8_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w256 | flash_rec | different | 100.000 | 95.654 | 94.531 | 7.666 | 0.54 |
| ctx1000_b8_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b8_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx1000_b8_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w64 | flash_rec | different | 100.000 | 97.656 | 96.875 | 26.562 | 0.29 |
| ctx1000_b8_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx1000_b8_w64 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b1_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w128 | vllm_gr | different | 100.000 | 95.312 | 95.312 | 17.969 | 0.198 |
| ctx5000_b1_w128 | flash_rec | different | 100.000 | 96.094 | 96.094 | 13.281 | 0.278 |
| ctx5000_b1_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b1_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | vllm_gr | different | 100.000 | 95.312 | 95.312 | 8.984 | 0.225 |
| ctx5000_b1_w256 | flash_rec | different | 100.000 | 96.094 | 96.094 | 4.297 | 0.287 |
| ctx5000_b1_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b1_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w64 | vllm_gr | different | 100.000 | 95.312 | 95.312 | 28.125 | 0.206 |
| ctx5000_b1_w64 | flash_rec | different | 100.000 | 93.750 | 93.750 | 32.812 | 0.297 |
| ctx5000_b1_w64 | tensorrt_llm | different | 0.000 | 96.875 | 96.875 | 14.062 | 0.352 |
| ctx5000_b2_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w128 | flash_rec | different | 50.000 | 94.531 | 93.750 | 10.547 | 0.269 |
| ctx5000_b2_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b2_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b2_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w256 | flash_rec | different | 100.000 | 95.508 | 94.141 | 8.594 | 0.247 |
| ctx5000_b2_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b2_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b2_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w64 | flash_rec | different | 50.000 | 92.188 | 90.625 | 25.000 | 0.479 |
| ctx5000_b2_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b2_w64 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b4_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w128 | flash_rec | different | 75.000 | 94.141 | 91.406 | 15.820 | 0.191 |
| ctx5000_b4_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b4_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b4_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w256 | flash_rec | different | 100.000 | 95.605 | 94.141 | 8.398 | 0.24 |
| ctx5000_b4_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b4_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b4_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w64 | flash_rec | different | 100.000 | 94.531 | 90.625 | 32.422 | 0.251 |
| ctx5000_b4_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b4_w64 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b8_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w128 | flash_rec | different | 100.000 | 94.531 | 92.188 | 14.062 | 0.246 |
| ctx5000_b8_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b8_w128 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b8_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w256 | flash_rec | different | 100.000 | 96.191 | 94.922 | 9.424 | 0.304 |
| ctx5000_b8_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b8_w256 | vllm_gr | unavailable | — | — | — | — | — |
| ctx5000_b8_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w64 | flash_rec | different | 100.000 | 95.508 | 93.750 | 27.344 | 0.246 |
| ctx5000_b8_w64 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b8_w64 | vllm_gr | unavailable | — | — | — | — | — |

</details>

<details>
<summary>Throughput, status and source runs</summary>

| Case | Candidate | Status | Requests/s | Reference | Source run |
| --- | --- | --- | --- | --- | --- |
| ctx1000_b1_w128 | sid_gr_inference | success | 93.590 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w128 | vllm_gr | success | 41.719 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w128 | flash_rec | success | 29.686 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w128 | tensorrt_llm | success | 13.970 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w256 | sid_gr_inference | success | 66.309 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w256 | vllm_gr | success | 30.380 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w256 | flash_rec | success | 23.066 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w256 | tensorrt_llm | success | 7.734 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w64 | sid_gr_inference | success | 107.422 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w64 | vllm_gr | success | 48.718 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w64 | flash_rec | success | 34.275 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b1_w64 | tensorrt_llm | success | 20.739 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w128 | sid_gr_inference | success | 127.429 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w128 | flash_rec | success | 37.557 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w256 | sid_gr_inference | success | 74.518 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w256 | flash_rec | success | 28.055 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w64 | sid_gr_inference | success | 180.368 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w64 | flash_rec | success | 46.763 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b2_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w128 | sid_gr_inference | success | 142.946 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w128 | flash_rec | success | 43.090 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w256 | sid_gr_inference | success | 84.826 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w256 | flash_rec | success | 31.161 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w64 | sid_gr_inference | success | 234.390 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w64 | flash_rec | success | 52.807 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b4_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w128 | sid_gr_inference | success | 161.674 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w128 | flash_rec | success | 47.628 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w256 | sid_gr_inference | success | 93.126 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w256 | flash_rec | success | 33.270 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w64 | sid_gr_inference | success | 261.080 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w64 | flash_rec | success | 59.500 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx1000_b8_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w128 | sid_gr_inference | success | 80.308 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w128 | vllm_gr | success | 29.606 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w128 | flash_rec | success | 9.425 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w256 | sid_gr_inference | success | 55.423 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w256 | vllm_gr | success | 23.915 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w256 | flash_rec | success | 7.128 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w64 | sid_gr_inference | success | 91.085 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w64 | vllm_gr | success | 32.726 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w64 | flash_rec | success | 10.652 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b1_w64 | tensorrt_llm | success | 6.947 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w128 | sid_gr_inference | success | 102.809 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w128 | flash_rec | success | 9.373 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w256 | sid_gr_inference | success | 63.018 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w256 | flash_rec | success | 7.795 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w64 | sid_gr_inference | success | 136.369 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w64 | flash_rec | success | 11.113 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b2_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w128 | sid_gr_inference | success | 112.850 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w128 | flash_rec | success | 9.245 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w256 | sid_gr_inference | success | 70.033 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w256 | flash_rec | success | 7.558 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w64 | sid_gr_inference | success | 163.407 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w64 | flash_rec | success | 10.260 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b4_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w128 | sid_gr_inference | success | 124.453 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w128 | flash_rec | success | 9.345 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w128 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w128 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w256 | sid_gr_inference | success | 73.533 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w256 | flash_rec | success | 7.532 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w256 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w256 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w64 | sid_gr_inference | success | 174.275 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w64 | flash_rec | success | 10.530 | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w64 | tensorrt_llm | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |
| ctx5000_b8_w64 | vllm_gr | unsupported | — | sid_gr_inference | [20261013T063653115757Z-b34f36](../results/20261013T063653115757Z-b34f36/report.md) (completed) |

</details>

## [NVIDIA RTX PRO 6000 Blackwell Server Edition](../config/hardware/rtx_pro_6000_blackwell_server.yml)

**CPU:** Intel(R) Xeon(R) Gold 6444Y · **Worker cores:** 10 · **Driver:** 615.71.09

| Solution | Exact beam/score match | Different beams or scores | Top-1 mismatches | Not compared |
| --- | --- | --- | --- | --- |
| sid_gr_inference | 0 of 24 cases | 24 of 24 cases | 7 cases | 0 cases |
| sglang_beam_search | 0 of 24 cases | 24 of 24 cases | 7 cases | 0 cases |
| sglang_beam_search_new | 0 of 24 cases | 24 of 24 cases | 7 cases | 0 cases |
| sglang | 24 of 24 cases | 0 of 24 cases | 0 cases | 0 cases |

| Top-1 mismatch case | Candidate |
| --- | --- |
| qwen3_1_7B/ctx5000_b1_w128 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b1_w128 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b1_w64 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b1_w64 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b2_w128 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b2_w256 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b2_w256 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b2_w256 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b2_w64 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b2_w64 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b4_w128 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b4_w128 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b4_w256 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b4_w256 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b4_w256 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b8_w128 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b8_w128 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b8_w128 | sglang_beam_search |
| qwen3_1_7B/ctx5000_b8_w64 | sid_gr_inference |
| qwen3_1_7B/ctx5000_b8_w64 | sglang_beam_search_new |
| qwen3_1_7B/ctx5000_b8_w64 | sglang_beam_search |



### Latency comparison

| Context | Batch | Beam | sid_gr_inference | sglang_beam_search | sglang_beam_search_new | sglang | Fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 64 | **9.682** | 18.255 | 23.063 | 31.425 | **sid_gr_inference** |
| 1000 | 1 | 128 | **11.090** | 21.990 | 24.534 | 34.607 | **sid_gr_inference** |
| 1000 | 1 | 256 | **15.868** | 34.228 | 31.601 | 46.678 | **sid_gr_inference** |
| 1000 | 2 | 64 | **11.382** | 23.220 | 21.694 | 45.733 | **sid_gr_inference** |
| 1000 | 2 | 128 | **16.459** | 35.183 | 28.216 | 56.902 | **sid_gr_inference** |
| 1000 | 2 | 256 | **28.105** | 53.195 | 39.290 | 73.205 | **sid_gr_inference** |
| 1000 | 4 | 64 | **17.817** | 37.061 | 29.677 | 74.804 | **sid_gr_inference** |
| 1000 | 4 | 128 | **29.217** | 55.465 | 40.427 | 89.991 | **sid_gr_inference** |
| 1000 | 4 | 256 | **49.799** | 85.638 | 60.272 | 123.284 | **sid_gr_inference** |
| 1000 | 8 | 64 | **31.808** | 58.644 | 43.731 | 129.553 | **sid_gr_inference** |
| 1000 | 8 | 128 | **52.064** | 90.123 | 63.774 | 160.896 | **sid_gr_inference** |
| 1000 | 8 | 256 | **88.593** | 155.543 | 102.508 | 230.797 | **sid_gr_inference** |
| 5000 | 1 | 64 | **11.346** | 29.101 | 28.674 | 82.708 | **sid_gr_inference** |
| 5000 | 1 | 128 | **12.881** | 41.748 | 32.419 | 94.604 | **sid_gr_inference** |
| 5000 | 1 | 256 | **18.476** | 83.531 | 44.315 | 130.143 | **sid_gr_inference** |
| 5000 | 2 | 64 | **15.161** | 44.654 | 29.683 | 156.851 | **sid_gr_inference** |
| 5000 | 2 | 128 | **20.143** | 79.591 | 41.549 | 191.279 | **sid_gr_inference** |
| 5000 | 2 | 256 | **33.430** | 117.011 | 50.243 | 234.368 | **sid_gr_inference** |
| 5000 | 4 | 64 | **25.283** | 89.309 | 43.326 | 305.269 | **sid_gr_inference** |
| 5000 | 4 | 128 | **36.623** | 120.496 | 51.847 | 347.044 | **sid_gr_inference** |
| 5000 | 4 | 256 | **59.507** | 216.492 | 70.795 | 443.267 | **sid_gr_inference** |
| 5000 | 8 | 64 | **46.440** | 127.440 | 57.388 | 589.925 | **sid_gr_inference** |
| 5000 | 8 | 128 | **66.478** | 224.088 | 77.105 | 683.185 | **sid_gr_inference** |
| 5000 | 8 | 256 | **110.685** | 422.067 | 119.025 | 886.314 | **sid_gr_inference** |

<details>
<summary>Output parity, including reference repeat consistency</summary>

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sid_gr_inference | different | 100.000 | 96.875 | 96.875 | 14.062 | 0.272 |
| ctx1000_b1_w128 | sglang_beam_search | different | 100.000 | 95.312 | 95.312 | 13.281 | 0.146 |
| ctx1000_b1_w128 | sglang_beam_search_new | different | 100.000 | 96.875 | 96.875 | 11.719 | 0.172 |
| ctx1000_b1_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 10.938 | 0.191 |
| ctx1000_b1_w256 | sglang_beam_search_new | different | 100.000 | 96.094 | 96.094 | 9.375 | 0.38 |
| ctx1000_b1_w256 | sglang_beam_search | different | 100.000 | 94.922 | 94.922 | 7.422 | 0.2 |
| ctx1000_b1_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w64 | sid_gr_inference | different | 100.000 | 95.312 | 95.312 | 21.875 | 0.19 |
| ctx1000_b1_w64 | sglang_beam_search | different | 100.000 | 95.312 | 95.312 | 32.812 | 0.187 |
| ctx1000_b1_w64 | sglang_beam_search_new | different | 100.000 | 93.750 | 93.750 | 26.562 | 0.207 |
| ctx1000_b1_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w128 | sid_gr_inference | different | 100.000 | 95.703 | 95.312 | 10.938 | 0.269 |
| ctx1000_b2_w128 | sglang_beam_search_new | different | 100.000 | 94.531 | 93.750 | 8.594 | 0.175 |
| ctx1000_b2_w128 | sglang_beam_search | different | 100.000 | 92.578 | 91.406 | 12.500 | 0.304 |
| ctx1000_b2_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w256 | sid_gr_inference | different | 100.000 | 95.898 | 94.922 | 7.812 | 0.24 |
| ctx1000_b2_w256 | sglang_beam_search_new | different | 100.000 | 96.094 | 95.312 | 8.398 | 0.266 |
| ctx1000_b2_w256 | sglang_beam_search | different | 100.000 | 97.266 | 96.484 | 9.570 | 0.266 |
| ctx1000_b2_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b2_w64 | sid_gr_inference | different | 100.000 | 92.969 | 92.188 | 22.656 | 0.231 |
| ctx1000_b2_w64 | sglang_beam_search_new | different | 100.000 | 93.750 | 92.188 | 18.750 | 0.189 |
| ctx1000_b2_w64 | sglang_beam_search | different | 100.000 | 95.312 | 95.312 | 33.594 | 0.196 |
| ctx1000_b2_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w128 | sid_gr_inference | different | 100.000 | 96.289 | 94.531 | 15.430 | 0.183 |
| ctx1000_b4_w128 | sglang_beam_search_new | different | 100.000 | 95.703 | 94.531 | 13.281 | 0.254 |
| ctx1000_b4_w128 | sglang_beam_search | different | 100.000 | 95.898 | 93.750 | 15.234 | 0.274 |
| ctx1000_b4_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w256 | sid_gr_inference | different | 100.000 | 95.801 | 95.312 | 8.594 | 0.218 |
| ctx1000_b4_w256 | sglang_beam_search_new | different | 100.000 | 95.801 | 92.188 | 6.738 | 0.277 |
| ctx1000_b4_w256 | sglang_beam_search | different | 100.000 | 95.410 | 92.969 | 9.668 | 0.274 |
| ctx1000_b4_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b4_w64 | sid_gr_inference | different | 100.000 | 94.922 | 93.750 | 30.078 | 0.177 |
| ctx1000_b4_w64 | sglang_beam_search_new | different | 100.000 | 94.922 | 92.188 | 27.734 | 0.196 |
| ctx1000_b4_w64 | sglang_beam_search | different | 100.000 | 96.094 | 95.312 | 27.734 | 0.159 |
| ctx1000_b4_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w128 | sid_gr_inference | different | 100.000 | 96.094 | 94.531 | 16.992 | 0.367 |
| ctx1000_b8_w128 | sglang_beam_search_new | different | 100.000 | 96.191 | 94.531 | 17.090 | 0.322 |
| ctx1000_b8_w128 | sglang_beam_search | different | 100.000 | 96.094 | 94.531 | 15.137 | 0.334 |
| ctx1000_b8_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w256 | sid_gr_inference | different | 100.000 | 95.703 | 94.922 | 8.887 | 0.321 |
| ctx1000_b8_w256 | sglang_beam_search_new | different | 100.000 | 96.387 | 95.312 | 9.033 | 0.368 |
| ctx1000_b8_w256 | sglang_beam_search | different | 100.000 | 96.436 | 95.312 | 8.154 | 0.349 |
| ctx1000_b8_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b8_w64 | sid_gr_inference | different | 100.000 | 96.680 | 93.750 | 30.469 | 0.236 |
| ctx1000_b8_w64 | sglang_beam_search_new | different | 100.000 | 95.117 | 90.625 | 28.711 | 0.222 |
| ctx1000_b8_w64 | sglang_beam_search | different | 100.000 | 95.508 | 89.062 | 28.320 | 0.292 |
| ctx1000_b8_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w128 | sid_gr_inference | different | 0.000 | 94.531 | 94.531 | 6.250 | 0.177 |
| ctx5000_b1_w128 | sglang_beam_search_new | different | 0.000 | 95.312 | 95.312 | 13.281 | 0.178 |
| ctx5000_b1_w128 | sglang_beam_search | different | 100.000 | 96.875 | 96.875 | 12.500 | 0.148 |
| ctx5000_b1_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | sid_gr_inference | different | 100.000 | 96.094 | 96.094 | 6.641 | 0.453 |
| ctx5000_b1_w256 | sglang_beam_search_new | different | 100.000 | 96.875 | 96.875 | 8.203 | 0.439 |
| ctx5000_b1_w256 | sglang_beam_search | different | 100.000 | 95.703 | 95.703 | 6.250 | 0.212 |
| ctx5000_b1_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w64 | sid_gr_inference | different | 0.000 | 96.875 | 96.875 | 23.438 | 0.168 |
| ctx5000_b1_w64 | sglang_beam_search_new | different | 0.000 | 95.312 | 95.312 | 18.750 | 0.212 |
| ctx5000_b1_w64 | sglang_beam_search | different | 100.000 | 95.312 | 95.312 | 21.875 | 0.21 |
| ctx5000_b1_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w128 | sid_gr_inference | different | 100.000 | 91.797 | 87.500 | 10.156 | 0.228 |
| ctx5000_b2_w128 | sglang_beam_search_new | different | 100.000 | 92.578 | 88.281 | 14.453 | 0.255 |
| ctx5000_b2_w128 | sglang_beam_search | different | 50.000 | 92.578 | 89.062 | 8.203 | 0.237 |
| ctx5000_b2_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w256 | sid_gr_inference | different | 50.000 | 95.312 | 95.312 | 5.664 | 0.331 |
| ctx5000_b2_w256 | sglang_beam_search_new | different | 50.000 | 96.094 | 96.094 | 8.008 | 0.251 |
| ctx5000_b2_w256 | sglang_beam_search | different | 50.000 | 93.945 | 92.188 | 5.273 | 0.388 |
| ctx5000_b2_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b2_w64 | sid_gr_inference | different | 50.000 | 92.969 | 90.625 | 17.969 | 0.344 |
| ctx5000_b2_w64 | sglang_beam_search_new | different | 100.000 | 92.188 | 92.188 | 17.969 | 0.176 |
| ctx5000_b2_w64 | sglang_beam_search | different | 50.000 | 94.531 | 93.750 | 18.750 | 0.188 |
| ctx5000_b2_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w128 | sid_gr_inference | different | 100.000 | 95.508 | 92.969 | 16.797 | 0.219 |
| ctx5000_b4_w128 | sglang_beam_search_new | different | 75.000 | 95.703 | 93.750 | 13.086 | 0.235 |
| ctx5000_b4_w128 | sglang_beam_search | different | 75.000 | 96.289 | 94.531 | 15.234 | 0.212 |
| ctx5000_b4_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w256 | sid_gr_inference | different | 75.000 | 95.605 | 94.531 | 7.812 | 0.339 |
| ctx5000_b4_w256 | sglang_beam_search_new | different | 75.000 | 95.312 | 94.141 | 9.277 | 0.235 |
| ctx5000_b4_w256 | sglang_beam_search | different | 75.000 | 95.801 | 94.922 | 8.594 | 0.238 |
| ctx5000_b4_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b4_w64 | sid_gr_inference | different | 100.000 | 94.531 | 90.625 | 23.828 | 0.204 |
| ctx5000_b4_w64 | sglang_beam_search_new | different | 100.000 | 95.703 | 95.312 | 27.344 | 0.218 |
| ctx5000_b4_w64 | sglang_beam_search | different | 100.000 | 96.094 | 95.312 | 32.031 | 0.209 |
| ctx5000_b4_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w128 | sid_gr_inference | different | 87.500 | 94.922 | 92.188 | 15.527 | 0.445 |
| ctx5000_b8_w128 | sglang_beam_search_new | different | 87.500 | 95.605 | 93.750 | 16.992 | 0.427 |
| ctx5000_b8_w128 | sglang_beam_search | different | 87.500 | 95.312 | 94.531 | 15.039 | 0.293 |
| ctx5000_b8_w128 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w256 | sid_gr_inference | different | 100.000 | 95.850 | 93.359 | 9.229 | 0.322 |
| ctx5000_b8_w256 | sglang_beam_search_new | different | 100.000 | 95.947 | 94.922 | 9.180 | 0.583 |
| ctx5000_b8_w256 | sglang_beam_search | different | 100.000 | 96.143 | 94.922 | 9.473 | 0.259 |
| ctx5000_b8_w256 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b8_w64 | sid_gr_inference | different | 87.500 | 96.094 | 93.750 | 25.586 | 0.445 |
| ctx5000_b8_w64 | sglang_beam_search_new | different | 87.500 | 96.289 | 95.312 | 31.641 | 0.427 |
| ctx5000_b8_w64 | sglang_beam_search | different | 87.500 | 96.484 | 93.750 | 24.805 | 0.221 |
| ctx5000_b8_w64 | sglang | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |

</details>

<details>
<summary>Throughput, status and source runs</summary>

| Case | Candidate | Status | Requests/s | Reference | Source run |
| --- | --- | --- | --- | --- | --- |
| ctx1000_b1_w128 | sid_gr_inference | success | 90.168 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w128 | sglang_beam_search | success | 45.475 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w128 | sglang_beam_search_new | success | 40.760 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w128 | sglang | success | 28.896 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w256 | sid_gr_inference | success | 63.021 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w256 | sglang_beam_search_new | success | 31.644 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w256 | sglang_beam_search | success | 29.216 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w256 | sglang | success | 21.424 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w64 | sid_gr_inference | success | 103.284 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w64 | sglang_beam_search | success | 54.779 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w64 | sglang_beam_search_new | success | 43.359 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b1_w64 | sglang | success | 31.822 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w128 | sid_gr_inference | success | 121.516 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w128 | sglang_beam_search_new | success | 70.881 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w128 | sglang_beam_search | success | 56.846 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w128 | sglang | success | 35.148 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w256 | sid_gr_inference | success | 71.162 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w256 | sglang_beam_search_new | success | 50.904 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w256 | sglang_beam_search | success | 37.597 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w256 | sglang | success | 27.321 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w64 | sid_gr_inference | success | 175.711 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w64 | sglang_beam_search_new | success | 92.193 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w64 | sglang_beam_search | success | 86.131 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b2_w64 | sglang | success | 43.732 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w128 | sid_gr_inference | success | 136.907 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w128 | sglang_beam_search_new | success | 98.945 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w128 | sglang_beam_search | success | 72.117 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w128 | sglang | success | 44.449 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w256 | sid_gr_inference | success | 80.323 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w256 | sglang_beam_search_new | success | 66.366 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w256 | sglang_beam_search | success | 46.708 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w256 | sglang | success | 32.445 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w64 | sid_gr_inference | success | 224.502 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w64 | sglang_beam_search_new | success | 134.784 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w64 | sglang_beam_search | success | 107.931 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b4_w64 | sglang | success | 53.473 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w128 | sid_gr_inference | success | 153.657 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w128 | sglang_beam_search_new | success | 125.444 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w128 | sglang_beam_search | success | 88.767 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w128 | sglang | success | 49.722 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w256 | sid_gr_inference | success | 90.301 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w256 | sglang_beam_search_new | success | 78.043 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w256 | sglang_beam_search | success | 51.433 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w256 | sglang | success | 34.662 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w64 | sid_gr_inference | success | 251.506 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w64 | sglang_beam_search_new | success | 182.936 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w64 | sglang_beam_search | success | 136.417 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx1000_b8_w64 | sglang | success | 61.751 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w128 | sid_gr_inference | success | 77.632 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w128 | sglang_beam_search_new | success | 30.846 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w128 | sglang_beam_search | success | 23.953 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w128 | sglang | success | 10.570 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w256 | sid_gr_inference | success | 54.123 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w256 | sglang_beam_search_new | success | 22.566 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w256 | sglang_beam_search | success | 11.972 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w256 | sglang | success | 7.684 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w64 | sid_gr_inference | success | 88.140 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w64 | sglang_beam_search_new | success | 34.875 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w64 | sglang_beam_search | success | 34.363 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b1_w64 | sglang | success | 12.091 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w128 | sid_gr_inference | success | 99.292 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w128 | sglang_beam_search_new | success | 48.135 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w128 | sglang_beam_search | success | 25.128 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w128 | sglang | success | 10.456 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w256 | sid_gr_inference | success | 59.827 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w256 | sglang_beam_search_new | success | 39.806 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w256 | sglang_beam_search | success | 17.092 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w256 | sglang | success | 8.534 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w64 | sid_gr_inference | success | 131.915 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w64 | sglang_beam_search_new | success | 67.378 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w64 | sglang_beam_search | success | 44.789 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b2_w64 | sglang | success | 12.751 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w128 | sid_gr_inference | success | 109.220 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w128 | sglang_beam_search_new | success | 77.150 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w128 | sglang_beam_search | success | 33.196 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w128 | sglang | success | 11.526 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w256 | sid_gr_inference | success | 67.219 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w256 | sglang_beam_search_new | success | 56.501 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w256 | sglang_beam_search | success | 18.476 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w256 | sglang | success | 9.024 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w64 | sid_gr_inference | success | 158.211 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w64 | sglang_beam_search_new | success | 92.324 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w64 | sglang_beam_search | success | 44.788 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b4_w64 | sglang | success | 13.103 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w128 | sid_gr_inference | success | 120.340 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w128 | sglang_beam_search_new | success | 103.755 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w128 | sglang_beam_search | success | 35.700 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w128 | sglang | success | 11.710 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w256 | sid_gr_inference | success | 72.277 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w256 | sglang_beam_search_new | success | 67.213 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w256 | sglang_beam_search | success | 18.954 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w256 | sglang | success | 9.026 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w64 | sid_gr_inference | success | 172.265 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w64 | sglang_beam_search_new | success | 139.402 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w64 | sglang_beam_search | success | 62.775 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |
| ctx5000_b8_w64 | sglang | success | 13.561 | sglang | [20261012T113832092061Z-c0ad33](../results/20261012T113832092061Z-c0ad33/report.md) (completed) |

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
