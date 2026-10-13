# SID-GR benchmark

Run `20261013T063653115757Z-b34f36` · [Run profile](../../config/runs/sid_gr_new_solutions.yml)

[Latency](#latency-comparison) · [Parity](#output-parity) · [GPU diagnosis](#gpu-diagnosis) · [Environment](#run-details) · [Artifacts](#artifacts)

| Run summary | Meaning |
| --- | --- |
| Status | completed |
| Workload | 48 context / batch / beam combinations |
| Solutions compared | sid_gr_inference, vllm_gr, tensorrt_llm, flash_rec |
| Progress | 192 of 192 solution–workload runs finished |
| Execution checks | 116 passed; 76 failed or unsupported |

Each solution runs every workload combination, with the warmups and repeats listed below.

Parity counts below include finished workload cases only; the reference row checks repeat consistency.

| Solution | Exact beam/score match | Different beams or scores | Top-1 mismatches | Not compared | Finished runs | Status |
| --- | --- | --- | --- | --- | --- | --- |
| sid_gr_inference | 48 of 48 cases | 0 of 48 cases | 0 cases | 0 cases | 48 of 48 | completed |
| vllm_gr | 0 of 48 cases | 12 of 48 cases | 1 cases | 36 cases | 48 of 48 | completed |
| tensorrt_llm | 0 of 48 cases | 8 of 48 cases | 1 cases | 40 cases | 48 of 48 | completed |
| flash_rec | 0 of 48 cases | 48 of 48 cases | 6 cases | 0 cases | 48 of 48 | completed |

| Top-1 mismatch case | Candidate |
| --- | --- |
| qwen3_0_6B/ctx1000_b8_w128 | flash_rec |
| qwen3_0_6B/ctx5000_b1_w128 | vllm_gr |
| qwen3_0_6B/ctx5000_b2_w128 | flash_rec |
| qwen3_0_6B/ctx5000_b4_w256 | flash_rec |
| qwen3_1_7B/ctx5000_b1_w64 | tensorrt_llm |
| qwen3_1_7B/ctx5000_b2_w64 | flash_rec |
| qwen3_1_7B/ctx5000_b2_w128 | flash_rec |
| qwen3_1_7B/ctx5000_b4_w128 | flash_rec |

**Reference:** `sid_gr_inference` · Offline batches on one GPU

Hardware profile: [rtx_pro_6000_blackwell_server](../../config/hardware/rtx_pro_6000_blackwell_server.yml).

## Latency comparison

Median batch **ms**; lower is better. **Bold** = lowest recorded median. Pending = no finished result. — = unavailable.

### qwen3_0_6B

| Context | Batch | Beam | sid_gr_inference | vllm_gr | tensorrt_llm | flash_rec | Fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 64 | **6.178** | 15.680 | 44.366 | 23.456 | **sid_gr_inference** |
| 1000 | 1 | 128 | **7.211** | 18.528 | 69.632 | 26.490 | **sid_gr_inference** |
| 1000 | 1 | 256 | **10.299** | 25.835 | 125.313 | 35.094 | **sid_gr_inference** |
| 1000 | 2 | 64 | **7.607** | unsupported | unsupported | 27.907 | **sid_gr_inference** |
| 1000 | 2 | 128 | **10.987** | unsupported | unsupported | 37.035 | **sid_gr_inference** |
| 1000 | 2 | 256 | **17.908** | unsupported | unsupported | 50.515 | **sid_gr_inference** |
| 1000 | 4 | 64 | **12.363** | unsupported | unsupported | 46.996 | **sid_gr_inference** |
| 1000 | 4 | 128 | **19.007** | unsupported | unsupported | 59.693 | **sid_gr_inference** |
| 1000 | 4 | 256 | **32.354** | unsupported | unsupported | 88.312 | **sid_gr_inference** |
| 1000 | 8 | 64 | **21.780** | unsupported | unsupported | 79.866 | **sid_gr_inference** |
| 1000 | 8 | 128 | **34.500** | unsupported | unsupported | 107.071 | **sid_gr_inference** |
| 1000 | 8 | 256 | **59.417** | unsupported | unsupported | 167.916 | **sid_gr_inference** |
| 5000 | 1 | 64 | **7.673** | 24.216 | 139.538 | 58.478 | **sid_gr_inference** |
| 5000 | 1 | 128 | **8.873** | 26.616 | unsupported | 70.574 | **sid_gr_inference** |
| 5000 | 1 | 256 | **12.881** | 34.142 | unsupported | 103.594 | **sid_gr_inference** |
| 5000 | 2 | 64 | **11.175** | unsupported | unsupported | 110.094 | **sid_gr_inference** |
| 5000 | 2 | 128 | **14.612** | unsupported | unsupported | 142.010 | **sid_gr_inference** |
| 5000 | 2 | 256 | **22.959** | unsupported | unsupported | 180.031 | **sid_gr_inference** |
| 5000 | 4 | 64 | **19.902** | unsupported | unsupported | 233.202 | **sid_gr_inference** |
| 5000 | 4 | 128 | **26.438** | unsupported | unsupported | 271.516 | **sid_gr_inference** |
| 5000 | 4 | 256 | **41.800** | unsupported | unsupported | 360.722 | **sid_gr_inference** |
| 5000 | 8 | 64 | **36.437** | unsupported | unsupported | 473.831 | **sid_gr_inference** |
| 5000 | 8 | 128 | **49.115** | unsupported | unsupported | 562.395 | **sid_gr_inference** |
| 5000 | 8 | 256 | **83.042** | unsupported | unsupported | 752.758 | **sid_gr_inference** |

### qwen3_1_7B

| Context | Batch | Beam | sid_gr_inference | vllm_gr | tensorrt_llm | flash_rec | Fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 64 | **9.309** | 20.526 | 48.218 | 29.176 | **sid_gr_inference** |
| 1000 | 1 | 128 | **10.685** | 23.970 | 71.584 | 33.686 | **sid_gr_inference** |
| 1000 | 1 | 256 | **15.081** | 32.917 | 129.292 | 43.354 | **sid_gr_inference** |
| 1000 | 2 | 64 | **11.088** | unsupported | unsupported | 42.768 | **sid_gr_inference** |
| 1000 | 2 | 128 | **15.695** | unsupported | unsupported | 53.253 | **sid_gr_inference** |
| 1000 | 2 | 256 | **26.839** | unsupported | unsupported | 71.290 | **sid_gr_inference** |
| 1000 | 4 | 64 | **17.066** | unsupported | unsupported | 75.748 | **sid_gr_inference** |
| 1000 | 4 | 128 | **27.983** | unsupported | unsupported | 92.828 | **sid_gr_inference** |
| 1000 | 4 | 256 | **47.156** | unsupported | unsupported | 128.364 | **sid_gr_inference** |
| 1000 | 8 | 64 | **30.642** | unsupported | unsupported | 134.454 | **sid_gr_inference** |
| 1000 | 8 | 128 | **49.482** | unsupported | unsupported | 167.969 | **sid_gr_inference** |
| 1000 | 8 | 256 | **85.905** | unsupported | unsupported | 240.457 | **sid_gr_inference** |
| 5000 | 1 | 64 | **10.979** | 30.557 | 143.953 | 93.877 | **sid_gr_inference** |
| 5000 | 1 | 128 | **12.452** | 33.777 | unsupported | 106.095 | **sid_gr_inference** |
| 5000 | 1 | 256 | **18.043** | 41.814 | unsupported | 140.299 | **sid_gr_inference** |
| 5000 | 2 | 64 | **14.666** | unsupported | unsupported | 179.976 | **sid_gr_inference** |
| 5000 | 2 | 128 | **19.453** | unsupported | unsupported | 213.387 | **sid_gr_inference** |
| 5000 | 2 | 256 | **31.737** | unsupported | unsupported | 256.567 | **sid_gr_inference** |
| 5000 | 4 | 64 | **24.479** | unsupported | unsupported | 389.847 | **sid_gr_inference** |
| 5000 | 4 | 128 | **35.445** | unsupported | unsupported | 432.661 | **sid_gr_inference** |
| 5000 | 4 | 256 | **57.116** | unsupported | unsupported | 529.208 | **sid_gr_inference** |
| 5000 | 8 | 64 | **45.904** | unsupported | unsupported | 759.759 | **sid_gr_inference** |
| 5000 | 8 | 128 | **64.281** | unsupported | unsupported | 856.089 | **sid_gr_inference** |
| 5000 | 8 | 256 | **108.794** | unsupported | unsupported | 1062.190 | **sid_gr_inference** |

### Failed or unsupported jobs

| Case | Candidate | Reason |
| --- | --- | --- |
| qwen3_0_6B/ctx1000_b2_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b2_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b2_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b2_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b2_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b2_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b4_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b4_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b4_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b4_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b4_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b4_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b8_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b8_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b8_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b8_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b8_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx1000_b8_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b1_w128 | tensorrt_llm | TensorRT-LLM KV capacity: The max KV cache length of input sequences (5000) exceeds the KV cache manager's maximum supported length (4640). |
| qwen3_0_6B/ctx5000_b1_w256 | tensorrt_llm | TensorRT-LLM KV capacity: The max KV cache length of input sequences (5000) exceeds the KV cache manager's maximum supported length (2272). |
| qwen3_0_6B/ctx5000_b2_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b2_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b2_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b2_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b2_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b2_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b4_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b4_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b4_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b4_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b4_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b4_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b8_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b8_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b8_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b8_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b8_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_0_6B/ctx5000_b8_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b2_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b2_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b2_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b2_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b2_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b2_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b4_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b4_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b4_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b4_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b4_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b4_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b8_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b8_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b8_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b8_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b8_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx1000_b8_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b1_w128 | tensorrt_llm | TensorRT-LLM KV capacity: The max KV cache length of input sequences (5000) exceeds the KV cache manager's maximum supported length (4512). |
| qwen3_1_7B/ctx5000_b1_w256 | tensorrt_llm | TensorRT-LLM KV capacity: The max KV cache length of input sequences (5000) exceeds the KV cache manager's maximum supported length (2208). |
| qwen3_1_7B/ctx5000_b2_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b2_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b2_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b2_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b2_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b2_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b4_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b4_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b4_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b4_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b4_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b4_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b8_w64 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b8_w64 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b8_w128 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b8_w128 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b8_w256 | vllm_gr | Pinned vLLM-GR CUSTOM beam KV pool supports batch_size=1 only |
| qwen3_1_7B/ctx5000_b8_w256 | tensorrt_llm | TensorRT-LLM 1.2.1 PyTorch beam search with cumulative scores supports batch_size=1 only |

<details>
<summary>All measurements</summary>

### qwen3_0_6B

| Case | Candidate | Status | Median ms | IQR ms | Δ latency % | Requests/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sid_gr_inference | success | 7.211 | 0.129 | 0.000 | 138.671 |
| ctx1000_b1_w128 | vllm_gr | success | 18.528 | 0.334 | 156.924 | 53.974 |
| ctx1000_b1_w128 | tensorrt_llm | success | 69.632 | 1.724 | 865.600 | 14.361 |
| ctx1000_b1_w128 | flash_rec | success | 26.490 | 0.390 | 267.344 | 37.750 |
| ctx1000_b1_w256 | sid_gr_inference | success | 10.299 | 0.113 | 0.000 | 97.097 |
| ctx1000_b1_w256 | vllm_gr | success | 25.835 | 0.462 | 150.850 | 38.707 |
| ctx1000_b1_w256 | tensorrt_llm | success | 125.313 | 1.739 | 1116.749 | 7.980 |
| ctx1000_b1_w256 | flash_rec | success | 35.094 | 0.397 | 240.748 | 28.495 |
| ctx1000_b1_w64 | sid_gr_inference | success | 6.178 | 0.039 | 0.000 | 161.874 |
| ctx1000_b1_w64 | vllm_gr | success | 15.680 | 0.490 | 153.811 | 63.777 |
| ctx1000_b1_w64 | tensorrt_llm | success | 44.366 | 0.960 | 618.172 | 22.540 |
| ctx1000_b1_w64 | flash_rec | success | 23.456 | 0.563 | 279.691 | 42.633 |
| ctx1000_b2_w128 | sid_gr_inference | success | 10.987 | 0.124 | 0.000 | 182.028 |
| ctx1000_b2_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b2_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b2_w128 | flash_rec | success | 37.035 | 0.350 | 237.069 | 54.003 |
| ctx1000_b2_w256 | sid_gr_inference | success | 17.908 | 0.154 | 0.000 | 111.681 |
| ctx1000_b2_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b2_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b2_w256 | flash_rec | success | 50.515 | 0.615 | 182.079 | 39.592 |
| ctx1000_b2_w64 | sid_gr_inference | success | 7.607 | 0.035 | 0.000 | 262.918 |
| ctx1000_b2_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b2_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b2_w64 | flash_rec | success | 27.907 | 0.121 | 266.858 | 71.667 |
| ctx1000_b4_w128 | sid_gr_inference | success | 19.007 | 0.157 | 0.000 | 210.448 |
| ctx1000_b4_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b4_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b4_w128 | flash_rec | success | 59.693 | 0.299 | 214.058 | 67.009 |
| ctx1000_b4_w256 | sid_gr_inference | success | 32.354 | 0.139 | 0.000 | 123.630 |
| ctx1000_b4_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b4_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b4_w256 | flash_rec | success | 88.312 | 0.385 | 172.951 | 45.294 |
| ctx1000_b4_w64 | sid_gr_inference | success | 12.363 | 0.045 | 0.000 | 323.551 |
| ctx1000_b4_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b4_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b4_w64 | flash_rec | success | 46.996 | 0.214 | 280.143 | 85.113 |
| ctx1000_b8_w128 | sid_gr_inference | success | 34.500 | 0.142 | 0.000 | 231.883 |
| ctx1000_b8_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b8_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b8_w128 | flash_rec | success | 107.071 | 0.365 | 210.349 | 74.717 |
| ctx1000_b8_w256 | sid_gr_inference | success | 59.417 | 0.256 | 0.000 | 134.641 |
| ctx1000_b8_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b8_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b8_w256 | flash_rec | success | 167.916 | 1.269 | 182.604 | 47.643 |
| ctx1000_b8_w64 | sid_gr_inference | success | 21.780 | 0.189 | 0.000 | 367.307 |
| ctx1000_b8_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b8_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b8_w64 | flash_rec | success | 79.866 | 0.329 | 266.691 | 100.168 |
| ctx5000_b1_w128 | sid_gr_inference | success | 8.873 | 0.027 | 0.000 | 112.705 |
| ctx5000_b1_w128 | vllm_gr | success | 26.616 | 0.270 | 199.980 | 37.571 |
| ctx5000_b1_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b1_w128 | flash_rec | success | 70.574 | 0.111 | 695.403 | 14.170 |
| ctx5000_b1_w256 | sid_gr_inference | success | 12.881 | 0.073 | 0.000 | 77.633 |
| ctx5000_b1_w256 | vllm_gr | success | 34.142 | 0.293 | 165.058 | 29.289 |
| ctx5000_b1_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b1_w256 | flash_rec | success | 103.594 | 0.445 | 704.229 | 9.653 |
| ctx5000_b1_w64 | sid_gr_inference | success | 7.673 | 0.085 | 0.000 | 130.335 |
| ctx5000_b1_w64 | vllm_gr | success | 24.216 | 0.647 | 215.618 | 41.295 |
| ctx5000_b1_w64 | tensorrt_llm | success | 139.538 | 2.417 | 1718.668 | 7.166 |
| ctx5000_b1_w64 | flash_rec | success | 58.478 | 0.185 | 662.178 | 17.100 |
| ctx5000_b2_w128 | sid_gr_inference | success | 14.612 | 0.050 | 0.000 | 136.874 |
| ctx5000_b2_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b2_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b2_w128 | flash_rec | success | 142.010 | 0.686 | 871.873 | 14.083 |
| ctx5000_b2_w256 | sid_gr_inference | success | 22.959 | 0.115 | 0.000 | 87.114 |
| ctx5000_b2_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b2_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b2_w256 | flash_rec | success | 180.031 | 0.724 | 684.158 | 11.109 |
| ctx5000_b2_w64 | sid_gr_inference | success | 11.175 | 0.044 | 0.000 | 178.963 |
| ctx5000_b2_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b2_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b2_w64 | flash_rec | success | 110.094 | 0.357 | 885.136 | 18.166 |
| ctx5000_b4_w128 | sid_gr_inference | success | 26.438 | 0.235 | 0.000 | 151.297 |
| ctx5000_b4_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b4_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b4_w128 | flash_rec | success | 271.516 | 0.635 | 926.991 | 14.732 |
| ctx5000_b4_w256 | sid_gr_inference | success | 41.800 | 0.122 | 0.000 | 95.693 |
| ctx5000_b4_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b4_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b4_w256 | flash_rec | success | 360.722 | 1.154 | 762.966 | 11.089 |
| ctx5000_b4_w64 | sid_gr_inference | success | 19.902 | 0.220 | 0.000 | 200.987 |
| ctx5000_b4_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b4_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b4_w64 | flash_rec | success | 233.202 | 0.851 | 1071.766 | 17.153 |
| ctx5000_b8_w128 | sid_gr_inference | success | 49.115 | 0.200 | 0.000 | 162.882 |
| ctx5000_b8_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b8_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b8_w128 | flash_rec | success | 562.395 | 2.106 | 1045.052 | 14.225 |
| ctx5000_b8_w256 | sid_gr_inference | success | 83.042 | 0.950 | 0.000 | 96.337 |
| ctx5000_b8_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b8_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b8_w256 | flash_rec | success | 752.758 | 5.867 | 806.477 | 10.628 |
| ctx5000_b8_w64 | sid_gr_inference | success | 36.437 | 0.307 | 0.000 | 219.556 |
| ctx5000_b8_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b8_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b8_w64 | flash_rec | success | 473.831 | 0.893 | 1200.405 | 16.884 |

### qwen3_1_7B

| Case | Candidate | Status | Median ms | IQR ms | Δ latency % | Requests/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sid_gr_inference | success | 10.685 | 0.042 | 0.000 | 93.590 |
| ctx1000_b1_w128 | vllm_gr | success | 23.970 | 0.766 | 124.333 | 41.719 |
| ctx1000_b1_w128 | tensorrt_llm | success | 71.584 | 1.253 | 569.956 | 13.970 |
| ctx1000_b1_w128 | flash_rec | success | 33.686 | 0.627 | 215.264 | 29.686 |
| ctx1000_b1_w256 | sid_gr_inference | success | 15.081 | 0.081 | 0.000 | 66.309 |
| ctx1000_b1_w256 | vllm_gr | success | 32.917 | 0.776 | 118.266 | 30.380 |
| ctx1000_b1_w256 | tensorrt_llm | success | 129.292 | 3.214 | 757.316 | 7.734 |
| ctx1000_b1_w256 | flash_rec | success | 43.354 | 0.220 | 187.475 | 23.066 |
| ctx1000_b1_w64 | sid_gr_inference | success | 9.309 | 0.043 | 0.000 | 107.422 |
| ctx1000_b1_w64 | vllm_gr | success | 20.526 | 0.210 | 120.495 | 48.718 |
| ctx1000_b1_w64 | tensorrt_llm | success | 48.218 | 1.996 | 417.961 | 20.739 |
| ctx1000_b1_w64 | flash_rec | success | 29.176 | 0.032 | 213.412 | 34.275 |
| ctx1000_b2_w128 | sid_gr_inference | success | 15.695 | 0.065 | 0.000 | 127.429 |
| ctx1000_b2_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b2_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b2_w128 | flash_rec | success | 53.253 | 0.319 | 239.296 | 37.557 |
| ctx1000_b2_w256 | sid_gr_inference | success | 26.839 | 0.238 | 0.000 | 74.518 |
| ctx1000_b2_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b2_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b2_w256 | flash_rec | success | 71.290 | 0.663 | 165.619 | 28.055 |
| ctx1000_b2_w64 | sid_gr_inference | success | 11.088 | 0.039 | 0.000 | 180.368 |
| ctx1000_b2_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b2_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b2_w64 | flash_rec | success | 42.768 | 0.124 | 285.704 | 46.763 |
| ctx1000_b4_w128 | sid_gr_inference | success | 27.983 | 0.262 | 0.000 | 142.946 |
| ctx1000_b4_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b4_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b4_w128 | flash_rec | success | 92.828 | 0.510 | 231.736 | 43.090 |
| ctx1000_b4_w256 | sid_gr_inference | success | 47.156 | 0.220 | 0.000 | 84.826 |
| ctx1000_b4_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b4_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b4_w256 | flash_rec | success | 128.364 | 0.558 | 172.215 | 31.161 |
| ctx1000_b4_w64 | sid_gr_inference | success | 17.066 | 0.062 | 0.000 | 234.390 |
| ctx1000_b4_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b4_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b4_w64 | flash_rec | success | 75.748 | 0.373 | 343.864 | 52.807 |
| ctx1000_b8_w128 | sid_gr_inference | success | 49.482 | 0.224 | 0.000 | 161.674 |
| ctx1000_b8_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b8_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b8_w128 | flash_rec | success | 167.969 | 0.342 | 239.454 | 47.628 |
| ctx1000_b8_w256 | sid_gr_inference | success | 85.905 | 0.711 | 0.000 | 93.126 |
| ctx1000_b8_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b8_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b8_w256 | flash_rec | success | 240.457 | 1.657 | 179.909 | 33.270 |
| ctx1000_b8_w64 | sid_gr_inference | success | 30.642 | 0.285 | 0.000 | 261.080 |
| ctx1000_b8_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx1000_b8_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx1000_b8_w64 | flash_rec | success | 134.454 | 0.747 | 338.789 | 59.500 |
| ctx5000_b1_w128 | sid_gr_inference | success | 12.452 | 0.037 | 0.000 | 80.308 |
| ctx5000_b1_w128 | vllm_gr | success | 33.777 | 0.184 | 171.258 | 29.606 |
| ctx5000_b1_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b1_w128 | flash_rec | success | 106.095 | 0.460 | 752.025 | 9.425 |
| ctx5000_b1_w256 | sid_gr_inference | success | 18.043 | 0.143 | 0.000 | 55.423 |
| ctx5000_b1_w256 | vllm_gr | success | 41.814 | 1.040 | 131.746 | 23.915 |
| ctx5000_b1_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b1_w256 | flash_rec | success | 140.299 | 0.440 | 677.574 | 7.128 |
| ctx5000_b1_w64 | sid_gr_inference | success | 10.979 | 0.043 | 0.000 | 91.085 |
| ctx5000_b1_w64 | vllm_gr | success | 30.557 | 0.712 | 178.326 | 32.726 |
| ctx5000_b1_w64 | tensorrt_llm | success | 143.953 | 2.243 | 1211.192 | 6.947 |
| ctx5000_b1_w64 | flash_rec | success | 93.877 | 0.410 | 755.078 | 10.652 |
| ctx5000_b2_w128 | sid_gr_inference | success | 19.453 | 0.148 | 0.000 | 102.809 |
| ctx5000_b2_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b2_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b2_w128 | flash_rec | success | 213.387 | 0.632 | 996.907 | 9.373 |
| ctx5000_b2_w256 | sid_gr_inference | success | 31.737 | 0.129 | 0.000 | 63.018 |
| ctx5000_b2_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b2_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b2_w256 | flash_rec | success | 256.567 | 0.378 | 708.412 | 7.795 |
| ctx5000_b2_w64 | sid_gr_inference | success | 14.666 | 0.049 | 0.000 | 136.369 |
| ctx5000_b2_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b2_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b2_w64 | flash_rec | success | 179.976 | 0.198 | 1127.152 | 11.113 |
| ctx5000_b4_w128 | sid_gr_inference | success | 35.445 | 0.199 | 0.000 | 112.850 |
| ctx5000_b4_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b4_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b4_w128 | flash_rec | success | 432.661 | 1.289 | 1120.644 | 9.245 |
| ctx5000_b4_w256 | sid_gr_inference | success | 57.116 | 0.461 | 0.000 | 70.033 |
| ctx5000_b4_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b4_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b4_w256 | flash_rec | success | 529.208 | 2.657 | 826.549 | 7.558 |
| ctx5000_b4_w64 | sid_gr_inference | success | 24.479 | 0.195 | 0.000 | 163.407 |
| ctx5000_b4_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b4_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b4_w64 | flash_rec | success | 389.847 | 0.512 | 1492.591 | 10.260 |
| ctx5000_b8_w128 | sid_gr_inference | success | 64.281 | 0.361 | 0.000 | 124.453 |
| ctx5000_b8_w128 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b8_w128 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b8_w128 | flash_rec | success | 856.089 | 4.765 | 1231.789 | 9.345 |
| ctx5000_b8_w256 | sid_gr_inference | success | 108.794 | 0.403 | 0.000 | 73.533 |
| ctx5000_b8_w256 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b8_w256 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b8_w256 | flash_rec | success | 1062.190 | 8.838 | 876.331 | 7.532 |
| ctx5000_b8_w64 | sid_gr_inference | success | 45.904 | 0.471 | 0.000 | 174.275 |
| ctx5000_b8_w64 | vllm_gr | unsupported | — | — | — | — |
| ctx5000_b8_w64 | tensorrt_llm | unsupported | — | — | — | — |
| ctx5000_b8_w64 | flash_rec | success | 759.759 | 2.771 | 1555.091 | 10.530 |

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
| ctx1000_b1_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w128 | flash_rec | different | 100.000 | 91.406 | 91.406 | 14.062 | 0.393 |
| ctx1000_b1_w128 | tensorrt_llm | different | 100.000 | 92.188 | 92.188 | 13.281 | 0.268 |
| ctx1000_b1_w128 | vllm_gr | different | 100.000 | 92.969 | 92.969 | 17.969 | 0.33 |
| ctx1000_b1_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w256 | flash_rec | different | 100.000 | 92.188 | 92.188 | 5.469 | 0.339 |
| ctx1000_b1_w256 | tensorrt_llm | different | 100.000 | 92.578 | 92.578 | 5.859 | 0.485 |
| ctx1000_b1_w256 | vllm_gr | different | 100.000 | 94.141 | 94.141 | 8.984 | 0.424 |
| ctx1000_b1_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w64 | flash_rec | different | 100.000 | 90.625 | 90.625 | 21.875 | 0.246 |
| ctx1000_b1_w64 | tensorrt_llm | different | 100.000 | 89.062 | 89.062 | 21.875 | 0.384 |
| ctx1000_b1_w64 | vllm_gr | different | 100.000 | 96.875 | 96.875 | 28.125 | 0.312 |
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
| ctx5000_b1_w128 | flash_rec | different | 100.000 | 95.312 | 95.312 | 15.625 | 0.317 |
| ctx5000_b1_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b1_w128 | vllm_gr | different | 0.000 | 96.094 | 96.094 | 11.719 | 0.413 |
| ctx5000_b1_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | flash_rec | different | 100.000 | 95.703 | 95.703 | 6.250 | 0.261 |
| ctx5000_b1_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b1_w256 | vllm_gr | different | 100.000 | 93.750 | 93.750 | 7.422 | 0.384 |
| ctx5000_b1_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w64 | flash_rec | different | 100.000 | 87.500 | 87.500 | 20.312 | 0.205 |
| ctx5000_b1_w64 | tensorrt_llm | different | 100.000 | 90.625 | 90.625 | 15.625 | 0.173 |
| ctx5000_b1_w64 | vllm_gr | different | 100.000 | 96.875 | 96.875 | 39.062 | 0.2 |
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

### qwen3_1_7B

| Case | Candidate | Parity | Top-1 % | Mean overlap % | Min overlap % | Rank match % | Max score Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ctx1000_b1_w128 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w128 | flash_rec | different | 100.000 | 93.750 | 93.750 | 17.188 | 0.246 |
| ctx1000_b1_w128 | tensorrt_llm | different | 100.000 | 94.531 | 94.531 | 14.844 | 0.176 |
| ctx1000_b1_w128 | vllm_gr | different | 100.000 | 93.750 | 93.750 | 7.812 | 0.389 |
| ctx1000_b1_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w256 | flash_rec | different | 100.000 | 93.359 | 93.359 | 7.812 | 0.236 |
| ctx1000_b1_w256 | tensorrt_llm | different | 100.000 | 93.750 | 93.750 | 8.594 | 0.461 |
| ctx1000_b1_w256 | vllm_gr | different | 100.000 | 95.703 | 95.703 | 6.250 | 0.248 |
| ctx1000_b1_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx1000_b1_w64 | flash_rec | different | 100.000 | 92.188 | 92.188 | 28.125 | 0.238 |
| ctx1000_b1_w64 | tensorrt_llm | different | 100.000 | 93.750 | 93.750 | 26.562 | 0.183 |
| ctx1000_b1_w64 | vllm_gr | different | 100.000 | 93.750 | 93.750 | 37.500 | 0.32 |
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
| ctx5000_b1_w128 | flash_rec | different | 100.000 | 96.094 | 96.094 | 13.281 | 0.278 |
| ctx5000_b1_w128 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b1_w128 | vllm_gr | different | 100.000 | 95.312 | 95.312 | 17.969 | 0.198 |
| ctx5000_b1_w256 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w256 | flash_rec | different | 100.000 | 96.094 | 96.094 | 4.297 | 0.287 |
| ctx5000_b1_w256 | tensorrt_llm | unavailable | — | — | — | — | — |
| ctx5000_b1_w256 | vllm_gr | different | 100.000 | 95.312 | 95.312 | 8.984 | 0.225 |
| ctx5000_b1_w64 | sid_gr_inference | exact | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| ctx5000_b1_w64 | flash_rec | different | 100.000 | 93.750 | 93.750 | 32.812 | 0.297 |
| ctx5000_b1_w64 | tensorrt_llm | different | 0.000 | 96.875 | 96.875 | 14.062 | 0.352 |
| ctx5000_b1_w64 | vllm_gr | different | 100.000 | 95.312 | 95.312 | 28.125 | 0.206 |
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

No GPU capture saved yet. Run `bench.py diagnose` for a successful case.

## Run details

<details>
<summary>Configuration environment</summary>

| Configuration variable | Solution(s) | Value used | Reproduce from |
| --- | --- | --- | --- |
| `BENCHMARK_GPU` | sid_gr_inference, vllm_gr, tensorrt_llm, flash_rec | `GPU-44b1e15f-891e-a997-ea3a-1887655925aa` | — |
| `FLASH_REC_IMAGE` | flash_rec | `sid-gr-benchmark:flashrec-1089682` | `sha256:bb1c130f725018cb6e48b837be790183c5dfcc9f5c42e3d01a48d5b66178296c` · [Build recipe](../../entrypoints/flash_rec/Dockerfile) |
| `FLASH_REC_PYTHON` | flash_rec | `python3` | — |
| `FLASH_REC_RUNTIME` | flash_rec | `docker` | — |
| `FLASH_REC_SOURCE` | flash_rec | `../../../FlashRec` | [1089682a087a](https://github.com/sohu-mptc/FlashRec/tree/1089682a087ae5493f3b357ed1ec501cb684717b) |
| `GR_DECODE_ATTEN_ROOT` | sid_gr_inference | `../../corelib/gr_decode_atten` | [b0183412e7b4](https://github.com/lauthu/recsys-examples/tree/b0183412e7b46dfca44711835e4997300ce860cb/corelib/gr_decode_atten) |
| `GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_DISABLE_FLATTEN_LINEAR` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_DISABLE_PREFILL_CUDA_GRAPH` | sid_gr_inference | `0` | — |
| `GR_INFERENCE_GR_TRTLLM_KERNELS_JIT` | sid_gr_inference | `1` | — |
| `GR_INFERENCE_PREFILL_BACKEND` | sid_gr_inference | `auto` | — |
| `GR_INFERENCE_TRTLLM_PACKED_GEMM_SCOPE` | sid_gr_inference | `all` | — |
| `GR_INFERENCE_TRTLLM_QK_NORM_ROPE_PHASE` | sid_gr_inference | `all` | — |
| `LD_LIBRARY_PATH` | tensorrt_llm | `/usr/local/tensorrt/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:/opt/nvidia/nvda_nixl/lib64:/usr/local/ucx/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64` | — |
| `QWEN3_0_6B_MODEL` | sid_gr_inference, vllm_gr, tensorrt_llm, flash_rec | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/repo/models/Qwen3-0.6B` | [Qwen/Qwen3-0.6B@c1899de289a0](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) |
| `QWEN3_1_7B_MODEL` | sid_gr_inference, vllm_gr, tensorrt_llm, flash_rec | `/home/scratch.xinyliu_gpu/sid-gr-qwen06-run/repo/models/Qwen3-1.7B` | [Qwen/Qwen3-1.7B@70d244cc86cc](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) |
| `SID_GR_IMAGE` | sid_gr_inference | `sid-gr-benchmark:beam-af4e3f4-cu130` | `sha256:95a5ad9ce211cc8fafd4d45f28897e5a61fdc4aa280dc61c0a3e4feb7bbd1834` · [Build recipe](../../entrypoints/sglang_beam_search/Dockerfile) |
| `SID_GR_PYTHON` | sid_gr_inference | `python3` | — |
| `SID_GR_RUNTIME` | sid_gr_inference | `docker` | — |
| `SID_GR_SOURCE` | sid_gr_inference | `../sid-gr-inference` | [b0183412e7b4](https://github.com/lauthu/recsys-examples/tree/b0183412e7b46dfca44711835e4997300ce860cb/examples/sid-gr-inference) |
| `TENSORRT_LLM_IMAGE` | tensorrt_llm | `nvcr.io/nvidia/tensorrt-llm/release@sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9` | `sha256:de257efe2b6ee6804d5609402b68c790a7219b3fb7d1e87b843126c2bb63adb6` · `nvcr.io/nvidia/tensorrt-llm/release@sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9` |
| `TENSORRT_LLM_LD_LIBRARY_PATH` | tensorrt_llm | `/usr/local/tensorrt/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:/opt/nvidia/nvda_nixl/lib64:/usr/local/ucx/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64` | — |
| `TENSORRT_LLM_PYTHON` | tensorrt_llm | `python3` | — |
| `TENSORRT_LLM_RUNTIME` | tensorrt_llm | `docker` | — |
| `TENSORRT_LLM_SOURCE` | tensorrt_llm | (unset) | — |
| `VLLM_GR_IMAGE` | vllm_gr | `sid-gr-benchmark:vllm-gr-4c7cd23` | `sha256:8bdccfccc178ca4a478e470efee2f2fd12b131dfa0209ff9a158a435ad6f6f6d` · [Build recipe](../../entrypoints/vllm_gr/Dockerfile) |
| `VLLM_GR_PYTHON` | vllm_gr | `python3` | — |
| `VLLM_GR_RUNTIME` | vllm_gr | `docker` | — |
| `VLLM_GR_SOURCE` | vllm_gr | `../../../je` | [4c7cd23e93c3](https://github.com/vllm-project/je/tree/4c7cd23e93c34d855b8d7aa2b43598f0c37bed1d) |
| `VLLM_NO_USAGE_STATS` | vllm_gr | `1` | — |
| `VLLM_USE_V2_MODEL_RUNNER` | vllm_gr | `0` | — |
| `VLLM_WORKER_MULTIPROC_METHOD` | vllm_gr | `spawn` | — |

| Solution | Engine option | Value used |
| --- | --- | --- |
| sid_gr_inference | `prefix_cache` | `true` |
| vllm_gr | `beam_graph_enabled` | `true` |
| vllm_gr | `enable_chunked_prefill` | `false` |
| vllm_gr | `enforce_eager` | `false` |
| vllm_gr | `gpu_memory_utilization` | `0.7` |
| vllm_gr | `prefix_cache` | `true` |
| tensorrt_llm | `disable_overlap_scheduler` | `true` |
| tensorrt_llm | `enable_chunked_prefill` | `false` |
| tensorrt_llm | `enable_cuda_graph` | `true` |
| tensorrt_llm | `free_gpu_memory_fraction` | `0.7` |
| tensorrt_llm | `prefix_cache` | `true` |
| flash_rec | `enable_cuda_graph` | `true` |
| flash_rec | `enable_fused_expand` | `true` |
| flash_rec | `enable_graph_expand` | `true` |
| flash_rec | `enable_prefill_batch` | `true` |
| flash_rec | `flashinfer_variant` | `fa2` |
| flash_rec | `host_worker_threads` | `4` |
| flash_rec | `mem_fraction_static` | `0.7` |
| flash_rec | `prefix_cache` | `true` |

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
| sid_gr_inference | Engine source | [b0183412e7b4](https://github.com/lauthu/recsys-examples/tree/b0183412e7b46dfca44711835e4997300ce860cb/examples/sid-gr-inference) |
| sid_gr_inference | GR decode kernels | [b0183412e7b4](https://github.com/lauthu/recsys-examples/tree/b0183412e7b46dfca44711835e4997300ce860cb/corelib/gr_decode_atten) |
| sid_gr_inference | Container image | `sha256:95a5ad9ce211cc8fafd4d45f28897e5a61fdc4aa280dc61c0a3e4feb7bbd1834` · [Build recipe](../../entrypoints/sglang_beam_search/Dockerfile) |
| vllm_gr | Engine source | [4c7cd23e93c3](https://github.com/vllm-project/je/tree/4c7cd23e93c34d855b8d7aa2b43598f0c37bed1d) |
| vllm_gr | Container image | `sha256:8bdccfccc178ca4a478e470efee2f2fd12b131dfa0209ff9a158a435ad6f6f6d` · [Build recipe](../../entrypoints/vllm_gr/Dockerfile) |
| tensorrt_llm | Engine source | `2d96fda13ab0dcdc4d8279b895e9c25565510a1b7800e276b778e20a68436b54` |
| tensorrt_llm | Container image | `sha256:de257efe2b6ee6804d5609402b68c790a7219b3fb7d1e87b843126c2bb63adb6` · `nvcr.io/nvidia/tensorrt-llm/release@sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9` |
| flash_rec | Engine source | [1089682a087a](https://github.com/sohu-mptc/FlashRec/tree/1089682a087ae5493f3b357ed1ec501cb684717b) |
| flash_rec | Container image | `sha256:bb1c130f725018cb6e48b837be790183c5dfcc9f5c42e3d01a48d5b66178296c` · [Build recipe](../../entrypoints/flash_rec/Dockerfile) |

Docker rows identify the actual image; build recipes link the pinned public base and branch dependencies.

| Solution | Source commit/tree | Dirty | Key packages |
| --- | --- | --- | --- |
| sid_gr_inference | [b0183412e7b4](https://github.com/lauthu/recsys-examples/tree/b0183412e7b46dfca44711835e4997300ce860cb/examples/sid-gr-inference) | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |
| vllm_gr | [4c7cd23e93c3](https://github.com/vllm-project/je/tree/4c7cd23e93c34d855b8d7aa2b43598f0c37bed1d) | False | torch 2.11.0+cu130, triton 3.6.0 |
| tensorrt_llm | `2d96fda13ab0dcdc4d8279b895e9c25565510a1b7800e276b778e20a68436b54` | False | torch 2.10.0a0+b4e4ee81d3.nv25.12, triton 3.5.1 |
| flash_rec | [1089682a087a](https://github.com/sohu-mptc/FlashRec/tree/1089682a087ae5493f3b357ed1ec501cb684717b) | False | torch 2.11.0+cu130, triton 3.6.0 |

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
tar -xzf results/20261013T063653115757Z-b34f36/raw_artifacts.tar.gz -C results/20261013T063653115757Z-b34f36
python3 bench.py report --run results/20261013T063653115757Z-b34f36
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
