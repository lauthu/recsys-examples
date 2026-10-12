# qwen3_1_7B leaderboard

Latest non-smoke result for each case and candidate, grouped by scenario/checkpoint revision and GPU/CPU/driver configuration. Each completed job refreshes this view; unexecuted cases retain their last result. Failed and unsupported attempts replace older successes. Source runs preserve raw evidence and execution status.

Parity compares with the reference from the same source run; results from different runs are never used to recompute parity. The reference row checks repeat consistency. Requests/s is batch size divided by median latency.

## NVIDIA RTX PRO 6000 Blackwell Server Edition; 97887.0 MiB; driver 595.58.03; Intel(R) Xeon(R) Platinum 8480+; 2 sockets, 112 physical cores, 224 logical CPUs, 28 CPUs in process affinity

Scenario/checkpoint revision: `c12dfcf838df`.

| Case | Candidate | Status | Median ms | Requests/s | Parity | Top-1 exact | Min overlap | Max score Δ | Reference | Source run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_1_7B/ctx1000_b1_w256 | sid_gr_inference | success | 27.635 | 36.186 | exact | True | 1.000 | 0.000 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx1000_b1_w256 | sglang_beam_search | success | 41.869 | 23.884 | different | True | 0.953 | 0.218 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx1000_b2_w256 | sid_gr_inference | success | 48.414 | 41.310 | exact | True | 1.000 | 0.000 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx1000_b2_w256 | sglang_beam_search | success | 69.253 | 28.880 | different | True | 0.953 | 0.213 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx1000_b4_w256 | sid_gr_inference | success | 85.085 | 47.012 | exact | True | 1.000 | 0.000 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx1000_b4_w256 | sglang_beam_search | success | 118.139 | 33.859 | different | True | 0.941 | 0.246 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx1000_b8_w256 | sid_gr_inference | success | 158.833 | 50.367 | exact | True | 1.000 | 0.000 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx1000_b8_w256 | sglang_beam_search | success | 226.693 | 35.290 | different | True | 0.941 | 0.391 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx5000_b1_w256 | sid_gr_inference | success | 74.757 | 13.377 | exact | True | 1.000 | 0.000 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx5000_b1_w256 | sglang_beam_search | success | 126.296 | 7.918 | different | True | 0.969 | 0.390 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx5000_b2_w256 | sid_gr_inference | success | 139.439 | 14.343 | exact | True | 1.000 | 0.000 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx5000_b2_w256 | sglang_beam_search | success | 213.576 | 9.364 | different | True | 0.938 | 0.334 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx5000_b4_w256 | sid_gr_inference | success | 275.301 | 14.530 | exact | True | 1.000 | 0.000 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx5000_b4_w256 | sglang_beam_search | success | 416.612 | 9.601 | different | False | 0.949 | 0.274 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx5000_b8_w256 | sid_gr_inference | success | 546.016 | 14.652 | exact | True | 1.000 | 0.000 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
| qwen3_1_7B/ctx5000_b8_w256 | sglang_beam_search | success | 829.489 | 9.644 | different | True | 0.949 | 0.322 | sid_gr_inference | [20261012T043440525028Z-9b035a](../20261012T043440525028Z-9b035a/report.md) (completed) |
