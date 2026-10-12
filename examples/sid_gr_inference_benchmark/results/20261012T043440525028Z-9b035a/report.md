# SID-GR benchmark — 20261012T043440525028Z-9b035a

16/16 jobs finished; 16 successful. Run status: completed. Mode: e2e.

**Environment:** NVIDIA RTX PRO 6000 Blackwell Server Edition; 97887.0 MiB; driver 595.58.03; Intel(R) Xeon(R) Platinum 8480+; 2 sockets, 112 physical cores, 224 logical CPUs, 28 CPUs in process affinity\
**Baseline:** sid_gr_inference\
**Measurement:** Offline batches on one GPU; protocol is recorded per scenario below.

## Scenario leaderboards

- [qwen3_1_7B](../leaderboards/qwen3_1_7B.md)

| Scenario | Warmups | Measurements |
| --- | --- | --- |
| qwen3_1_7B | 5 | 30 |

## End-to-end results

| Case | Solution | Status | Median ms | IQR ms | Δ latency % | Requests/s |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3_1_7B/ctx1000_b1_w256 | sid_gr_inference | success | 27.635 | 0.148 | 0.000 | 36.186 |
| qwen3_1_7B/ctx1000_b1_w256 | sglang_beam_search | success | 41.869 | 0.264 | 51.510 | 23.884 |
| qwen3_1_7B/ctx1000_b2_w256 | sglang_beam_search | success | 69.253 | 0.275 | 43.042 | 28.880 |
| qwen3_1_7B/ctx1000_b2_w256 | sid_gr_inference | success | 48.414 | 0.236 | 0.000 | 41.310 |
| qwen3_1_7B/ctx1000_b4_w256 | sid_gr_inference | success | 85.085 | 0.143 | 0.000 | 47.012 |
| qwen3_1_7B/ctx1000_b4_w256 | sglang_beam_search | success | 118.139 | 0.707 | 38.848 | 33.859 |
| qwen3_1_7B/ctx1000_b8_w256 | sglang_beam_search | success | 226.693 | 1.916 | 42.724 | 35.290 |
| qwen3_1_7B/ctx1000_b8_w256 | sid_gr_inference | success | 158.833 | 0.654 | 0.000 | 50.367 |
| qwen3_1_7B/ctx5000_b1_w256 | sid_gr_inference | success | 74.757 | 0.076 | 0.000 | 13.377 |
| qwen3_1_7B/ctx5000_b1_w256 | sglang_beam_search | success | 126.296 | 1.241 | 68.944 | 7.918 |
| qwen3_1_7B/ctx5000_b2_w256 | sglang_beam_search | success | 213.576 | 0.384 | 53.169 | 9.364 |
| qwen3_1_7B/ctx5000_b2_w256 | sid_gr_inference | success | 139.439 | 0.118 | 0.000 | 14.343 |
| qwen3_1_7B/ctx5000_b4_w256 | sid_gr_inference | success | 275.301 | 0.273 | 0.000 | 14.530 |
| qwen3_1_7B/ctx5000_b4_w256 | sglang_beam_search | success | 416.612 | 2.050 | 51.329 | 9.601 |
| qwen3_1_7B/ctx5000_b8_w256 | sglang_beam_search | success | 829.489 | 4.843 | 51.917 | 9.644 |
| qwen3_1_7B/ctx5000_b8_w256 | sid_gr_inference | success | 546.016 | 1.958 | 0.000 | 14.652 |

Latency is milliseconds per completed batch. Negative change means faster. IQR describes sample variation; this offline batch statistic is not online request-tail latency. Timings include input submission, engine execution and conversion to completed CPU beam IDs/scores. Setup, validation and report serialization are excluded.

## Output parity

| Case | Solution | Validity | Top-1 exact | Min overlap % | Max score Δ |
| --- | --- | --- | --- | --- | --- |
| qwen3_1_7B/ctx1000_b1_w256 | sid_gr_inference | structural pass | True | 100.000 | 0 |
| qwen3_1_7B/ctx1000_b1_w256 | sglang_beam_search | structural pass | True | 95.312 | 0.218 |
| qwen3_1_7B/ctx1000_b2_w256 | sglang_beam_search | structural pass | True | 95.312 | 0.213 |
| qwen3_1_7B/ctx1000_b2_w256 | sid_gr_inference | structural pass | True | 100.000 | 0 |
| qwen3_1_7B/ctx1000_b4_w256 | sid_gr_inference | structural pass | True | 100.000 | 0 |
| qwen3_1_7B/ctx1000_b4_w256 | sglang_beam_search | structural pass | True | 94.141 | 0.246 |
| qwen3_1_7B/ctx1000_b8_w256 | sglang_beam_search | structural pass | True | 94.141 | 0.391 |
| qwen3_1_7B/ctx1000_b8_w256 | sid_gr_inference | structural pass | True | 100.000 | 0 |
| qwen3_1_7B/ctx5000_b1_w256 | sid_gr_inference | structural pass | True | 100.000 | 0 |
| qwen3_1_7B/ctx5000_b1_w256 | sglang_beam_search | structural pass | True | 96.875 | 0.39 |
| qwen3_1_7B/ctx5000_b2_w256 | sglang_beam_search | structural pass | True | 93.750 | 0.334 |
| qwen3_1_7B/ctx5000_b2_w256 | sid_gr_inference | structural pass | True | 100.000 | 0 |
| qwen3_1_7B/ctx5000_b4_w256 | sid_gr_inference | structural pass | True | 100.000 | 0 |
| qwen3_1_7B/ctx5000_b4_w256 | sglang_beam_search | structural pass | False | 94.922 | 0.274 |
| qwen3_1_7B/ctx5000_b8_w256 | sglang_beam_search | structural pass | True | 94.922 | 0.322 |
| qwen3_1_7B/ctx5000_b8_w256 | sid_gr_inference | structural pass | True | 100.000 | 0 |

Structural validity and agreement with a reference are separate checks. Synthetic inputs do not establish recommendation quality. Score differences are reported only for shared beam sequences.

## Speed-of-light estimate

| Case | Solution | Floor ms | Floor/measured % | Gap × | Coverage |
| --- | --- | --- | --- | --- | --- |
| qwen3_1_7B/ctx1000_b1_w256 | sid_gr_inference | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx1000_b1_w256 | sglang_beam_search | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx1000_b2_w256 | sglang_beam_search | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx1000_b2_w256 | sid_gr_inference | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx1000_b4_w256 | sid_gr_inference | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx1000_b4_w256 | sglang_beam_search | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx1000_b8_w256 | sglang_beam_search | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx1000_b8_w256 | sid_gr_inference | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx5000_b1_w256 | sid_gr_inference | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx5000_b1_w256 | sglang_beam_search | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx5000_b2_w256 | sglang_beam_search | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx5000_b2_w256 | sid_gr_inference | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx5000_b4_w256 | sid_gr_inference | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx5000_b4_w256 | sglang_beam_search | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx5000_b8_w256 | sglang_beam_search | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |
| qwen3_1_7B/ctx5000_b8_w256 | sid_gr_inference | — | — | — | No sourced dense BF16 ceiling in the selected hardware profile |

The current estimator supplies a partial matrix-compute floor, not a complete model of the physical latency limit. Memory traffic, reductions and host overhead remain unmodeled. A gap to this floor is not a promised speedup.

## Versions and limitations

| Solution | Source commit/tree | Dirty | Key packages |
| --- | --- | --- | --- |
| sid_gr_inference | 949f2bbdd258e1e2 | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |
| sglang_beam_search | af4e3f42a5a1a79f | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |

- Frameworks run serially, with case order rotating the solution order; clocks are not locked by this suite.
- GPU memory is unavailable until a comparable process-tree collector is configured.
- Historical reports use a different timing boundary and are not merged into this leaderboard.
- Structural validity does not imply exact beam agreement.
- Small sample counts are smoke evidence, not a regression conclusion.

## Reproduce

Use the frozen `jobs/*/*/*/request.json` and `command.json` for exact execution settings.

```bash
python bench.py report --run results/20261012T043440525028Z-9b035a
python bench.py diagnose --run results/20261012T043440525028Z-9b035a --case <case-id>
```
The manifest records checkpoint hashes, source changes, image IDs and shared input checksums.

Raw artifacts: [manifest](manifest.json), [comparison data](summary.json). Executed jobs contain their frozen requests, commands and worker logs; successful jobs also retain latency samples and measured beam outputs. Profiling results are separate from these unprofiled headline timings.
