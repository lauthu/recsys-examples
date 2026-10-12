# SID-GR benchmark

Run `20261012T043440525028Z-9b035a`

[Latency](#latency-comparison) · [Parity](#output-parity) · [Details](#run-details) · [Artifacts](#artifacts)

| Status | Jobs finished | Successful | Other outcomes | Mode |
| --- | --- | --- | --- | --- |
| completed | 16/16 | 16 | 0 | e2e |

| Exact | Different | Top-1 mismatches | Unavailable |
| ---: | ---: | ---: | ---: |
| 8 | 8 | 1 | 0 |

| Top-1 mismatch case | Candidate |
| --- | --- |
| qwen3_1_7B/ctx5000_b4_w256 | sglang_beam_search |

**Reference:** `sid_gr_inference` · Offline batches on one GPU

| Hardware | Recorded configuration |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory / driver | 97887.0 MiB / 595.58.03 |
| CPU | Intel(R) Xeon(R) Platinum 8480+ |
| CPU topology / affinity | 2 sockets · 112 physical cores · 224 logical CPUs · 28 allowed CPUs |

## Latency comparison

Median batch **ms**; lower is better. **Bold** = lowest recorded median. — = no result.

### qwen3_1_7B

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
<summary>All measurements</summary>

### qwen3_1_7B

| Case | Candidate | Status | Median ms | IQR ms | Δ latency % | Requests/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ctx1000_b1_w256 | sid_gr_inference | success | 27.635 | 0.148 | 0.000 | 36.186 |
| ctx1000_b1_w256 | sglang_beam_search | success | 41.869 | 0.264 | 51.510 | 23.884 |
| ctx1000_b2_w256 | sid_gr_inference | success | 48.414 | 0.236 | 0.000 | 41.310 |
| ctx1000_b2_w256 | sglang_beam_search | success | 69.253 | 0.275 | 43.042 | 28.880 |
| ctx1000_b4_w256 | sid_gr_inference | success | 85.085 | 0.143 | 0.000 | 47.012 |
| ctx1000_b4_w256 | sglang_beam_search | success | 118.139 | 0.707 | 38.848 | 33.859 |
| ctx1000_b8_w256 | sid_gr_inference | success | 158.833 | 0.654 | 0.000 | 50.367 |
| ctx1000_b8_w256 | sglang_beam_search | success | 226.693 | 1.916 | 42.724 | 35.290 |
| ctx5000_b1_w256 | sid_gr_inference | success | 74.757 | 0.076 | 0.000 | 13.377 |
| ctx5000_b1_w256 | sglang_beam_search | success | 126.296 | 1.241 | 68.944 | 7.918 |
| ctx5000_b2_w256 | sid_gr_inference | success | 139.439 | 0.118 | 0.000 | 14.343 |
| ctx5000_b2_w256 | sglang_beam_search | success | 213.576 | 0.384 | 53.169 | 9.364 |
| ctx5000_b4_w256 | sid_gr_inference | success | 275.301 | 0.273 | 0.000 | 14.530 |
| ctx5000_b4_w256 | sglang_beam_search | success | 416.612 | 2.050 | 51.329 | 9.601 |
| ctx5000_b8_w256 | sid_gr_inference | success | 546.016 | 1.958 | 0.000 | 14.652 |
| ctx5000_b8_w256 | sglang_beam_search | success | 829.489 | 4.843 | 51.917 | 9.644 |

| Metric | Meaning |
| --- | --- |
| Δ latency % | Relative to reference; negative = faster |
| IQR ms | Sample variation; not online tail latency |
| Requests/s | Batch size / median batch latency |
| Bold median | Lowest observed value; no significance or parity claim |

</details>

## Output parity

<details>
<summary>All parity checks</summary>

### qwen3_1_7B

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

| Check | Meaning |
| --- | --- |
| Success | Structural checks passed |
| Exact | Ordered beams and scores match the first same-run reference output |
| Top-1 | Leading sequence matches; scores may differ |
| Overlap % | Minimum shared-beam percentage across measured outputs |
| Max score Δ | Largest score difference among shared sequences |
| Reference row | Repeat consistency |

</details>

## Run details

<details>
<summary>Protocol and source versions</summary>

| Scenario | Warmups | Measurements |
| --- | --- | --- |
| qwen3_1_7B | 5 | 30 |

| Solution | Source commit/tree | Dirty | Key packages |
| --- | --- | --- | --- |
| sid_gr_inference | 949f2bbdd258e1e2 | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |
| sglang_beam_search | af4e3f42a5a1a79f | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |

</details>

<details>
<summary>Speed-of-light estimate</summary>

| Estimate | Reason |
| --- | --- |
| Unavailable | No sourced dense BF16 ceiling in the selected hardware profile |

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
tar -xzf results/20261012T043440525028Z-9b035a/raw_artifacts.tar.gz -C results/20261012T043440525028Z-9b035a
python3 bench.py report --run results/20261012T043440525028Z-9b035a
```

| Saved data | Location inside the run |
| --- | --- |
| Frozen configuration and hardware | `manifest.json` |
| Derived measurements | `summary.json` |
| Exact inputs | `workloads/` |
| Requests, commands, logs, samples and beams | `jobs/<candidate>/<scenario>/<case>/` |



**Scenario leaderboards**

- [qwen3_1_7B](../leaderboards/qwen3_1_7B.md)
