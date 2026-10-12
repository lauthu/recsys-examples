# SID-GR benchmark

Run `20261012T043440525028Z-9b035a`

[Latency](#latency-comparison) · [Output parity](#output-parity) · [Run details](#run-details) · [Reproduce](#reproduce)

| Status | Jobs finished | Successful | Other outcomes | Mode |
| --- | --- | --- | --- | --- |
| completed | 16/16 | 16 | 0 | e2e |

> **Results with output differences: 8.** **Top-1 mismatches: 1.** Compared with the same-run reference.
> - `qwen3_1_7B/ctx5000_b4_w256` · `sglang_beam_search`

**Reference:** `sid_gr_inference` · Offline batches on one GPU

| Hardware | Recorded configuration |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory / driver | 97887.0 MiB / 595.58.03 |
| CPU | Intel(R) Xeon(R) Platinum 8480+ |
| CPU topology / affinity | 2 sockets · 112 physical cores · 224 logical CPUs · 28 allowed CPUs |

## Latency comparison

Median batch latency in **milliseconds**; lower is better. **Bold** marks the lowest recorded median in each row, without implying statistical significance or output equivalence. A dash means no recorded result.

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
<summary>All measurements: latency, variation and throughput</summary>

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

Δ latency compares each candidate with the reference; negative means faster. IQR describes sample variation, not online request-tail latency. Requests/s is batch size divided by median batch latency.

</details>

## Output parity

Successful jobs pass structural checks. Exact parity additionally requires ordered beams and scores to match the reference. Top-1 agreement alone does not establish exact parity or recommendation quality.

<details>
<summary>All parity checks, including reference repeat consistency</summary>

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

Overlap is the minimum shared-beam percentage across measured outputs. Score differences use only shared sequences. Each result is checked against the first measured reference output from the same run; the reference's own row checks repeat consistency.

</details>

## Run details

<details>
<summary>Measurement protocol and source versions</summary>

| Scenario | Warmups | Measurements |
| --- | --- | --- |
| qwen3_1_7B | 5 | 30 |

The timer includes input submission, engine execution and conversion to completed CPU beam IDs/scores. Setup, warmup, validation and serialization are excluded.

| Solution | Source commit/tree | Dirty | Key packages |
| --- | --- | --- | --- |
| sid_gr_inference | 949f2bbdd258e1e2 | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |
| sglang_beam_search | af4e3f42a5a1a79f | False | sglang 0.0.0.dev1+ge6072935b, torch 2.9.1+cu130, triton 3.5.1 |

</details>

<details>
<summary>Speed-of-light estimate</summary>

- No sourced dense BF16 ceiling in the selected hardware profile.

This is a partial matrix-compute floor. Memory traffic, reductions and host overhead remain unmodeled. A gap to this floor is not a promised speedup.

</details>

<details>
<summary>Limitations</summary>

- Frameworks run serially, with case order rotating the solution order; clocks are not locked by this suite.
- GPU memory is unavailable until a comparable process-tree collector is configured.
- Historical reports use a different timing boundary and are not merged into this leaderboard.
- Structural validity does not imply exact beam agreement.
- Small sample counts are smoke evidence, not a regression conclusion.

</details>

## Reproduce

[Raw artifact archive](raw_artifacts.tar.gz) (extract into this run directory before regenerating) · [Manifest](manifest.json) · [Summary JSON](summary.json)

Use the frozen `jobs/*/*/*/request.json` and `command.json` for exact execution settings.

```bash
python bench.py report --run results/20261012T043440525028Z-9b035a
python bench.py diagnose --run results/20261012T043440525028Z-9b035a --case <case-id>
```
The manifest records checkpoint hashes, source changes, image IDs and shared input checksums.

Executed jobs retain frozen requests, commands and worker logs; successful jobs also retain raw latency samples and measured beam outputs. Profiling artifacts remain separate from unprofiled headline timings.

**Scenario leaderboards**

- [qwen3_1_7B](../leaderboards/qwen3_1_7B.md)
