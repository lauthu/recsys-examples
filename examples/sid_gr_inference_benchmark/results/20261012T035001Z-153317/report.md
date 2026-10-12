# SID-GR benchmark

Run `20261012T035001Z-153317`

[Latency](#latency-comparison) · [Parity](#output-parity) · [Details](#run-details) · [Artifacts](#artifacts)

| Status | Jobs finished | Successful | Other outcomes | Mode |
| --- | --- | --- | --- | --- |
| completed | 1/1 | 1 | 0 | smoke |

> **Smoke run:** small-sample validation only; excluded from scenario rankings.

| Exact | Different | Top-1 mismatches | Unavailable |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 |

**Reference:** `sid_gr_inference` · Offline batches on one GPU

| Hardware | Recorded configuration |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory / driver | 97887.0 MiB / 595.58.03 |
| CPU | unrecorded |

## Latency comparison

Median batch **ms**; lower is better. **Bold** = lowest recorded median. — = no result.

### qwen3_0_6B

| Context | Batch | Beam | sid_gr_inference |
| ---: | ---: | ---: | ---: |
| 1000 | 1 | 64 | **15.334** |



<details>
<summary>All measurements</summary>

### qwen3_0_6B

| Case | Candidate | Status | Median ms | IQR ms | Δ latency % | Requests/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ctx1000_b1_w64 | sid_gr_inference | success | 15.334 | 0.058 | 0.000 | 65.216 |

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

### qwen3_0_6B

| Case | Candidate | Parity | Top-1 | Overlap % | Max score Δ |
| --- | --- | --- | --- | ---: | ---: |
| ctx1000_b1_w64 | sid_gr_inference | exact | Match | 100.000 | 0 |

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
| qwen3_0_6B | 2 | 3 |

| Solution | Source commit/tree | Dirty | Key packages |
| --- | --- | --- | --- |
| sid_gr_inference | db75723cc18d2bda | False | torch 2.12.0a0+5aff3928d8.nv26.5.50603568, triton 3.7.0+gitb4e20bb.nv26.5 |

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
tar -xzf results/20261012T035001Z-153317/raw_artifacts.tar.gz -C results/20261012T035001Z-153317
python3 bench.py report --run results/20261012T035001Z-153317
```

| Saved data | Location inside the run |
| --- | --- |
| Frozen configuration and hardware | `manifest.json` |
| Derived measurements | `summary.json` |
| Exact inputs | `workloads/` |
| Requests, commands, logs, samples and beams | `jobs/<candidate>/<scenario>/<case>/` |

| Saved GPU diagnosis |
| --- |
| [20261012T035247Z-f176c8](diagnostics/20261012T035247Z-f176c8/diagnosis.md) |

**Scenario leaderboards**

Smoke runs do not update scenario leaderboards.
