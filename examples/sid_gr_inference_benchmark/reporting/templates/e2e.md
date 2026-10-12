# SID-GR benchmark

Run `$run_id`

[Latency](#latency-comparison) · [Parity](#output-parity) · [Details](#run-details) · [Artifacts](#artifacts)

$summary

$attention

**Reference:** `$baseline` · Offline batches on one GPU

$hardware

## Latency comparison

Median batch **ms**; lower is better. **Bold** = lowest recorded median. — = no result.

$results

$failures

<details>
<summary>All measurements</summary>

$measurements

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

$validation

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

$protocols

$versions

</details>

<details>
<summary>Speed-of-light estimate</summary>

$sol

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

$artifacts

```bash
${extract}python3 bench.py report --run $directory
```

| Saved data | Location inside the run |
| --- | --- |
| Frozen configuration and hardware | `manifest.json` |
| Derived measurements | `summary.json` |
| Exact inputs | `workloads/` |
| Requests, commands, logs, samples and beams | `jobs/<candidate>/<scenario>/<case>/` |

$diagnostics

**Scenario leaderboards**

$leaderboards
