# SID-GR benchmark

Run `$run_id` · $run_profile

[Latency](#latency-comparison) · [Parity](#output-parity) · [GPU diagnosis](#gpu-diagnosis) · [Environment](#run-details) · [Artifacts](#artifacts)

$summary

Each solution runs every workload combination, with the warmups and repeats listed below.

Parity counts below include finished workload cases only; the reference row checks repeat consistency.

$attention

**Reference:** `$baseline` · Offline batches on one GPU

$hardware

## Latency comparison

Median batch **ms**; lower is better. **Bold** = lowest recorded median. Pending = no finished result. — = unavailable.

$results

$failures

<details>
<summary>All measurements</summary>

$measurements

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

$validation

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

$diagnostics

## Run details

<details>
<summary>Configuration environment</summary>

$environment

$options

</details>

<details>
<summary>Protocol and source versions</summary>

$protocols

$sources

Docker rows identify the actual image; build recipes link the pinned public base and branch dependencies.

$versions

</details>

<details>
<summary>Optional compute-floor estimate</summary>

$sol

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

**Scenario leaderboards**

$leaderboards
