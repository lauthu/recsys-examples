# SID-GR benchmark

Run `$run_id`

[Latency](#latency-comparison) · [Output parity](#output-parity) · [Run details](#run-details) · [Reproduce](#reproduce)

$summary

$attention

**Reference:** `$baseline` · Offline batches on one GPU

$hardware

## Latency comparison

Median batch latency in **milliseconds**; lower is better. **Bold** marks the lowest recorded median in each row, without implying statistical significance or output equivalence. A dash means no recorded result.

$results

$failures

<details>
<summary>All measurements: latency, variation and throughput</summary>

$measurements

Δ latency compares each candidate with the reference; negative means faster. IQR describes sample variation, not online request-tail latency. Requests/s is batch size divided by median batch latency.

</details>

## Output parity

Successful jobs pass structural checks. Exact parity additionally requires ordered beams and scores to match the reference. Top-1 agreement alone does not establish exact parity or recommendation quality.

<details>
<summary>All parity checks, including reference repeat consistency</summary>

$validation

Overlap is the minimum shared-beam percentage across measured outputs. Score differences use only shared sequences. Each result is checked against the first measured reference output from the same run; the reference's own row checks repeat consistency.

</details>

## Run details

<details>
<summary>Measurement protocol and source versions</summary>

$protocols

The timer includes input submission, engine execution and conversion to completed CPU beam IDs/scores. Setup, warmup, validation and serialization are excluded.

$versions

</details>

<details>
<summary>Speed-of-light estimate</summary>

$sol

This is a partial matrix-compute floor. Memory traffic, reductions and host overhead remain unmodeled. A gap to this floor is not a promised speedup.

</details>

<details>
<summary>Limitations</summary>

$limitations

</details>

## Reproduce

$artifacts

$reproduce

Executed jobs retain frozen requests, commands and worker logs; successful jobs also retain raw latency samples and measured beam outputs. Profiling artifacts remain separate from unprofiled headline timings.

**Scenario leaderboards**

$leaderboards
