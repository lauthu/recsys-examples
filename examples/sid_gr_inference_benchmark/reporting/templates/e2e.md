# SID-GR benchmark — $run_id

$summary

**Environment:** $hardware\
**Baseline:** $baseline\
**Measurement:** $measurement

## Scenario leaderboards

$leaderboards

$protocols

## End-to-end results

$results

Latency is milliseconds per completed batch. Negative change means faster. IQR describes sample variation; this offline batch statistic is not online request-tail latency. Timings include input submission, engine execution and conversion to completed CPU beam IDs/scores. Setup, validation and report serialization are excluded.

## Output parity

$validation

Structural validity and agreement with a reference are separate checks. Synthetic inputs do not establish recommendation quality. Score differences are reported only for shared beam sequences.

## Speed-of-light estimate

$sol

The current estimator supplies a partial matrix-compute floor, not a complete model of the physical latency limit. Memory traffic, reductions and host overhead remain unmodeled. A gap to this floor is not a promised speedup.

## Versions and limitations

$versions

$limitations

## Reproduce

$reproduce

Raw artifacts: [manifest](manifest.json), [comparison data](summary.json). Executed jobs contain their frozen requests, commands and worker logs; successful jobs also retain latency samples and measured beam outputs. Profiling results are separate from these unprofiled headline timings.
