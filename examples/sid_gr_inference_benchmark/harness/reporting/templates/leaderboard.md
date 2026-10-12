# $scenario leaderboard

Compare each solution's latest latency on the same workload and hardware.

Median batch **ms**; lower is better. The **Fastest** column names the winner for each workload.

Parity counts are workload cases; each reference row checks repeat consistency.

$sections

<details>
<summary>Ranking rules</summary>

| Rule | Behavior |
| --- | --- |
| Latest result | Each finished job replaces its case/candidate row |
| Partial run | Unexecuted rows keep their last result |
| Failed / unsupported | Replaces earlier success |
| Grouping | Scenario, checkpoint, GPU, CPU and driver |
| Source runs | Recorded per row; latency cells may come from different runs |
| Parity | Same-run reference only; never recomputed across runs |
| Reference parity | Repeat consistency |
| Requests/s | Batch size / median batch latency |
| Bold median | No significance or output-equivalence claim |
| Excluded | Short validation runs (`--smoke`) and profiled timings |

</details>

Generated from the [leaderboard template](../harness/reporting/templates/leaderboard.md).
