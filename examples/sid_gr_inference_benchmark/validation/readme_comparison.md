# README comparison validation

Run `20261012T043440525028Z-9b035a`, measured with suite commit `949f2bbdd258e1e2e287ed5179f74a246ef25f2f`.
All eight README cases completed for both engines: Qwen3-1.7B BF16, contexts
1000/5000, batches 1/2/4/8, beam width 256 and three output tokens. Every job used
five warmups and thirty measurements. All measured outputs passed structural checks.
Cross-engine exact parity was not achieved: every case has beam/score differences,
and one case also has a top-1 difference.

Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 595.58.03.
CPU: Intel(R) Xeon(R) Platinum 8480+, 2 sockets,
112 physical cores and 224 logical CPUs.
Both engines were restricted to the controller's 28-CPU affinity set.
Both used image `sha256:e95aae5f9b3fdeab685d7045b559bd59a1bf971195f6a943a33dc6160adfb78a` and PyTorch
`2.9.1+cu130`. SGLang source:
`af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d` (`feature/beam_search`).

| Context | Batch | SID-GR ms | SGLang ms | SGLang/SID-GR | SID-GR req/s | SGLang req/s | Top-1 exact | Min overlap % | Max score delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1 | 27.635 | 41.869 | 1.52 | 36.19 | 23.88 | True | 95.31 | 0.2175 |
| 1000 | 2 | 48.414 | 69.253 | 1.43 | 41.31 | 28.88 | True | 95.31 | 0.2128 |
| 1000 | 4 | 85.085 | 118.139 | 1.39 | 47.01 | 33.86 | True | 94.14 | 0.2459 |
| 1000 | 8 | 158.833 | 226.693 | 1.43 | 50.37 | 35.29 | True | 94.14 | 0.3908 |
| 5000 | 1 | 74.757 | 126.296 | 1.69 | 13.38 | 7.92 | True | 96.88 | 0.3896 |
| 5000 | 2 | 139.439 | 213.576 | 1.53 | 14.34 | 9.36 | True | 93.75 | 0.3341 |
| 5000 | 4 | 275.301 | 416.612 | 1.51 | 14.53 | 9.60 | False | 94.92 | 0.2738 |
| 5000 | 8 | 546.016 | 829.489 | 1.52 | 14.65 | 9.64 | True | 94.92 | 0.3221 |

At context 5000 / batch 4, `request-0` differs in all thirty repetitions.
SID-GR ranks `[220, 17, 15]` first (score -8.464215) and `[17, 15, 17]` second
(-8.475200). SGLang ranks `[17, 15, 17]` first (-8.452927) and `[220, 17, 15]`
second (-8.505817). The saved JSON includes this evidence under `top1_mismatches`.
The cause of the output difference has not been isolated.

Latency is median completed-batch wall time; throughput is batch size divided by
that latency. Agreement covers every measured repetition against the reference's
first output. Shared-beam score differences use cumulative log-probabilities.
Exact parity additionally requires ordered beams and scores to match exactly;
consult the per-job `parity` values in the [validation data](readme_comparison.json).

The original README used H100; this is a workflow validation on RTX PRO 6000
Blackwell, not reproduction of its absolute latencies. GPU clocks were not locked.
Inputs use the new scenario's seeded generator and common timing boundary.
These synthetic inputs do not establish recommendation quality.

Reproduce with the shared image and environment in the [benchmark README](../README.md#reproduce-the-sid-gr-readme-comparison):

```bash
python3 bench.py e2e --config run.readme.yml --case 'qwen3_1_7B/*_w256'
```

The saved invocation used a copy of this template with `continue_on_error: false`.
The generated [run report](../results/20261012T043440525028Z-9b035a/report.md) and
[scenario leaderboard](../results/leaderboards/qwen3_1_7B.md) are committed alongside
the manifest and summary. Raw inputs, outputs, logs and frozen requests are in the
[artifact archive](../results/20261012T043440525028Z-9b035a/raw_artifacts.tar.gz);
the [example instructions](../README.md#completed-example) show how to extract it
and regenerate the reports without a GPU. The checked-in validation JSON includes
all latency samples, agreement metrics, output checksums, model/workload checksums
and source/image/hardware identities.
