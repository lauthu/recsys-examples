# SID-GR diagnosis — $case_id

Source benchmark: $original

## GPU timeline

$timelines

$measured

Only kernels within complete `sid_bench/measured` NVTX ranges enter this summary.
Startup and warmup are excluded. Kernel durations can overlap; their sum is not
request latency. The GPU-active union counts overlapping intervals once.

Use these captures to locate expensive work, then rerun the unprofiled benchmark
to measure an optimization. Profiled timings do not enter the e2e report.

## Reproduce

$reproduce

Requests, commands, logs, native `.nsys-rep` files and exported SQLite traces are
saved alongside `diagnosis.json`.
