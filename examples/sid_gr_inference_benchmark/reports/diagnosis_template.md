# SID-GR diagnosis — $case_id

Source benchmark: $original

## Execution timelines

$timelines

GPU-duration sums can overlap and do not sum to request latency. Kernel names are reported as captured; no heuristic stage attribution is treated as authoritative. Only kernels contained in the complete measured-request NVTX ranges enter the summary. Startup and warmup are excluded.

## Optional operator comparisons

$kernels

Isolated eager and CUDA-graph timings have separate labels. Repeated inputs are cache-warm screening evidence, not a prediction of full-model speedup. Counter captures cover selected invocations and can perturb cache state and execution.

## Speed-of-light context

$sol

## Interpretation

**Measured:** $measured

**Inferred:** Compare the dominant work and recorded versions before assigning a cause. This report does not automatically claim causality from kernel timing differences.

**Confirmed by A/B:** Not established by this capture. Change one factor and rerun the unprofiled end-to-end case to confirm its effect.

## Reproduce and artifacts

$reproduce
