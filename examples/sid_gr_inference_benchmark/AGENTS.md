# Benchmark suite conventions

- Keep the controller free of GPU framework imports; load frameworks inside workers.
- Keep workload semantics in scenarios and framework API details in entrypoints.
- Preserve raw samples, exact inputs, source identities, and failed/unsupported results.
- Never replace beam search with sampling or silently change precision/constraints.
- Keep profiled timings separate from unprofiled end-to-end results.
- Test configuration, output validation, timing boundaries, and failure handling with `python3 -m unittest discover -s tests`.
- Run a GPU smoke case after adapter changes when the appropriate environment is available.
