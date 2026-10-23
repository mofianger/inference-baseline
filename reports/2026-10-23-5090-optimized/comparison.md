# Model-only benchmark

Accuracy gates are not implied by latency measurements.

| Engine | Mean ms | p50 ms | p95 ms | p99 ms | Max abs error | Gate |
|---|---:|---:|---:|---:|---:|---|
| embodiinfer_optimized | 28.482 | 28.485 | 28.520 | 28.535 | 0.160156 | not_configured |
| flashrt_fp8 | 14.418 | 14.421 | 14.442 | 14.449 | 1.300781 | not_configured |
| flashrt_nvfp4 | 8.974 | 8.976 | 9.014 | 9.039 | 1.343750 | not_configured |
