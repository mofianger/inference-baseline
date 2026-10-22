# Model-only benchmark

Accuracy gates are not implied by latency measurements.

| Engine | Mean ms | p50 ms | p95 ms | p99 ms | Max abs error | Gate |
|---|---:|---:|---:|---:|---:|---|
| reference | 83.068 | 82.964 | 85.449 | 86.550 | 0.000000 | not_configured |
| flashrt | 16.541 | 16.540 | 16.571 | 16.587 | 1.296875 | not_configured |
| embodiinfer | 70.439 | 66.626 | 102.992 | 106.242 | 0.126953 | not_configured |
| phyai | 26.773 | 26.753 | 26.934 | 27.068 | 0.566406 | not_configured |
