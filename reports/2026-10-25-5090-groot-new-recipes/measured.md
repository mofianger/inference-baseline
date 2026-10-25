# 本轮有效测量

全部采用相同三视角评估 fixture；数值门槛未设置。误差为所有校验样本的最大绝对误差。

| 配置 / Recipe | mean (ms) | p95 (ms) | max abs error | 原始记录 |
|---|---:|---:|---:|---|
| `groot-trt-fp8` | 14.946 | 14.987 | 0.364136 | [summary](runs/20261025-groot-trt-fp8-r3/run/summary.json) |
| `groot-trt-nvfp4` | 16.048 | 16.158 | 0.323730 | [summary](runs/20261025-groot-trt-nvfp4-r3/run/summary.json) |
| `groot-trt-optimized` | 22.403 | 22.523 | 0.072266 | [summary](runs/20261025-groot-trt-optimized-r1/summary.json) |
| `foldquant-smoothquant` | 26.428 | 26.550 | 0.432617 | [summary](runs/20261025-foldquant-smoothquant-r4/run/summary.json) |
| `phyai-llm-int4-a16` | 26.956 | 27.139 | 1.148438 | [summary](runs/20261025-phyai-new-recipes-r2/phyai-llm-int4-a16/summary.json) |
| `phyai-llm-nvfp4-a16` | 26.979 | 27.246 | 0.681641 | [summary](runs/20261025-phyai-new-recipes-r2/phyai-llm-nvfp4-a16/summary.json) |
| `phyai-llm-fp8-a16` | 26.990 | 27.209 | 0.760742 | [summary](runs/20261025-phyai-new-recipes-r2/phyai-llm-fp8-a16/summary.json) |
| `groot-trt-baseline` | 26.995 | 27.403 | 0.128906 | [summary](runs/20261025-groot-trt-baseline-r1/run/summary.json) |
| `foldquant-w8a8` | 29.941 | 30.111 | 0.128906 | [summary](runs/20261025-foldquant-w8a8-r2/run/summary.json) |
| `groot-trt-compile` | 46.034 | 46.897 | 0.031250 | [summary](runs/20261025-groot-trt-compile-r1/summary.json) |
| `foldquant-awq` | 57.523 | 57.950 | 1.064453 | [summary](runs/20261025-foldquant-awq-r3/run/summary.json) |
| `phyai-llm-fp8-a8-eager` | 334.574 | 347.139 | 0.868164 | [summary](runs/20261025-phyai-new-recipes-eager-r1/phyai-llm-fp8-a8-eager/summary.json) |

候选 ID 的覆盖状态与每次尝试见 [candidate-coverage.json](candidate-coverage.json)；包括重试的全部记录见 [run-index.json](run-index.json)。
