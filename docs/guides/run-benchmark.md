# 单次测试操作手册

## 1. 固定实验

使用 [实验任务模板](../../.github/ISSUE_TEMPLATE/benchmark.md) 写明模型、协议、数据集、reference、Recipe、硬件、执行者和统计计划。根据 [Recipe 清单](../../recipes/README.md) 选择实际存在的执行配置。

在任务开始前分配唯一 run ID，例如 `20261023T120000Z-alice-5090a-flashrt-fp8-r1`。时间使用 UTC，同名不可覆盖，失败重试使用新 ID。已测历史目录不要复用。

## 2. 准备与预检

拉取约定 commit；检查仓库和引擎源码变更；核对环境、权重和 frozen dataset hash；确认所有目标 layout 可预热。使用原数据集时不要重新生成再冒充同一数据集。

在 5090-machine 上，现有 FlashRT 环境安装入口为 `bash scripts/install_flashrt.sh`；EmbodiInfer 为 `bash scripts/install_embodiinfer.sh`；PhyAI 为 `bash scripts/install_phyai.sh`。检查脚本适用平台后再运行，已安装且版本匹配时无须每次重装。

确认 GPU 无其他负载，约定 CPU threads、功耗/时钟策略和重复次数；不要与 kernel 编译并行正式测量。记录预检信息和偏离项。

## 3. 执行

以下为服务器已支持的命令。将示例 ID 换为自己的唯一 ID：

```bash
cd /data/inference-baseline
git pull --ff-only
bash scripts/run_5090.sh embodiinfer artifacts/UNIQUE_RUN_ID
# 或 flashrt、flashrt_nvfp4、phyai、reference，一次只运行一组。
```

默认 10 次预热、100 次计时。当前 Runner 验证 checkpoint/fixture、输出 shape/finite、A/B/A、图像/state/noise 响应、输入未被修改，计算误差并检查图状态。阈值 null 时只报告误差。失败会生成 `failure.json`；进程硬退出/OOM 等情况还需要保存日志和退出码，未必能生成 failure 文件。

编译/捕获不在正式计时区间，但当前 `setup_seconds` 主要覆盖 Adapter 构造，不包含随后全部 warmup，不能把它解释为完整冷启动耗时。

## 4. 导出与复核

```bash
/data/5090-machine/Isaac-GR00T/.venv/bin/python scripts/export_report.py \
  artifacts/UNIQUE_RUN_ID --output results/UNIQUE_RUN_ID
```

当前 exporter 使用 config.label（缺省 engine）命名文件，产生 `<label>.json`、`<label>-latency.jsonl` 和 comparison.md。每个 Run 用独立目录，同一 label 的不同重复运行不要导出到同一目录。失败运行应另存 failure.json/日志摘要及原因，不能伪造 measured summary。

在结果目录补充 `README.md`，使用 [Run 模板](../../results/run-template.md)：关联 Issue、各对象 ID、runner commit、具体命令、制品位置/hash、误差、限制和失败重试。现有 summary 已保存大量实际配置，但没有所有对象 ID 和完整环境锁，需补充。

复算 count/mean/分位数，检查单位、边界、样本数、版本、实际精度、图状态和输入哈希。当前 report 命令只核对 dataset manifest，不会自动审查硬件、参考精度或引擎计算变化。

## 5. 提交

提交小型 manifest、summary、逐请求延迟、环境锁/构建参数、说明；大型 tensor/权重/缓存留在外部并给出可访问方式及哈希。按 [协作手册](collaborate.md) 提交 PR，复现者应能从记录重建实验，而不是依赖聊天历史。
