"""Archive a completed pi05 suite without weights or tensor fixtures."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    suite = json.loads((args.source / "suite.json").read_text())
    if suite["status"] != "finished":
        raise ValueError("Only a fully successful suite may be published")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.source / "suite.json", args.output / "suite.json")
    names = ["reference", "embodiinfer", "phyai", "flashrt-bf16", "flashrt-fp8", "flashrt-nvfp4"]
    rows = []
    for name in names:
        shutil.copytree(args.source / name, args.output / name)
        # Keep setup diagnostics (including calibration warnings) as tracked text.
        shutil.copy2(args.source / (name + ".log"), args.output / (name + "-run.txt"))
        s = json.loads((args.source / name / "summary.json").read_text())
        unique = {e["sample_id"]: e for e in s["correctness"]}.values()
        rows.append({"recipe": name, **s["latency"],
                     "mean_mae": sum(e["mae"] for e in unique) / len(unique),
                     "worst_max_abs": max(e["max_abs"] for e in unique),
                     "min_cosine": min(e["cosine"] for e in unique)})
    shutil.copy2("datasets/pi05-base-v3-v1/manifest.json", args.output / "dataset-manifest.json")
    for name in ("official-source.json", "verified-files.json"):
        shutil.copy2(Path("/data/huggingface/openpi/pi05_base") / name, args.output / name)
    shutil.copy2("/data/huggingface/openpi/pi05_base_pytorch_fp32/conversion-provenance.json",
                 args.output / "conversion-provenance.json")
    (args.output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# π0.5 / RTX 5090 实测（2026-10-23）", "",
        "官方 pi05_base；B1；3 路有效 RGB 224×224；token 容量 200；10 步 Euler；完整 normalized action [1,50,32]。",
        "同一 FP32 canonical checkpoint、fixture、初始噪声。计时从 GPU 预处理输入到 GPU action 完成，包含 embeddings、prefix、全部去噪、内部复制和同步，不含预处理、H2D/D2H、初始化、校准、capture。",
        "每项 10 次预热、100 次串行测量；4 个合成评估样本轮换，另有 1 个独立合成校准样本。误差相对冻结的官方 OpenPI PyTorch 输出，MAE 对 4 个唯一评估样本等权平均。", "",
        "| Recipe | Mean ms | P50 ms | P95 ms | MAE | 最大绝对误差 | 最低 cosine |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f'| [{r["recipe"]}]({r["recipe"]}/summary.json) | {r["mean_ms"]:.3f} | {r["p50_ms"]:.3f} | {r["p95_ms"]:.3f} | {r["mean_mae"]:.6f} | {r["worst_max_abs"]:.6f} | {r["min_cosine"]:.6f} |')
    lines += ["", "## 解读与边界", "",
        "未设置数值验收阈值，所有结果均为 measured / accuracy_gate=not_configured，不能宣称精度合格。合成输入上的误差不等于机器人任务成功率；本轮不接模拟器。",
        "EmbodiInfer 保留完整 vision/projector FP32；OpenPI 与 PhyAI 使用各自混合 BF16/FP32 路径；FlashRT BF16、FP8、NVFP4 也并非所有算子使用单一精度。差异属于 Recipe，需要同时查看延迟与误差。",
        "FlashRT 量化仅用 1 个合成样本校准，FP8 日志存在 scale outlier 警告；低精度结果不能外推到真实观测分布。",
        "A/B/A 重复输入、图像/状态 token/噪声敏感性、输入不变性、输出形状与设备检查均由 runner 执行；图缓存统计在计时前后保持一致。这些是执行契约检查，不是数值精度验收。",
        "GPU 未锁频；硬件/驱动/时钟前后快照见 summary。单次运行的分位数不代表跨运行置信区间。显存字段仅统计 PyTorch allocator，可能漏掉引擎直接 CUDA 分配。", "",
        "## 复现与追溯", "",
        "完整操作见 [运行指引](../../docs/guides/pi05-5090.md)。各目录包含 summary.json 和 100 行 latency.jsonl；*-run.txt 保留初始化和校准日志。summary 记录配置、源码提交、环境版本、checkpoint/fixture 哈希、运行时图状态；FlashRT 另记录实际 kernel 二进制 SHA256。",
        "official-source.json 固定下载来源和版本，verified-files.json 记录逐文件校验；conversion-provenance.json 记录官方转换来源。权重和 tensor fixtures 保留在服务器，不提交 Git。",
        "首轮失败原因及修复：PhyAI 直接调用 venv Python 时 ninja 不在 PATH（CLI 加入环境 bin），FlashInfer workspace 由 128 MiB 调整为 256 MiB；FlashRT 安装 ml-dtypes，并开启 π0.5 SKINNY/NVFP4 kernels 重新构建。最终正式测量在这些修复后顺序执行。", ""]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
