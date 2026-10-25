"""Normalize the collaborator HTML; optional dependency: beautifulsoup4.

Research records are not executable recipes. Preserve source metric wording
and record IDs; never infer missing batch/checkpoint/timing information.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    "llama": "GGML/GGUF 驱动的本地 LLM 推理框架。",
    "vllm": "以请求调度和 KV 管理为核心的 LLM 服务引擎。",
    "trtllm": "NVIDIA GPU 上的 LLM 优化与部署引擎。",
    "flashrt": "面向固定模型和硬件，以融合低精度 kernel 和 CUDA Graph 执行推理。",
    "rtvla": "以固定形状 Triton 算子和 CUDA Graph 加速 VLA。",
    "vlacpp": "将 VLA 封装为 GGUF 并通过 C++/GGML 执行。",
    "embodied": "以 C++/GGML 执行具身模型，并桥接 llama.cpp 骨干。",
    "phyai": "通过模型 Runner、图捕获和 kernel policy 提供低延迟具身推理。",
    "apx": "将准备阶段与执行阶段分离的推理栈；APXinf-robo 为相关封装。",
    "embodiinfer": "结合原生模型、编译、CUDA Graph 与去噪状态管理的具身引擎。",
}
MODELS = {"q": "Qwen3-8B", "p": "pi0.5", "g": "GR00T N1.7"}


def txt(node):
    return node.get_text(" ", strip=True)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def repo(url):
    u = urlparse(url)
    return "/".join(u.path.strip("/").split("/")[:2]).lower() if u.netloc == "github.com" else u.netloc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=ROOT / "AI高速推理框架性能总表_V3.html")
    p.add_argument("--output", type=Path, default=ROOT / "reports/2026-10-25-v3-baseline")
    args = p.parse_args()
    raw = args.source.read_bytes()
    soup = BeautifulSoup(raw.decode("utf-8"), "html.parser")
    engines = {}
    for tr in soup.select("tr[data-method-framework]"):
        a = tr.select_one("a.framework-name")
        key = tr["data-method-framework"]
        engines[key] = {"name": txt(a), "repository": a["href"],
            "description": DESCRIPTIONS[key], "evidence_status": "collaborator_reported_capabilities",
            "methods": [{"name": txt(x.select_one('.method-name')),
                         "description": txt(x.select_one('.method-explanation')),
                         "source": x.select_one('.method-name')["href"]}
                        for x in tr.select('.method-item')]}
    records = []
    for tr in soup.select("tr[data-record]"):
        td = tr.select("td")
        if len(td) != 5:
            raise ValueError("Unexpected table layout")
        source = tr.select_one("a.source")["href"]
        context = tr.select_one('.context')
        context_text = " ".join(t.strip() for t in context.find_all(string=True)
                                if t.parent.name != 'a' and t.strip())
        rid, engine = tr["data-record"], tr["data-framework"]
        records.append({"id": rid, "status": "reported", "platform": tr["data-gpu"],
            "model": MODELS[tr["data-model"]], "engine": engine,
            "precision_label": txt(td[2]), "source_profile": tr["data-profile"],
            "recipe_candidate_id": f"v3-{engine}-{tr['data-gpu'].lower().replace(' ', '-')}-{rid.lower()}",
            "recipe_status": "reported_variant_requires_protocol_and_config",
            "metric_original": txt(tr.select_one('.metric')), "conditions_original": context_text,
            "source_url": source, "source_html_record": rid,
            "source_html_classification": "official" if 'official' in td[4].get('class', []) else "external",
            "publisher": repo(source), "publisher_is_engine_repository": repo(source) == repo(engines[engine]['repository']),
            "executable_config": None, "checkpoint_revision": None,
            "protocol_id": None, "dataset_manifest_sha256": None,
            "comparable_to_repository_protocol": False})
    assert len({r['id'] for r in records}) == len(records)
    for key, e in engines.items():
        rows = [r for r in records if r['engine'] == key]
        e['models_with_records'] = sorted({r['model'] for r in rows})
        e['platforms_with_records'] = sorted({r['platform'] for r in rows})
        e['precision_labels_with_records'] = sorted({r['precision_label'] for r in rows})
    measured = []
    paths = [ROOT / 'reports/2026-10-23-5090-optimized' / name for name in
             ('flashrt_fp8.json', 'flashrt_nvfp4.json', 'embodiinfer_optimized.json')]
    paths += [ROOT / 'reports/2026-10-23-5090-smoke' / name for name in ('phyai.json', 'reference.json')]
    paths += sorted((ROOT / 'reports/2026-10-23-5090-pi05').glob('*/summary.json'))
    for path in paths:
        s = json.loads(path.read_text(encoding='utf-8'))
        unique = {r['sample_id']: r for r in s['correctness']}.values()
        measured.append({'id': path.relative_to(ROOT).as_posix(), 'status': s['status'],
            'platform': 'RTX 5090', 'model': s['protocol'].get('model', 'GR00T N1.7'),
            'label': s['config'].get('label', path.stem), 'engine': s['config']['engine'],
            'protocol': s['protocol'], 'latency': s['latency'],
            'mean_mae': sum(r['mae'] for r in unique) / len(unique),
            'max_abs': max(r['max_abs'] for r in unique), 'accuracy_gate': s['accuracy_gate'],
            'dataset_manifest_sha256': s['dataset_manifest_sha256'],
            'checkpoint': s['checkpoint'], 'engine_source': s['engine_source']})
    data = {'schema_version': 1, 'document_only': True, 'organized_date': '2026-10-25',
        'source': {'path': args.source.name, 'sha256': hashlib.sha256(raw).hexdigest(),
                   'declared_check_date': '2026-10-23'},
        'scope': {'platforms_with_records': sorted({r['platform'] for r in records}),
                  'models_with_records': sorted({r['model'] for r in records}),
                  'mentioned_without_records': ['Llama'],
                  'historical_requested_platform_without_v3_records': ['Thor']},
        'engines': engines, 'reported_records': records, 'repository_measurements': measured}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'baseline-data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# V3 推理引擎 Baseline 总表', '',
        '整理日期：2026-10-25；原 HTML 自述核查日期：2026-10-23。本次是资料整理和现有结果归档，没有启动新的 GPU 测试。', '',
        '**优先入口：[RTX 5090 × GR00T N1.7：FP8 / NVFP4 Baseline](5090-groot-n17.md)。**', '',
        '来源：[协作者 HTML](../../'+args.source.name+')；结构化数据：[baseline-data.json](baseline-data.json)。保留全部原始记录 ID、指标、条件和链接；源文件 SHA256：`'+data['source']['sha256']+'`。', '',
        '## 按仓库约定解释', '',
        'Setting 用平台 × 模型 × 精度族筛选；真正可比较的 Baseline 还必须固定 checkpoint、协议、测试集和工作量。Recipe 描述分模块精度、执行、校准及硬件约束，Run 才是一次测量。FP8 与 NVFP4 是两个精度族，不代表整个模型所有算子都使用该精度。', '',
        '`reported` 是协作者汇编的外部数字；`measured` 是本仓库有原始 Run 的测量；缺失的组合保持待调研。HTML 的 profile 是筛选标签，不能当作可运行 Recipe。本次每个外部记录保留独立 recipe_candidate_id，避免把 eager/graph/fusion 或不同测量边界误合并；它们不是 41 个已验证配方。', '',
        '原 HTML 的 official 标色不一定代表被测引擎自己发布：例如 PhyAI/vla.cpp/Embodied.cpp 的部分记录来自 EmbodiInfer benchmark。JSON 同时保留原标色、发布者及是否与引擎仓库一致；其余外部行未逐一重新核查上游，不能称为本次验证。', '',
        '## 平台与模型', '',
        '| 维度 | 整理结果 |', '|---|---|',
        '| 性能表平台 | RTX 4090、RTX 5090；本仓库已有 Run 为 RTX 5090 |',
        '| 有性能记录的模型 | Qwen3-8B、π0.5、GR00T N1.7；具体 checkpoint 不能仅凭模型名称判定相同 |',
        '| 仅规划提及 | Llama：无型号/性能行；Thor：历史研究关注平台，本 V3 无性能行 |',
        '| 精度 | BF16/FP32 混合、FP8、INT8 W8A8、AWQ W4A16、GGUF Q8_0、NVFP4 混合；INT8/Q8_0 不归为 FP8，AWQ 不归为 NVFP4 |', '',
        '## 引擎与 Recipe 范围', '',
        '以下模型和平台列仅表示 V3 有相应记录，不能扩展为框架全部支持范围。APXInf/APXinf-robo 按原表作为一个家族，封装与底层仍需分别固定版本。', '',
        '| 引擎 | 一句话描述 | 有记录模型 | 有记录平台 | 有记录精度 / Recipe 线索 |', '|---|---|---|---|---|']
    for e in engines.values():
        vals = [f"[{e['name']}]({e['repository']})", e['description'], '、'.join(e['models_with_records']),
                '、'.join(e['platforms_with_records']), '；'.join(e['precision_labels_with_records'])]
        lines.append('| '+' | '.join(map(cell, vals))+' |')
    lines += ['', '各框架的方法名称与解释、逐项来源完整保存在 JSON 的 engines.methods。方法清单只是能力线索，不表示每个测速配置都启用了这些技术。', '',
        '## 外部 Baseline：完整 41 条记录', '',
        '保留原指标：LLM 的吞吐/TTFT、具身模型的完整调用、GPU 图重放不能混排。并发 16 和 1 QPS 也不自动等于单请求 B1。空缺条件不补猜；不得据本表直接计算跨行加速比。', '',
        '| ID | 平台 | 模型 | 引擎 | 精度 / Recipe 变体 | 原始指标 | 输入 / 边界 / 统计条件 |', '|---|---|---|---|---|---|---|']
    for r in records:
        vals = [f"[{r['id']}]({r['source_url']})", r['platform'], r['model'], engines[r['engine']]['name'],
                r['precision_label'], r['metric_original'], r['conditions_original']]
        lines.append('| '+' | '.join(map(cell, vals))+' |')
    lines += ['', '## 本仓库实测 Baseline', '',
        '以下 11 条均来自 2026-10-23 已有 Run，不是本次重跑。只在相同模型协议与资产下比较；GR00T 的 reference/PhyAI 来自首轮，其他 GR00T 路径来自优化轮。旧 FlashRT 和旧 EmbodiInfer 不再作为当前 Baseline，历史记录仍保留。', '',
        '| 模型 | 引擎 / 路径 | Mean ms | P50 ms | P95 ms | 平均 MAE | 原始 Run |', '|---|---|---:|---:|---:|---:|---|']
    for m in measured:
        l = m['latency']
        vals = [m['model'], m['label'], f"{l['mean_ms']:.3f}", f"{l['p50_ms']:.3f}", f"{l['p95_ms']:.3f}",
                f"{m['mean_mae']:.6f}", f"[summary](../../{m['id']})"]
        lines.append('| '+' | '.join(map(cell, vals))+' |')
    lines += ['', '全部 accuracy_gate=not_configured；误差不代表任务成功率。π0.5 的详细协议见 [运行指引](../../docs/guides/pi05-5090.md)，GR00T 重点分组见 [专项 Baseline](5090-groot-n17.md)。', '',
        '## 维护与接入', '',
        '候选配方先补齐 checkpoint/revision、硬件 kernel 支持、分模块 W/A/KV 精度、校准集、步数/H/A、计时端点与缓存规则，再新增 recipes 文档、configs 和 Adapter；实际运行完成后才标 measured。无法映射当前模型协议的候选应建立新协议，不能覆盖已有结果。', '',
        '重新导入（只生成本总表和 JSON，不覆盖专项人工审查）：`python scripts/import_baseline_v3.py`。依赖 beautifulsoup4，仅用于资料导入，不加入 GPU 引擎环境。', '']
    (args.output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'engines': len(engines), 'reported_records': len(records), 'repository_measurements': len(measured)}))


if __name__ == '__main__':
    main()
