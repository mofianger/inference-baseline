"""Adapt the existing reviewed atlas layout for the aux-only Nsys report."""
from pathlib import Path

run = Path(__file__).resolve().parents[1]
s = (run.parent/'flashrt-nvfp4-5090-20261030/harness/site.html').read_text(encoding='utf-8')
s = s.replace('<title>', '<title>Nsys · ')
s = s.replace('INFERENCE BASELINE · EXECUTION ATLAS', 'INFERENCE BASELINE · NSIGHT SYSTEMS')
s = s.replace('真实采集 · 独立静态 HTML', 'cuda-hw · aux → action')
a = s.index('<div class="notice">'); b = s.index('</div>', a)+6
s = s[:a]+'''<div class="notice"><b>计时范围：GPU 上已就绪的 aux、state 和初始噪声 → GPU action tensor。</b> 图像 embedding 已在采集前准备好。页面展示 FlashRT 自身执行的 ViT / DeepStack、LLM、VL 与动作生成；包含前端的精度转换和 GPU 内拷贝。模型图中金色部分属于计时范围外。</div>'''+s[b:]
s = s.replace('外部输入准备 · 官方 BF16', '计时外 · GPU 输入已就绪').replace('① 完整视觉主输出', '① aux 视觉输入').replace('data-model="support"', 'data-excluded="support" disabled')
s = s.replace('官方 ViT24 + Merger</text>', 'aux 中的视觉 embedding</text>').replace('最终视觉 embedding</text>', '已准备好 · 计时外</text>')
s = s.replace('<option value="integrated">完整调用 · 含官方视觉准备</option>', '')
s = s.replace("'Official support':'#bb975e'", "'GPU copy':'#bb975e'")
s = s.replace('30 次同步 wall time 记录实际调用延迟；另用 torch.profiler 采集 10 次请求', '50 次同步 wall time 在独立进程中记录实际调用延迟；另用 Nsight Systems 2025.3.2 的 cuda-hw,nvtx 模式采集 10 次请求')
s = s.replace('本页没有 NVBit 插桩，也没有使用旧的 SM–CTA 数据。', '采用 CUDA Graph 节点级追踪；PyTorch Profiler 在本次运行中关闭。')
s = s.replace('源码与张量说明来自单独的 eager 映射采集，其耗时不用于报告。', '源码与张量说明复用已审计映射，并逐项核对新 Trace 的完整名称、顺序、Grid 和 Block；时间戳全部来自本次 Nsys。')
s = s.replace('5 个样本逐一比较', '本次检查当前 evaluation 样本')
s = s.replace('全部 1,575 个计算节点通过名称、顺序、Grid 与 Block 关联校验，10 次 replay 的 Kernel 序列一致。', '1,578 个 Kernel（含 3 个精度转换）和 8 个 Graph 拷贝逐项核对，10 次 replay 的执行序列一致。')
s = s.replace("['完整调用 · 同步 wall',fmt(S.metadata.timings.integrated.median_ms)+' ms','含官方 BF16 视觉输入准备']", "['采集方式','Nsys cuda-hw','Graph node · 10 次完整请求']")
s = s.replace("fmt(N.length,0),'1,575 个计算节点已关联源码'", "fmt(S.kernel_count,0),'另有 8 个 Graph 内拷贝节点'")
s = s.replace("n.grid.join('×')", "(n.grid.length?n.grid.join('×'):'—')").replace("n.block.join('×')", "(n.block.length?n.block.join('×'):'—')")
s = s.replace('纵向分行只为展示重叠，不代表不同 stream 或 SM。', '纵向分行只为展示记录区间的交叠，不代表不同 stream 或 SM；亚微秒交叠不能直接判断为有效计算并行。')
s = s.replace('点击条带查看输入、输出及融合计算。', '点击条带查看输入、输出及融合计算。节点编号沿用前版，8 个 Graph Memcpy 单独标明。')
s = s.replace("n.group+' · #'+n.i+' · stream '+n.stream", "n.group+' · #'+n.i+' · '+n.event_type+' · stream '+n.stream")
s = s.replace('checkpoint_protocol:S.metadata.protocol,', 'profiler:S.metadata.profiler,checkpoint_protocol:S.metadata.protocol,')
s = s.replace("for(let i=0;i<=5;i++){x.fillStyle='#65756c';", "for(const cp of D.boundary_copies){x.fillStyle='#bb975e';x.fillRect(cp.s/total*w,60,Math.max(2,cp.d/total*w),8)}for(let i=0;i<=5;i++){x.fillStyle='#65756c';")
s = s.replace('上方 wall time 来自单独的未开启 profiler 测量。', '上方 wall time 来自单独的未开启 profiler 测量。下方金色短线为 5 次边界 D2D 拷贝，均在计时范围内。')
marker = '<section class="card"><div class="section-no">04 / 测量与可复现性</div>'
s = s.replace(marker, '''<section class="card"><div class="section-no">04 / 时间边界对照</div><h2>Gate 与 Up：两种采集方式的结果</h2><p>比较 LLM 第 1 层的 #211 Gate 与 #212 Up。两者在同一 stream 顺序提交。正数表示记录区间交叠，负数表示间隔。</p><div id="overlap-check"></div><p class="note">cuda-hw：10 次中 1 次交叠 0.096 μs；常规 cuda：10 次中 9 次交叠。结果支持采集方式会影响微小边界，但还不能把任意交叠都解释为计时误差或 PDL 并行。保留原始时间戳，未做裁剪或平移。</p></section>'''+marker.replace('04 /', '05 /'))
insert = '''const comp=S.comparison;$('overlap-check').innerHTML='<div class="table-wrap"><table><thead><tr><th>回放</th><th>cuda-hw 交叠 μs</th><th>常规 cuda 交叠 μs</th></tr></thead><tbody>'+comp.cuda_hw.gate_up_signed_overlap_us.map((v,i)=>`<tr><td>${i+1}</td><td>${fmt(v,3)}</td><td>${fmt(comp.cuda_sw.gate_up_signed_overlap_us[i],3)}</td></tr>`).join('')+'</tbody></table></div>';
'''
s = s.replace('const err=S.metadata.reference_error;', insert+'const err=S.metadata.reference_error;')
s = s.replace('全部 ${N.length} 个 Kernel', '全部 ${N.length} 个节点（含 8 次 Graph Memcpy）')
s = s.replace("$('count').textContent=list.length+' 个节点'", "$('count').textContent=list.length+' 个节点'")
s = s.replace('<a href="REPORT.md">采集说明</a>', '<a href="REPORT.md">采集说明</a> · <a href="reports/frontend-hw.nsys-rep">Nsys 原始报告（本地附件）</a>')
assert 'timings.integrated' not in s
assert 'torch.profiler 采集' not in s
(run/'harness/site.html').write_text(s,encoding='utf-8')
