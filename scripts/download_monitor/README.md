# 本地下载监控

在仓库根目录运行，无第三方 Python 依赖：

```bash
python scripts/download_monitor/server.py --port 8765 --interval 30
```

打开 <http://localhost:8765>。服务仅监听 `127.0.0.1`，只监控 π0.5 官方下载，不控制下载、权重转换或 benchmark。

结构：HTTP 主进程展示页面，独立 worker 子进程每 30 秒通过 `ssh -o ClearAllForwardings=yes 5090-machine python3 -` 执行只读采集，再向本地 `/api/update` POST JSON。上报使用随机进程级令牌；浏览器只读取 `/api/status`。无需在远端写入脚本或打开端口。需本机已配置 SSH alias 和免交互认证。

下载量按官方清单逐文件计算：完整文件计一次，未完成文件的 `.parts` 与 `.partial` 取较大值，不把临时副本重复相加。部分重叠时该值是保守估计。校验量基于 downloader 校验通过后才生成的最终文件；只有完整 `verified-files.json` 与清单对应且文件齐全时标记完成。采集器不会重复哈希整个模型。

速度是本次监控最近最多 30 次成功采样的有效字节增长率，不是网络链路吞吐；ETA 只估计剩余下载，不包括最终校验、转换或测试。连接异常保留最近数据并显示警告；超过 90 秒无更新显示过期状态。完成后 worker 退出，网页保留最终结果。

前台运行时 Ctrl+C 结束服务。后台运行时终止启动日志中列出的 server PID；worker 检测到本地服务不可达后自行退出。运行日志保存在 `artifacts/download-monitor.stdout.log` 与 `artifacts/download-monitor.stderr.log`（若使用重定向启动）。
