基金分析仪 VPS 版
================

入口为 `http://192.243.114.13/fund/`，已有上证页仍在根目录。Nginx
转发到 127.0.0.1:8010；443 由现有代理使用。

数据归属
--------

- GitHub Actions 的 `update-daily.yml` 继续刷新净值、指标、行情和季度规模。
- 搬瓦工原有 QVIX cron 继续独占产出 `qvix.db`，原有上证页 cron 保留。
- `fund-web-sync.timer` 每 15 分钟检查 Release，仅下载变更的文件。校验
  SQLite 完整性、净值/指标版本及下载前后资产版本后，原子切换整个快照。
- API 和计算任务只读快照；持仓和 ETF 映射使用跨版本保留的独立缓存，绝不推回 Release。
- 同步成功后预热当天候选及全市场的诊断行。查代码按索引读缓存，绕过计算队列。
  上游延迟时页面显示数据日期，数据不足时拒绝计算。

目录与服务
----------

`/opt/fund-web/releases/<version>` 存放代码；`current` 指向当前版本。
`/opt/fund-web/venv` 是独立环境。`/var/lib/fund-web/snapshots` 保存最近
数据代次，`current` 原子切换；`cache` 由专用的 `fund-web` 用户写入。

主进程不加载 pandas。分析和持仓请求各有一个子进程，慢数据源不阻塞分析队列。
按代码版本、数据代次及参数缓存结果。候选计算每批读取 100 只基金。
分析队列上限 8、持仓队列上限 4，单任务 240 秒，服务内存上限 448MB、CPU 上限一个核。
持仓默认只加载最新年度，历史年份由页面按钮按需加载。查询缓存上限
约 128MB，旧结果保留至多 7 天；快照至少留 4 代，便于回退。

部署
----

执行 `python3 deploy_web.py`，默认部署搬瓦工；打包仅包含代码和网页，
自动排除 macOS 元数据，不上传配置密钥或本地数据库。每次生成独立 release。
安装脚本保留 nginx 备份，先生成当天缓存、健康检查再开放入口。它不改原有跑批环境和 cron。

检查：

```bash
systemctl status fund-web fund-web-sync.timer
journalctl -u fund-web -u fund-web-sync -n 80 --no-pager
curl http://127.0.0.1:8010/api/v1/health
curl http://127.0.0.1:8010/api/v1/status
```

升级后若需回退代码，将 `/opt/fund-web/current` 指回上一 release，再
`systemctl restart fund-web`。数据回退则把 `/var/lib/fund-web/current`
指回保留的快照；API 下一次请求即使用该代数据。两者均不影响原有数据生产者。

本地验证：`FUND_WEB_ROOT=<测试数据目录> FUND_WEB_CACHE=<缓存目录>`，使用
`uvicorn web_service:app --host 127.0.0.1 --port 8010` 启动。测试命令为
`python -m unittest discover -s tests -v`。
