# 基金分析仪

基金查询使用 FastAPI + 原生网页; 上证行情及策略复盘使用静态网页。

FOF 基金按类型及名称识别，不参与筛选、选基、回测及净值完整性检查；每日跑批不再维护其净值和规模。
定开（含“定期开放”）基金与持有期基金统一排除，不参与筛选、选基、回测及对应的净值完整性检查。

筛选范围内 C 份额的实际成立日保存在 `rank.fund_inception`，来自东财 `ESTABDATE`，附带来源及抓取时间。每日跑批只补缺失日期，手动补全可选择 Action 的 `inception-only` 模式。今日选基先按决策日排除成立未满 6 个自然月的 C 份额，再检查参选基金近 3 月净值是否完整。

## 代码入口

- `web/`、`web_service.py`: 基金分析网站及 HTTP 接口。
- `web_tasks.py`、`web_cache.py`: 查询计算和持久缓存。
- `web_sync.py`、`web_prepare.py`: 同步只读数据快照、预热查询。
- `fetcher.py`、`update_daily.py`: 数据采集和每日跑批。
- `today_pick.py`、`backtest_qvix.py`、`pct_rank.py`: 候选诊断、策略回测和分位补算。
- `build_site.py`、`sse_view.py`、`site/`: 上证静态页。
- `qvix_core.py`、`qvix_calc.py`、`qvix_now.py`: QVIX 计算及本地实时查询。
- `backfill_qvix_history.py`: QVIX 历史回填工具。
- `sync_down.py`、`push_dbs.py`、`cloud_peek.py`、`cloud_assets.py`: 数据同步、发布和检查工具。
- `deploy/`、`deploy_web.py`: 基金网站部署; 详见 [部署说明](deploy/README.md)。
- `.github/workflows/`、`vps_*.sh`: 云端每日跑批、VPS QVIX 跑批和静态页生成。
- `tests/`: 查询服务和数据指标检查。

## 本地使用

完整开发及跑批依赖:

```bash
python3 -m pip install -r requirements.txt
```

仅运行基金网站时安装 `requirements-web.txt`, 不需要 AKShare。

已有快照时启动网站:

```bash
FUND_WEB_ROOT=/path/to/fund-web uvicorn web_service:app --host 127.0.0.1 --port 8010
```

快照目录结构见部署说明。本地回测前同步数据:

```bash
python3 sync_down.py --all
python3 backtest_qvix.py
```

本地实时 QVIX 可双击 `qvix.command`(Windows 使用 `qvix.bat`), 或执行
`python3 qvix_now.py`。

运行现有检查(需额外安装 `httpx`):

```bash
python3 -m unittest discover -s tests -v
```
