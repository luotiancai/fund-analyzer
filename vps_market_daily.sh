#!/bin/bash
# root cron (UTC)，北京时间周一至周五 18:10:
# 10 10 * * 1-5 /opt/fund-analyzer/repo/vps_market_daily.sh >> /var/log/fund-market-daily.log 2>&1
set -euo pipefail

BASE=/opt/fund-analyzer
export FUND_ANALYZER_DATA=$BASE/data
PY=$BASE/venv/bin/python
cd "$BASE/repo"

exec 9>/run/fund-market-daily.lock
flock -w 300 9

echo "───────── $(TZ=Asia/Shanghai date '+%F %T CST') ─────────"
git pull -q --ff-only || echo "⚠️ git pull 失败, 用本地现有代码继续"

"$PY" update_daily.py --market-only
"$BASE/repo/vps_site_build.sh" || echo "⚠️ 静态页生成失败, 等下次定时重建"
"$PY" push_dbs.py market
