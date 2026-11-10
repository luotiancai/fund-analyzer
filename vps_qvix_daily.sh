#!/bin/bash
# QVIX 和上证日线每日跑批, 由 VPS 的 cron 调用。
#
# 为什么这活儿不在 GitHub Actions 上跑: 上交所的期权风险指标接口挡 runner
# 的地址段, 见 CLAUDE.md「为什么 qvix 不归云端跑批管」。换时间点重试解决
# 不了, 原来那个 qvix-retry.yml 已经删了。
#
# 只发布 qvix.db 和 market.db。净值等基金库在这台 VPS 上只有空表, 切勿用 --all 发布。
#
# 装在哪: /opt/fund-analyzer/{repo,venv,data}, cron 见本文件末尾注释。
set -euo pipefail

BASE=/opt/fund-analyzer
export FUND_ANALYZER_DATA=$BASE/data
PY=$BASE/venv/bin/python
cd "$BASE/repo"

exec 9>/run/fund-qvix-daily.lock
flock -w 300 9

echo "───────── $(date -u '+%F %T UTC') / $(TZ=Asia/Shanghai date '+%F %T CST') ─────────"

# 跟着仓库走, 免得代码改了这台还在跑旧的。--ff-only: 这台机器只消费不提交,
# 真出现分叉说明有人在这儿改过代码, 那该停下来看看而不是自动合并。
git pull -q --ff-only || echo "⚠️ git pull 失败, 用本地现有代码继续"

status=0
# QVIX 先跑, 实时口径收盘值必须在 09:15 集合竞价前补算。
if ! "$PY" sync_down.py qvix ||
   ! "$PY" update_daily.py --qvix-only ||
   ! "$PY" push_dbs.py qvix; then
    echo "⚠️ QVIX 更新或发布失败"
    status=1
fi

if ! "$PY" update_daily.py --market-only ||
   ! "$PY" push_dbs.py market; then
    echo "⚠️ 上证日线更新或发布失败"
    status=1
fi

"$BASE/repo/vps_site_build.sh" || { echo "⚠️ 静态页生成失败"; status=1; }
exit "$status"

# ── cron(root, 时区 UTC)────────────────────────────────────────────────────
#   0 23 * * 1-5  /opt/fund-analyzer/repo/vps_qvix_daily.sh >> /var/log/qvix-daily.log 2>&1
#
# 23:00 UTC = 北京次日 07:00, 周二至周六早上(对应周一至周五的收盘)。
# 上交所那份风险指标当晚就发布, 07:00 早就有了; 原来 GitHub 上那次补跑定在
# 06:23 也一直取得到。
#
# 跟云端跑批的时间**不需要错开**: 分库之后两边写的是不同文件, 谁先谁后都不
# 会盖掉对方 —— 这正是拆库要买的东西。
