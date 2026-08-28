#!/bin/bash
# QVIX 每日跑批 —— 跑在国内那台 VPS 上, 由 cron 调。
#
# 为什么这活儿不在 GitHub Actions 上跑: 上交所的期权风险指标接口挡 runner
# 的地址段, 见 CLAUDE.md「为什么 qvix 不归云端跑批管」。换时间点重试解决
# 不了, 原来那个 qvix-retry.yml 已经删了。
#
# ⚠️ 只碰 qvix.db 一个库(约 124KB)。**别在这台机器上跑 push_dbs.py --all**:
# 数据目录里其余几个库都是 fetcher.init_db() 顺手建的空壳(几 KB), 推上去
# 就是拿空库盖掉线上几十 MB 的真数据。push_dbs 的行数检查会拦下来, 但别去
# 试那道防线 —— 它是最后一道, 不是第一道。
#
# 装在哪: /opt/fund-analyzer/{repo,venv,data}, cron 见本文件末尾注释。
set -euo pipefail

BASE=/opt/fund-analyzer
export FUND_ANALYZER_DATA=$BASE/data
PY=$BASE/venv/bin/python
cd "$BASE/repo"

echo "───────── $(date -u '+%F %T UTC') / $(TZ=Asia/Shanghai date '+%F %T CST') ─────────"

# 跟着仓库走, 免得代码改了这台还在跑旧的。--ff-only: 这台机器只消费不提交,
# 真出现分叉说明有人在这儿改过代码, 那该停下来看看而不是自动合并。
git pull -q --ff-only || echo "⚠️ git pull 失败, 用本地现有代码继续"

# 拉最新的 qvix.db。条件下载, 远端没变就零流量。
$PY sync_down.py qvix

# 算"上一个交易日"的 QVIX, 顺带补最近15天的空值(接口抽风留下的洞)。
# 幂等: INSERT OR REPLACE, 重复跑是同一个数。
$PY update_daily.py --qvix-only

# 推回去。库才 124KB, 不值得为"这次到底补上没有"去解析日志 —— 无条件推,
# push_dbs 自己会做推前行数检查和上传后回读验证(2026-08-11 加的, 那次
# gh upload 返回 0 但资产根本没变)。
$PY push_dbs.py qvix

# ── cron(root, 时区 UTC)────────────────────────────────────────────────────
#   0 23 * * 1-5  /opt/fund-analyzer/repo/vps_qvix_daily.sh >> /var/log/qvix-daily.log 2>&1
#
# 23:00 UTC = 北京次日 07:00, 周二至周六早上(对应周一至周五的收盘)。
# 上交所那份风险指标当晚就发布, 07:00 早就有了; 原来 GitHub 上那次补跑定在
# 06:23 也一直取得到。
#
# 跟云端跑批的时间**不需要错开**: 分库之后两边写的是不同文件, 谁先谁后都不
# 会盖掉对方 —— 这正是拆库要买的东西。
