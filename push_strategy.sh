#!/usr/bin/env bash
# 只把策略库(回测跑批结果)推到云端,不碰行情数据。
#
# 为什么单独一个脚本:策略结果几十KB,行情主库 400MB。以前混在一张库里,
# 发一次回测结果得跑 push_db.sh 整库上传——2026-08-05 就因此出过事故,
# 本地行情停在 07-30,一推把云端每日跑批攒到 08-04 的数据整个盖回去了。
# 分库之后这两条发布线互不干扰:
#   · 行情:每日跑批(GitHub Actions)传 fund_cache*.db.gz 那三个资产;
#   · 策略:这个脚本传 fund_strategy.db.gz,十几KB,几秒钟。
# 两边可以同时跑,谁也盖不到谁。
#
# 什么时候跑:本地 backtest_qvix.py 跑完、想让线上页面看到这次跑批时。
#
# 依赖 gh CLI 且已登录(gh auth login)。
set -euo pipefail

DB="${FUND_ANALYZER_DATA:-$HOME/.local/share/fund-analyzer}/fund_strategy.db"
[ -f "$DB" ] || { echo "找不到策略库: $DB(先跑一次 backtest_qvix.py)" >&2; exit 1; }

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

python3 - "$DB" <<'PY'
import sqlite3, sys, datetime
c = sqlite3.connect(sys.argv[1]); c.row_factory = sqlite3.Row
rows = c.execute("SELECT id, run_at, label, is_standard, n_trades, win_rate,"
                 " cum_return FROM strategy_runs ORDER BY run_at DESC").fetchall()
print(f"策略库里有 {len(rows)} 次跑批:")
for r in rows[:10]:
    t = datetime.datetime.fromtimestamp(r["run_at"]).strftime("%m-%d %H:%M")
    std = "【标准】" if r["is_standard"] else "        "
    w = f"{r['win_rate']:.0f}%" if r["win_rate"] is not None else "—"
    cum = f"{r['cum_return']:+.0f}%" if r["cum_return"] is not None else "—"
    print(f"  #{r['id']:>3} {t} {std} {r['label']}  ({r['n_trades']}笔 胜率{w} 复利{cum})")
if len(rows) > 10:
    print(f"  … 另有 {len(rows) - 10} 次")
PY

gzip -9 -c "$DB" > "$TMPDIR/fund_strategy.db.gz"
echo "上传 $(du -h "$TMPDIR/fund_strategy.db.gz" | cut -f1)…"
gh release upload data "$TMPDIR/fund_strategy.db.gz" --clobber
echo "✅ 已上传,线上页面最迟 1 小时内自动换用新策略库"
