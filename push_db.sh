#!/usr/bin/env bash
# 把本地数据库(含净值历史与模拟盘记录)上传为云端数据快照
# (GitHub Release tag `data`,每日跑批与线上页面都从这里取数)。
#
# 什么时候跑:
#   · 首次部署,给云端播种子数据(否则 Actions 首跑要从零回填数小时);
#   · 本地做了模拟盘操作,想把状态同步到线上页面时。
#
# 依赖 gh CLI 且已登录(gh auth login)。
set -euo pipefail

DB="${FUND_ANALYZER_DATA:-$HOME/.local/share/fund-analyzer}/fund_cache.db"
[ -f "$DB" ] || { echo "找不到数据库: $DB" >&2; exit 1; }

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT
echo "压缩 $(du -h "$DB" | cut -f1) 的数据库…"
gzip -9 -c "$DB" > "$TMPDIR/fund_cache.db.gz"

gh release create data --title "数据快照" \
  --notes "每日跑批产物 fund_cache.db.gz,应用启动时自动拉取,请勿手动改动" \
  2>/dev/null || true
echo "上传 $(du -h "$TMPDIR/fund_cache.db.gz" | cut -f1)…"
gh release upload data "$TMPDIR/fund_cache.db.gz" --clobber
echo "✅ 已上传,线上页面最迟 1 小时内自动换用新快照"
