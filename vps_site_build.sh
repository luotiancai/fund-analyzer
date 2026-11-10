#!/bin/bash
# 上证页静态网页 —— 跑在 BWG 上, 生成 /opt/fund-analyzer/site/index.html,
# nginx 直接发这个目录。页面本身见 build_site.py / site/template.html。
#
# cron 每小时检查本地推送的 strategy, VPS 更新指数日线或 QVIX 后也会调用。
# market 和 qvix 直接读 VPS 本地库。
set -euo pipefail

BASE=/opt/fund-analyzer
export FUND_ANALYZER_DATA=$BASE/data
PY=$BASE/venv/bin/python
SITE=$BASE/site
cd "$BASE/repo"

# 定时和跑批触发的重建共用一把锁。
exec 9>/run/fund-site-build.lock
flock -w 300 9 || { echo "等锁超时, 跳过"; exit 0; }

git pull -q --ff-only || echo "⚠️ git pull 失败, 用本地现有代码继续"

# 条件下载, 远端没变零流量
$PY sync_down.py strategy

# 代码和三个库都没变就不重新生成
mkdir -p "$SITE"
STAMP=$SITE/.inputs
NOW="$(git rev-parse HEAD) $(stat -c %Y "$FUND_ANALYZER_DATA"/{market,qvix,fund_strategy}.db | tr '\n' ' ')"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$NOW" ]; then
    exit 0
fi
echo "───────── $(TZ=Asia/Shanghai date '+%F %T CST') ─────────"
$PY build_site.py --out "$SITE"
echo "$NOW" > "$STAMP"

# ── 部署(一次性, 已做过) ────────────────────────────────────────────────────
# cron(root, 时区 UTC):
#   20 * * * *  /opt/fund-analyzer/repo/vps_site_build.sh >> /var/log/fund-site.log 2>&1
#
# nginx(dnf install nginx; systemctl enable --now nginx), /etc/nginx/conf.d/fund-site.conf:
#   server {
#       listen 80 default_server;
#       listen [::]:80 default_server;
#       server_name 192.243.114.13;
#       root /opt/fund-analyzer/site;
#       index index.html;
#       server_tokens off;
#       gzip on;
#       gzip_types application/javascript;
#       location ^~ /.well-known/acme-challenge/ { root /var/www/acme; default_type text/plain; }
#       location = /index.html { add_header Cache-Control "no-cache"; }
#       location ~ \.js$ { add_header Cache-Control "public, max-age=31536000, immutable"; }
#       location ~ /\. { deny all; }
#   }
#
# 443 被 xray(REALITY)占着, 所以走 80。不设密码: 页面上的数据本来就在公开
# 仓库的 Release 里。
#
# ⚠️ 80 端口原来不是空着的: acme.sh 给 x-ui 面板续那张 IP 证书(shortlived,
# 约5天一续)用的是 standalone 模式, 续的时候自己临时占 80。nginx 一常驻就
# 续不上, 证书几天后过期。所以那张证书改成了 webroot 模式(Le_Webroot=
# /var/www/acme, 原配置备份在 .conf.bak-standalone), 走上面那个
# acme-challenge location。2026-10-14 用 staging 实签一次验证通过。
