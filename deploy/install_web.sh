#!/bin/bash
# Run on BWG with an unpacked, reviewed release directory as the only argument.
set -euo pipefail
RELEASE=$(realpath "$1")
BASE=/opt/fund-web
STATE=/var/lib/fund-web

id fund-web >/dev/null 2>&1 || useradd --system --no-create-home --shell /sbin/nologin fund-web
install -d -m 755 "$BASE" "$STATE" "$STATE/snapshots"
install -d -m 750 -o fund-web -g fund-web "$STATE/cache"
if [ ! -x "$BASE/venv/bin/python" ]; then
    /opt/fund-analyzer/venv/bin/python -m venv "$BASE/venv"
fi
"$BASE/venv/bin/python" -m pip install --quiet -r "$RELEASE/requirements-web.txt"
"$BASE/venv/bin/python" -m compileall -q -x '(^|/)\._' "$RELEASE"

install -m 644 "$RELEASE"/deploy/fund-web*.service "$RELEASE"/deploy/fund-web-sync.timer /etc/systemd/system/
install -m 644 "$RELEASE/deploy/fund-web-location.inc" /etc/nginx/fund-web-location.inc

# Preserve a concrete rollback target before touching nginx or switching code.
PREVIOUS=$(readlink -f "$BASE/current" || true)
ln -s "$RELEASE" "$BASE/.current-$$"
mv -Tf "$BASE/.current-$$" "$BASE/current"
"$BASE/venv/bin/python" "$BASE/current/web_sync.py"
systemctl daemon-reload
systemctl enable --now fund-web.service
systemctl restart fund-web.service
if ! "$BASE/venv/bin/python" - <<'PY'
import time, urllib.request
for _ in range(20):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8010/api/v1/health',timeout=2) as r:
            if r.status == 200:
                break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('new web service is not healthy')
PY
then
    if [ -n "$PREVIOUS" ]; then
        ln -s "$PREVIOUS" "$BASE/.rollback-$$"
        mv -Tf "$BASE/.rollback-$$" "$BASE/current"
        systemctl restart fund-web.service
    fi
    exit 1
fi

python3 - <<'PY'
from pathlib import Path
p=Path('/etc/nginx/conf.d/fund-site.conf')
before=p.read_text()
include='    include /etc/nginx/fund-web-location.inc;'
if include not in before:
    backup=p.with_name(p.name+'.before-fund-web')
    if not backup.exists():
        backup.write_text(before)
    head,tail=before.rsplit('}',1)
    p.write_text(head+include+'\n}'+tail)
PY
if ! nginx -t; then
    cp /etc/nginx/conf.d/fund-site.conf.before-fund-web /etc/nginx/conf.d/fund-site.conf
    exit 1
fi
systemctl reload nginx
systemctl enable --now fund-web-sync.timer
systemctl start --no-block fund-web-sync.service
echo "Ready: http://192.243.114.13/fund/"
