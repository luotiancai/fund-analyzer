#!/bin/bash
# Run on the VPS with an unpacked release. --code-only reuses the running setup.
set -euo pipefail
RELEASE=$(realpath "$1")
BASE=/opt/fund-web
STATE=/var/lib/fund-web
MODE=${2:-}
if [[ -n "$MODE" && "$MODE" != --code-only ]]; then
    echo "Unknown installation mode: $MODE" >&2
    exit 2
fi
exec 9>/run/fund-web-install.lock
flock -w 300 9
PREVIOUS=$(readlink -e "$BASE/current" || true)

if [[ "$MODE" == --code-only ]]; then
    if [[ -z "$PREVIOUS" || ! -L "$STATE/current" ]]; then
        echo "Initial web installation and data sync are required" >&2
        exit 1
    fi
else
    id fund-web >/dev/null 2>&1 || useradd --system --no-create-home --shell /sbin/nologin fund-web
    install -d -m 755 "$BASE" "$STATE" "$STATE/snapshots"
    install -d -m 750 -o fund-web -g fund-web "$STATE/cache"
fi

PREVIOUS_VENV="$BASE/venv"
if [[ -n "$PREVIOUS" && -x "$PREVIOUS/.venv/bin/uvicorn" ]]; then
    PREVIOUS_VENV=$(readlink -e "$PREVIOUS/.venv")
fi
requirements_hash() {
    sha256sum "$1" | cut -d ' ' -f 1
}
REQUIREMENTS=$(requirements_hash "$RELEASE/requirements-web.txt")
if [[ -n "$PREVIOUS" && -x "$PREVIOUS_VENV/bin/uvicorn" && -f "$PREVIOUS/requirements-web.txt" ]] &&
   [[ "$REQUIREMENTS" == "$(requirements_hash "$PREVIOUS/requirements-web.txt")" ]]; then
    VENV="$PREVIOUS_VENV"
else
    VENV="$BASE/venvs/$REQUIREMENTS"
    if [[ ! -f "$VENV/.ready" ]]; then
        /opt/fund-analyzer/venv/bin/python -m venv --clear "$VENV"
        "$VENV/bin/python" -m pip install --quiet -r "$RELEASE/requirements-web.txt"
        touch "$VENV/.ready"
    fi
fi
PY="$VENV/bin/python"
"$PY" -m compileall -q -x '(^|/)\._' "$RELEASE"
ln -sfnT "$VENV" "$RELEASE/.venv"

if [[ "$MODE" != --code-only ]]; then
    "$PY" "$RELEASE/web_sync.py"
fi

# Prepare read caches before activation; the existing release keeps serving.
systemd-run --quiet --wait --pipe --collect --unit="fund-web-prepare-$$" \
    -p User=fund-web -p Group=fund-web -p MemoryMax=320M -p CPUQuota=75% \
    -p RuntimeMaxSec=240 -E FUND_WEB_ROOT="$STATE" \
    "$PY" "$RELEASE/web_prepare.py"

wait_for_web() {
    "$PY" - "$1" "${2:-}" <<'PY'
import json
import sys
import time, urllib.request

port, expected = sys.argv[1:]
base = f"http://127.0.0.1:{port}/api/v1/"
for _ in range(15):
    try:
        with urllib.request.urlopen(base + "health", timeout=2) as response:
            ready = response.status == 200
        if ready:
            with urllib.request.urlopen(base + "status", timeout=2) as response:
                version = json.load(response)["code_version"]
            if not expected or version == expected:
                print(version)
                break
    except (OSError, ValueError, KeyError):
        pass
    time.sleep(1)
else:
    raise SystemExit(f"Fund Analyzer health/version check failed on port {port}")
PY
}

switch_release() {
    ln -sfnT "$1" "$BASE/.current-$$"
    mv -Tf "$BASE/.current-$$" "$BASE/current"
}

CANDIDATE="fund-web-candidate-$$"
CANDIDATE_RUNNING=0
SWITCHED=0
cleanup() {
    local result=$?
    if [[ "$CANDIDATE_RUNNING" == 1 ]]; then
        systemctl stop "$CANDIDATE.service" || true
    fi
    if [[ "$SWITCHED" == 1 ]]; then
        echo "Deployment failed; restoring the previous release" >&2
        if [[ -n "$PREVIOUS" ]]; then
            switch_release "$PREVIOUS" && systemctl restart fund-web.service ||
                echo "Fund Analyzer rollback failed" >&2
        else
            systemctl stop fund-web.service || true
            unlink "$BASE/current" || true
        fi
    fi
    return "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Give the candidate a separate cache so its startup cannot touch live jobs.
CANDIDATE_RUNNING=1
systemd-run --quiet --collect --unit="$CANDIDATE" \
    -p User=fund-web -p Group=fund-web -p WorkingDirectory="$RELEASE" \
    -p RuntimeDirectory="$CANDIDATE" -p RuntimeMaxSec=90 -p MemoryMax=128M \
    -E FUND_WEB_ROOT="$STATE" -E FUND_WEB_CACHE="/run/$CANDIDATE" \
    "$PY" -m uvicorn web_service:app --host 127.0.0.1 --port 8011
EXPECTED=$(wait_for_web 8011)
systemctl is-active --quiet "$CANDIDATE.service"
systemctl stop "$CANDIDATE.service"
CANDIDATE_RUNNING=0

if [[ "$MODE" != --code-only ]]; then
    install -m 644 "$RELEASE"/deploy/{fund-web.service,fund-web-sync.service,fund-web-sync.timer} /etc/systemd/system/
    install -m 644 "$RELEASE/deploy/fund-web-location.inc" /etc/nginx/fund-web-location.inc
    systemctl daemon-reload
fi

SWITCHED=1
switch_release "$RELEASE"
systemctl restart fund-web.service
wait_for_web 8010 "$EXPECTED" >/dev/null
SWITCHED=0
echo "Active Fund Analyzer code: $EXPECTED"
if [[ "$MODE" == --code-only ]]; then
    exit 0
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
systemctl enable fund-web.service
systemctl enable --now fund-web-sync.timer
systemctl start --no-block fund-web-sync.service
echo "Ready: http://192.243.114.13/fund/"
