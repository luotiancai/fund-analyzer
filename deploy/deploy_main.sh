#!/bin/bash
# Called by fund-web-deploy.service on the VPS.
set -euo pipefail
BASE=/opt/fund-web
SOURCE="$BASE/deploy-source.git"
REPO=https://github.com/luotiancai/fund-analyzer.git

if [[ ! -d "$SOURCE" ]]; then
    git init --quiet --bare "$SOURCE"
fi
git --git-dir="$SOURCE" fetch --quiet --depth=1 --no-tags "$REPO" refs/heads/main
revision=$(git --git-dir="$SOURCE" rev-parse FETCH_HEAD)
if [[ $(cat "$BASE/current/.git-revision" 2>/dev/null || true) == "$revision" ]]; then
    exit 0
fi

install -d -m 755 "$BASE/releases"
release=$(mktemp -d "$BASE/releases/main-${revision:0:12}.XXXXXXXX")
chmod 755 "$release"
cleanup() {
    local result=$?
    if [[ "$result" != 0 && $(readlink -e "$BASE/current" || true) != "$release" ]]; then
        rm -rf -- "$release"
    fi
    return "$result"
}
trap cleanup EXIT

echo "Deploying main at $revision"
git --git-dir="$SOURCE" archive "$revision" | tar -x -C "$release"
bash "$release/deploy/install_web.sh" "$release" --code-only
printf '%s\n' "$revision" > "$release/.git-revision"
echo "Deployed main at $revision"
