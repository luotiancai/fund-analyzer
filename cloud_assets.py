"""云端资产的共用底座:每个库一个 Release 资产,下载/缓存/看版本戳。

被 cloud_peek.py(看线上)、sync_down.py(拉到本地)、push_dbs.py(推上去)
共用。资产名直接由 fetcher.DB_LAYOUT 推出来 —— 加一个新库只要改那张表,
这三个工具自动跟上。

下载缓存放 ~/.cache/fund-analyzer-peek,跟真实数据目录分开:看线上和用线上
是两回事,前者绝不该覆盖你本地正在用的库。
"""

import gzip
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetcher   # noqa: E402

REPO = "luotiancai/fund-analyzer"
RELEASE_TAG = "data"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "fund-analyzer-peek")
_STAMP_FILE = os.path.join(CACHE_DIR, "asset-stamps.json")

# 库名 → 资产名。fund_rank.db → fund_rank.db.gz
ASSET_OF = {name: fn + ".gz" for name, fn, _t, _l in fetcher.DB_LAYOUT}
DB_OF_ASSET = {v: k for k, v in ASSET_OF.items()}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def asset_times() -> dict:
    """各资产在 Release 上的更新时间 {资产名: ISO 时间}。取不到返回 {}。"""
    try:
        out = subprocess.run(
            ["gh", "release", "view", RELEASE_TAG, "--repo", REPO,
             "--json", "assets", "--jq",
             '.assets[] | "\\(.name)\\t\\(.updatedAt)\\t\\(.size)"'],
            capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return {}
    got = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        name, ts, size = line.split("\t")
        got[name] = {"updated": ts, "size": int(size)}
    return got


def _stamps() -> dict:
    try:
        with open(_STAMP_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_stamp(key: str, value: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_ = _stamps()
    all_[key] = value
    tmp = _STAMP_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(all_, f, indent=1)
    os.replace(tmp, _STAMP_FILE)


def fetch(db_name: str, dest: str = None, refresh: bool = False,
          remote_times: dict = None) -> str:
    """下载并解压某个库的资产, 返回落地路径。

    条件下载: 记住上次拿到的 Release updatedAt, 远端没变就直接跳过、零流量。
    这是本地同步省流量的关键 —— 净值库 60MB, 一天没变就不该再下一次。
    dest 不给时落到缓存目录(只看不用), 给了就落到那儿(比如真实数据目录)。
    """
    asset = ASSET_OF[db_name]
    dest = dest or os.path.join(CACHE_DIR, os.path.basename(fetcher.DB_PATH[db_name]))
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    remote = (remote_times if remote_times is not None else asset_times()).get(asset)
    key = f"{asset}@{dest}"
    if not refresh and remote and os.path.exists(dest) \
            and _stamps().get(key) == remote["updated"]:
        log(f"  {asset}: 远端未变({remote['updated'][:16]}), 跳过")
        return dest
    if not remote:
        log(f"  {asset}: Release 上没有这个资产")
        if os.path.exists(dest):
            return dest
        raise FileNotFoundError(asset)

    gz = dest + ".gz"
    log(f"  {asset}: 下载 {remote['size']/1048576:.1f}MB…")
    t0 = time.time()
    subprocess.run(
        ["gh", "release", "download", RELEASE_TAG, "--pattern", asset,
         "--output", gz, "--clobber", "--repo", REPO],
        check=True, stdout=subprocess.DEVNULL)
    tmp = dest + ".tmp"
    with gzip.open(gz, "rb") as f_in, open(tmp, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.replace(tmp, dest)          # 原子换上, 避免半截文件被别的进程读到
    os.unlink(gz)
    _save_stamp(key, remote["updated"])
    log(f"     解出 {os.path.getsize(dest)/1048576:.1f}MB, 用时 {time.time()-t0:.1f}s")
    return dest


def upload(paths: list):
    """把若干个 .gz 传上 Release(--clobber 覆盖同名资产)。"""
    subprocess.run(
        ["gh", "release", "create", RELEASE_TAG, "--title", "数据快照",
         "--notes", "每日跑批产物,应用启动时自动拉取,请勿手动改动",
         "--repo", REPO],
        check=False, capture_output=True)
    subprocess.run(
        ["gh", "release", "upload", RELEASE_TAG, *paths,
         "--clobber", "--repo", REPO], check=True)
