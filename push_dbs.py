#!/usr/bin/env python3
"""把本地的某个/某几个库推到云端 Release,推之前先确认不会盖掉更新的数据。

    python3 push_dbs.py scale              # 只推规模库
    python3 push_dbs.py rank market        # 推多个
    python3 push_dbs.py --all              # 除 cache 外全推
    python3 push_dbs.py nav --force        # 跳过安全检查(想清楚再用)
    python3 push_dbs.py scale --dry-run    # 只做检查, 不上传

**这个脚本存在的唯一理由是防止 2026-08-05 那类事故**:那天本地库的行情停在
07-30, 为了发两张回测表跑了 push_db.sh(整库上传), 把云端每日跑批攒到
08-04 的数据整个盖回 07-30。事后看有两个独立的错误:

  1. 发布单元太大 —— 想推几 KB 的回测结果, 却只能整库推。这个已经由分库
     解决(见 fetcher.DB_LAYOUT), 现在推 scale 碰不到 nav。
  2. **没人知道本地比云端旧多少**。分库只解决了第一个, 第二个要靠这里:
     每次推之前下载云端那份, 逐表比行数, 本地少了就中止并打印差异。

安全检查会下载云端对应的库(条件下载, 远端没变就用缓存)。推 nav(60MB)时
这一步不便宜, 但正是那种最不该盖错的场合。
"""

import os
import sqlite3
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_assets   # noqa: E402
import fetcher        # noqa: E402

# cache 库不参与推送: 筛选结果缓存, 丢了自动重算, 推上去只是给别人塞一份
# 无用的历史查询。
PUSHABLE = tuple(n for n, _f, _t, _l in fetcher.DB_LAYOUT if n != "cache")


def _counts(path: str, tables) -> dict:
    """某个库文件里各表的行数; 文件不存在或表不存在给 None。"""
    if not os.path.exists(path):
        return {t: None for t in tables}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {}
    for t in tables:
        try:
            out[t] = conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
        except sqlite3.Error:
            out[t] = None
    conn.close()
    return out


def check(db_name: str, tables, remote_times) -> bool:
    """本地这个库是不是"不比云端旧"。返回 True 表示可以安全推送。"""
    local = fetcher.DB_PATH[db_name]
    if not os.path.exists(local):
        print(f"  ❌ 本地没有 {local}")
        return False
    try:
        cloud = cloud_assets.fetch(db_name, remote_times=remote_times)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  云端还没有这个资产 —— 首次发布, 跳过对比")
        return True
    c_local, c_cloud = _counts(local, tables), _counts(cloud, tables)
    worse = []
    print(f"  {'表':22}{'本地':>12}{'云端':>12}{'差值':>12}")
    for t in tables:
        a, b = c_local[t], c_cloud[t]
        if a is None or b is None:
            print(f"  {t:22}{str(a):>12}{str(b):>12}{'—':>12}")
            continue
        d = a - b
        flag = "  ← 本地更旧" if d < 0 else ""
        if d < 0:
            worse.append(f"{t}({d:+,})")
        print(f"  {t:22}{a:>12,}{b:>12,}{d:>+12,}{flag}")
    if worse:
        print(f"  ❌ 本地这些表比云端少: {', '.join(worse)}")
        return False
    print("  ✅ 本地不比云端旧")
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    dry = "--dry-run" in sys.argv
    names = list(PUSHABLE) if "--all" in sys.argv else args
    if not names:
        raise SystemExit(__doc__)
    bad = [n for n in names if n not in dict((x, 1) for x in PUSHABLE)]
    if bad:
        raise SystemExit(f"未知的库: {', '.join(bad)}\n可推的: {', '.join(PUSHABLE)}")

    tables_of = {n: t for n, _f, t, _l in fetcher.DB_LAYOUT}
    remote_times = cloud_assets.asset_times()

    ok_names = []
    for n in names:
        print(f"\n=== {n}  ({os.path.basename(fetcher.DB_PATH[n])}) ===")
        if check(n, tables_of[n], remote_times) or force:
            if force:
                print("  (--force: 无视检查结果)")
            ok_names.append(n)
        else:
            print("  跳过。确实要覆盖就加 --force。")

    if not ok_names:
        raise SystemExit("\n没有可推的库。")
    if dry:
        print(f"\n--dry-run: 本来会推 {', '.join(ok_names)}")
        return

    tmp = tempfile.mkdtemp()
    paths = []
    for n in ok_names:
        src = fetcher.DB_PATH[n]
        gz = os.path.join(tmp, os.path.basename(src) + ".gz")
        print(f"\n压缩 {os.path.basename(src)} "
              f"({os.path.getsize(src)/1048576:.1f}MB)…", end="", flush=True)
        subprocess.run(f"gzip -9 -c '{src}' > '{gz}'", shell=True, check=True)
        print(f" → {os.path.getsize(gz)/1048576:.1f}MB")
        paths.append(gz)
    print(f"\n上传 {len(paths)} 个资产…")
    cloud_assets.upload(paths)
    print("✅ 已上传,线上页面最迟 1 小时内自动换用新库")


if __name__ == "__main__":
    main()
