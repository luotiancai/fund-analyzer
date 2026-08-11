#!/usr/bin/env python3
"""把本地的某个/某几个库推到云端 Release,推之前先确认不会盖掉更新的数据。

    python3 push_dbs.py scale              # 只推规模库
    python3 push_dbs.py rank market        # 推多个
    python3 push_dbs.py --all              # 除 cache 外全推
    python3 push_dbs.py nav --force        # 跳过安全检查(想清楚再用)
    python3 push_dbs.py scale --dry-run    # 只做检查, 不上传
    python3 push_dbs.py nav --skip-verify  # 不回读验证(省一次下载, 别常用)

**这个脚本存在的唯一理由是防止 2026-08-05 那类事故**:那天本地库的行情停在
07-30, 为了发两张回测表跑了 push_db.sh(整库上传), 把云端每日跑批攒到
08-04 的数据整个盖回 07-30。事后看有两个独立的错误:

  1. 发布单元太大 —— 想推几 KB 的回测结果, 却只能整库推。这个已经由分库
     解决(见 fetcher.DB_LAYOUT), 现在推 scale 碰不到 nav。
  2. **没人知道本地比云端旧多少**。分库只解决了第一个, 第二个要靠这里:
     每次推之前下载云端那份, 逐表比行数, 本地少了就中止并打印差异。

安全检查会下载云端对应的库(条件下载, 远端没变就用缓存)。推 nav(60MB)时
这一步不便宜, 但正是那种最不该盖错的场合。

**2026-08-11 补的第三个错误**: 上面两条都只管"推之前", 没人管"推完了到底
有没有上去"。那天推一条跑批(#29 事后最优上限), `gh release upload --clobber`
返回码 0、脚本照常打「✅ 已上传」, 而 Release 上的资产**根本没变**(API 报的
updatedAt 停在上一次推送, 文件大小一字节没动)。结果是页面上死活看不到新
跑批, 重启多少次都没用 —— ETag 没变, 连下载都被跳过, 从表现上完全等同于
"缓存没刷新", 查了半天才发现是上传本身没生效。所以现在:

  · 推完**必回读**: 把刚传上去的资产重新拉一份, 跟本地文件比 sha256,
    对不上就非零退出。CDN 有几秒延迟, 所以失败会重试几次再判死。
  · 行数检查同时比**内容摘要**: 原来只比行数, 对"行数不变、内容原地改写"
    (比如改 backtest_notes 的点评)等于没检查, 绿灯是假的。现在小表会
    多算一个内容摘要, 行数相同但内容不同时明确打出来, 让人自己判断
    (谁新谁旧机器判断不了, 但至少不能假装检查过了)。
"""

import hashlib
import os
import sqlite3
import subprocess
import sys
import tempfile
import time

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


# 内容摘要只给"小表"算: 摘要要整表扫一遍, nav 那 490 万行扫一次几十秒,
# 而它本来就不是会被原地改写的表(只追加净值)。真正需要这道检查的是
# backtest_notes / strategy_runs 这种几十行、经常整条重写的小表。
DIGEST_MAX_ROWS = 500_000


def _digests(path: str, tables, counts: dict) -> dict:
    """各表的内容摘要; 表太大/不存在的给 None(表示"没比")。"""
    if not os.path.exists(path):
        return {t: None for t in tables}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {}
    for t in tables:
        n = counts.get(t)
        if n is None or n > DIGEST_MAX_ROWS:
            out[t] = None
            continue
        h = hashlib.sha256()
        try:
            # ORDER BY rowid 固定行序, 免得同样的内容因为物理顺序不同
            # (VACUUM 过、或先后插入顺序不一样)摘出两个不同的值。
            # WITHOUT ROWID 的表没有 rowid, 退回不排序的整表扫。
            try:
                rows = conn.execute(f"SELECT * FROM [{t}] ORDER BY rowid")
            except sqlite3.Error:
                rows = conn.execute(f"SELECT * FROM [{t}]")
            for row in rows:
                h.update(repr(tuple(row)).encode())
            out[t] = h.hexdigest()[:12]
        except sqlite3.Error:
            out[t] = None
    conn.close()
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(db_name: str, tries: int = 4, wait: float = 4.0) -> bool:
    """推完回读: 重新拉一份刚传的资产, 跟本地文件比 sha256。

    为什么要重试: Release 资产走 CDN, 刚 clobber 完的头几秒可能还回旧
    副本, 一次对不上不能立刻判死。重试几次仍对不上, 那就是真没传上去
    (2026-08-11 那次就是这种, 重试到底都是旧的)。
    """
    local = fetcher.DB_PATH[db_name]
    want = _sha256(local)
    for i in range(1, tries + 1):
        try:
            # refresh=True 绕开"远端未变就跳过"的条件下载 —— 这里要的正是
            # 无条件重新拉一份, 缓存里那份恰恰是我们要证伪的对象。
            got_path = cloud_assets.fetch(db_name, refresh=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"  第{i}次回读失败({e.__class__.__name__}), 等 {wait:.0f}s 再试…")
            time.sleep(wait)
            continue
        got = _sha256(got_path)
        if got == want:
            print(f"  ✅ 回读一致(sha256 {got[:12]}…)")
            return True
        print(f"  第{i}次回读对不上: 本地 {want[:12]}… / 云端 {got[:12]}…"
              + (f", 等 {wait:.0f}s 再试…" if i < tries else ""))
        if i < tries:
            time.sleep(wait)
    print(f"  ❌ {db_name}: 传完回读仍然对不上 —— 这次上传**没有生效**。"
          f"别信上面那句已上传, 重跑一次。")
    return False


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
    d_local, d_cloud = _digests(local, tables, c_local), _digests(cloud, tables, c_cloud)
    worse, rewritten = [], []
    print(f"  {'表':22}{'本地':>12}{'云端':>12}{'差值':>12}  内容")
    for t in tables:
        a, b = c_local[t], c_cloud[t]
        if a is None or b is None:
            print(f"  {t:22}{str(a):>12}{str(b):>12}{'—':>12}")
            continue
        d = a - b
        flag = "  ← 本地更旧" if d < 0 else ""
        if d < 0:
            worse.append(f"{t}({d:+,})")
        # 内容一栏: 行数相同才有意义 —— 行数都不一样了, 内容当然不同,
        # 那种情况看差值那列就够了。
        if d == 0 and d_local[t] is None:
            note = "(表大, 未比)"
        elif d == 0 and d_local[t] == d_cloud[t]:
            note = "一致"
        elif d == 0:
            note = "**原地改写**"
            rewritten.append(t)
        else:
            note = ""
        print(f"  {t:22}{a:>12,}{b:>12,}{d:>+12,}{flag}  {note}")
    if worse:
        print(f"  ❌ 本地这些表比云端少: {', '.join(worse)}")
        return False
    if rewritten:
        # 不拦: 改备注/改点评这类原地重写是正常操作。但必须说出来 ——
        # 行数相同时那句"本地不比云端旧"是**没有验证过的**, 谁新谁旧
        # 只有人知道。以前这里默不作声地打绿灯, 是假的安全感。
        print(f"  ⚠️ 行数相同但内容不同的表: {', '.join(rewritten)}")
        print("     ——这几张表机器判断不了谁新谁旧, 确认本地是最新的再推。")
    print("  ✅ 本地不比云端旧" + ("(行数口径)" if rewritten else ""))
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    dry = "--dry-run" in sys.argv
    skip_verify = "--skip-verify" in sys.argv
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

    # 上传命令返回 0 ≠ 资产真的换了(2026-08-11 踩过, 见模块 docstring)。
    # 回读比 sha256 才算数; 对不上就非零退出, 别让调用方以为成功了。
    if skip_verify:
        print("--skip-verify: 没有回读验证, 这次是否真的上去了不知道")
        print("✅ 已上传(未验证),线上页面最迟 1 小时内自动换用新库")
        return
    print("\n回读验证…")
    bad = [n for n in ok_names if not verify(n)]
    if bad:
        raise SystemExit(f"\n❌ 这些库没推上去: {', '.join(bad)}。"
                         f"重跑一次; 连着几次都这样就去 Release 页面看看。")
    print("✅ 已上传并回读验证,线上页面最迟 1 小时内自动换用新库")


if __name__ == "__main__":
    main()
