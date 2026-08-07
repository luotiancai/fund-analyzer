#!/usr/bin/env python3
"""看一眼线上数据到底是什么状态,不用手动下载解压。

    python3 cloud_peek.py              # 体检: 各表行数 + 新鲜度(只下小库)
    python3 cloud_peek.py --full       # 连净值/规模一起看(要下 60MB)
    python3 cloud_peek.py --diff       # 跟本地库逐表对比
    python3 cloud_peek.py --sql "SELECT ..."   # 任意只读查询

为什么要这个:线上数据是 gz 压缩的 SQLite,存在 GitHub Release 上。想确认
"线上净值到哪天了""规模表覆盖多少只""模拟盘还在不在",过去只能手动
gh release download 80MB、gunzip 出 400MB、再开 sqlite3 敲 SQL。一次几分钟,
于是实际上从来没人看 —— 本地和云端差多少全靠猜,2026-08-05 那次把云端攒了
五天的行情盖回去,直接原因就是没人知道两边差了多少。

分库之后连"下 80MB"这一步也省了:默认只拉几个小库(合计约 2MB),净值和
规模那两个大库要 --full 才拉。下载走 cloud_assets 的条件下载,远端没变直接
用缓存、零流量。

只读:全程不写线上,也不碰本地正在用的库(缓存落在 ~/.cache 下)。
"""

import argparse
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_assets   # noqa: E402
import fetcher        # noqa: E402

# 默认拉的小库(合计约 2MB)。大库(nav/scale)要 --full。cache 不看。
SMALL_DBS = ("rank", "market", "strategy", "sim")
BIG_DBS = ("nav", "scale")

# 体检项: (表名, 说明, 新鲜度 SQL 或 None)
CHECKS = [
    ("fund_nav_daily", "基金净值(日)", "SELECT MAX(date) FROM fund_nav_daily"),
    ("fund_nav_meta", "净值覆盖的基金", None),
    ("fund_scale_hist", "基金季度规模", "SELECT MAX(quarter_end) FROM fund_scale_hist"),
    ("fund_scale_miss", "标记为「无规模数据」", None),
    ("fund_holdings", "基金季度持仓", None),
    ("fund_sharpe", "年化/回撤/区间收益", None),
    ("qvix_self_history", "QVIX 自算历史", "SELECT MAX(date) FROM qvix_self_history"),
    ("index_daily_cache", "指数日线缓存", None),
    ("fund_list", "榜单快照(整块1行)", None),
    ("sim_trades", "模拟盘交易", None),
    ("sim_archives", "模拟盘存档", None),
]


def open_cloud(full: bool, refresh: bool) -> sqlite3.Connection:
    """把线上各库下下来, ATTACH 成一个连接。

    库名跟 fetcher._conn() 里一致, 所以不带库名的表名解析方式也一致 ——
    从应用代码里抄来的 SQL 直接能跑。
    """
    cloud_assets.log("取线上数据:")
    remote = cloud_assets.asset_times()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    names = list(SMALL_DBS) + (list(BIG_DBS) if full else [])
    for name in names:
        try:
            path = cloud_assets.fetch(name, refresh=refresh, remote_times=remote)
        except (FileNotFoundError, subprocess.CalledProcessError):
            cloud_assets.log(f"  {name}: 云端没有, 跳过")
            continue
        # 直接给路径而不是 file:...?mode=ro —— ATTACH 只在全局开了 URI 文件名
        # 时才认 file: 前缀。这些都是缓存副本, 本脚本从不写。
        conn.execute(f"ATTACH DATABASE ? AS {name}", (path,))
    return conn


def _count(conn, table):
    try:
        return conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
    except sqlite3.Error:
        return None


def report(conn, full: bool):
    print("\n=== 线上资产 ===")
    times = cloud_assets.asset_times()
    if not times:
        print("  (取不到 Release 资产列表, gh 登录了吗?)")
    for name, fn, _t, _l in fetcher.DB_LAYOUT:
        info = times.get(cloud_assets.ASSET_OF[name])
        if not info:
            print(f"  {fn:22} —(云端没有)")
            continue
        t = datetime.strptime(info["updated"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc).astimezone()
        age = (datetime.now(timezone.utc)
               - t.astimezone(timezone.utc)).total_seconds() / 3600
        print(f"  {fn:22}{info['size']/1048576:7.1f}MB  更新于 "
              f"{t:%Y-%m-%d %H:%M} ({age:.0f} 小时前)")

    print("\n=== 各表状态 ===")
    print(f"  {'表':22}{'说明':22}{'行数':>12}  新鲜度")
    for table, desc, freshness_sql in CHECKS:
        n = _count(conn, table)
        if n is None:
            mark = "—(大库, 加 --full)" if not full else "—(表不存在)"
            print(f"  {table:22}{desc:22}{mark:>12}")
            continue
        extra = ""
        if freshness_sql:
            try:
                extra = conn.execute(freshness_sql).fetchone()[0] or ""
            except sqlite3.Error:
                extra = ""
        print(f"  {table:22}{desc:22}{n:>12,}  {extra}")

    try:
        rows = conn.execute(
            "SELECT id, label, is_standard, n_trades, win_rate, cum_return "
            "FROM strategy_runs ORDER BY run_at DESC LIMIT 5").fetchall()
        print("\n=== 策略跑批(最近5次)===")
        for r in rows:
            std = "【标准】" if r["is_standard"] else "        "
            print(f"  #{r['id']:>3} {std} {r['label'][:44]:46}"
                  f"{r['n_trades']}笔 胜率{r['win_rate']:.0f}% 复利{r['cum_return']:+.0f}%")
    except sqlite3.Error:
        pass

    n_miss = _count(conn, "fund_scale_miss")
    if n_miss and n_miss > 500:
        print(f"\n⚠️  fund_scale_miss 有 {n_miss:,} 条 —— 正常应该只有几十条。"
              "\n    是批量抓取被限流、失败被误记成「该基金没有规模表」, 后果是"
              "\n    这些基金的规模门槛失效。见 fetcher.fetch_fund_scale_hist。")


def diff_local(conn, full: bool):
    """跟本地逐表比行数——推送前跑一下, 免得拿旧库盖掉新数据。
    (push_dbs.py 每次推送前会自动做这件事, 这里是手动看的入口。)"""
    loc = fetcher._conn()
    print(f"\n=== 本地 vs 线上 ===  本地: {fetcher._DATA_DIR}")
    print(f"  {'表':22}{'本地':>12}{'线上':>12}{'差值':>12}")
    worse = []
    for table, _desc, _f in CHECKS:
        c_cloud, c_local = _count(conn, table), _count(loc, table)
        if c_cloud is None or c_local is None:
            continue
        d = c_local - c_cloud
        flag = "  ← 本地更旧" if d < 0 else ""
        if d < 0:
            worse.append(table)
        print(f"  {table:22}{c_local:>12,}{c_cloud:>12,}{d:>+12,}{flag}")
    loc.close()
    if worse:
        print(f"\n⚠️  本地这些表比线上少: {', '.join(worse)}")
        print("    对应的库别推 —— push_dbs.py 会自己拦下来。")
    else:
        print("\n✅ 本地没有比线上旧的表。")
    if not full:
        print("  (净值/规模在大库里, 加 --full 才会一起比)")


def main():
    ap = argparse.ArgumentParser(
        description="看一眼线上数据状态(只读)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--full", action="store_true", help="连净值/规模一起看")
    ap.add_argument("--diff", action="store_true", help="跟本地逐表对比行数")
    ap.add_argument("--sql", metavar="SQL", help="跑一条只读 SQL")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存强制重下")
    args = ap.parse_args()

    conn = open_cloud(args.full, args.refresh)
    try:
        if args.sql:
            rows = conn.execute(args.sql).fetchall()
            if not rows:
                print("(空结果)")
                return
            print(" | ".join(rows[0].keys()))
            print("-" * 60)
            for r in rows[:200]:
                print(" | ".join("" if v is None else str(v) for v in tuple(r)))
            if len(rows) > 200:
                print(f"… 共 {len(rows)} 行, 只显示前 200")
            return
        report(conn, args.full)
        if args.diff:
            diff_local(conn, args.full)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
