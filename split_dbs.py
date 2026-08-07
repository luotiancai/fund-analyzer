#!/usr/bin/env python3
"""一次性迁移:把分库前的 fund_cache.db 拆成 fetcher.DB_LAYOUT 定义的那几个库。

    python3 split_dbs.py                       # 拆本地库
    python3 split_dbs.py --check               # 只对比行数, 不写
    python3 split_dbs.py --src X.db --out DIR  # 拆任意库到任意目录

为什么要拆见 fetcher.DB_LAYOUT 上面那段。这里只说迁移本身的几个取舍:

  · **不动源库**。全程只读 fund_cache.db, 拆完它原样留着——它就是备份。
    确认新库没问题之后再自己删(401MB)。所以这个脚本可以反复跑。

  · **先建表再灌数据**, 不用 CREATE TABLE AS SELECT。后者会丢掉
    WITHOUT ROWID、主键这些, 漏一个就是隐性劣化(整表退化成有 rowid 的
    堆表, 净值表会白白多占几十 MB)。这里让 fetcher 按 _DDL 建好正确的
    表结构, 再逐表 INSERT ... SELECT。

  · **按列名交集灌**。fund_sharpe 那八列 mdd/ret 是历史上一路 ALTER 加的
    (夏普三列 2026-08-07 又 ALTER 掉了), 源库和新库的列顺序不一定一致,
    所以显式列名而不是 SELECT *。源库没有的列留 NULL。

  · **僵尸表自然消失**。不在 DB_LAYOUT 里的表(backtest_notes 在 core 里
    那份、backtest_trades、backtest_regime_1m/3m)不会被复制——这正是要的
    效果: 那 4 张是策略拆库时本地删掉、但云端那份库一路继承下来的残留,
    其中 core.backtest_notes 还在遮蔽 strategy 库里的真表。
"""

import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetcher   # noqa: E402


def _cols(conn, table, db="main"):
    return [r[1] for r in conn.execute(f"PRAGMA {db}.table_info([{table}])")]


def _count(conn, table, db="main"):
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {db}.[{table}]").fetchone()[0]
    except sqlite3.Error:
        return None


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    check_only = "--check" in sys.argv
    # --src/--out 用来拆**云端**那份库: 上线时不能拿本地库播种线上(本地净值
    # 通常比线上旧几天, 直接推等于把跑批攒的数据盖回去, 2026-08-05 的事故就
    # 是这么来的)。正确做法是下载云端整库、在它身上拆、再上传。
    src = _arg("--src", fetcher._LEGACY_SINGLE_DB)
    out = _arg("--out")
    if out:
        os.makedirs(out, exist_ok=True)
        fetcher._DATA_DIR = out
        fetcher.DB_PATH = {n: os.path.join(out, fn)
                           for n, fn, _t, _l in fetcher.DB_LAYOUT}
    if not os.path.exists(src):
        raise SystemExit(f"找不到源库: {src}\n"
                         "(已经拆过了? 那就不用再跑这个脚本。)")
    print(f"源库 {src}  {os.path.getsize(src)/1048576:.0f}MB")
    if out:
        print(f"输出到 {out}")

    # 让 fetcher 把 7 个库和表结构建好(含 ALTER 补列)。
    fetcher._IN_MIGRATION = True
    fetcher.init_db()

    # 普通连接而不是 file:...?mode=ro —— 只读模式打不开处于 WAL 状态、又还
    # 没有 -shm 伴生文件的库(SQLite 需要建 shm 才能读 WAL, 只读时建不了,
    # 报 "unable to open database file")。本脚本对源库只 SELECT。
    src_conn = sqlite3.connect(src)
    src_tables = {r[0] for r in src_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}

    dst = fetcher._conn()
    # 直接给路径而不是 file:...?mode=ro —— ATTACH 只在全局开了 URI 文件名时
    # 才认 file: 前缀。本脚本对 legacy 只 SELECT, 不写。
    # (此时连接上挂着 7 个库 + legacy = 8 个, 在 SQLite 默认上限 10 之内。)
    dst.execute("ATTACH DATABASE ? AS legacy", (src,))

    total_rows = 0
    t0 = time.time()
    print(f"\n{'表':22}{'源库':>12}{'新库(前)':>12}{'新库(后)':>12}  归属")
    for db_name, _fn, tables, _lazy in fetcher.DB_LAYOUT:
        for t in tables:
            n_src = _count(dst, t, "legacy") if t in src_tables else None
            n_before = _count(dst, t, db_name)
            if n_src is None:
                print(f"{t:22}{'—(源库没有)':>12}{n_before:>12}{n_before:>12}  {db_name}")
                continue
            if check_only:
                print(f"{t:22}{n_src:>12,}{n_before:>12,}{'(--check)':>12}  {db_name}")
                continue
            cols = [c for c in _cols(dst, t, db_name)
                    if c in set(_cols(dst, t, "legacy"))]
            collist = ", ".join(f"[{c}]" for c in cols)
            # INSERT OR REPLACE 让脚本可重复跑: 已经灌过的行原样覆盖。
            dst.execute(f"INSERT OR REPLACE INTO {db_name}.[{t}] ({collist}) "
                        f"SELECT {collist} FROM legacy.[{t}]")
            dst.commit()
            n_after = _count(dst, t, db_name)
            total_rows += n_src
            print(f"{t:22}{n_src:>12,}{n_before:>12,}{n_after:>12,}  {db_name}")

    # 源库里有、但不属于任何库的表 = 僵尸表, 明确列出来让人看见没被带过去。
    known = {t for _n, _f, tabs, _l in fetcher.DB_LAYOUT for t in tabs}
    zombies = sorted(src_tables - known - {"sqlite_sequence", "fund_nav"})
    if zombies:
        print(f"\n未迁移(不属于任何库, 视为历史残留): {', '.join(zombies)}")
        for z in zombies:
            print(f"    {z}: {_count(dst, z, 'legacy'):,} 行")

    if not check_only:
        # 派生索引: 净值按日期查(模拟盘的交易日历、回测的区间取数都要)。
        print("\n建 nav.idx_nav_date …", end="", flush=True)
        ti = time.time()
        dst.execute("CREATE INDEX IF NOT EXISTS nav.idx_nav_date "
                    "ON fund_nav_daily(date)")
        dst.commit()
        print(f" {time.time()-ti:.1f}s")

    dst.close()
    src_conn.close()

    if check_only:
        return
    print(f"\n✅ 迁移完成: {total_rows:,} 行, 用时 {(time.time()-t0)/60:.1f} 分钟")
    print("\n各库体量:")
    for name, fn, _t, _l in fetcher.DB_LAYOUT:
        p = fetcher.DB_PATH[name]
        if os.path.exists(p):
            print(f"  {fn:20}{os.path.getsize(p)/1048576:8.1f} MB")
    print(f"\n源库原样留着没动: {src}")
    print("  跑一遍 app / 回测确认没问题之后, 自己删掉它回收 "
          f"{os.path.getsize(src)/1048576:.0f}MB。")


if __name__ == "__main__":
    main()
