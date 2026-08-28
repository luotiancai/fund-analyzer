#!/usr/bin/env python3
"""一次性迁移:把 qvix_self_history 从 market.db 搬进独立的 qvix.db。

    python3 migrate_qvix_db.py                # 阶段一: 复制到 qvix.db(不动 market)
    python3 migrate_qvix_db.py --check        # 只比行数, 什么都不写
    python3 migrate_qvix_db.py --drop-legacy  # 阶段二: 从 market.db 删掉旧表

为什么要搬见 fetcher.DB_LAYOUT 上面那段(一句话: 上交所接口挡 GitHub runner
的境外 IP, QVIX 改由国内 VPS 产出, 而两个写入方不能共用一个库文件)。

**两个阶段必须分开跑, 中间夹着一次代码发布**, 顺序是为了让线上应用全程不断:

  1. 阶段一 + 推 qvix.db。新资产对**旧代码**完全隐形 —— 旧的 DB_LAYOUT 里
     没有 qvix 这个库, 应用根本不会去找它。此刻线上照常读 market.db 里那张。
  2. 推代码。应用重新部署, 新的 DB_LAYOUT 把 qvix.db 挂在 market 前面, 于是
     读到的是新库那张; market.db 里那张同名表被 ATTACH 顺序遮蔽, 只是死重量
     (先挂的先命中, 实测过)。这一步之所以安全, 全靠阶段一已经把资产传上去了。
  3. 阶段二 + 推 market.db。确认线上好用之后再删旧表, 清掉那份死重量。

反过来先推代码会开一个天窗: 新代码要下 qvix.db, 而那个资产还不存在。
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetcher   # noqa: E402

TABLE = "qvix_self_history"


def _count(conn, db, table):
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {db}.[{table}]").fetchone()[0]
    except sqlite3.Error:
        return None


def _has_table(path, table):
    if not os.path.exists(path):
        return False
    c = sqlite3.connect(path)
    try:
        return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                         (table,)).fetchone() is not None
    finally:
        c.close()


def main():
    check = "--check" in sys.argv
    drop = "--drop-legacy" in sys.argv
    market_path = fetcher.DB_PATH["market"]
    qvix_path = fetcher.DB_PATH["qvix"]

    legacy = _has_table(market_path, TABLE)
    print(f"market.db  {market_path}")
    print(f"  旧表还在: {'是' if legacy else '否(已经搬过了)'}")
    print(f"qvix.db    {qvix_path}")

    # init_db 会按新的 _DDL 把 qvix.db 建好(空表)。
    fetcher.init_db()
    conn = fetcher._conn()
    n_new = _count(conn, "qvix", TABLE)

    # market 库现在的 DB_LAYOUT 里已经没有这张表了, 得显式挂上去读旧的那份。
    conn.execute("ATTACH DATABASE ? AS legacy_market", (market_path,))
    n_old = _count(conn, "legacy_market", TABLE)
    print(f"\n行数  market.db(旧): {n_old}   qvix.db(新): {n_new}")

    if drop:
        if not legacy:
            print("\nmarket.db 里已经没有这张表了, 无事可做。")
            return
        if n_new is None or (n_old or 0) > n_new:
            raise SystemExit(
                f"\n❌ 拒绝删除: 新库 {n_new} 行 < 旧库 {n_old} 行。\n"
                "   先把阶段一跑通(不带 --drop-legacy)。")
        conn.close()
        print(f"\n新库 {n_new} 行 ≥ 旧库 {n_old} 行, 可以删。")
        c = sqlite3.connect(market_path)
        with c:
            c.execute(f"DROP TABLE {TABLE}")
        before = os.path.getsize(market_path)
        c.execute("VACUUM")     # 不 VACUUM 的话文件不会缩, 白传一份空洞
        c.close()
        after = os.path.getsize(market_path)
        print(f"✅ 已从 market.db 删除并 VACUUM: "
              f"{before/1048576:.2f}MB → {after/1048576:.2f}MB")
        print("\n下一步: python3 push_dbs.py market")
        return

    if check:
        print("\n--check: 什么都没写。")
        return

    if not legacy:
        print("\nmarket.db 里没有旧表, 不用复制。")
        return

    cols = [r[1] for r in conn.execute(f"PRAGMA qvix.table_info([{TABLE}])")]
    old_cols = [r[1] for r in conn.execute(f"PRAGMA legacy_market.table_info([{TABLE}])")]
    use = [c for c in cols if c in set(old_cols)]
    collist = ", ".join(f"[{c}]" for c in use)
    print(f"\n复制列: {', '.join(use)}")
    # INSERT OR REPLACE: 脚本可以反复跑, 已经灌过的行原样覆盖。
    conn.execute(f"INSERT OR REPLACE INTO qvix.[{TABLE}] ({collist}) "
                 f"SELECT {collist} FROM legacy_market.[{TABLE}]")
    conn.commit()
    n_after = _count(conn, "qvix", TABLE)
    conn.close()
    print(f"✅ qvix.db 现在 {n_after} 行 (原 {n_new} 行, 源 {n_old} 行)")
    print(f"   文件 {os.path.getsize(qvix_path)/1024:.0f}KB")
    print("\n下一步: python3 push_dbs.py qvix   然后推代码, 最后 --drop-legacy")


if __name__ == "__main__":
    main()
