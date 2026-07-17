#!/usr/bin/env python3
"""一次性维护:删除库内债券型基金(用户不做债基,筛选已整体排除)。

删除类型前缀为「债券型」的基金(长债/中短债/混合一级/混合二级/利率债/
信用债)的净值/指标/持仓数据,并清空筛选结果缓存。日常管线的回填过滤已
排除该类型(见 fetcher.run_pipeline 的 is_bond 检查),删后不会再拉回来。

用法:
  python3 purge_bonds.py --dry-run   # 只看统计,不动数据
  python3 purge_bonds.py             # 实际执行
"""

import argparse

import fetcher


def main():
    parser = argparse.ArgumentParser(description="删除库内债券型基金")
    parser.add_argument("--dry-run", action="store_true", help="只统计不执行")
    args = parser.parse_args()

    fetcher.init_db()
    df = fetcher.fetch_fund_list()
    bond_codes = set(df[df["type"].map(fetcher.is_bond)]["code"].dropna())

    conn = fetcher._conn()
    stored = {r["code"] for r in conn.execute("SELECT DISTINCT code FROM fund_nav_daily")}
    # 模拟盘交易涉及的基金保留净值,否则持仓估值/图表会断(nav_series 只读库)。
    try:
        sim_codes = {r["code"] for r in conn.execute("SELECT DISTINCT code FROM sim_trades")}
    except Exception:
        sim_codes = set()
    kept = sorted(bond_codes & stored & sim_codes)
    to_del = sorted((bond_codes & stored) - sim_codes)
    n_filter, = conn.execute("SELECT COUNT(*) FROM filter_results").fetchone()
    print(f"债券型代码 {len(bond_codes)} 个 · 库内命中 {len(to_del) + len(kept)} 只 · "
          f"筛选缓存 {n_filter} 条")
    if kept:
        print(f"保留 {len(kept)} 只(模拟盘交易涉及): {', '.join(kept)}")

    if args.dry_run:
        conn.close()
        print("dry-run 结束,未修改任何数据")
        return

    params = [(c,) for c in to_del]
    for table in ("fund_nav_daily", "fund_nav_meta", "fund_sharpe", "fund_holdings"):
        conn.executemany(f"DELETE FROM {table} WHERE code=?", params)
    conn.execute("DELETE FROM filter_results")
    conn.commit()
    left, = conn.execute("SELECT COUNT(DISTINCT code) FROM fund_nav_daily").fetchone()
    conn.execute("VACUUM")
    conn.close()
    print(f"✅ 已删除 {len(to_del)} 只债券型基金,清空筛选缓存,库内剩余 {left} 只")


if __name__ == "__main__":
    main()
