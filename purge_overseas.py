#!/usr/bin/env python3
"""一次性维护:删除库内海外股基金(QDII 额度限购,实际买不进有意义的仓位)。

删除 fetcher.OVERSEAS_EQUITY_TYPES 三类(指数型-海外股票 / QDII-普通股票 /
QDII-混合偏股)的净值/指标/持仓数据,并清空筛选结果缓存。日常管线的回填
过滤已排除这些类型(见 fetcher.run_daily_pipeline),删后不会再拉回来。

用法:
  python3 purge_overseas.py --dry-run   # 只看统计,不动数据
  python3 purge_overseas.py             # 实际执行
"""

import argparse

import fetcher


def main():
    parser = argparse.ArgumentParser(description="删除库内海外股基金")
    parser.add_argument("--dry-run", action="store_true", help="只统计不执行")
    args = parser.parse_args()

    fetcher.init_db()
    df = fetcher.fetch_fund_list()
    ov_codes = set(df[df["type"].isin(fetcher.OVERSEAS_EQUITY_TYPES)]["code"].dropna())

    conn = fetcher._conn()
    stored = {r["code"] for r in conn.execute("SELECT DISTINCT code FROM fund_nav_daily")}
    to_del = sorted(ov_codes & stored)
    n_filter, = conn.execute("SELECT COUNT(*) FROM filter_results").fetchone()
    print(f"海外股代码 {len(ov_codes)} 个 · 库内命中 {len(to_del)} 只 · 筛选缓存 {n_filter} 条")

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
    print(f"✅ 已删除 {len(to_del)} 只海外股,清空筛选缓存,库内剩余 {left} 只")


if __name__ == "__main__":
    main()
