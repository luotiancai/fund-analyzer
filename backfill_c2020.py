#!/usr/bin/env python3
"""一次性维护:只保留 C 类基金,并把其净值历史回填到 2020-01-01。

两步:
  ① 清理:删除库里所有非 C 类基金的净值/指标/持仓缓存(用户只买 C 类,
     日常管线此后也只回填 C 类,不会再把它们拉回来);
  ② 回填:对 2025 年前已存在的 C 类基金,每只 1 次 pingzhongdata 请求拉回
     2020-01-01 至今的完整序列(平均 ~145KB/只)。曾试过按日期段的历史净值
     接口只拉缺失区间,但它每只要串行翻 ~25 页且易被限流(实测 ~4 秒/只、
     失败率 ~15%),单请求整段反而快 25 倍、更稳,重叠部分是同一响应顺带的。

最后重算全部指标并 VACUUM 回收空间。

用法:
  python3 backfill_c2020.py --dry-run   # 只看统计,不动数据
  python3 backfill_c2020.py             # 实际执行
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_c2020")

# 首条净值在此日期之前的基金视为 2025 年前已存在,需要回填
EXISTED_CUTOFF = "2025-01-10"


def _stored_first_dates() -> dict:
    conn = fetcher._conn()
    rows = conn.execute(
        "SELECT code, MIN(date) AS first_d FROM fund_nav_daily GROUP BY code"
    ).fetchall()
    conn.close()
    return {r["code"]: r["first_d"] for r in rows}


def purge_non_c(non_c: list):
    conn = fetcher._conn()
    params = [(c,) for c in non_c]
    for table in ("fund_nav_daily", "fund_nav_meta", "fund_sharpe", "fund_holdings"):
        conn.executemany(f"DELETE FROM {table} WHERE code=?", params)
    conn.commit()
    conn.close()


def backfill_one(code: str, first_date: str) -> bool:
    """一次请求拉回 2020 至今整段并入库;重叠由 INSERT OR REPLACE 兜底。"""
    df = fetcher._fetch_nav_full(code)
    if df is None:
        return False
    if not df.empty:
        conn = fetcher._conn()
        fetcher._write_nav_rows(conn, code, df)
        conn.commit()
        conn.close()
    return True  # 空结果=该基金 2020 年后才成立,不算失败


def backfill(targets: dict, workers: int) -> list:
    """threaded 回填,返回失败的 code 列表。"""
    failed, done, t_last = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(backfill_one, c, d): c for c, d in targets.items()}
        for fut in as_completed(futures):
            done += 1
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if not ok:
                failed.append(futures[fut])
            if done == len(targets) or time.time() - t_last > 5:
                log.info("   回填 %d/%d(失败 %d)", done, len(targets), len(failed))
                t_last = time.time()
    return failed


def main():
    parser = argparse.ArgumentParser(description="清理非C类 + C类净值回填到2020")
    parser.add_argument("--dry-run", action="store_true", help="只统计不执行")
    parser.add_argument("--workers", type=int, default=fetcher.MAX_WORKERS)
    args = parser.parse_args()

    t0 = time.time()
    fetcher.init_db()

    list_df = fetcher.fetch_fund_list()
    names = list_df.dropna(subset=["code"]).drop_duplicates("code") \
        .set_index("code")["name"]
    c_codes = {c for c, n in names.items() if fetcher.is_c_class(n)}

    first_dates = _stored_first_dates()
    stored = set(first_dates)
    non_c = sorted(stored - c_codes)
    unknown = sorted(stored - set(names.index))  # 已不在基金列表里的(含在 non_c 内)
    targets = {c: d for c, d in first_dates.items()
               if c in c_codes and d <= EXISTED_CUTOFF}

    log.info("库内基金 %d · C类 %d · 待清理非C类 %d(其中已下架/不在列表 %d)"
             " · 待回填C类 %d",
             len(stored), len(stored & c_codes), len(non_c), len(unknown),
             len(targets))

    if args.dry_run:
        log.info("dry-run 结束,未修改任何数据")
        return

    log.info("① 清理非C类 %d 只…", len(non_c))
    purge_non_c(non_c)

    log.info("② 回填 %d 只C类基金 2020-01-01 ~ 首条已存日期…", len(targets))
    failed = backfill(targets, args.workers)
    if failed:
        log.info("   重试失败的 %d 只…", len(failed))
        failed = backfill({c: first_dates[c] for c in failed}, max(2, args.workers // 2))
    if failed:
        log.warning("   最终失败 %d 只: %s%s", len(failed),
                    ",".join(failed[:20]), " …" if len(failed) > 20 else "")

    log.info("③ 重算全部指标…")
    saved = fetcher.recompute_all()
    log.info("   写入 %d 只指标", saved)

    log.info("④ VACUUM 回收空间…")
    conn = fetcher._conn()
    conn.execute("VACUUM")
    conn.close()

    log.info("✅ 完成,总耗时 %.1f 分钟", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
