#!/usr/bin/env python3
"""Daily batch: keep NAV history fresh and recompute Sharpe/drawdown for all funds.

The Streamlit app only *reads* the precomputed metrics, so all the slow network
work lives here and runs out of band.

Steps
  ① 拉取基金列表(1 次批量调用,带回全部基金的最新净值点)
  ② 历史回填:对还没有净值历史的基金,逐只下载近一年序列(慢,一次性)
  ③ 增量追加:把当日最新净值点追加到已有历史的基金(快,无逐只请求)
       缺口过大的基金转入全量重拉
  ④ 重算:用存好的净值,对全部基金重算夏普 + 最大回撤(纯 CPU,几秒)

用法
  手动:   python3 update_daily.py
  仅重算:  python3 update_daily.py --recompute-only
  cron:    0 18 * * 1-5  cd /path/to/fund-analyzer && python3 update_daily.py >> update.log 2>&1
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
log = logging.getLogger("update_daily")


def _backfill(codes, workers=fetcher.MAX_WORKERS):
    """Download full ~1y NAV history for `codes` (threaded)."""
    total, done, ok = len(codes), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetcher.fetch_nav, c): c for c in codes}
        for fut in as_completed(futures):
            done += 1
            try:
                if fut.result() is not None:
                    ok += 1
            except Exception:
                pass
            if done % 200 == 0 or done == total:
                log.info("   回填 %d/%d(成功 %d)", done, total, ok)
    return ok


def main():
    parser = argparse.ArgumentParser(description="基金净值每日跑批")
    parser.add_argument("--recompute-only", action="store_true",
                        help="跳过下载,只用已存净值重算夏普/回撤")
    args = parser.parse_args()

    t0 = time.time()
    fetcher.init_db()

    if not args.recompute_only:
        log.info("① 拉取基金列表(批量)…")
        list_df = fetcher.fetch_fund_list(force_refresh=True)
        all_codes = list_df["code"].dropna().unique().tolist()
        log.info("   共 %d 只基金", len(all_codes))

        have = fetcher.list_nav_codes()
        to_backfill = [c for c in all_codes if c not in have]
        log.info("② 历史回填:%d 只缺历史", len(to_backfill))
        if to_backfill:
            _backfill(to_backfill)

        log.info("③ 增量追加当日净值…")
        res = fetcher.append_latest_nav(list_df)
        log.info("   追加 %d,跳过 %d,缺口需重拉 %d",
                 res["updated"], res["skipped"], len(res["gapped"]))
        if res["gapped"]:
            _backfill(res["gapped"])

    log.info("④ 重算夏普 + 回撤…")
    saved = fetcher.recompute_all(
        progress_callback=lambda d, t: log.info("   重算 %d/%d", d, t)
    )
    log.info("   写入 %d 只指标", saved)

    log.info("✅ 完成,总耗时 %.1f 分钟", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
