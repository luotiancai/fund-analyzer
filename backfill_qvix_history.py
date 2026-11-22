#!/usr/bin/env python3
"""用上交所逐合约收盘价回填日线 QVIX，同时更新滚动阈值。

2015–2017 年合约较少，按现有策略口径从 2018 年开始保留历史。
例如：python3 backfill_qvix_history.py --start 2024-01-01
"""

import argparse
import datetime as dt
import logging

import fetcher
import qvix_calc
from qvix_core import CST


log = logging.getLogger(__name__)
_MIN_DATE = dt.date(2018, 1, 1)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="用上交所期权收盘价重算历史 QVIX")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=_MIN_DATE,
                        help="起始日 YYYY-MM-DD，最早 2018-01-01")
    parser.add_argument("--end", type=dt.date.fromisoformat,
                        help="截止日 YYYY-MM-DD，默认最近已收盘日")
    parser.add_argument("--delay", type=float, default=0.15, help="交易日合约名单查询间隔秒数")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay 不能为负")

    trading_days = fetcher._trading_days()
    if not trading_days:
        raise SystemExit("拿不到交易日历")
    now = dt.datetime.now(CST)
    latest = now.date() if now.time() >= dt.time(15) else now.date() - dt.timedelta(days=1)
    start, end = max(args.start, _MIN_DATE), min(args.end or latest, latest)
    dates = sorted(day for day in trading_days if start <= day <= end)
    if not dates:
        raise SystemExit("指定区间内没有已收盘的交易日")

    data = qvix_calc.compute_qvix_history(dates, trading_days=trading_days, delay=args.delay)
    fetcher.init_db()
    fetcher.save_qvix_self_history([{"date": day.isoformat(), "qvix": value, "note": note}
                                   for day, (value, note) in data.items()], trading_days=trading_days)
    for day, (value, note) in data.items():
        if value is None:
            log.warning("%s：%s", day, note)
    valid = sum(value is not None for value, _ in data.values())
    broken = sum(qvix_calc.is_chain_broken(note) for _, note in data.values())
    log.info("已更新 QVIX 及阈值：%d 天，%d 天链失效", valid, broken)
    if valid + broken != len(data):
        raise SystemExit(f"另有 {len(data) - valid - broken} 天未算出，已有有效记录保留，可重试")


if __name__ == "__main__":
    main()
