#!/usr/bin/env python3
"""一次性批量重算近10年QVIX历史,不依赖optbbs——用上交所官方期权风险
指标接口(option_risk_indicator_sse,官方数据,2015-02-09起可查)反推
每个历史交易日的隐含波动率指数,写入独立的 qvix_self_history 表(跟
optbbs 的 index_daily_cache 互不覆盖,方便并排比对)。

起因:optbbs(唯一免费QVIX源)偶发返回整天空值、日线收盘价发布常年
延迟到次日上午,而且某个历史极端行情日(2026-03-23,上证单日-3.63%)
的官方发布收盘值(42.16)用标准CBOE公式配合上交所官方IV反推价格交叉
验证怎么都对不上(两条独立路径都落在~23),怀疑那天的发布值本身有误。
干脆自己按标准方法论把近10年历史全部重算一遍,全面不再依赖 optbbs。

实际计算逻辑在 qvix_calc.compute_qvix_for_date() 里(跟每日跑批增量
更新 fetcher.update_qvix_self_daily() 共用同一套实现),这里只是批量
循环 + 提前一次性拉好整段历史(50ETF价格、SHIBOR)避免每天重复拉取。

用法:
  python3 backfill_qvix_history.py                # 近10年
  python3 backfill_qvix_history.py --years 5      # 近5年
  python3 backfill_qvix_history.py --start 2020-01-01
"""

import argparse
import datetime as dt
import logging
import time
from zoneinfo import ZoneInfo

import pandas as pd

import fetcher
import qvix_calc as qc

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backfill_qvix_history")

_CST = ZoneInfo("Asia/Shanghai")


def main():
    parser = argparse.ArgumentParser(description="批量重算历史QVIX(上交所官方数据源)")
    parser.add_argument("--years", type=int, default=10, help="回算最近几年,默认10")
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD,优先于 --years")
    parser.add_argument("--delay", type=float, default=0.15, help="每个交易日请求间隔秒数,别对官方接口太猛")
    args = parser.parse_args()

    fetcher.init_db()
    today = dt.datetime.now(_CST).date()
    start = (dt.datetime.strptime(args.start, "%Y-%m-%d").date()
             if args.start else today.replace(year=today.year - args.years))

    log.info("拉交易日历…")
    cal = fetcher.ak.tool_trade_date_hist_sina()
    cal["trade_date"] = pd.to_datetime(cal["trade_date"]).dt.date
    dates = sorted(d for d in cal["trade_date"] if start <= d <= today)
    log.info("共 %d 个交易日(%s ~ %s)", len(dates), dates[0], dates[-1])

    log.info("拉50ETF历史价格…")
    spot_df = fetcher.ak.fund_etf_hist_sina(symbol="sh510050")
    spot_df["date"] = pd.to_datetime(spot_df["date"]).dt.date
    spot_map = dict(zip(spot_df["date"], spot_df["close"]))

    log.info("拉SHIBOR历史利率…")
    shibor = fetcher.ak.macro_china_shibor_all()
    shibor["日期"] = pd.to_datetime(shibor["日期"]).dt.date
    shibor_map = {}
    for _, row in shibor.iterrows():
        pts = []
        for tenor, days in qc._SHIBOR_TENOR_DAYS:
            col = f"{tenor}-定价"
            if col in row.index and pd.notna(row[col]):
                pts.append((float(days), float(row[col]) / 100.0))
        pts.sort()
        if len(pts) >= 2:
            shibor_map[row["日期"]] = pts

    results = []
    ok = fail = 0
    t0 = time.time()
    for i, d in enumerate(dates):
        spot = spot_map.get(d)
        curve = shibor_map.get(d)
        if spot is None:
            results.append({"date": d.isoformat(), "qvix": None, "note": "无50ETF当日收盘价"})
            fail += 1
        else:
            vix, err = qc.compute_qvix_for_date(d, spot=spot, shibor_curve=curve)
            if vix is None:
                results.append({"date": d.isoformat(), "qvix": None, "note": err})
                fail += 1
            else:
                results.append({"date": d.isoformat(), "qvix": vix, "note": None})
                ok += 1
        if (i + 1) % 100 == 0 or i == len(dates) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dates) - i - 1)
            log.info("  %d/%d(成功%d 失败%d),已耗时 %.1f 分钟,预计还需 %.1f 分钟",
                     i + 1, len(dates), ok, fail, elapsed / 60, eta / 60)
        time.sleep(args.delay)

    fetcher.save_qvix_self_history(results)

    log.info("重算滚动3年95分位恐慌阈值…")
    hist = fetcher.load_qvix_self_history()
    hist = hist.sort_values("date").reset_index(drop=True)
    hist["threshold"] = hist["qvix"].rolling(720, min_periods=240).quantile(0.95)
    fetcher.save_qvix_self_threshold(hist["date"].tolist(), hist["threshold"].tolist())

    log.info("✅ 完成,写入 %d 条(成功%d 失败%d),总耗时 %.1f 分钟",
             len(results), ok, fail, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
