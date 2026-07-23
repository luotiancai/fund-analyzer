#!/usr/bin/env python3
"""Daily batch: keep NAV history fresh and recompute Sharpe/drawdown for all funds.

The Streamlit app only *reads* the precomputed metrics, so all the slow network
work lives here and runs out of band. The same pipeline is also exposed as the
in-app「🔄 更新数据」button via fetcher.run_pipeline().

Pipeline (fetcher.run_pipeline):
  ① 拉取基金列表(1 次批量调用,带回全部基金的最新净值点)
  ② 历史回填:对还没有净值历史的 C 类基金,每只 1 次请求下载 2020-01-01 至今
     的序列(基本一次性;非 C 类不存净值,见 fetcher.is_c_class)
  ③ 增量补净值:只差一个交易日的基金直接追加基金列表带回的当日净值点(零请求);
     缺口更大的用天天基金历史净值接口按日期段拉取(每只一次几 KB 的请求)
  ④ 重算:用存好的净值对全部基金重算夏普 + 最大回撤(纯 CPU,几秒)

用法
  手动:   python3 update_daily.py
  仅重算:  python3 update_daily.py --recompute-only
  cron:    0 18 * * 1-5  cd /path/to/fund-analyzer && python3 update_daily.py >> update.log 2>&1
"""

import argparse
import logging
import time

import fetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("update_daily")

# Throttle per-phase logging so a 20k-fund phase doesn't spam one line per 50.
_last_log = {}


def _log_progress(phase, done, total):
    if done == total or time.time() - _last_log.get(phase, 0) > 2:
        log.info("   %s %d/%d", phase, done, total)
        _last_log[phase] = time.time()


def main():
    parser = argparse.ArgumentParser(description="基金净值每日跑批")
    parser.add_argument("--recompute-only", action="store_true",
                        help="跳过下载,只用已存净值重算夏普/回撤")
    args = parser.parse_args()

    t0 = time.time()
    fetcher.init_db()

    if args.recompute_only:
        log.info("仅重算:用已存净值重算夏普 + 回撤…")
        saved = fetcher.recompute_all(progress_callback=lambda d, t: _log_progress("重算", d, t))
        log.info("   写入 %d 只指标", saved)
    else:
        summary = fetcher.run_pipeline(progress=_log_progress)
        log.info("基金 %d · 回填 %d · 当日追加 %d · 补缺口 %d（失败 %d）· 重算 %d",
                 summary["funds"], summary["backfilled"], summary["appended"],
                 summary["patched"], summary["failed"], summary["recomputed"])

    # ── 指数刷新 + QVIX自算 + 动态恐慌阈值 ──────────────────────────────────
    # 上证指数刷新照旧(app 侧无过期时间,只在这里force_refresh才变)。QVIX
    # 不再用 optbbs——那是免费QVIX源里唯一一家,却发布延迟常年到次日上午、
    # 还偶发整天返回空值,而且历史极端行情日交叉验证对不上(见
    # qvix_calc.py 顶部说明),已不再信任。改用
    # fetcher.update_qvix_self_daily():上交所官方期权风险指标现算,写入
    # 独立的 qvix_self_history 表,阈值也是照这份自算历史现算的。
    log.info("刷新指数缓存(上证)+ 自算QVIX…")
    try:
        sse = fetcher.fetch_sse_daily(force_refresh=True)
        vix, note = fetcher.update_qvix_self_daily()
        hist = fetcher.load_qvix_self_history()
        thr = None
        if hist is not None:
            row = hist.dropna(subset=["threshold"])
            if not row.empty:
                thr = float(row["threshold"].iloc[-1])
        if sse is not None and vix is not None and thr is not None:
            s_last = sse.iloc[-1]
            triggered = vix > thr
            log.info("   上证 %s 收 %.0f(%+.2f%%) · QVIX(自算) %.2f · 恐慌阈值(3年95分位) %.2f%s",
                     s_last["date"], s_last["close"], s_last["pct"],
                     vix, thr,
                     " · 🔔 B点触发(QVIX破阈值)" if triggered else "")
        elif vix is None:
            log.warning("   QVIX 自算未成功: %s", note)
    except Exception as e:
        log.warning("   指数刷新失败(不影响基金跑批): %s", e)

    log.info("✅ 完成,总耗时 %.1f 分钟", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
