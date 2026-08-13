#!/usr/bin/env python3
"""回填/更新 qvix_self_history 的 skew 列(期权认沽-认购隐含波动率偏度)。

    python3 backfill_skew.py            # 补所有还没算的交易日
    python3 backfill_skew.py --limit 50 # 只补50天(试跑)
    python3 backfill_skew.py --redo     # 已有值的也重算

**它解决的是 QVIX 不分方向的问题。** QVIX 是隐含波动率, 只回答"预期波动大不
大", 暴跌和暴涨都会把它推高 —— 实测 QVIX 日变化与上证日涨跌的相关系数只有
-0.19, 而 97 个信号日里 55% 发生在**上涨日**。想区分"跌出来的恐慌"和"涨出来
的亢奋", 得看同一天认沽和认购的 IV 谁更贵:

    skew = 虚值认沽IV − 虚值认购IV   (取 |delta|≈0.25 的那两只, 单位: 波动率点)

  · skew 为正 = 资金在抢买下跌保护 → 恐慌
  · skew 为负 = 资金在抢买上涨门票 → 亢奋/追涨

实测(83个信号日): skew 与上证当日涨跌相关系数 **-0.463**, 判别力是 QVIX 自身
(-0.19)的 2.4 倍。skew>0 的日子里 68% 是下跌日; skew<0 的日子里只有 33%。

⚠️ 用 delta 而不是行权价定"虚值程度": 标的价格会漂移, 同一个行权价今天虚值
10%、明天可能只虚值 3%, 而 delta 是自适应的, 横向可比。
⚠️ 国内 50ETF 期权的 skew **中位数是负的**(-3.30, 认购IV 平时反而更高), 跟
美股"put skew 常年为正"的经验相反 —— 可能跟备兑开仓、雪球产品、散户偏好买
call 博反弹有关。所以绝对值不能照搬美股, 阈值要按本土分布来定。

数据源 ak.option_risk_indicator_sse(date=) —— 与 QVIX 自算同源(上交所官方
期权风险指标), 可按日期回查到 2015 年。这里只补 qvix_self_history 里已有的
交易日(2018 起, 2015-2017 期权流动性薄已整段剔除, 见 fetcher)。
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetcher   # noqa: E402

log = fetcher.logger


def skew_on(date_str: str, tries: int = 3):
    """某个交易日的 25-delta skew(波动率点)。取不到返回 None。"""
    import akshare as ak
    d = None
    for i in range(tries):
        try:
            d = ak.option_risk_indicator_sse(date=date_str.replace("-", ""))
            break
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(2.0 * (i + 1))
    if d is None or d.empty:
        return None
    d = d[d["CONTRACT_SYMBOL"].str.startswith("50ETF")].copy()
    if d.empty:
        return None
    d["IV"] = pd.to_numeric(d["IMPLC_VOLATLTY"], errors="coerce")
    d["DELTA"] = pd.to_numeric(d["DELTA_VALUE"], errors="coerce")
    # 近月合约: 合约代码里 [CP] 后面四位是到期年月, 取最小的那个
    d["ym"] = d["CONTRACT_ID"].str.extract(r"[CP](\d{4})")[0]
    if d["ym"].isna().all():
        return None
    near = d[d["ym"] == d["ym"].min()]
    calls = near[near["CONTRACT_SYMBOL"].str.contains("购")].dropna(
        subset=["IV", "DELTA"])
    puts = near[near["CONTRACT_SYMBOL"].str.contains("沽")].dropna(
        subset=["IV", "DELTA"])
    if calls.empty or puts.empty:
        return None
    c = calls.iloc[(calls["DELTA"] - 0.25).abs().argsort()[:1]]
    p = puts.iloc[(puts["DELTA"] + 0.25).abs().argsort()[:1]]
    return round((float(p["IV"].iloc[0]) - float(c["IV"].iloc[0])) * 100, 2)


def main():
    ap = argparse.ArgumentParser(description="回填 QVIX 表的 skew 列")
    ap.add_argument("--limit", type=int, default=None, help="只补前 N 天")
    ap.add_argument("--redo", action="store_true", help="已有值的也重算")
    args = ap.parse_args()

    conn = fetcher._conn()
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(qvix_self_history)")]
    if "skew" not in cols:
        conn.execute("ALTER TABLE strategy_placeholder_noop" if False else
                     "ALTER TABLE qvix_self_history ADD COLUMN skew REAL")
        conn.commit()
        log.info("已给 qvix_self_history 加 skew 列")

    where = "" if args.redo else " WHERE skew IS NULL"
    days = [r["date"] for r in conn.execute(
        f"SELECT date FROM qvix_self_history{where} ORDER BY date")]
    if args.limit:
        days = days[:args.limit]
    log.info("待补 %d 个交易日", len(days))

    ok = fail = 0
    t0 = time.time()
    for i, d in enumerate(days, 1):
        s = skew_on(d)
        if s is None:
            fail += 1
        else:
            conn.execute("UPDATE qvix_self_history SET skew=? WHERE date=?",
                         (s, d))
            conn.commit()          # 逐日提交: 中断了也不白跑
            ok += 1
        if i % 50 == 0 or i == len(days):
            el = time.time() - t0
            log.info("  %d/%d  成功%d 失败%d  用时%.0fs  预计剩余%.0fs",
                     i, len(days), ok, fail, el, el / i * (len(days) - i))
        time.sleep(0.35)

    n = conn.execute(
        "SELECT COUNT(*) c FROM qvix_self_history WHERE skew IS NOT NULL"
    ).fetchone()["c"]
    log.info("完成: 本次成功 %d / 失败 %d; 全表已有 skew 的共 %d 天", ok, fail, n)


if __name__ == "__main__":
    main()
