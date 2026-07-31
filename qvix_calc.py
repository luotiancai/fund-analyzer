"""自算 QVIX(CBOE VIX 白皮书方法论)。两条独立路径,共享同一套核心公式
(_term_variance/K0选取/1-K²加权/30天插值),数据来源不同:

  ① compute_qvix() ——盘中实时值。用上交所50ETF期权的新浪实时报价
     (bid/ask)现算。fetcher.fetch_qvix_now() 用这个。
  ② compute_qvix_for_date() ——收盘后的历史/日线值。实时买卖价查不到
     历史,改用上交所官方期权风险指标接口(option_risk_indicator_sse,
     2015-02-09起可查)已经算好的隐含波动率反推 Black-Scholes 理论价格
     再代入同一套公式。backfill_qvix_history.py(一次性批量回算)和
     fetcher.update_qvix_self_daily()(每日跑批增量更新)都用这个。

起因:optbbs(1.optbbs.com,唯一免费QVIX源,akshare里所有QVIX变体背后
都是这一家)偶发返回整天空值、日线收盘价发布常年延迟到次日上午,而且
2026-03-23(上证单日-3.63%)那天发布的收盘值 42.16 根本不是30天期口径
——那天3月合约只剩2天到期,自算它的隐含波动率是 41.31,跟 42.16 基本
重合,而真正30天位置上的4月合约(正好剩30天)是 22.95。也就是说他们
那天的值塌成了"临到期合约的波动率"。已不再信任 optbbs,全面改用自算。

方法论: 近月+次近月期权,用 put-call parity 反推远期价格 F,K0 取不
超过 F 的最大行权价,以 K0 为界选虚值认沽(K<K0)+虚值认购(K>K0)+K0
处认购认沽均价,按 1/K² 加权求和,再按到期时间插值成 30 天期方差。
到期时间精确到秒(到期日15:00收盘 - 当前时刻,数学上等价于 CBOE 白皮书
按分钟分段累加的写法);无风险利率按 SHIBOR 期限结构对近月/次近月各自
的剩余天数线性插值(而不是不分期限统一用一个1年期利率),取不到 SHIBOR
时退回 fetcher.get_risk_free_rate()(1年期国债收益率)。

到期换月:按现行版白皮书口径,只要求"取夹住30天的两个到期日",近月一直
用到它到期为止,不做"剩余不足N天就整体跳到后两个月"的换月。这里踩过坑,
记下来免得再改回去:2009版白皮书(周合约2014年纳入VIX之前那版)确实写了
"近月剩不足8天就 roll 到第二、第三个合约月",中文圈讲复现VIX的教程抄的
基本都是那一版,本项目最初也照抄了(_MIN_ROLL_DAYS=7)。但CBOE后来把
规则改掉、并为此专门把周合约纳进VIX,原因正是老规则会退化:50ETF只有
月合约,近月一旦被跳过,次月往往已经是35~45天,30天目标落在它之前,
插值变外推、次月权重变负。恐慌时期限结构倒挂(短期波动率>长期),沿倒挂
斜率往回外推会把值压低——恰好在该往上冲的时候压低。实测2026-04~07共74
个交易日,对 optbbs 的平均绝对偏差 0.53→0.41、最大偏差 3.45→2.20;
2026-07-17(上证-3.05%)从 19.02 修正到 21.08(optbbs 22.47),那天恐慌
全在只剩5天的7月合约里(IV 26.30),被老规则整个丢掉了。

已知跟官方方法论的差距,均为有意识的取舍而非疏漏:
  - 风险利率用 SHIBOR(银行间同业拆借利率,含银行信用风险)而非 CBOE
    原版用的短期国债收益率(纯无风险)——境内没有可比的短期限国债收益
    率曲线,SHIBOR 是量化定价里通行的替代,概念上不完全等价但业内公认
    可用。
  - 50ETF期权没有周合约,凑不出"永远夹住30天"的一对。每月合约到期后的
    头3~4天,最近的合约还剩33~35天,30天目标落在它之前,这几天仍然是
    外推(次月权重约-0.15)。这个是没有周合约的硬限制、避不掉,但外推
    跨度只有3~5天、且不集中在恐慌日,跟上面那条老换月规则的性质不同。
  - 每次现算要拉近月+次月整月期权链的实时报价。新浪 hq.sinajs.cn/list=
    支持逗号拼多个 CON_OP_ 代码一次返回(2026-07 实测整月~50合约一个请求
    15KB 全回来),所以每月链只发 1 个批量请求(近月+次月共 2 个),不再
    是原来的近50连发——单请求容错高、不会因"瞬时几十并发"被反爬当爬虫
    掐掉,境外主机(GitHub Actions/Streamlit云)上尤其稳。返回不全时
    _fetch_chain 会在日志留下"只拿到 X/Y 个合约"的记录,可顺着排查。
算出来的数量级和走势应该跟官方 QVIX 一致,但不保证分毫不差——报价取中
还是取最新成交、零买价的裁剪时机等实现细节,不同实现之间本来就会有出入。
"""

import datetime as dt
import logging
import math
import re
import threading
import time
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import fetcher
import qvix_core

log = logging.getLogger(__name__)

_CST = ZoneInfo("Asia/Shanghai")
_UNDERLYING = "510050"


_SHIBOR_TENOR_DAYS = [
    ("O/N", 1), ("1W", 7), ("2W", 14), ("1M", 30),
    ("3M", 90), ("6M", 180), ("9M", 270), ("1Y", 365),
]


# ── 实时路径:全部委托给 qvix_core ──────────────────────────────────────────
# qvix_core 是这套算法唯一的实现(只依赖 requests, 见那边的模块说明)。这里
# 只做转发, **不要**在本文件里另写一份——今天就是因为"同一套算法两处实现"
# 的风险, 才把核心抽过去的; 在这里复制一份等于把那个风险又请回来。
# 历史路径(下方 compute_qvix_for_date)复用 qvix_core 的纯计算函数, 取数走
# 上交所官方接口, 所以仍留在本文件。

norm_cdf = qvix_core.norm_cdf
bs_price = qvix_core.bs_price
expiry_date_for_yymm = qvix_core.expiry_date_for_yymm
_rate_for_days = qvix_core.rate_for_days
_years_to_expiry = qvix_core.years_to_expiry
_candidate_pairs = qvix_core.candidate_pairs
_term_variance = qvix_core.term_variance
_expiry_candidates = qvix_core.expiry_candidates
_fetch_chain = qvix_core.fetch_chain


def _shibor_curve() -> Optional[list]:
    """今天最新一行 SHIBOR 各期限报价(年化小数)。见 qvix_core.shibor_curve。"""
    return qvix_core.shibor_curve()


def compute_qvix(as_of: Optional[dt.datetime] = None) -> Optional[tuple]:
    """现算 QVIX → (qvix, "HH:MM:SS");算不出来返回 None。

    as_of 用于把结果"冻结"在某一刻(中午休市11:30 / 收盘后15:00), 调用方见
    fetcher.qvix_phase()。SHIBOR 取不到时的兜底利率仍走本项目的
    fetcher.get_risk_free_rate()(1年期国债收益率, 带DB缓存); qvix_core 独立
    部署到云函数时用它自己的默认常数。"""
    try:
        fallback = fetcher.get_risk_free_rate()
    except Exception:
        fallback = 0.0113
    try:
        return qvix_core.compute_qvix(as_of=as_of, fallback_rate=fallback)
    except Exception as e:
        log.warning("self-computed QVIX failed: %s", e)
        return None


_CONTRACT_RE = re.compile(r"^510050([CP])(\d{4})([A-Z])(\d{5})$")


def spot_price_for_date(target_date: dt.date) -> Optional[float]:
    """指定交易日 510050 收盘价;当天数据还没发布/非交易日返回 None。"""
    try:
        df = fetcher.ak.fund_etf_hist_sina(symbol="sh510050")
    except Exception:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    row = df[df["date"] == target_date]
    return float(row["close"].iloc[0]) if not row.empty else None


def shibor_curve_for_date(target_date: dt.date) -> Optional[list]:
    """指定交易日的 SHIBOR 期限结构,格式同 _shibor_curve()。当天没有
    发布返回 None(调用方回退到 fetcher.get_risk_free_rate())。"""
    try:
        df = fetcher.ak.macro_china_shibor_all()
    except Exception:
        return None
    df["日期"] = pd.to_datetime(df["日期"]).dt.date
    row = df[df["日期"] == target_date]
    if row.empty:
        return None
    row = row.iloc[0]
    pts = []
    for tenor, days in _SHIBOR_TENOR_DAYS:
        col = f"{tenor}-定价"
        if col in row.index and pd.notna(row[col]):
            pts.append((float(days), float(row[col]) / 100.0))
    pts.sort()
    return pts if len(pts) >= 2 else None


def _listed_expiries(df, target_date: dt.date):
    """当天数据里实际出现过的到期月代码 → (到期日期, 剩余自然天数),
    按剩余天数升序;只看普通(非分红调整)合约。"""
    out = {}
    for cid in df["CONTRACT_ID"]:
        m = _CONTRACT_RE.match(cid)
        if not m or m.group(3) != "M":
            continue
        yymm = m.group(2)
        if yymm in out:
            continue
        try:
            expiry = expiry_date_for_yymm(yymm)
        except Exception:
            continue
        days = (expiry - target_date).days
        if days > 0:
            out[yymm] = (expiry, days)
    return sorted(out.items(), key=lambda kv: kv[1][1])


def _build_chain_from_risk_indicator(df, yymm: str, S: float, T: float, r: float):
    rows = []
    for _, row in df.iterrows():
        m = _CONTRACT_RE.match(row["CONTRACT_ID"])
        if not m or m.group(3) != "M" or m.group(2) != yymm:
            continue
        sigma = row["IMPLC_VOLATLTY"]
        if pd.isna(sigma):
            continue
        strike = int(m.group(4)) / 1000.0
        price = bs_price(m.group(1), S, strike, T, r, float(sigma))
        if price is None or price <= 0:
            continue
        rows.append({"kind": m.group(1), "strike": strike, "bid": price, "mid": price})
    # 返回 list of dict 而不是 DataFrame:qvix_core.term_variance 吃的是这个
    # 结构(它不依赖 pandas)。两条路径共用同一个方差函数, 才谈得上"一份实现"。
    return rows


def _fetch_risk_indicator_with_retry(date_str: str, attempts: int = 3):
    for i in range(attempts):
        try:
            return fetcher.ak.option_risk_indicator_sse(date=date_str)
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(1.0)


def compute_qvix_for_date(target_date: dt.date, spot: Optional[float] = None,
                           shibor_curve: Optional[list] = None) -> tuple:
    """算某个收盘后交易日的QVIX(上交所官方期权风险指标反推,不依赖
    optbbs)。spot/shibor_curve 不传时现查当天的(单天用;批量回算时
    调用方应该一次性拉整段历史自己传,不然每天都要重新拉一遍全history)。
    失败返回 (None, 原因字符串)。"""
    if spot is None:
        spot = spot_price_for_date(target_date)
    if spot is None:
        return None, "拿不到50ETF当日收盘价(可能还没发布/非交易日)"
    if shibor_curve is None:
        shibor_curve = shibor_curve_for_date(target_date)

    date_str = target_date.strftime("%Y%m%d")
    try:
        df = _fetch_risk_indicator_with_retry(date_str)
    except Exception as e:
        return None, f"接口失败: {e}"
    if df is None or df.empty:
        return None, "当天无数据(非交易日/上市前/尚未发布)"
    df = df[df["CONTRACT_ID"].str.startswith("510050")]
    if df.empty:
        return None, "当天无50ETF期权数据"

    pairs = _candidate_pairs(_listed_expiries(df, target_date))
    if not pairs:
        return None, "找不到满足条件的近月/次近月(合约月份不足)"

    last_err = "方差算不出来(合约或IV数据不足)"
    for (near_ms, (_, near_days)), (next_ms, (_, next_days)) in pairs:
        T1, T2 = near_days / 365.0, next_days / 365.0
        r1 = _rate_for_days(shibor_curve, near_days, 0.02)
        r2 = _rate_for_days(shibor_curve, next_days, 0.02)

        near_chain = _build_chain_from_risk_indicator(df, near_ms, spot, T1, r1)
        next_chain = _build_chain_from_risk_indicator(df, next_ms, spot, T2, r2)
        near = _term_variance(near_chain, r1, T1)
        nxt = _term_variance(next_chain, r2, T2)
        if near is None or nxt is None:
            continue
        sigma1, _, _ = near
        sigma2, _, _ = nxt

        n30 = 30.0
        w1 = (next_days - n30) / (next_days - near_days)
        w2 = (n30 - near_days) / (next_days - near_days)
        sigma2_30 = (T1 * sigma1 * w1 + T2 * sigma2 * w2) * (365.0 / n30)
        if sigma2_30 <= 0:
            last_err = "插值方差非正"
            continue
        vix = 100.0 * (sigma2_30 ** 0.5)
        if not (1.0 < vix < 150.0):
            last_err = f"结果 {vix:.2f} 超出合理区间,判为脏数据"
            continue
        return round(vix, 2), None
    return None, last_err
