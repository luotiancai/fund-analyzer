"""自算 QVIX(CBOE VIX 白皮书方法论,用上交所50ETF期权实时行情现算)。

只是 fetcher.fetch_qvix_now() 的备用路径:optbbs 分钟接口(1.optbbs.com)
是免费公开数据里唯一的现成 QVIX 源,挂掉/返回空值时没有第二家可切换
(akshare 里所有 QVIX 变体——50/300/500ETF等9个——背后都是同一个站)。
这里改为自己用期权真实报价按标准公式现算,不依赖任何第三方已经算好
的指标。

方法论: 近月+次近月期权,用 put-call parity 反推远期价格 F,K0 取不
超过 F 的最大行权价,以 K0 为界选虚值认沽(K<K0)+虚值认购(K>K0)+K0
处认购认沽均价,按 1/K² 加权求和,再按到期时间插值成 30 天期方差。
用日历天数近似替代 CBOE 原版的分钟精度(误差可忽略);无风险利率复用
fetcher.get_risk_free_rate()(1年期国债收益率,不按期限精确插值,近似)。
算出来的数量级和走势应该跟官方 QVIX 一致,但不保证分毫不差——不同
实现细节(报价取中还是取最新成交、零买价的裁剪时机等)本身就会有出入。
"""

import datetime as dt
import logging
import threading
import time
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import fetcher

log = logging.getLogger(__name__)

_CST = ZoneInfo("Asia/Shanghai")
_UNDERLYING = "510050"


def _parallel_fetch(items, fn, timeout=8):
    """对 items 并发跑 fn(item),daemon 线程、共享超时预算,超时的直接丢弃。
    不用 concurrent.futures.ThreadPoolExecutor——它的 worker 线程不是
    daemon,会被 atexit 钩子 join,一次性脚本(GitHub Actions)进程退出时
    被晾着的慢线程拖住(同 fetcher._fetch_with_timeout 的理由)。返回收集
    到的结果列表,顺序不保证,超时/失败的直接跳过。"""
    results = []
    lock = threading.Lock()

    def _run(item):
        try:
            r = fn(item)
        except Exception:
            return
        if r is not None:
            with lock:
                results.append(r)

    threads = [threading.Thread(target=_run, args=(it,), daemon=True)
               for it in items]
    for t in threads:
        t.start()
    deadline = time.time() + timeout
    for t in threads:
        remaining = deadline - time.time()
        if remaining > 0:
            t.join(remaining)
    return results


def _next_month_str(base: dt.date, offset: int) -> str:
    y, m = base.year, base.month + offset
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f"{y:04d}{m:02d}"


def _two_expiries(today: dt.date):
    """选近月/次近月到期合约:(月份代码, 到期日, 剩余自然天数) × 2。

    近月剩余不足7天时跳到下两个月,避开临近到期的报价噪音(标准 VIX
    方法论遇到近月太近时的处理方式)。50ETF期权当前只挂近两个月+两个
    季月,月份代码探测 6 个月足够覆盖近月/次近月这两个连续月份。
    """
    candidates = []
    for off in range(6):
        ms = _next_month_str(today, off)
        try:
            expiry_date, days = fetcher.ak.option_sse_expire_day_sina(
                trade_date=ms, symbol="50ETF")
        except Exception:
            continue
        if expiry_date is None or days is None or days <= 0:
            continue
        candidates.append((ms, expiry_date, days))
        if len(candidates) >= 4:
            break

    if len(candidates) < 2:
        return None
    near, nxt = candidates[0], candidates[1]
    if near[2] < 7 and len(candidates) >= 3:
        near, nxt = candidates[1], candidates[2]
    return near, nxt


def _fetch_chain(month_str: str) -> pd.DataFrame:
    """近月/次近月看涨+看跌合约代码列表 → 并发拉实时报价。"""
    codes = []
    for label, kind in (("看涨期权", "C"), ("看跌期权", "P")):
        try:
            df = fetcher.ak.option_sse_codes_sina(
                symbol=label, trade_date=month_str, underlying=_UNDERLYING)
        except Exception:
            continue
        for c in df["期权代码"]:
            codes.append((c, kind))

    def _fetch_one(item):
        code, kind = item
        d = fetcher.ak.option_sse_spot_price_sina(symbol=code)
        vals = dict(zip(d["字段"], d["值"]))
        bid = float(vals["买价"])
        ask = float(vals["卖价"])
        last = float(vals["最新价"])
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        return {"kind": kind, "strike": float(vals["行权价"]),
                "bid": bid, "mid": mid}

    rows = _parallel_fetch(codes, _fetch_one, timeout=8)
    return pd.DataFrame(rows)


def _term_variance(chain: pd.DataFrame, r: float, T: float):
    """单个到期日的方差贡献(CBOE VIX 白皮书公式)。返回 (sigma2, F, K0),
    合约不足/报价缺失时返回 None。"""
    if chain.empty:
        return None
    calls = chain[chain["kind"] == "C"].set_index("strike")["mid"]
    puts = chain[chain["kind"] == "P"].set_index("strike")["mid"]
    bid_calls = chain[chain["kind"] == "C"].set_index("strike")["bid"]
    bid_puts = chain[chain["kind"] == "P"].set_index("strike")["bid"]

    common = sorted(set(calls.index) & set(puts.index))
    if len(common) < 3:
        return None

    # 远期价格:C-P 差最小的那个行权价上用 put-call parity 反推。
    k_f = min(common, key=lambda k: abs(calls[k] - puts[k]))
    F = k_f + np.exp(r * T) * (calls[k_f] - puts[k_f])
    k0_candidates = [k for k in common if k <= F]
    if not k0_candidates:
        return None
    K0 = max(k0_candidates)

    def _trim(strikes, bid_table):
        """扫描远离 K0 的方向,遇到连续两个零买价行权价就截断(CBOE 规则)。"""
        out, zero_run = [], 0
        for k in strikes:
            bid = bid_table.get(k, 0)
            if bid <= 0:
                zero_run += 1
                if zero_run >= 2:
                    break
                continue
            zero_run = 0
            out.append(k)
        return out

    put_side = _trim(sorted((k for k in puts.index if k < K0), reverse=True),
                     bid_puts)
    call_side = _trim(sorted(k for k in calls.index if k > K0), bid_calls)
    selected = sorted(set(put_side) | {K0} | set(call_side))
    if len(selected) < 3:
        return None

    def _price_at(k):
        if k < K0:
            return puts[k]
        if k > K0:
            return calls[k]
        vals = [t[k] for t in (calls, puts) if k in t.index]
        return sum(vals) / len(vals)

    total = 0.0
    n = len(selected)
    for i, k in enumerate(selected):
        if i == 0:
            dk = selected[1] - selected[0]
        elif i == n - 1:
            dk = selected[-1] - selected[-2]
        else:
            dk = (selected[i + 1] - selected[i - 1]) / 2
        total += (dk / k ** 2) * _price_at(k)

    sigma2 = (2 / T) * np.exp(r * T) * total - (1 / T) * (F / K0 - 1) ** 2
    return sigma2, F, K0


def compute_qvix() -> Optional[tuple]:
    """现算当前 QVIX。失败(到期月探测失败/合约或报价拿不全等)返回 None,
    由调用方(fetcher.fetch_qvix_now)决定要不要继续找别的路子。
    返回 (qvix, "HH:MM:SS")。"""
    now = dt.datetime.now(_CST)
    try:
        expiries = _two_expiries(now.date())
        if expiries is None:
            return None
        (near_ms, _, near_days), (next_ms, _, next_days) = expiries

        r = fetcher.get_risk_free_rate()
        T1, T2 = near_days / 365.0, next_days / 365.0

        near_chain = _fetch_chain(near_ms)
        next_chain = _fetch_chain(next_ms)
        near = _term_variance(near_chain, r, T1)
        nxt = _term_variance(next_chain, r, T2)
        if near is None or nxt is None:
            return None
        sigma1, _, _ = near
        sigma2, _, _ = nxt

        n30 = 30.0
        w1 = (next_days - n30) / (next_days - near_days)
        w2 = (n30 - near_days) / (next_days - near_days)
        sigma2_30 = (T1 * sigma1 * w1 + T2 * sigma2 * w2) * (365.0 / n30)
        if sigma2_30 <= 0:
            return None
        vix = float(100.0 * np.sqrt(sigma2_30))
        # 粗粒度合理性校验:历史 QVIX 大致落在个位数到三位数以内,离谱的
        # 结果多半是报价缺失/行权价选取出错,宁可返回 None 也不展示假数。
        if not (1.0 < vix < 150.0):
            return None
        return round(vix, 2), now.strftime("%H:%M:%S")
    except Exception as e:
        log.debug("self-computed QVIX failed: %s", e)
        return None
