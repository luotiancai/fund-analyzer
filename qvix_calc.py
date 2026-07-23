"""自算 QVIX(CBOE VIX 白皮书方法论,用上交所50ETF期权实时行情现算)。

只是 fetcher.fetch_qvix_now() 的备用路径:optbbs 分钟接口(1.optbbs.com)
是免费公开数据里唯一的现成 QVIX 源,挂掉/返回空值时没有第二家可切换
(akshare 里所有 QVIX 变体——50/300/500ETF等9个——背后都是同一个站)。
这里改为自己用期权真实报价按标准公式现算,不依赖任何第三方已经算好
的指标。

方法论: 近月+次近月期权,用 put-call parity 反推远期价格 F,K0 取不
超过 F 的最大行权价,以 K0 为界选虚值认沽(K<K0)+虚值认购(K>K0)+K0
处认购认沽均价,按 1/K² 加权求和,再按到期时间插值成 30 天期方差。
到期时间精确到秒(到期日15:00收盘 - 当前时刻,数学上等价于 CBOE 白皮书
按分钟分段累加的写法);无风险利率按 SHIBOR 期限结构对近月/次近月各自
的剩余天数线性插值(而不是不分期限统一用一个1年期利率),取不到 SHIBOR
时退回 fetcher.get_risk_free_rate()(1年期国债收益率)。

已知跟官方方法论的差距,均为有意识的取舍而非疏漏:
  - 风险利率用 SHIBOR(银行间同业拆借利率,含银行信用风险)而非 CBOE
    原版用的短期国债收益率(纯无风险)——境内没有可比的短期限国债收益
    率曲线,SHIBOR 是量化定价里通行的替代,概念上不完全等价但业内公认
    可用。
  - 50ETF期权只有月度合约、没有周合约,每月合约到期换月后的头几天,
    目标的30天期限会落在近月合约到期日之前(即 N30 < N1),标准插值
    公式此时退化成外推而非真正的插值,数学上仍然成立(官方 QVIX 遇到
    同样情况大概率也是同样处理),只是不如"30天被近月/次近月夹住"时
    直觉。
  - 每次现算要发起近 50 个单合约实时报价请求(新浪逐合约接口,没有批量
    接口可用),偏"抓取密集型",这是免费数据源的结构性限制,不是实现
    取巧。请求之间做了错峰启动(_parallel_fetch 的 stagger 参数)而不是
    瞬间并发炸出去——新浪这类接口的反爬限流通常按瞬时并发连接数识别,
    实测 optbbs 全天不可用时这条备用路径会被高频触发(每5分钟一次),
    错峰能把"看起来像爬虫"的特征降下来,代价是单次现算耗时从约2秒
    涨到约5~8秒。只在 optbbs 失败时才触发、结果缓存5分钟,可以接受
    但要知道这个成本;如果新浪那边还是限流,_fetch_chain 会在日志里
    留下"只拿到 X/Y 个合约"的记录,可以顺着排查。
算出来的数量级和走势应该跟官方 QVIX 一致,但不保证分毫不差——报价取中
还是取最新成交、零买价的裁剪时机等实现细节,不同实现之间本来就会有出入。
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


def _parallel_fetch(items, fn, timeout=12, stagger=0.04):
    """对 items 并发跑 fn(item),daemon 线程、共享超时预算,超时的直接丢弃。
    不用 concurrent.futures.ThreadPoolExecutor——它的 worker 线程不是
    daemon,会被 atexit 钩子 join,一次性脚本(GitHub Actions)进程退出时
    被晾着的慢线程拖住(同 fetcher._fetch_with_timeout 的理由)。

    stagger:每个请求线程错开这么多秒启动,不是瞬间炸出几十个连接——
    新浪这类免费接口的反爬限流通常按"瞬时并发连接数"识别爬虫,不是按
    全天总量,近50个合约同一瞬间发请求比全天分散发更容易触发限流/临时
    封IP(实测撞过:早上重启后能用,挂了几小时后又不行了,很可能就是
    optbbs 全天不可用导致这条备用路径高频触发、把新浪那边打出限流)。
    错峰对总耗时影响很小——大部分请求早在错峰启动完之前就已经拿到
    结果了。返回收集到的结果列表,顺序不保证,超时/失败的直接跳过。"""
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

    deadline = time.time() + timeout
    threads = []
    for it in items:
        t = threading.Thread(target=_run, args=(it,), daemon=True)
        threads.append(t)
        t.start()
        if stagger and time.time() < deadline:
            time.sleep(stagger)
    for t in threads:
        remaining = deadline - time.time()
        if remaining > 0:
            t.join(remaining)
    return results


_SHIBOR_TENOR_DAYS = [
    ("O/N", 1), ("1W", 7), ("2W", 14), ("1M", 30),
    ("3M", 90), ("6M", 180), ("9M", 270), ("1Y", 365),
]


def _shibor_curve() -> Optional[list]:
    """今天最新一行 SHIBOR 各期限报价(年化小数),按天数排序,供插值用。
    量化定价里给短期限期权算无风险利率,SHIBOR 是境内通行的替代——
    A股期权到期通常只有一两个月,用 fetcher.get_risk_free_rate() 那个
    统一的1年期国债收益率给近月/次近月共用不够精确;这里多一次请求
    (免费、约0.5秒)换成按各自期限插值。取不到时调用方回退到那个
    1年期利率。"""
    try:
        df = fetcher.ak.macro_china_shibor_all()
    except Exception:
        return None
    if df is None or df.empty:
        return None
    row = df.iloc[-1]
    pts = []
    for tenor, days in _SHIBOR_TENOR_DAYS:
        col = f"{tenor}-定价"
        if col in row.index and pd.notna(row[col]):
            pts.append((float(days), float(row[col]) / 100.0))
    pts.sort()
    return pts if len(pts) >= 2 else None


def _rate_for_days(curve: Optional[list], days: float, fallback: float) -> float:
    """SHIBOR 期限结构线性插值;超出曲线两端用端点值(不外推),曲线
    拿不到时用 fallback(1年期国债收益率)。"""
    if not curve:
        return fallback
    if days <= curve[0][0]:
        return curve[0][1]
    if days >= curve[-1][0]:
        return curve[-1][1]
    for (d0, r0), (d1, r1) in zip(curve, curve[1:]):
        if d0 <= days <= d1:
            w = (days - d0) / (d1 - d0)
            return r0 + w * (r1 - r0)
    return fallback


def _years_to_expiry(expiry_date: str, now: dt.datetime):
    """到期日当天15:00(收盘,期权停止交易的时刻)到 now 的精确年数/天数。
    直接拿 datetime 相减取秒级精度,数学上等价于 CBOE 白皮书里"当天剩余
    分钟+到期日分钟+中间整天分钟"分段累加的写法,只是实现更直接;不是
    对分钟精度的近似,是同一个数字的另一种算法。返回 (年, 天),天带
    小数,交易日当天/临近到期时不会是整数。"""
    y, m, d = map(int, expiry_date.split("-"))
    settle = dt.datetime(y, m, d, 15, 0, 0, tzinfo=_CST)
    frac_days = (settle - now).total_seconds() / 86400.0
    return frac_days / 365.0, frac_days


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

    rows = _parallel_fetch(codes, _fetch_one)
    if len(rows) < len(codes):
        # 拿到的合约数明显少于请求数,大概率是新浪那边限流/连接被拒——
        # 留个可见记录,不然这种"部分失败"不报异常,只是悄悄少几行数据,
        # 排查起来无从下手。
        log.warning("QVIX %s 月合约行情只拿到 %d/%d 个,可能被限流",
                    month_str, len(rows), len(codes))
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
            log.warning("QVIX 自算失败:到期月份探测拿不到近月/次近月")
            return None
        (near_ms, near_date, _), (next_ms, next_date, _) = expiries

        T1, N1 = _years_to_expiry(near_date, now)
        T2, N2 = _years_to_expiry(next_date, now)
        if T1 <= 0 or T2 <= T1:
            log.warning("QVIX 自算失败:到期时间异常 T1=%.4f T2=%.4f", T1, T2)
            return None

        curve = _shibor_curve()
        fallback_r = fetcher.get_risk_free_rate()
        r1 = _rate_for_days(curve, N1, fallback_r)
        r2 = _rate_for_days(curve, N2, fallback_r)

        near_chain = _fetch_chain(near_ms)
        next_chain = _fetch_chain(next_ms)
        near = _term_variance(near_chain, r1, T1)
        nxt = _term_variance(next_chain, r2, T2)
        if near is None or nxt is None:
            log.warning("QVIX 自算失败:%s月合约%d个报价/%s月合约%d个报价,"
                       "方差算不出来(near=%s, next=%s)",
                       near_ms, len(near_chain), next_ms, len(next_chain),
                       near is not None, nxt is not None)
            return None
        sigma1, _, _ = near
        sigma2, _, _ = nxt

        n30 = 30.0
        w1 = (N2 - n30) / (N2 - N1)
        w2 = (n30 - N1) / (N2 - N1)
        sigma2_30 = (T1 * sigma1 * w1 + T2 * sigma2 * w2) * (365.0 / n30)
        if sigma2_30 <= 0:
            log.warning("QVIX 自算失败:插值方差非正 sigma2_30=%.6f", sigma2_30)
            return None
        vix = float(100.0 * np.sqrt(sigma2_30))
        # 粗粒度合理性校验:历史 QVIX 大致落在个位数到三位数以内,离谱的
        # 结果多半是报价缺失/行权价选取出错,宁可返回 None 也不展示假数。
        if not (1.0 < vix < 150.0):
            log.warning("QVIX 自算失败:结果 %.2f 超出合理区间,判为脏数据", vix)
            return None
        return round(vix, 2), now.strftime("%H:%M:%S")
    except Exception as e:
        log.warning("self-computed QVIX failed: %s", e)
        return None
