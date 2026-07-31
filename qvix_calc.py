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

log = logging.getLogger(__name__)

_CST = ZoneInfo("Asia/Shanghai")
_UNDERLYING = "510050"


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


def expiry_date_for_yymm(yymm: str) -> dt.date:
    """'2604' 这种到期月代码 → 该月第4个周三(50ETF期权标准到期日规则)。
    供历史回算用:未来月份走 fetcher.ak.option_sse_expire_day_sina 直接
    查到期日,但那个接口只认还没到期的月份,回算历史(backfill_qvix_history.py)
    只能靠这条规则自己算,不依赖任何实时接口。"""
    year = 2000 + int(yymm[:2])
    month = int(yymm[2:])
    weds = []
    d = 1
    while True:
        try:
            date = dt.date(year, month, d)
        except ValueError:
            break
        if date.weekday() == 2:
            weds.append(date)
        d += 1
    return weds[3]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(kind: str, S: float, K: float, T: float, r: float, sigma: float):
    """Black-Scholes 价格,供历史回算用:上交所官方风险指标接口只给隐含
    波动率、不给报价,拿官方 IV 反推回价格(数学上就是官方算 IV 时用的
    同一个模型倒着走一遍),再喂给 _term_variance 走标准 CBOE 公式。
    sigma<=0 或 T<=0 时返回 None(数据缺失/已到期)。"""
    if sigma is None or sigma <= 0 or T <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "C":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def _next_month_str(base: dt.date, offset: int) -> str:
    y, m = base.year, base.month + offset
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f"{y:04d}{m:02d}"


def _expiry_candidates(today: dt.date):
    """还没到期的合约月份,按到期先后:[(月份代码, 到期日, 剩余自然天数), …]。
    50ETF期权当前只挂近两个月+两个季月,月份代码探测 6 个月足够覆盖。
    近月/次近月的配对和顺延交给 _candidate_pairs(),这里不做换月裁剪。"""
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
    return candidates


def _candidate_pairs(listed):
    """按优先级列出 (近月, 次近月) 候选组合:相邻两个月为一对,最靠前的
    优先,算不出来再整体往后顺延一个月。实时路径(_expiry_candidates)和
    历史路径(_listed_expiries)共用。

    不做"近月剩余不足N天就跳过"的换月(理由见模块顶部"到期换月"一节):
    近月只剩一两天时报价确实噪,但它在30天插值里的权重是 T1×w1,T1=1/365
    这一项本身就趋近于0,数学上自然被压掉,不需要额外规则;真正报价烂到
    算不出方差时,由调用方顺延到下一对兜底。"""
    return list(zip(listed, listed[1:]))


_SINA_OPT_URL = "https://hq.sinajs.cn/list="
_SINA_OPT_HEADERS = {"Referer": "https://stock.finance.sina.com.cn/",
                     "User-Agent": "Mozilla/5.0"}

_sina_sess = None
_sina_sess_lock = threading.Lock()


def _sina_session():
    """拉行情用的会话:保持长连接 + 建连失败重试。

    一次现算要拉近月、次月两条链,打的是**同一个 host**(hq.sinajs.cn)。
    原来每条链各发一次独立请求,等于每次都重新做一遍跨境 TCP 握手,多赌
    一次运气。境外主机上这条链路抖动明显——Streamlit Cloud(GCP)实测过
    近月链 22 个报价拿到了、次月链却卡在建连超时(connect timeout=12),
    于是整次现算作废、退回 optbbs。
    改用 Session 后第二条链直接复用已建立的连接,不再重新握手;再对建连
    失败重试 2 次(指数退避),跨境偶发丢包也能兜住。连接超时放宽到 20 秒:
    跨境握手本来就比境内慢得多,12 秒对 GCP 这条路太紧。"""
    global _sina_sess
    if _sina_sess is not None:
        return _sina_sess
    with _sina_sess_lock:
        if _sina_sess is None:
            s = requests.Session()
            s.headers.update(_SINA_OPT_HEADERS)
            # 不重试。证据: GCP 那条链路要么很快连上(原来12秒超时的配置
            # 成功过), 要么根本连不上, 不存在"需要更久才握上手"的中间态。
            # 既然如此, 重试和长超时都只是白等——而且调用方(fetch_qvix_now)
            # 拿不到自算值时会立刻退 optbbs, 早失败比晚失败好。
            retry = Retry(total=0, connect=0, read=0,
                          status_forcelist=(429, 500, 502, 503, 504))
            s.mount("https://", HTTPAdapter(max_retries=retry,
                                            pool_connections=4, pool_maxsize=4))
            _sina_sess = s
    return _sina_sess


# (连接超时, 读取超时)。连接超时刻意压到 8 秒:境外主机上这条链路要么秒连、
# 要么连不上, 等 18 秒也等不来一个成功的握手, 只会把页面卡住。两条链最坏
# 16 秒, 留在外层 20 秒预算内; 失败后立刻退 optbbs, 用户几秒就能看到数。
_SINA_TIMEOUT = (8, 12)


def _fetch_chain(month_str: str) -> pd.DataFrame:
    """近月/次近月看涨+看跌整月合约链 → 一次批量拉实时报价。

    新浪 hq.sinajs.cn/list= 支持逗号拼多个 CON_OP_<代码> 一次返回(2026-07
    实测整月~50个合约一个请求15KB全回来),所以不再像原来那样逐合约近50
    连发——单请求容错高得多、也不会因"瞬时并发几十连接"被反爬当爬虫掐掉,
    境外主机(GitHub Actions/Streamlit云)上尤其稳。批量行情行格式:
      var hq_str_CON_OP_<code>="买量,买价,最新价,卖价,卖量,持仓,涨幅,行权价,…"
    """
    codes = []
    for label, kind in (("看涨期权", "C"), ("看跌期权", "P")):
        try:
            df = fetcher.ak.option_sse_codes_sina(
                symbol=label, trade_date=month_str, underlying=_UNDERLYING)
        except Exception:
            continue
        for c in df["期权代码"]:
            codes.append((str(c), kind))
    if not codes:
        return pd.DataFrame()

    kind_of = {c: k for c, k in codes}
    syms = ",".join("CON_OP_" + c for c, _ in codes)
    try:
        r = _sina_session().get(_SINA_OPT_URL + syms, timeout=_SINA_TIMEOUT)
        text = r.text
    except Exception as e:
        log.warning("QVIX %s 月合约批量行情请求失败: %s", month_str, e)
        return pd.DataFrame()

    rows = []
    for m in re.finditer(r'CON_OP_(\d+)="([^"]*)"', text):
        code, parts = m.group(1), m.group(2).split(",")
        if len(parts) < 8:
            continue
        try:
            bid, last, ask = float(parts[1]), float(parts[2]), float(parts[3])
            strike = float(parts[7])
        except ValueError:
            continue
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        rows.append({"kind": kind_of.get(code, "C"), "strike": strike,
                     "bid": bid, "mid": mid})

    if len(rows) < len(codes):
        # 部分失败(新浪限流/返回不全)不报异常,只会悄悄少几行——留个记录。
        log.warning("QVIX %s 月合约批量行情只拿到 %d/%d 个",
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
    all_strikes = sorted(set(calls.index) | set(puts.index))
    if not common or len(all_strikes) < 3:
        return None

    # 远期价格:C-P 差最小的那个行权价上用 put-call parity 反推。反推必须
    # 认购认沽都有报价,所以只能在 common 里选。
    k_f = min(common, key=lambda k: abs(calls[k] - puts[k]))
    F = k_f + np.exp(r * T) * (calls[k_f] - puts[k_f])
    # K0 = 不超过 F 的最大行权价,范围是整条链的全部行权价(白皮书口径),
    # 不能限制在 common 里:上交所风险指标接口对没成交的实值认购报 IV=0,
    # 这些合约被 _build_chain_from_risk_indicator 丢掉后,common 会整段
    # 缩到平值以上,于是"不超过 F 的行权价"一个都不剩、整天算不出来
    # (历史上 11 天空值都是这么来的,如 2026-07-09/07-14)。认沽那侧的
    # 行权价是全的,用并集就没这个问题;K0 处只有单边报价时,下面的
    # _price_at 本来就会退化成取那一边,不需要额外处理。
    k0_candidates = [k for k in all_strikes if k <= F]
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


def compute_qvix(as_of: Optional[dt.datetime] = None) -> Optional[tuple]:
    """现算 QVIX。失败(到期月探测失败/合约或报价拿不全等)返回 None,
    由调用方(fetcher.fetch_qvix_now)决定要不要继续找别的路子。
    返回 (qvix, "HH:MM:SS")。

    as_of:用哪个时刻算剩余到期时间 T,默认当下。传定值是为了把结果
    "冻结"在某一刻——中午休市(11:30~13:00)和收盘后,新浪返回的都是
    11:30 / 15:00 那一刻的静态报价,如果 T 还跟着真实时间一直缩小,
    同一批报价会随着时间推移算出越来越不一样的数(下午2点看和晚上10点
    看不一致)。把 as_of 钉在 11:30 / 15:00,算出来的就稳定是"上午收盘值"
    /"今日收盘值"。调用方见 fetcher.qvix_phase()。"""
    now = as_of or dt.datetime.now(_CST)
    try:
        pairs = _candidate_pairs(_expiry_candidates(now.date()))
        if not pairs:
            log.warning("QVIX 自算失败:到期月份探测拿不到近月/次近月")
            return None

        curve = _shibor_curve()
        fallback_r = fetcher.get_risk_free_rate()
        chains = {}   # 顺延重试时同一个月不重复拉行情

        for (near_ms, near_date, _), (next_ms, next_date, _) in pairs:
            T1, N1 = _years_to_expiry(near_date, now)
            T2, N2 = _years_to_expiry(next_date, now)
            if T1 <= 0 or T2 <= T1:
                log.warning("QVIX 自算:%s/%s 到期时间异常 T1=%.4f T2=%.4f,顺延",
                            near_ms, next_ms, T1, T2)
                continue

            r1 = _rate_for_days(curve, N1, fallback_r)
            r2 = _rate_for_days(curve, N2, fallback_r)

            for ms in (near_ms, next_ms):
                if ms not in chains:
                    chains[ms] = _fetch_chain(ms)
            near = _term_variance(chains[near_ms], r1, T1)
            nxt = _term_variance(chains[next_ms], r2, T2)
            if near is None or nxt is None:
                log.warning("QVIX 自算:%s月合约%d个报价/%s月合约%d个报价,"
                            "方差算不出来(near=%s, next=%s),顺延到下一对",
                            near_ms, len(chains[near_ms]), next_ms,
                            len(chains[next_ms]), near is not None, nxt is not None)
                continue
            sigma1, _, _ = near
            sigma2, _, _ = nxt

            n30 = 30.0
            w1 = (N2 - n30) / (N2 - N1)
            w2 = (n30 - N1) / (N2 - N1)
            sigma2_30 = (T1 * sigma1 * w1 + T2 * sigma2 * w2) * (365.0 / n30)
            if sigma2_30 <= 0:
                log.warning("QVIX 自算:插值方差非正 sigma2_30=%.6f,顺延", sigma2_30)
                continue
            vix = float(100.0 * np.sqrt(sigma2_30))
            # 粗粒度合理性校验:历史 QVIX 大致落在个位数到三位数以内,离谱的
            # 结果多半是报价缺失/行权价选取出错,宁可返回 None 也不展示假数。
            if not (1.0 < vix < 150.0):
                log.warning("QVIX 自算:结果 %.2f 超出合理区间,判为脏数据,顺延", vix)
                continue
            return round(vix, 2), now.strftime("%H:%M:%S")

        log.warning("QVIX 自算失败:所有近月/次近月组合都算不出来")
        return None
    except Exception as e:
        log.warning("self-computed QVIX failed: %s", e)
        return None


# ── 历史/日线自算(收盘后,上交所官方期权风险指标反推) ────────────────────────
# 用于 backfill_qvix_history.py(一次性批量回算)和
# fetcher.update_qvix_self_daily()(每日跑批增量更新)共用。

# 510050C2604M02900: 标的 + C/P + 到期月(YYMM) + 调整标记(M=普通,
# A/B..=分红调整) + 行权价×1000。只用 M(普通合约),调整合约的行权价
# 跟标的除权前后对不上,排除掉更干净。
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
    return pd.DataFrame(rows)


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
