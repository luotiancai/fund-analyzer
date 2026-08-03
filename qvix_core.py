"""自算 QVIX 的核心实现:**只用 Python 标准库**,零第三方依赖——没有
requests / pandas / numpy / akshare。

为什么单独抽出来:
  这套算法要能跑在两个地方——本项目(Streamlit + 每日跑批)和境内云函数。
  境外主机连不上新浪 hq.sinajs.cn(实测 Streamlit Cloud/GCP 建连成功率约50%,
  且新浪只认 *.sina.com.cn 的 Referer, 浏览器无法伪造, 所以客户端方案也走
  不通), 唯一可靠的办法是把这段算在境内跑。而云函数上带着 akshare(它又拖着
  pandas/numpy/lxml 一大堆)打包会非常笨重。而腾讯云 SCF 的 Python 运行时连
  requests 都不带(实测 python3.9 直接 ModuleNotFoundError), 所以干脆把依赖砍
  到零——只用标准库, 整个文件原样粘进在线编辑器就能跑, 不装依赖、不传层、
  不打包 zip。

  更重要的是: 抽出来是为了**只有一份实现**。如果为手机端另写一份 JS/Swift,
  两边的 K0 选取、零买价裁剪、合约过滤这些手工判断迟早会不一致——而恐慌阈值
  是拿这套算法产出的序列算的, 一旦不一致就等于拿两把尺子量同一件事。今天就
  刚改过换月规则和 K0 选取, 要是有第二份实现, 它现在已经悄悄错了。

数据源(全部是普通 HTTP, 均需 *.sina.com.cn 的 Referer 才不被 403):
  · 到期日   stock.finance.sina.com.cn/futures/api/.../getRemainderDay  (JSON)
  · 合约清单 hq.sinajs.cn/list=OP_UP_510050<yymm> / OP_DOWN_...          (JS文本)
  · 实时报价 hq.sinajs.cn/list=CON_OP_<代码>,...                          (JS文本)
  · SHIBOR   cdn.jin10.com/data_center/reports/il_1.json                 (JSON)

方法论、换月规则、已知取舍等背景, 见 qvix_calc.py 顶部的长注释——那里是这套
算法的"说明书", 本文件只是它的无依赖实现。**改算法时两个文件要一起看。**
"""

import datetime as dt
import json
import math
import re
import time
from typing import Optional
from zoneinfo import ZoneInfo

from urllib.parse import urlencode
from urllib.request import Request, urlopen

CST = ZoneInfo("Asia/Shanghai")
UNDERLYING = "510050"

_HEADERS = {"Referer": "https://stock.finance.sina.com.cn/",
            "User-Agent": "Mozilla/5.0"}
_HQ = "https://hq.sinajs.cn/list="
_EXPIRE_API = ("https://stock.finance.sina.com.cn/futures/api/openapi.php/"
               "StockOptionService.getRemainderDay")
_SHIBOR_JSON = "https://cdn.jin10.com/data_center/reports/il_1.json"

SHIBOR_TENOR_DAYS = [("O/N", 1), ("1W", 7), ("2W", 14), ("1M", 30),
                     ("3M", 90), ("6M", 180), ("9M", 270), ("1Y", 365)]

# 超时(秒)。境内跑握手很快, 这个值绰绰有余; 境外(本项目本地兜底用)也够——
# 那条链路要么秒连、要么连不上, 等更久也等不来一个成功的握手。
TIMEOUT = 10


# 每个请求最多试几次、每次间隔多久。新浪按 IP 限流, 而云厂商出口 IP 是共享的
# ——实测腾讯云(上海)和 Streamlit Cloud(GCP)上单个请求都只有约一半成功率, 而
# 本机同一时刻百发百中。两个完全不同的云厂商、境内境外各一个, 都是一半, 基本
# 排除了网络路由问题, 指向"按 IP 段限流": 机房 IP 一律降级, 家用宽带放行。
# 提不高单次成功率, 就只能重试 + 少发请求(见 fetch_chains 如何把 6 次压成 2 次)。
# 5 次全失败的概率约 1/32。
_RETRY = 5
_RETRY_SLEEP = 0.6


def _get(url: str, params: Optional[dict] = None) -> str:
    """GET → 文本。失败或空响应会重试;试完还不行就抛异常, **不返回空串**。

    用标准库 urllib 而不是 requests:腾讯云 SCF 的 Python 运行时不自带 requests
    (实测 python3.9 报 ModuleNotFoundError), 这个文件要能原样粘进在线编辑器
    就跑, 不装依赖、不传层、不打包 zip。

    "空响应当失败"很重要:新浪限流时返回的是 200 + 空 body, 如果当成正常结果,
    上层会静默地少拿一半合约, 然后照样算出一个像模像样、实际完全不对的数。
    宁可抛错让上层知道, 也不要悄悄用残缺数据。

    新浪那两个接口必须带 *.sina.com.cn 的 Referer, 否则一律 403 Forbidden。
    hq.sinajs.cn 返回 GBK 文本, 但我们只用正则抠数字和代码, 中文解码失败可忽略。
    """
    if params:
        url = url + ("&" if "?" in url else "?") + urlencode(params)
    last = "未知"
    for i in range(_RETRY):
        try:
            req = Request(url, headers=_HEADERS)
            with urlopen(req, timeout=TIMEOUT) as r:
                text = r.read().decode("gbk", errors="ignore")
            if text.strip():
                return text
            last = "空响应(多半是被限流)"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if i < _RETRY - 1:
            time.sleep(_RETRY_SLEEP * (i + 1))
    raise RuntimeError(f"GET 失败({_RETRY}次): {url.split('?')[0][:70]} — {last}")


def _get_json(url: str, params: Optional[dict] = None):
    return json.loads(_get(url, params))


# ── 纯计算部分(无网络) ────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(kind: str, S: float, K: float, T: float, r: float, sigma: float):
    """Black-Scholes 价格。sigma<=0 或 T<=0 返回 None(数据缺失/已到期)。"""
    if sigma is None or sigma <= 0 or T <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "C":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def rate_for_days(curve: Optional[list], days: float, fallback: float) -> float:
    """SHIBOR 期限结构线性插值;超出曲线两端用端点值(不外推),曲线拿不到时
    用 fallback。"""
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


def years_to_expiry(expiry_date: str, now: dt.datetime):
    """到期日当天15:00(期权停止交易的时刻)到 now 的精确年数/天数。
    返回 (年, 天), 天带小数。"""
    y, m, d = map(int, expiry_date.split("-"))
    settle = dt.datetime(y, m, d, 15, 0, 0, tzinfo=CST)
    frac_days = (settle - now).total_seconds() / 86400.0
    return frac_days / 365.0, frac_days


def expiry_date_for_yymm(yymm: str) -> dt.date:
    """'2604' → 该月第4个周三(50ETF期权标准到期日规则)。"""
    year, month = 2000 + int(yymm[:2]), int(yymm[2:])
    weds, d = [], 1
    while True:
        try:
            date = dt.date(year, month, d)
        except ValueError:
            break
        if date.weekday() == 2:
            weds.append(date)
        d += 1
    return weds[3]


def candidate_pairs(listed):
    """相邻两个月为一对, 最靠前的优先, 算不出来再整体往后顺延一个月。
    不做"近月剩余不足N天就跳过"的换月, 理由见 qvix_calc.py 顶部。"""
    return list(zip(listed, listed[1:]))


def term_variance(chain, r: float, T: float):
    """单个到期日的方差贡献(CBOE VIX 白皮书公式)。chain 是
    [{"kind":"C"/"P", "strike":float, "bid":float, "mid":float}, ...]。
    返回 (sigma2, F, K0);合约不足/报价缺失时返回 None。"""
    if not chain:
        return None
    calls, puts, bid_calls, bid_puts = {}, {}, {}, {}
    for row in chain:
        k = row["strike"]
        if row["kind"] == "C":
            calls[k], bid_calls[k] = row["mid"], row["bid"]
        else:
            puts[k], bid_puts[k] = row["mid"], row["bid"]

    common = sorted(set(calls) & set(puts))
    all_strikes = sorted(set(calls) | set(puts))
    if not common or len(all_strikes) < 3:
        return None

    # 远期价格:C-P 差最小的那个行权价上用 put-call parity 反推(要求该行权价
    # 认购认沽都有报价, 所以只能在 common 里选)。
    k_f = min(common, key=lambda k: abs(calls[k] - puts[k]))
    F = k_f + math.exp(r * T) * (calls[k_f] - puts[k_f])
    # K0 = 不超过 F 的最大行权价, 范围是**整条链**(白皮书口径), 不能限制在
    # common 里:上交所对没成交的实值认购报 IV=0, 那些合约被丢掉后 common 会
    # 整段缩到平值以上, 于是一个候选都不剩、整天算不出来。
    k0_candidates = [k for k in all_strikes if k <= F]
    if not k0_candidates:
        return None
    K0 = max(k0_candidates)

    def _trim(strikes, bid_table):
        """扫描远离 K0 的方向, 遇到连续两个零买价行权价就截断(CBOE 规则)。"""
        out, zero_run = [], 0
        for k in strikes:
            if bid_table.get(k, 0) <= 0:
                zero_run += 1
                if zero_run >= 2:
                    break
                continue
            zero_run = 0
            out.append(k)
        return out

    put_side = _trim(sorted((k for k in puts if k < K0), reverse=True), bid_puts)
    call_side = _trim(sorted(k for k in calls if k > K0), bid_calls)
    selected = sorted(set(put_side) | {K0} | set(call_side))
    if len(selected) < 3:
        return None

    def _price_at(k):
        if k < K0:
            return puts[k]
        if k > K0:
            return calls[k]
        vals = [t[k] for t in (calls, puts) if k in t]
        return sum(vals) / len(vals)

    total, n = 0.0, len(selected)
    for i, k in enumerate(selected):
        if i == 0:
            dk = selected[1] - selected[0]
        elif i == n - 1:
            dk = selected[-1] - selected[-2]
        else:
            dk = (selected[i + 1] - selected[i - 1]) / 2
        total += (dk / k ** 2) * _price_at(k)

    sigma2 = (2 / T) * math.exp(r * T) * total - (1 / T) * (F / K0 - 1) ** 2
    return sigma2, F, K0


def interpolate_30d(sigma1, T1, N1, sigma2, T2, N2):
    """把近月/次月方差插值成30天期方差(CBOE 公式)。返回年化方差, 非正返回 None。"""
    n30 = 30.0
    w1 = (N2 - n30) / (N2 - N1)
    w2 = (n30 - N1) / (N2 - N1)
    s30 = (T1 * sigma1 * w1 + T2 * sigma2 * w2) * (365.0 / n30)
    return s30 if s30 > 0 else None


# ── 取数部分 ─────────────────────────────────────────────────────────────────

def shibor_curve() -> Optional[list]:
    """今天最新一行 SHIBOR 各期限报价(年化小数), 按天数排序, 供插值用。
    数据源同 akshare.macro_china_shibor_all 的上游(金十的静态 JSON), 这里直接
    取, 免掉 akshare 依赖。取不到返回 None, 调用方回退到 fallback 利率。"""
    try:
        data = _get_json(_SHIBOR_JSON, {"_": int(time.time())})
    except Exception:
        return None
    values = data.get("values") or {}
    if not values:
        return None
    latest = values[max(values)]     # 键是 "YYYY-MM-DD", 字典序即时间序
    pts = []
    for tenor, days in SHIBOR_TENOR_DAYS:
        v = latest.get(tenor)
        if isinstance(v, (list, tuple)) and v:
            try:
                pts.append((float(days), float(v[0]) / 100.0))
            except (TypeError, ValueError):
                continue
    pts.sort()
    return pts if len(pts) >= 2 else None


def _next_month_str(base: dt.date, offset: int) -> str:
    y, m = base.year, base.month + offset
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f"{y:04d}{m:02d}"


def expiry_candidates(today: dt.date, limit: int = 2):
    """还没到期的合约月份, 按到期先后:[(月份代码, 到期日, 剩余自然天数), …]。

    默认只取 2 个:加了"近月不得超过40天"的护栏后, 只有最靠前的那一对可能
    有效(顺延一对的话近月就变成次月、五十多天, 必然被护栏挡掉), 再往后探
    纯属白发请求——而在被限流的机房 IP 上, 每个请求都是一次翻车机会。
    50ETF期权只挂近两个月+两个季月, 中间月份不存在, 所以要多试几个 offset
    才凑得够 limit 个。"""
    out = []
    for off in range(6):
        ms = _next_month_str(today, off)
        try:
            # date 要 "YYYY-MM" 形式(ms 是 "202608" → "2026-08")
            j = _get_json(_EXPIRE_API, {"exchange": "null", "cate": "50ETF",
                                        "date": f"{ms[:4]}-{ms[4:]}"})
            d = (j.get("result") or {}).get("data") or {}
            expiry, days = d.get("expireDay"), d.get("remainderDays")
        except Exception:
            continue
        if not expiry or days is None:
            continue
        try:
            days = int(days)
        except (TypeError, ValueError):
            continue
        if days <= 0:
            continue
        out.append((ms, expiry, days))
        if len(out) >= limit:
            break
    return out


def _contract_codes_multi(months):
    """一次请求拿多个月的双边合约清单 → {月份: [(代码, "C"/"P"), ...]}。

    新浪 list= 支持逗号拼接多个符号, 所以 N 个月的认购+认沽清单(原本 2N 次
    请求)压成 1 次。在被限流的机房 IP 上, 请求次数就是成功率——少发一次就少
    一次翻车机会。
    任一月任一侧缺失, 该月不进结果:只有认购没有认沽时 put-call parity 反推
    不出远期价格, 后面 K0 选取、虚值筛选全建立在错误前提上。"""
    syms = []
    for ms in months:
        syms += [f"OP_UP_{UNDERLYING}{ms[-4:]}", f"OP_DOWN_{UNDERLYING}{ms[-4:]}"]
    text = _get(_HQ + ",".join(syms))
    out = {}
    for ms in months:
        sides = {}
        for tag, kind in (("OP_UP_", "C"), ("OP_DOWN_", "P")):
            m = re.search(f'{tag}{UNDERLYING}{ms[-4:]}="([^"]*)"', text)
            codes = re.findall(r"CON_OP_(\d+)", m.group(1)) if m else []
            if codes:
                sides[kind] = codes
        if len(sides) == 2:
            out[ms] = [(c, k) for k, cs in sides.items() for c in cs]
    return out


def fetch_chains(months):
    """一次请求拿多个月的全部实时报价 → {月份: [{"kind","strike","bid","mid"}…]}。

    连同上面的清单接口, 一次现算对 hq.sinajs.cn 只发 **2 个**请求(原本每月
    2次清单+1次报价 = 6次)。行情行格式:
      var hq_str_CON_OP_<code>="买量,买价,最新价,卖价,卖量,持仓,涨幅,行权价,…"
    某月报价没拿全就不放进结果——少几个行权价会让 1/K² 加权求和漏掉整段虚值
    期权, 算出来偏低且看不出异常。宁可那个月不算。

    只收标准(M)合约, 丢掉分红调整(A)合约——第43字段是 M/A 标志。510050 每年
    约11~12月分红一次, 之后数月 A 合约与 M 合约并行挂牌: A 的行权价非标准
    (如 2.701 夹在 2.70 和 2.75 之间)且基本无人交易、常年零买价, 混进来会在
    行权价阶梯**中段**触发 term_variance 的"连续两个零买价截断", 把后面报价
    正常的虚值段整个砍掉, 值悄悄偏低。历史路径(qvix_calc._CONTRACT_RE 只认
    M)一直就滤, 这里不滤的话实时值和阈值就不是同一把尺子。"""
    codes_by_month = _contract_codes_multi(months)
    if not codes_by_month:
        return {}
    meta = {c: (k, ms) for ms, lst in codes_by_month.items() for c, k in lst}
    text = _get(_HQ + ",".join("CON_OP_" + c for c in meta))

    got = {ms: [] for ms in codes_by_month}
    seen = {ms: 0 for ms in codes_by_month}   # 收到的行数, 含被过滤的 A 合约
    for m in re.finditer(r'CON_OP_(\d+)="([^"]*)"', text):
        code, parts = m.group(1), m.group(2).split(",")
        if code not in meta or len(parts) < 8:
            continue
        kind, ms = meta[code]
        try:
            bid, last, ask = float(parts[1]), float(parts[2]), float(parts[3])
            strike = float(parts[7])
        except ValueError:
            continue
        seen[ms] += 1
        # 字段不足44个时视同 M 放行: 宁可退回旧行为, 也不要因为新浪改了字段
        # 数就把整月合约全滤光、天天算不出来。
        if len(parts) > 43 and parts[43].strip() != "M":
            continue
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        got[ms].append({"kind": kind, "strike": strike,
                        "bid": bid, "mid": mid})
    # 完整性看"收到的行数"而不是"留下的行数": A 合约是拿到后主动丢的, 不算缺。
    return {ms: rows for ms, rows in got.items()
            if rows and seen[ms] == len(codes_by_month[ms])}


def fetch_chain(month_str: str):
    """单月版, 内部走 fetch_chains。拿不到返回 []。"""
    try:
        return fetch_chains([month_str]).get(month_str, [])
    except Exception:
        return []


def compute_qvix(as_of: Optional[dt.datetime] = None,
                 fallback_rate: float = 0.0113):
    """现算 QVIX → (qvix, "HH:MM:SS");算不出来返回 None。

    as_of:用哪个时刻算剩余到期时间 T, 默认当下。传定值是为了把结果"冻结"在
    某一刻(中午休市 11:30、收盘后 15:00)——那两段新浪返回的是静态报价, 若 T
    还跟着真实时间缩小, 同一批报价会随时间算出不一样的数。
    fallback_rate:SHIBOR 取不到时用的无风险利率。"""
    now = as_of or dt.datetime.now(CST)
    cands = expiry_candidates(now.date())
    pairs = candidate_pairs(cands)
    if not pairs:
        return None

    curve = shibor_curve()
    # 一次性把要用到的月份全取回来(对 hq.sinajs.cn 只 2 个请求), 而不是每月
    # 分别去取——机房 IP 被限流时, 请求次数直接决定成功率。
    try:
        chains = fetch_chains([c[0] for c in cands])
    except Exception:
        return None

    for (near_ms, near_date, _), (next_ms, next_date, _) in pairs:
        T1, N1 = years_to_expiry(near_date, now)
        T2, N2 = years_to_expiry(next_date, now)
        if T1 <= 0 or T2 <= T1:
            continue
        # 近月都已经比 30 天远这么多, 说明本该用的那个月没取到数, 此时的
        # "插值"其实是大幅外推, 会给出一个像模像样、实际完全不对的数——实测
        # 云端 8 月链(26天)拿不到时顺延到 (9月54天, 12月145天), 算出 22.54,
        # 而正确值是 19.72。宁可返回失败也不给假数。
        # 50ETF 合约月间隔约30天, 到期后新近月最远也就 35 天左右, 所以正常
        # 情况下 N1 不会超过 40。
        if N1 > 40:
            continue
        if near_ms not in chains or next_ms not in chains:
            continue
        r1 = rate_for_days(curve, N1, fallback_rate)
        r2 = rate_for_days(curve, N2, fallback_rate)
        near = term_variance(chains[near_ms], r1, T1)
        nxt = term_variance(chains[next_ms], r2, T2)
        if near is None or nxt is None:
            continue
        s30 = interpolate_30d(near[0], T1, N1, nxt[0], T2, N2)
        if s30 is None:
            continue
        vix = 100.0 * math.sqrt(s30)
        # 粗粒度合理性校验:历史 QVIX 大致落在个位数到三位数以内, 离谱的结果
        # 多半是报价缺失/行权价选取出错, 宁可返回 None 也不给假数。
        if not (1.0 < vix < 150.0):
            continue
        return round(vix, 2), now.strftime("%H:%M:%S")
    return None


# ── 腾讯云云函数入口 ─────────────────────────────────────────────────────────
# 部署方式:把**本文件**整个粘进 SCF 的在线编辑器(函数类型选"事件函数",
# 运行环境 Python 3.x, 执行方法填 index.main_handler), 加一个 API 网关触发器
# 就能拿到 URL。只依赖 requests, SCF 的 Python 运行时自带, 不用装任何东西。
#
# 为什么要跑在境内:境外主机连不上 hq.sinajs.cn(Streamlit Cloud 实测建连成功率
# 约50%), 而新浪只认 *.sina.com.cn 的 Referer、浏览器无法伪造, 所以"让手机
# 直接抓"也走不通。境内云函数是唯一既能连上、又不用你自己开着机器的办法。
# 页面(在境外)去调境内的这个 URL 方向是通的——反过来才不通。
#
# 可选加固:这个接口是公开的, 每次调用都会打一次新浪。URL 本身很难猜到,
# 真担心被人乱调可以在网关上加个鉴权, 或在下面校验一个约定的 query 参数。

def diagnose(as_of: Optional[dt.datetime] = None) -> dict:
    """逐步跑一遍并记录每步结果, 供云端排查。函数URL 加 ?debug=1 即可看到。

    "算不出来"这个结果本身不带原因(compute_qvix 只返回 None), 在本地能一行行
    print, 在云函数里只能靠这个。四个数据源逐个探一遍, 哪步断了一目了然——
    尤其能区分"网络出不去"和"数据解析不对", 这两种表现一样、修法完全不同。"""
    out = {}
    now = as_of or dt.datetime.now(CST)
    out["now"] = now.strftime("%Y-%m-%d %H:%M:%S")
    out["tz_ok"] = str(now.tzinfo)

    # ① 最基础的连通性:随便打一个新浪行情
    try:
        out["probe_hq"] = _get(_HQ + "sh000001")[:100]
    except Exception as e:
        out["probe_hq_error"] = f"{type(e).__name__}: {e}"

    # ② 到期日接口
    try:
        out["expiry_candidates"] = expiry_candidates(now.date())
    except Exception as e:
        out["expiry_error"] = f"{type(e).__name__}: {e}"

    # ③ 合约清单 + ④ 批量报价(拿第一个到期月试)
    cands = out.get("expiry_candidates") or []
    if cands:
        months = [c[0] for c in cands]
        try:
            codes = _contract_codes_multi(months)
            out["contract_codes_n"] = {k: len(v) for k, v in codes.items()}
        except Exception as e:
            out["contract_codes_error"] = f"{type(e).__name__}: {e}"
        try:
            chains = fetch_chains(months)
            out["chain_n"] = {k: len(v) for k, v in chains.items()}
        except Exception as e:
            out["chain_error"] = f"{type(e).__name__}: {e}"

    # ⑤ SHIBOR(拿不到不致命, 会退到 fallback 利率)
    try:
        c = shibor_curve()
        out["shibor"] = c[:3] if c else None
    except Exception as e:
        out["shibor_error"] = f"{type(e).__name__}: {e}"

    out["result"] = compute_qvix(as_of=as_of)
    return out


def parse_qs_flat(qs: str) -> dict:
    """"a=1&b=2" → {"a":"1","b":"2"}。用标准库, 不引额外依赖。"""
    from urllib.parse import parse_qs
    return {k: v[0] for k, v in parse_qs(qs.lstrip("?")).items() if v}


def main_handler(event, context):
    """HTTP 触发:返回 {"qvix":19.71,"time":"15:00:00","ok":true}。

    query 参数 as_of=HH:MM 可以把计算时刻钉死(页面在午休/收盘后会传 11:30 /
    15:00, 理由见 compute_qvix 的 as_of 说明)。不传就用当下。"""
    # query 参数的字段名各触发方式不一样(函数URL / API网关 / 直接测试),
    # 全都兼容一下, 免得换个接入方式就取不到。
    e = event or {}
    params = {}
    for key in ("queryString", "queryStringParameters"):
        v = e.get(key)
        if isinstance(v, dict):
            params.update(v)
        elif isinstance(v, str) and v:
            params.update(parse_qs_flat(v))
    for key in ("rawQueryString", "rawQuery"):
        if isinstance(e.get(key), str) and e[key]:
            params.update(parse_qs_flat(e[key]))
    params.update({k: v for k, v in e.items() if k == "as_of"})  # 控制台直接测试

    as_of = None
    raw = params.get("as_of")
    if raw:
        try:
            hh, mm = (int(x) for x in str(raw).split(":")[:2])
            as_of = dt.datetime.now(CST).replace(hour=hh, minute=mm, second=0,
                                                 microsecond=0)
        except Exception:
            as_of = None
    if str(params.get("debug", "")).lower() in ("1", "true", "yes"):
        try:
            body = diagnose(as_of=as_of)
        except Exception as e:
            body = {"ok": False, "error": f"diagnose 崩了: {type(e).__name__}: {e}"}
        return {"isBase64Encoded": False, "statusCode": 200,
                "headers": {"Content-Type": "application/json; charset=utf-8"},
                "body": json.dumps(body, ensure_ascii=False, default=str)}

    try:
        r = compute_qvix(as_of=as_of)
    except Exception as e:
        body = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        body = ({"ok": True, "qvix": r[0], "time": r[1]} if r
                else {"ok": False, "error": "算不出来(合约或报价拿不全)"})
    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(body, ensure_ascii=False),
    }


if __name__ == "__main__":
    print(json.dumps({"result": compute_qvix()}, ensure_ascii=False))
