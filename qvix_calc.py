"""50ETF 日线 QVIX。上交所风险指标表只提供当日合约编码，取价用逐合约日 K 线的收盘价。

2026-11-20 两张有正常成交的认购合约被风险指标表填了零隐波。用它反推
期权价格会漏掉这两张，算出 13.99，而上交所、东方财富和新浪收盘价都得到
15.66。历史日线和每日更新共用这个模块；新浪盘中买卖价仍走 qvix_core。

只用标准合约，使用最近两个到期月并按 30 天插值。近月到期前不提前换月；
50ETF 没有周合约，换月后偶尔需要小幅外推，最近可用近月超过 40 天就停。
到期日为第四个星期三，休市顺延。现货价格和上交所隐波均不参与计算，
远期价通过认购认沽平价反推。SHIBOR 期限结构按到期天数插值。

沪深300期权曾以同口径试跑，2026-08-31 的回测不如 50ETF，已撤。
"""

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import logging
import math
import re
import time

import requests

import fetcher
import qvix_core


log = logging.getLogger(__name__)
_UNDERLYING = "510050"
_CONTRACT_RE = re.compile(rf"^{_UNDERLYING}([CP])(\d{{4}})M(\d{{5}})$")
_SSE_HEADERS = {"Referer": "https://www.sse.com.cn/", "User-Agent": "Mozilla/5.0"}
_CONTRACT_URL = "https://query.sse.com.cn/commonQuery.do"
_CLOSE_URL = "https://yunhq.sse.com.cn:32042/v1/sho/dayk/"
_SHIBOR_BACKUP_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
CHAIN_BROKEN = "链失效"


def is_chain_broken(note) -> bool:
    """只有期权链确实无法覆盖远期价的日子才可作为恐慌信号。"""
    return isinstance(note, str) and note.startswith(CHAIN_BROKEN)


def _get_json(url, params=None, *, headers=None):
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=8)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))


def _fetch_contracts(day):
    """风险指标表当日挂牌快照，仅读交易代码和数字合约编码。"""
    params = {
        "isPagination": "false",
        "trade_date": day.strftime("%Y%m%d"),
        "sqlId": "SSE_ZQPZ_YSP_GGQQZSXT_YSHQ_QQFXZB_DATE_L",
        "contractSymbol": "",
    }
    payload = _get_json(_CONTRACT_URL, params, headers=_SSE_HEADERS)
    rows = payload["result"]
    if not isinstance(rows, list) or any(row["TRADE_DATE"] != day.isoformat() for row in rows):
        raise ValueError("合约名单日期不符")
    return [(row["CONTRACT_ID"], row["SECURITY_ID"]) for row in rows]


def _fetch_closes(symbol, begin, end):
    params = {"begin": begin.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
              "select": "date,open,high,low,close,volume"}
    payload = _get_json(_CLOSE_URL + symbol, params, headers=_SSE_HEADERS)
    rows = payload.get("kline")
    if str(payload.get("code")) != symbol or not isinstance(rows, list):
        raise ValueError("合约日线响应异常")
    closes = {}
    for raw_day, _, _, _, raw_close, _ in rows:
        day = dt.datetime.strptime(str(raw_day), "%Y%m%d").date()
        if not begin <= day <= end or day in closes:
            raise ValueError("合约日线日期越界或重复")
        close = float(raw_close) if raw_close is not None else 0.0
        closes[day] = close if math.isfinite(close) and close > 0 else None
    return closes


def _shibor_curves(dates):
    """金十日线为主。若尚未收录近期报价，用东方财富的 SHIBOR 当日报价补齐。"""
    tenors = qvix_core.SHIBOR_TENOR_DAYS
    needed = {d.isoformat() for d in dates}
    values = {}
    try:
        data = _get_json(qvix_core._SHIBOR_JSON)["values"]
        values = {d: {tenor: point[0] for tenor, point in data.get(d, {}).items()
                      if isinstance(point, list) and point}
                  for d in needed}
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        log.warning("日线 QVIX 获取金十 SHIBOR 失败：%s", exc)

    def curve(row):
        try:
            points = [(days, float(row[tenor]) / 100.0) for tenor, days in tenors]
        except (KeyError, ValueError, TypeError):
            return None
        return points if all(math.isfinite(rate) and rate > 0 for _, rate in points) else None

    missing = {d for d in needed if curve(values.get(d, {})) is None}
    if missing:
        try:
            params = {"reportName": "RPT_IMP_INTRESTRATEN", "columns": "ALL",
                      "filter": '(MARKET_CODE="001")', "pageSize": 200,
                      "sortColumns": "REPORT_DATE", "sortTypes": "-1"}
            for row in _get_json(_SHIBOR_BACKUP_URL, params)["result"]["data"]:
                d = row["REPORT_DATE"][:10]
                tenor = re.search(r"\(([^)]+)\)$", row["REPORT_PERIOD"])
                if d in missing and tenor:
                    values.setdefault(d, {})[tenor[1]] = row["IR_RATE"]
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            log.warning("日线 QVIX 获取东方财富 SHIBOR 失败：%s", exc)
    return {d: curve(values.get(d.isoformat(), {})) for d in dates}


def _expiry_for_yymm(month, trading_days):
    base = qvix_core.expiry_date_for_yymm(month)
    return next((base + dt.timedelta(days=i) for i in range(10)
                 if base + dt.timedelta(days=i) in trading_days), base)


def _plan(day, trading_days):
    months = {}
    for code, symbol in _fetch_contracts(day):
        match = _CONTRACT_RE.fullmatch(code)
        if not match:
            continue
        kind, month, strike = match.groups()
        symbol = str(symbol)
        if not re.fullmatch(r"\d{8}", symbol):
            raise ValueError("合约编码异常")
        key = kind, int(strike) / 1000
        contracts = months.setdefault(month, {})
        if key in contracts and contracts[key] != symbol:
            raise ValueError("同月、同方向、同行权价存在不同合约")
        contracts[key] = symbol
    listed = sorted(((expires - day).days, contracts)
                    for month, contracts in months.items()
                    if (expires := _expiry_for_yymm(month, trading_days)) > day)
    return [pair for pair in qvix_core.candidate_pairs(listed) if pair[0][0] <= 40]


def _term(day, term, closes, curve):
    days, contracts = term
    years = days / 365.0
    rate = qvix_core.rate_for_days(curve, days, 0.02)
    if any(symbol not in closes for symbol in contracts.values()):
        return None, False, True
    prices = {key: closes[symbol][day] for key, symbol in contracts.items()
              if closes[symbol].get(day) is not None}
    missing = len(contracts) - len(prices)
    if missing > max(3, len(contracts) // 10):
        return None, False, True
    if missing:
        log.warning("日线 QVIX %s（%d 天合约）缺 %d/%d 个收盘价",
                    day, days, missing, len(contracts))
    chain = [{"kind": kind, "strike": strike, "bid": price, "mid": price}
             for (kind, strike), price in prices.items()]
    result = qvix_core.term_variance(chain, rate, years)
    if result is not None:
        return result[0], False, False

    calls = {k: price for (kind, k), price in prices.items() if kind == "C"}
    puts = {k: price for (kind, k), price in prices.items() if kind == "P"}
    forward = qvix_core.forward_price(calls, puts, rate, years) if len(calls.keys() & puts.keys()) >= 2 else None
    broken = forward is not None and forward < min(k for _, k in contracts)
    return None, broken, False


def _calculate(day, pairs, closes, curve):
    if not pairs:
        return None, "无可用近月/次月合约"
    if not curve:
        return None, "未取得当日 SHIBOR"
    broken = False
    for near, nxt in pairs:
        var1, b1, missing1 = _term(day, near, closes, curve)
        var2, b2, missing2 = _term(day, nxt, closes, curve)
        if missing1 or missing2:
            return None, "上交所期权收盘数据不全"
        broken |= b1 or b2
        if var1 is None or var2 is None:
            continue
        n1, n2 = near[0], nxt[0]
        var30 = qvix_core.interpolate_30d(var1, n1 / 365, n1, var2, n2 / 365, n2)
        if var30:
            vix = 100.0 * math.sqrt(var30)
            if 1.0 < vix < 150.0:
                return round(vix, 2), None
    if broken:
        return None, f"{CHAIN_BROKEN}: 远期价低于合约最低挂牌行权价"
    return None, "收盘价计算未得到有效的30天方差"


def compute_qvix_history(dates, *, trading_days=None, delay=0.0):
    """按日计算 QVIX。每张合约只查一次日线，回填和每日跑批共用。"""
    dates = sorted(set(dates))
    if not dates:
        return {}
    if trading_days is None:
        trading_days = fetcher._trading_days() or set()
    curves = _shibor_curves(dates)
    plans, results = {}, {}
    for i, day in enumerate(dates, 1):
        try:
            plans[day] = _plan(day, trading_days)
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            results[day] = (None, f"上交所合约名单获取失败: {exc}")
        if i % 100 == 0:
            log.info("日线 QVIX 已检查 %d/%d 天合约名单", i, len(dates))
        if delay and i < len(dates):
            time.sleep(delay)
    symbols = {symbol for day, pairs in plans.items() if curves[day]
               for pair in pairs for _, contracts in pair for symbol in contracts.values()}

    def load(symbol):
        try:
            return symbol, _fetch_closes(symbol, dates[0], dates[-1])
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            log.warning("日线 QVIX 合约 %s 获取收盘价失败：%s", symbol, exc)
            return symbol, None

    log.info("日线 QVIX %d 个交易日，取 %d 张合约的收盘价", len(dates), len(symbols))
    closes = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (symbol, values) in enumerate(pool.map(load, sorted(symbols)), 1):
            if values is not None:
                closes[symbol] = values
            if i % 250 == 0:
                log.info("日线 QVIX 已请求 %d/%d 张合约", i, len(symbols))
    for day, pairs in plans.items():
        results[day] = _calculate(day, pairs, closes, curves[day])
    return {day: results[day] for day in dates}


def compute_qvix_for_date(target_date: dt.date) -> tuple:
    """指定收盘日的日线 QVIX，失败时返回 (None, 原因)。"""
    return compute_qvix_history([target_date])[target_date]
