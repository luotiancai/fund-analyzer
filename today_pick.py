#!/usr/bin/env python3
"""按标准策略算"今天会买哪只", 连带走到它之前被刷掉的那些。供 app 页面调用。

**只在线上页面用, 不做本地命令行入口**: 这个功能最容易出的错不是算错, 是
拿旧数据算出一个看起来很正常的答案 —— 本地跑回测习惯带 --no-sync, 净值库
停在几天前也不会报错。实测就踩过: 本地库停在 08-07, 而线上跑批早已更新到
08-12, 同一只基金的近3月差了 2.5 个百分点(-29.78% vs -27.32%), 选出来的
标的也不一样。线上页面每小时从 Release 拉最新库, 数据新鲜度由跑批保证,
再加上 data_freshness() 这道显式校验。

为什么要列出被淘汰的: 策略只输出一个代码, 但那个代码是从跌幅榜一路往下走、
被波动率和规模两道门槛筛掉一批之后才落到的。只看结果不知道"它凭什么是它",
也看不出今天这批候选整体是什么成色 —— 比如 2026-07-20 那天第一顺位是鹏华
新能源汽车(跌 29.84%、波动 2.81), 只因为规模 22.39 亿超了当时的上限被刷掉,
这种事只看最终结果完全看不见。

结果按**近3月跌幅从深到浅**排(跟策略的遍历顺序一致), **走到选中那只就截止**
—— 后面的候选策略根本没看过。每行标注被哪道门槛刷掉。跌幅超过上限的那批只
报个数、不进表: 它们被一条跟基金本身无关的规则一刀切掉, 逐只列会把表刷满
(实测某天有 48 只)。

不显示"QVIX 是否触发信号": 页面拿到的只是昨收, 而盘中实时值要跑 qvix_now.py
(qvix.command), 在页面上摆一个基于昨收的"未触发"反而误导。

⚠️ 所有指标都是**信号日 T-1 口径**(基金净值当晚才公布, 决策时看不到当天),
跟回测完全一致; 规模用信号日当时**已披露**的最新季报(无未来函数)。
QVIX 用的是库里最后一个收盘值 —— 盘中要看实时值请跑 qvix_now.py。
"""
import inspect
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_qvix as B      # noqa: E402
import fetcher                 # noqa: E402

def _std_params():
    """标准策略的参数, **直接读 run_backtest 的签名默认值**。

    不在这里抄一份字面量: 抄了就有两个真相来源, 改完策略忘了同步这边, 这个
    脚本会安安静静地按旧口径给出答案 —— 而它恰恰是拿来决定"今天买哪只"的,
    错了不会有任何报错。读签名的代价是依赖参数名不变; 万一改名, 下面的
    KeyError 会当场炸出来, 比静默用错值好。
    """
    d = {k: v.default for k, v in
         inspect.signature(B.run_backtest).parameters.items()
         if v.default is not inspect.Parameter.empty}
    return {k: d[k] for k in ("min_vol_ratio", "min_aum", "max_aum",
                              "max_drop", "dd_divisor", "window", "pct")}


STD = _std_params()


def _threshold(asof: pd.Timestamp):
    """信号日当天该比的阈值(截至前一交易日的 window/pct 分位)。"""
    q = fetcher.load_qvix_self_history().rename(columns={"qvix": "close"})
    q["date"] = pd.to_datetime(q["date"])
    q["close"] = pd.to_numeric(q["close"], errors="coerce")
    q = q.sort_values("date").reset_index(drop=True)
    thr = q["close"].rolling(STD["window"],
                             min_periods=int(STD["window"] * 0.97)
                             ).quantile(STD["pct"]).shift(1)
    q["thr"] = thr
    row = q[q["date"] <= asof]
    if row.empty:
        return None, None, None
    r = row.iloc[-1]
    return r["date"], r["close"], r["thr"]


def data_freshness(conn):
    """(净值库最新日, 上证最新日, 落后几个交易日)。

    这个功能最容易出的错不是算错, 是**拿旧数据算出一个看起来很正常的答案**
    —— 本地跑回测习惯带 --no-sync, 净值库停在几天前也不会报错。所以调用方
    必须先看这个: 净值库比上证行情落后几个交易日, 就是答案过期几天。
    """
    nav_max = conn.execute(
        "SELECT MAX(date) d FROM fund_nav_daily").fetchone()["d"]
    sse = B.load_cached_json(conn, "sse")
    sse_max = sse["date"].max()
    if not nav_max:
        return None, sse_max, None
    nav_max = pd.Timestamp(nav_max)
    behind = int((sse["date"] > nav_max).sum())
    return nav_max, sse_max, behind


def _ret3m_window(conn, code: str, asof_s: str):
    """这只基金「近3月」实际用的窗口 (起点, 终点)。

    口径照抄 fetcher._window_by_date: 终点 = 该基金**早于 asof** 的最后一个
    净值日(当日净值当晚才公布, 决策时看不到); 起点 = 终点往回 91 个自然日、
    取该日或之前最近的一个净值日。所以窗口是**逐只**算的 —— 哪只基金净值
    更新慢, 它的窗口就整体往前挪, 跟别人不完全对齐。
    ⚠️ 这跟支付宝/东财的"近3月"不是一回事: 它们按**整3个自然月**取起点。
    实测 025778 同一个终点(2026-08-12), 91天口径起点落在 05-13 得 -27.32%,
    而支付宝的起点是 05-12、得 -28.13%, 差一个交易日就差 0.8 个百分点。
    """
    end = conn.execute(
        "SELECT MAX(date) d FROM fund_nav_daily WHERE code=? AND date<?",
        (code, asof_s)).fetchone()["d"]
    if not end:
        return None, None
    start_limit = (pd.Timestamp(end)
                   - pd.Timedelta(days=fetcher.RETURN_DAYS["ret_3m"])
                   ).strftime("%Y-%m-%d")
    anchor = conn.execute(
        "SELECT MAX(date) d FROM fund_nav_daily WHERE code=? AND date<=?",
        (code, start_limit)).fetchone()["d"]
    return anchor, end


def scan(asof, top: int = 40) -> dict:
    """按标准策略走一遍当天的候选, 返回结构化结果(CLI 和页面共用)。

    rows 是**实际遍历路径**: 从跌幅最深处往下, 每行 (代码,名称,近3月,波动比,
    规模,淘汰原因或None,是否选中,近3月窗口起点,窗口终点), 走到选中那只就截止
    —— 之后的候选策略压根没看过。跌幅超上限的那批只计数(too_deep), 不进 rows。
    """
    conn = B.get_conn()
    names, types = {}, {}
    for it in json.loads(conn.execute(
            "SELECT data FROM fund_list").fetchone()[0]):
        c = it.get("code", "")
        names[c] = it.get("name") or c
        types[c] = it.get("type") or ""
    exclude = ({c for c, t in types.items() if ("QDII" in t or "海外" in t)}
               | {c for c, n in names.items() if B._HOLD_PERIOD_RE.search(n)})

    sse = B.load_cached_json(conn, "sse")
    sse["close"] = pd.to_numeric(sse["close"], errors="coerce")
    sse = sse.sort_values("date").reset_index(drop=True)

    asof = pd.Timestamp(asof)
    qdate, qvix, thr = _threshold(asof)
    nav_max, sse_max, behind = data_freshness(conn)
    out = {"asof": asof, "qdate": qdate, "qvix": qvix, "thr": thr,
           "hit": (qvix is not None and thr is not None
                   and not pd.isna(thr) and qvix > thr),
           "nav_max": nav_max, "sse_max": sse_max, "behind": behind,
           "std": STD, "rows": [], "too_deep": 0, "chosen": None,
           "fund_line": None, "sse_line": None, "names": names}
    if thr is None or pd.isna(thr):
        return out

    asof_s = asof.strftime("%Y-%m-%d")
    metrics = fetcher.compute_metrics_asof(asof_s, cols={"ret_3m"})
    cand = {c: m["ret_3m"] for c, m in metrics.items()
            if m.get("ret_3m") is not None and c not in exclude}
    if not cand:
        return out
    aums = fetcher.funds_aum_asof(sorted(cand), asof_s, merge_classes=True)

    rows, chosen, too_deep = [], None, 0
    for code in sorted(cand, key=cand.get):        # 跌幅从深到浅
        r3 = cand[code]
        if r3 >= 0:
            break                                   # 跌幅耗尽, 后面全是涨的
        if r3 < -STD["max_drop"]:
            too_deep += 1
            continue
        why, beta, aum = None, None, aums.get(code)
        if not B._has_full_window(conn, code, asof_s,
                                  fetcher.RETURN_DAYS["ret_3m"]):
            why = "回看窗口不完整"
        else:
            beta = B.compute_beta(conn, sse, code, asof_s)
            if beta < STD["min_vol_ratio"]:
                why = f"波动{beta:.2f}<{STD['min_vol_ratio']}"
            elif aum is None:
                why = "规模查不到"
            elif aum < STD["min_aum"]:
                why = f"规模{aum:.2f}亿<{STD['min_aum']}"
            elif STD["max_aum"] and aum > STD["max_aum"]:
                why = f"规模{aum:.2f}亿>{STD['max_aum']}"
        if why is None and chosen is None:
            end = pd.Timestamp(asof_s)
            if B._has_stale_catchup(conn, code, end - pd.Timedelta(days=101),
                                    end - pd.Timedelta(days=1)):
                why = "净值僵化-补涨"
            else:
                chosen = code
        _a, _e = _ret3m_window(conn, code, asof_s)
        rows.append((code, names.get(code, code), r3, beta, aum, why,
                     code == chosen, _a, _e))
        if chosen is not None or len(rows) >= top:
            break

    out.update(rows=rows, too_deep=too_deep, chosen=chosen)
    if chosen is not None:
        b = B.compute_beta(conn, sse, chosen, asof_s)
        out["fund_line"] = thr / STD["dd_divisor"] * b
        out["sse_line"] = thr / STD["dd_divisor"]
    return out
