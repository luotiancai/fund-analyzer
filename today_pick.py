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


def _row(code, name, r3, beta, aum, why, role, win_a, win_e, rank):
    """遍历路径/代码查询共用的一行。两边必须同结构, 页面才能用同一张表渲染
    —— 查一个代码的意义就是"把它摆到今天这批候选里比", 表长得不一样就没法比。
    """
    return {"code": code, "name": name, "ret3m": r3, "beta": beta, "aum": aum,
            "why": why, "role": role, "win_start": win_a, "win_end": win_e,
            "rank": rank}


def _gates(conn, sse, code, asof_s, aum):
    """三道门槛(窗口完整/波动率/规模)+净值僵化, 按策略的短路顺序判。
    → (淘汰原因 or None, 波动率比值 or None, 规模 or None)。

    僵化-补涨这道排在最后: 前面几道全是便宜的库内判断, 它要扫一段净值。"""
    if not B._has_full_window(conn, code, asof_s,
                              fetcher.RETURN_DAYS["ret_3m"]):
        return "回看窗口不完整", None, aum
    beta = B.compute_beta(conn, sse, code, asof_s)
    if beta < STD["min_vol_ratio"]:
        return f"波动{beta:.2f}<{STD['min_vol_ratio']}", beta, aum
    if aum is None:
        return "规模查不到", beta, aum
    if aum < STD["min_aum"]:
        return f"规模{aum:.2f}亿<{STD['min_aum']}", beta, aum
    if STD["max_aum"] and aum > STD["max_aum"]:
        return f"规模{aum:.2f}亿>{STD['max_aum']}", beta, aum
    end = pd.Timestamp(asof_s)
    if B._has_stale_catchup(conn, code, end - pd.Timedelta(days=101),
                            end - pd.Timedelta(days=1)):
        return "净值僵化-补涨", beta, aum
    return None, beta, aum


def scan(asof, top: int = 40) -> dict:
    """按标准策略走一遍当天的候选, 返回结构化结果(CLI 和页面共用)。

    rows 是**实际遍历路径**: 从跌幅最深处往下, 每行一个 dict(见 _row)。
    走到选中那只就截止 —— 之后的候选策略压根没看过。跌幅超上限的那批只
    计数(too_deep), 不进 rows。

    想知道某只具体基金在今天这批候选里的位置, 用 probe() 单查, 不要把这条
    遍历路径拉长: 路径的意义就是"策略实际看过哪些", 多列的部分不属于它。"""
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

    # 跌幅榜名次: 分母是「真跌」的那批(跟策略的遍历顺序同一个序列), 不含
    # 涨的 —— 涨的根本不在这条路径上, 混进分母只会让名次看起来更靠前。
    order = sorted((c for c in cand if cand[c] < 0), key=cand.get)
    ranks = {c: i + 1 for i, c in enumerate(order)}
    out["n_drop"] = len(order)

    rows, chosen, too_deep = [], None, 0
    for code in order:                              # 跌幅从深到浅
        r3 = cand[code]
        if r3 < -STD["max_drop"]:
            too_deep += 1
            continue
        why, beta, aum = _gates(conn, sse, code, asof_s, aums.get(code))
        role = ""
        if why is None:
            chosen, role = code, "★ 选中"
        _a, _e = _ret3m_window(conn, code, asof_s)
        rows.append(_row(code, names.get(code, code), r3, beta, aum, why,
                         role, _a, _e, ranks.get(code)))
        if chosen is not None or len(rows) >= top:
            break

    out.update(rows=rows, too_deep=too_deep, chosen=chosen, cand=cand,
               ranks=ranks)
    if chosen is not None:
        b = B.compute_beta(conn, sse, chosen, asof_s)
        out["fund_line"] = thr / STD["dd_divisor"] * b
        out["sse_line"] = thr / STD["dd_divisor"]
    return out


def probe(asof, codes, res: dict = None) -> list:
    """指定基金代码在**今天这套口径**下的诊断行, 结构与 scan 的 rows 一致。

    res 传上一次 scan 的返回值就直接复用它的 cand/ranks(全市场重算近3月要
    ~10秒, 查个代码没必要再跑一遍); 不传就自己算一遍。

    跟 scan 的差别只有两处, 都是故意的:
      · 跌幅超上限的照样出行(why 写明超了多少), 不像 scan 那样并进 too_deep
        计数 —— 查一个具体代码时"它为什么没被选"正是要问的;
      · 涨的(近3月为正)也出行, why="没真跌"。策略遍历到这就收工了, 但用户
        手上拿着的那只未必在跌幅带里。
    """
    conn = B.get_conn()
    names, types = {}, {}
    for it in json.loads(conn.execute(
            "SELECT data FROM fund_list").fetchone()[0]):
        c = it.get("code", "")
        names[c] = it.get("name") or c
        types[c] = it.get("type") or ""
    sse = B.load_cached_json(conn, "sse")
    sse["close"] = pd.to_numeric(sse["close"], errors="coerce")
    sse = sse.sort_values("date").reset_index(drop=True)

    asof_s = pd.Timestamp(asof).strftime("%Y-%m-%d")
    cand = (res or {}).get("cand")
    ranks = (res or {}).get("ranks")
    if cand is None:
        metrics = fetcher.compute_metrics_asof(asof_s, cols={"ret_3m"})
        cand = {c: m["ret_3m"] for c, m in metrics.items()
                if m.get("ret_3m") is not None}
        ranks = {c: i + 1 for i, c in enumerate(
            sorted((c for c in cand if cand[c] < 0), key=cand.get))}

    codes = [c.strip() for c in codes if c and c.strip()]
    aums = fetcher.funds_aum_asof(codes, asof_s, merge_classes=True)
    out = []
    for code in codes:
        r3 = cand.get(code)
        if r3 is None:
            # 「没进池子」要说清是哪一种 —— 这个功能是拿来跟支付宝对账的,
            # 一句笼统的"不在候选池"没法判断到底是规则排掉的还是数据漏了。
            t, n = types.get(code), names.get(code)
            if n is None:
                miss = "榜单里没有这个代码(不是C类/已清盘/代码敲错)"
            elif t and ("QDII" in t or "海外" in t):
                miss = f"规则排除: {t}(跟踪境外市场)"
            elif B._HOLD_PERIOD_RE.search(n):
                miss = "规则排除: 名称含持有期锁定"
            elif fetcher.is_bond(t or ""):
                miss = f"规则排除: 债券类({t})"
            else:
                _last = conn.execute(
                    "SELECT MAX(date) d, COUNT(*) n FROM fund_nav_daily "
                    "WHERE code=? AND date<?", (code, asof_s)).fetchone()
                miss = (f"净值库里只有 {_last['n']} 条(最新 {_last['d']}),"
                        "算不出近3月" if _last and _last["n"]
                        else "净值库里没有这只的任何净值(候选池是**C类**"
                        "全市场, A/E 等别的份额类别不在库里 —— 支付宝上"
                        "拿到的常是 A 类代码, 换成同一只的 C 类再查)")
            out.append(_row(code, n or code, None, None, aums.get(code),
                            miss, "", None, None, None))
            continue
        if r3 >= 0:
            why, beta, aum = "没真跌(近3月为正)", None, aums.get(code)
        elif r3 < -STD["max_drop"]:
            # 门槛照样跑一遍再把跌幅上限写在最前面: 用户要知道的是"除了跌太多
            # 之外它还差在哪", 只报一条最先短路的没用。
            _w, beta, aum = _gates(conn, sse, code, asof_s, aums.get(code))
            why = f"跌幅{r3:.2f}%超上限{STD['max_drop']:.0f}%" + (f" + {_w}" if _w else "")
        else:
            why, beta, aum = _gates(conn, sse, code, asof_s, aums.get(code))
        a, e = _ret3m_window(conn, code, asof_s)
        out.append(_row(code, names.get(code, code), r3, beta, aum, why,
                        "" if why else "过了全部门槛", a, e, ranks.get(code)))
    return out
