#!/usr/bin/env python3
"""按标准策略, 今天(或指定日期)会买哪只? 顺带把被筛选淘汰的一并列出来。

    python3 today_pick.py                # 今天
    python3 today_pick.py 2026-07-20     # 指定日期(复盘用)
    python3 today_pick.py --top 40       # 多列几行, 默认 25

为什么要列出被淘汰的: 策略只输出一个代码, 但那个代码是从跌幅榜一路往下走、
被波动率和规模两道门槛筛掉一批之后才落到的。只看结果不知道"它凭什么是它",
也看不出今天这批候选整体是什么成色 —— 比如 2026-07-20 那天第一顺位是鹏华
新能源汽车(跌 29.84%、波动 2.81), 只因为规模 22.39 亿超了当时的上限被刷掉,
这种事只看最终结果完全看不见。

输出按**近3月跌幅从深到浅**排(跟策略的遍历顺序一致), **走到选中那只就截止**
—— 后面的候选策略根本没看过。每行标注被哪道门槛刷掉, 选中的那只用 ★ 标出。
跌幅超过上限的那批只报个数、不进表: 它们被一条跟基金本身无关的规则一刀切掉,
逐只列出来会把表刷满(实测某天有 48 只)。

⚠️ 所有指标都是**信号日 T-1 口径**(基金净值当晚才公布, 决策时看不到当天),
跟回测完全一致; 规模用信号日当时**已披露**的最新季报(无未来函数)。
QVIX 用的是库里最后一个收盘值 —— 盘中要看实时值请跑 qvix_now.py。
"""
import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_qvix as B      # noqa: E402
import fetcher                 # noqa: E402

# 标准策略的默认参数(跟 backtest_qvix.run_backtest 的签名默认值保持一致,
# 改那边记得也看这里 —— 这里显式写出来是为了打印时能说清"按什么口径选的")
STD = dict(min_vol_ratio=1.5, min_aum=2.0, max_aum=None, max_drop=30.0,
           dd_divisor=5.0, window=490, pct=0.90)


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


def main() -> int:
    ap = argparse.ArgumentParser(description="按标准策略看今天会买哪只")
    ap.add_argument("date", nargs="?", default=None, help="默认今天")
    ap.add_argument("--top", type=int, default=40,
                    help="安全上限: 一直找不到合格候选时最多列多少行, 默认40")
    args = ap.parse_args()
    asof = pd.Timestamp(args.date) if args.date else pd.Timestamp(
        dt.date.today())

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

    qdate, qvix, thr = _threshold(asof)
    print()
    print(f"  === {asof.date()} 按标准策略的候选 ===")
    if thr is None or pd.isna(thr):
        print("  ⚠️ 当天算不出恐慌阈值(QVIX 数据不足)")
        return 1
    hit = qvix > thr
    print(f"  QVIX {qvix:.2f} (收于 {qdate.date()})  阈值 {thr:.2f}  →  "
          + ("★ 触发买入信号" if hit else "未触发, 以下仅为假设推演"))
    print(f"  口径: 近3月跌幅最大, 跌幅≤{STD['max_drop']:.0f}%, "
          f"波动率比值≥{STD['min_vol_ratio']}, 规模≥{STD['min_aum']}亿"
          + (f"~{STD['max_aum']}亿" if STD["max_aum"] else "(无上限)"))
    print()

    asof_s = asof.strftime("%Y-%m-%d")
    metrics = fetcher.compute_metrics_asof(asof_s, cols={"ret_3m"})
    cand = {c: m["ret_3m"] for c, m in metrics.items()
            if m.get("ret_3m") is not None and c not in exclude}
    if not cand:
        print("  当天没有可用候选(净值数据不足?)")
        return 1

    # 批量取规模: 一次 SQL, 纯读库不触网
    aums = fetcher.funds_aum_asof(sorted(cand), asof_s, merge_classes=True)

    rows, chosen, too_deep = [], None, 0
    for code in sorted(cand, key=cand.get):        # 跌幅从深到浅
        r3 = cand[code]
        if r3 >= 0:
            break                                   # 跌幅耗尽, 后面全是涨的
        # 跌超上限的那批只计数不列表: 它们被一条**跟基金本身无关**的规则一刀
        # 切掉, 逐只列出来会把表刷满(实测 2026-07-20 那天前 18 行全是这个),
        # 而真正值得看的是"进了跌幅带、却被波动率/规模刷掉"的那些。
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
            # 僵化-补涨这道最贵, 只在真要选中时才查
            end = pd.Timestamp(asof_s)
            if B._has_stale_catchup(conn, code,
                                    end - pd.Timedelta(days=101),
                                    end - pd.Timedelta(days=1)):
                why = "净值僵化-补涨"
            else:
                chosen = code
        rows.append((code, r3, beta, aum, why, code == chosen))
        # 走到选中那只就停 —— 表要回答的是"从跌得最惨的往下找, 一路上被什么
        # 刷掉、最后停在谁身上", 选中之后的候选策略压根没看过, 列出来是噪声。
        if chosen is not None or len(rows) >= args.top:
            break

    print(f"  从跌幅最深处往下找(跌超 {STD['max_drop']:.0f}% 的 {too_deep} "
          f"只已跳过), 直到选出一只为止:")
    print()
    print(f"  {'':2}{'基金':<30}{'近3月':>9}{'波动比':>7}{'规模(亿)':>9}  结论")
    print("  " + "─" * 74)
    for code, r3, beta, aum, why, is_pick in rows[:args.top]:
        nm = f"{names.get(code, code)[:22]} ({code})"
        print(f"  {'★' if is_pick else ' '} {nm:<30}{r3:>8.2f}%"
              f"{(f'{beta:.2f}' if beta is not None else '—'):>7}"
              f"{(f'{aum:.2f}' if aum is not None else '—'):>9}  "
              + ("★ 选中" if is_pick else (why or "通过")))
    if chosen is None:
        print("\n  今天没有合格候选 —— 按标准策略放弃当天, 顺延到下一个信号日")
    else:
        beta = B.compute_beta(conn, sse, chosen, asof_s)
        print(f"\n  ★ {names.get(chosen, chosen)} ({chosen})")
        print(f"     基金回撤控制线 {thr / STD['dd_divisor'] * beta:.2f}%  "
              f"大盘回撤线 {thr / STD['dd_divisor']:.2f}%")
        if not hit:
            print("     ⚠️ 今天并未触发信号, 上面是假设推演")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
