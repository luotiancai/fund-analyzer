#!/usr/bin/env python3
"""给标准策略的每一笔算「全市场分位」, 写回 strategy_runs 的 trades。

    python3 pct_rank.py            # 算最新的标准跑批
    python3 pct_rank.py 43         # 指定跑批 id
    python3 pct_rank.py --dry-run  # 只算不写库

回答的问题是: 策略选中的那只, 在**当天全市场能买的基金**里排第几?
分位 88% = 打败了 88% 的候选; 50% = 跟闭眼瞎选没区别。这比"占最佳收益的
百分之多少"有意义得多 —— 后者的分母是事后最大值(极端序统计量), 任何规则
都不可能逼近; 而分位回答的是"这套选基规则相比随机到底强多少"。

候选池口径(与「无条件天花板」那条对照跑批一致):
  · 只排除三类**这笔交易本身不成立**的标的 —— QDII/海外(跟QVIX恐慌-反弹
    逻辑脱钩, 策略一贯排除)、持有期锁定(赎不出来, 止损卖不掉)、净值僵化-
    补涨(脏数据);
  · **不设**任何选基条件: 不要求真跌、无跌幅上限、无波动率门槛、无规模区间。
  · 每只候选都用**它自己的**双止损线(基金线=当日阈值/5×自身波动率比值,
    大盘线=阈值/5)逐日盯出卖点, 再按实现收益(扣完赎回费)排名 —— 跟策略
    那笔是同一套买卖口径, 可比。
  · 收益/波动一律走 daily_ret_pct(复权), 不碰单位净值(见 compute_beta 说明)。

⚠️ **换了标准策略就要重跑这个脚本**, 否则页面上那一列是上一版的陈旧数据。
每个信号日要把两三千到近五千只基金各跑一遍逐日止损, 实测约 15 分钟。
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import backtest_qvix as B      # noqa: E402
import fetcher                 # noqa: E402

COL = "全市场分位"


def _pool_universe(conn):
    """(名称表, 类型表, 排除集合)。"""
    names, types = {}, {}
    for it in json.loads(conn.execute(
            "SELECT data FROM fund_list").fetchone()[0]):
        c = it.get("code", "")
        names[c] = it.get("name") or c
        types[c] = it.get("type") or ""
    exclude = ({c for c, t in types.items() if ("QDII" in t or "海外" in t)}
               | {c for c, n in names.items() if B._HOLD_PERIOD_RE.search(n)})
    return names, types, exclude


def _fee(days):
    return 1.5 if days < 7 else (0.5 if days < 30 else 0.0)


def compute(run_id: int, dry_run: bool = False):
    conn = B.get_conn()
    names, _types, exclude = _pool_universe(conn)

    sse = B.load_cached_json(conn, "sse")
    sse["close"] = pd.to_numeric(sse["close"], errors="coerce")
    sse = sse.sort_values("date").reset_index(drop=True)
    sse_close = sse.set_index("date")["close"]
    sse_ret = sse_close.pct_change()
    need = fetcher.RETURN_DAYS["ret_3m"]

    row = conn.execute("SELECT label, trades FROM strategy_runs WHERE id=?",
                       (run_id,)).fetchone()
    trades = json.loads(row["trades"])
    print(f"跑批 #{run_id}「{row['label']}」{len(trades)} 笔\n", flush=True)

    for t in trades:
        buy, thr = pd.Timestamp(t["买入日"]), t["恐慌阈值"]
        metrics = fetcher.compute_metrics_asof(t["买入日"], cols={"ret_3m"})
        cand = {c for c, m in metrics.items()
                if m.get("ret_3m") is not None and c not in exclude}

        lo = (buy - pd.Timedelta(days=need + 20)).strftime("%Y-%m-%d")
        hi = (buy + pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        raw = pd.read_sql_query(
            "SELECT code, date, nav, daily_ret_pct FROM fund_nav_daily "
            "WHERE date>=? AND date<=?", conn, params=(lo, hi))
        raw["date"] = pd.to_datetime(raw["date"])
        raw["nav"] = pd.to_numeric(raw["nav"], errors="coerce")
        raw["r"] = pd.to_numeric(raw["daily_ret_pct"], errors="coerce") / 100.0
        raw = raw[raw["code"].isin(cand)]

        win_lo = buy - pd.Timedelta(days=need)
        m_pre = sse_ret[(sse_ret.index >= win_lo)
                        & (sse_ret.index < buy)].dropna().values
        m_std = np.std(m_pre) if len(m_pre) >= 20 else None
        days_after = sse[sse["date"] >= buy]["date"].values
        buy_sse = float(sse_close.loc[buy])

        rets = []
        for code, g in raw.groupby("code", sort=False):
            g = g.sort_values("date")
            pre = g[(g["date"] >= win_lo) & (g["date"] < buy)]["r"].dropna()
            if len(pre) < 20 or not m_std:
                continue
            v = pre.values[np.abs(pre.values) <= 0.35]
            if len(v) < 20:
                continue
            beta = round(float(np.std(v) / m_std), 2)
            # 净值僵化-补涨: 连续多日几乎不动后单日跳变, 脏数据
            run, stale = 0, False
            for x in pre.values:
                if abs(x) < B.STALE_FLAT_EPS:
                    run += 1
                    continue
                if run >= B.STALE_MIN_RUN and abs(x) > B.STALE_JUMP_THRESH:
                    stale = True
                    break
                run = 0
            if stale:
                continue
            post = g[g["date"] >= buy]
            if post.empty or pd.isna(post["nav"].iloc[0]) or not post["nav"].iloc[0]:
                continue
            cur = float(post["nav"].iloc[0])
            series = {}
            for i, (d, r) in enumerate(zip(post["date"].values, post["r"].values)):
                if i and not pd.isna(r):
                    cur = cur * (1 + r)
                series[d] = cur
            buy_nav = series[post["date"].values[0]]
            fund_lim, sse_lim = thr / 5.0 * beta, thr / 5.0
            peak = last = buy_nav
            peak_sse = buy_sse
            for d in days_after:
                nav = series.get(d, last)
                last = nav
                peak = max(peak, nav)
                sc = float(sse_close.loc[d])
                peak_sse = max(peak_sse, sc)
                if ((peak - nav) / peak * 100 >= fund_lim
                        or (peak_sse - sc) / peak_sse * 100 >= sse_lim):
                    n_days = (pd.Timestamp(d) - buy).days
                    rets.append((nav / buy_nav - 1) * 100 - _fee(n_days))
                    break
        del raw

        mine = float(t["费后收益"])
        pct = 100.0 * sum(1 for v in rets if v < mine) / len(rets)
        t[COL] = f"{pct:.1f}%"
        print(f"  {t['买入日']} 池{len(rets):>5} | 本笔 {mine:+7.2f}% → "
              f"分位 {pct:5.1f}% | 池中位 {np.median(rets):+6.2f}%", flush=True)

    if dry_run:
        print("\n--dry-run: 不写库")
        return
    conn.execute("UPDATE strategy_runs SET trades=? WHERE id=?",
                 (json.dumps(trades, ensure_ascii=False, default=str), run_id))
    conn.commit()
    print(f"\n已写回跑批 #{run_id} 的「{COL}」列")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    conn = B.get_conn()
    rid = int(args[0]) if args else conn.execute(
        "SELECT id FROM strategy_runs WHERE is_standard=1 "
        "ORDER BY id DESC LIMIT 1").fetchone()["id"]
    compute(rid, dry_run="--dry-run" in sys.argv)
