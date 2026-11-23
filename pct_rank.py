#!/usr/bin/env python3
"""给标准策略的每一笔算「全市场分位」, 写回 strategy_runs 的 trades。

    python3 pct_rank.py            # 算最新的标准跑批
    python3 pct_rank.py 43         # 指定跑批 id
    python3 pct_rank.py --dry-run  # 只算不写库

回答的问题是: 策略选中的那只, 在**当天全市场能买的基金**里排第几?
分位 88% = 打败了 88% 的候选; 50% = 跟闭眼瞎选没区别。这比"占最佳收益的
百分之多少"有意义得多 —— 后者的分母是事后最大值(极端序统计量), 任何规则
都不可能逼近; 而分位回答的是"这套选基规则相比随机到底强多少"。

候选池口径(与「可达天花板」那条对照跑批一致, 2026-08-13 起)。注意它**不是**
策略自己的候选池 —— 策略还要过"必须真跌 + 跌幅≤30% + 波动率比值≥1.5"三道
筛选, 这里全都不设。这是**故意的**: 分母若改成策略筛完的名单, 衡量的就只剩
"在已经选好的候选里排第几", 等于给前面那几道筛选免票; 而那几道恰恰是策略的
一部分、也最需要被检验。用"那天本来可以买的所有基金"当分母, 衡量的才是整套
选基规则(筛选+排序)合起来创造了多少价值。所以列名叫「可买池」而不是「同池」。
  · 保留**规模≥2亿**这一条选基条件, 与标准策略同口径 —— 分母里混进策略
    根本不会买的迷你基金没有意义, 而且那批基金的净值本身就不可信(015111
    惠升领先优选 2024-03-28 单日 +20.18%, 而它当季末净资产只剩 100 万,
    那 20% 是 93% 份额被短期赎回、赎回费分摊给剩下 7% 造成的);
  · 另排除**这笔交易本身不成立**的标的 —— QDII/海外(跟QVIX恐慌-反弹
    逻辑脱钩, 策略一贯排除)、FOF(净值披露滞后)、持有期锁定(赎不出来, 止损卖不掉)、净值僵化-
    补涨(脏数据);
  · 排除买入日已核实的历史限购，包括大额限额、暂停申购及暂停个人申购；
    公告证据和逐日快照保存在 fund_purchase_limits.json，不按基金永久拉黑。
    当前覆盖标准 #54 的九个买入日；新增买入日需先补公告核查，不能默认不限购。
  · 对有成立年限门槛的跑批，只纳入买入日已满 6 个自然月的份额。
  · 其余**不设**条件: 不要求真跌、无跌幅上限、无波动率门槛、无规模上限。
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
from fund_purchase_rules import restricted_codes_on  # noqa: E402

COL = "可买池分位"


def _pool_universe(conn):
    """(名称表, 类型表, 排除集合)。"""
    names, types = {}, {}
    for it in json.loads(conn.execute(
            "SELECT data FROM fund_list").fetchone()[0]):
        c = it.get("code", "")
        names[c] = it.get("name") or c
        types[c] = it.get("type") or ""
    exclude = ({c for c, t in types.items() if ("QDII" in t or "海外" in t)}
               | {c for c, t in types.items() if fetcher.is_fof(t, names[c])}
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

    row = conn.execute("SELECT label, trades, params FROM strategy_runs WHERE id=?",
                       (run_id,)).fetchone()
    params = json.loads(row["params"] or "{}")
    trades = json.loads(row["trades"])
    maturity_dates = fetcher.fund_maturity_dates(conn) if params.get("min_fund_age_months") else None
    print(f"跑批 #{run_id}「{row['label']}」{len(trades)} 笔\n", flush=True)

    for t in trades:
        buy, thr = pd.Timestamp(t["买入日"]), t["恐慌阈值"]
        asof = t["买入日"]
        restricted = restricted_codes_on(asof)
        metrics = fetcher.compute_metrics_asof(asof, cols={"ret_3m"})
        cand = {c for c, m in metrics.items()
                if m.get("ret_3m") is not None and c not in exclude and c not in restricted
                and (maturity_dates is None or (c in maturity_dates and maturity_dates[c] <= asof))}
        # 规模≥2亿: 用批量版一次 SQL 取完, 纯读库不触网(逐只 fund_aum_asof
        # 要查两万多次, 会把这个脚本拖到小时级)。
        aums = fetcher.funds_aum_asof(sorted(cand), asof,
                                    merge_classes=(params.get("aum_basis", "single") == "merged"))
        cand = [c for c in sorted(cand) if aums.get(c) is not None and aums[c] >= 2.0]

        hi = (buy + pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        days_after = sse[sse["date"] >= buy]["date"].values
        buy_sse = float(sse_close.loc[buy])
        market_returns = sse_close.pct_change(fill_method=None)

        rets = []
        for code in cand:
            window = fetcher.nav_period_bounds(conn, code, asof, fetcher.PERIOD_OFFSETS["3m"])
            if window[0] is None:
                continue
            pre = B._fund_period_returns(conn, code, window)
            if B._is_stale_catchup(pre):
                continue
            beta = B._volatility_ratio(pre, market_returns)
            if beta is None:
                continue
            post = pd.read_sql_query(
                "SELECT date, nav, daily_ret_pct FROM fund_nav_daily "
                "WHERE code=? AND date>=? AND date<=? ORDER BY date", conn, params=(code, asof, hi))
            if post.empty or pd.isna(post["nav"].iloc[0]) or not post["nav"].iloc[0]:
                continue
            post["date"] = pd.to_datetime(post["date"])
            post["r"] = pd.to_numeric(post["daily_ret_pct"], errors="coerce") / 100.0
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
        if not rets:
            raise ValueError(f"{asof} 没有可计算分位的已平仓候选。")
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
