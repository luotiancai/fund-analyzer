"""Fund Analyzer — Streamlit dashboard."""

import datetime as dt
import hashlib
import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import fetcher
import simulator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="基金夏普比率分析仪",
    page_icon="📈",
    layout="wide",
)

# init_db / risk-free rate run once per server (resp. hourly), not on every
# rerun — both hit SQLite on /mnt/c (slow Windows-disk I/O under WSL), which
# used to tax every single click.
@st.cache_resource
def _init_db_once():
    fetcher.init_db()
    simulator.init_sim_db()
    return True


@st.cache_data(ttl=3600, show_spinner=False)
def _get_rf() -> float:
    return fetcher.get_risk_free_rate()


_init_db_once()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 控制面板")

    rf_rate = _get_rf()
    st.metric("无风险利率", f"{rf_rate*100:.2f}%")
    st.caption("1年期国债收益率，自动取、每月刷新")

    st.markdown("---")
    update_btn = st.button("🔄 更新数据（拉当日净值+重算）", type="primary")

    st.markdown("---")
    st.caption("数据来源：天天基金（AKShare）")
    st.caption("夏普比率 = (年化收益 − 无风险利率) / 年化波动率")

# ── Load fund list ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="正在加载基金列表…")
def load_fund_list():
    return fetcher.fetch_fund_list(force_refresh=False)


# Per-fund detail data. fetcher caches in SQLite, but these wrappers matter on
# reruns: every click anywhere (e.g. 「开始筛选」) re-executes the detail tab,
# and without them each rerun re-read NAV, re-fetched holdings and — worst —
# recomputed + re-WROTE the fund's Sharpe row on the slow /mnt/c disk.
@st.cache_data(ttl=3600, show_spinner=False)
def load_holdings(code: str):
    return fetcher.fetch_holdings(code)


@st.cache_data(ttl=3600, show_spinner=False)
def load_nav(code: str):
    return fetcher.fetch_nav(code)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fund_metrics(code: str, rf: float):
    return fetcher.compute_sharpe_for_fund(code, rf=rf)


# Precomputed Sharpe/drawdown as a merge-ready DataFrame, built once and shared
# across sessions/reruns (reading ~20k SQLite rows + dict→DataFrame on every
# filter click is what made 筛选 feel slow). `cache_key` is last_update_time(),
# so a pipeline run naturally invalidates it; the buttons also clear it.
@st.cache_data(ttl=3600, show_spinner=False)
def load_metrics_df(cache_key):
    data = fetcher.load_all_precomputed()
    if not data:
        return None
    return pd.DataFrame.from_dict(data, orient="index") \
        .reset_index().rename(columns={"index": "code"})

# Update button: run the same daily pipeline as update_daily.py in-process,
# streaming progress into a bar. Refreshes the list cache and reloads the
# precomputed metrics so the table reflects the new data without a manual rerun.
if update_btn:
    _bar = st.progress(0.0, text="开始更新…")

    def _on_progress(phase: str, done: int, total: int):
        frac = (done / total) if total else 1.0
        _bar.progress(frac, text=f"{phase}… {done}/{total}")

    with st.spinner("正在更新数据（增量补净值 + 重算指标）…"):
        summary = fetcher.run_pipeline(progress=_on_progress, rf=rf_rate)
    load_fund_list.clear()
    load_metrics_df.clear()
    _bar.progress(1.0, text="完成")
    st.success(
        f"更新完成 · 基金 {summary['funds']:,} · 回填 {summary['backfilled']} · "
        f"当日追加 {summary['appended']:,} · 补缺口 {summary['patched']:,}"
        + (f"（失败 {summary['failed']:,}）" if summary["failed"] else "")
        + f" · 重算 {summary['recomputed']:,} · 无风险利率 {summary['rf']*100:.2f}%"
    )

with st.spinner("加载基金列表中…"):
    fund_df = load_fund_list()

if fund_df is None or fund_df.empty:
    st.error("无法获取基金列表，请检查网络连接。")
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
# Each time-period maps to its return column (from the fund list), its computed
# max-drawdown column, and its computed Sharpe column (only 6m/1y have Sharpe;
# shorter windows are too noisy, so None means "no Sharpe column").
PERIODS = {
    "近1月": ("ret_1m", "mdd_1m", None),
    "近3月": ("ret_3m", "mdd_3m", None),
    "近6月": ("ret_6m", "mdd_6m", "sharpe_6m"),
    "近1年": ("ret_1y", "mdd_1y", "sharpe_1y"),
}

# Ensure period return columns are numeric (cache may store them as strings).
for _c in ("ret_1m", "ret_3m", "ret_6m", "ret_1y"):
    if _c in fund_df.columns:
        fund_df[_c] = pd.to_numeric(fund_df[_c], errors="coerce")

all_types = sorted(fund_df["type"].dropna().unique().tolist()) if "type" in fund_df.columns else []
# Filters live in a form: changing a widget does NOT rerun/refilter — everything
# applies at once when 「开始筛选」 is pressed (expensive as-of recomputes stay
# off until then). Until the first submit, the defaults below are in effect.
with st.form("filter_form"):
    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([3, 1, 1, 1, 1.2])
    with col_f1:
        selected_types = st.multiselect(
            "基金类型筛选（不选则显示全部）",
            options=all_types,
            default=[],
            placeholder="全部类型",
        )
    with col_f2:
        period_label = st.selectbox("时间区间", options=list(PERIODS.keys()), index=3)
    with col_f3:
        min_ret = st.number_input("所选区间最低收益率 %", value=20.0, step=1.0)
    with col_f4:
        max_dd = st.number_input(
            "所选区间最大回撤率 %", value=15.0, min_value=0.0, step=1.0,
        )
    with col_f5:
        asof_date = st.date_input(
            "截至日期（不选=今天）",
            value=None,
            min_value=dt.date(2026, 1, 1),
            max_value=dt.date.today(),
            help="按该日及之前的净值历史重算收益/回撤/夏普，还原当天的筛选结果。"
                 "本地净值从 2025-01-01 起，因此最早可选 2026-01-01，"
                 "保证近1年窗口有完整数据。",
        )
    submitted = st.form_submit_button("🔍 开始筛选", type="primary")

# Nothing is filtered/merged/rendered until the user explicitly runs a filter —
# a fresh page load stops at the (cached) fund list. The flag persists in the
# session so later reruns (tab switches, detail lookups) keep the last result.
if submitted:
    st.session_state.filter_applied = True
filter_ready = st.session_state.get("filter_applied", False)

ret_col, mdd_col, sharpe_col = PERIODS[period_label]

# ── As-of snapshot mode ───────────────────────────────────────────────────────
# A past 截至日期 swaps the live rank-list returns and precomputed metrics for
# ones recomputed from stored NAV truncated to that date, so the filters below
# reproduce what the screen would have shown back then. Cached per (date, rf).
asof_mode = asof_date is not None and asof_date < dt.date.today()


# Manual cross-session snapshot cache: a plain dict held by st.cache_resource.
# Not st.cache_data — the progress bar is updated from inside the computation,
# and cache_data would record that element write and crash replaying it on
# later cache hits (CacheReplayClosureError). Here the bar runs in ordinary
# script code and only while real work is happening; hits return instantly.
@st.cache_resource(show_spinner=False)
def _asof_cache() -> dict:
    return {}


if not (filter_ready and asof_mode):
    _last_update = fetcher.last_update_time()
    if _last_update:
        _age_h = (time.time() - _last_update) / 3600
        _fresh = "🟢" if _age_h < 30 else "🟠"
        st.caption(
            f"{_fresh} 指标数据更新于 {time.strftime('%Y-%m-%d %H:%M', time.localtime(_last_update))}"
            f"（{_age_h:.0f} 小时前）"
        )
    else:
        st.caption("⚠️ 还没有指标数据。请点击「🔄 更新数据（增量+重算）」，或运行 `python3 update_daily.py`。")

display = None
table = None
_hit = None
if filter_ready:
    # Persistent result cache: the top-100 rows of every distinct filter run
    # are stored in SQLite, so repeating one (even after a restart — notably
    # the ~1min as-of snapshots) is a single read instead of a recompute.
    # Live-mode keys embed the metrics version, so a daily data update
    # naturally starts a fresh entry; as-of snapshots are immutable history.
    _asof_iso = asof_date.strftime("%Y-%m-%d") if asof_mode else None
    _fparams = {
        "types": sorted(selected_types), "period": period_label,
        "min_ret": min_ret, "max_dd": max_dd, "asof": _asof_iso,
        "data_ver": None if asof_mode else fetcher.last_update_time(),
    }
    _fkey = hashlib.md5(json.dumps(
        _fparams, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    _hit = fetcher.load_filter_result(_fkey)
    if _hit is not None:
        table, _fmeta, _fsaved = _hit
        st.caption(
            f"⚡ 本次筛选命中缓存（{time.strftime('%Y-%m-%d %H:%M', time.localtime(_fsaved))} 计算）"
            f" · 共 {_fmeta.get('total', 0):,} 条匹配，显示前 {len(table)} 条"
        )

if filter_ready and _hit is None:
    if asof_mode:
        _cache = _asof_cache()
        _key = (_asof_iso, round(rf_rate, 6))
        if _key not in _cache:
            _bar = st.progress(0.0, text=f"📅 正在按 {_asof_iso} 重算全市场指标（约1分钟）…")
            _cache[_key] = fetcher.compute_metrics_asof(
                _asof_iso, rf_rate,
                progress_callback=lambda d, t: _bar.progress(
                    (d / t) if t else 1.0,
                    text=f"📅 正在按 {_asof_iso} 重算全市场指标… {d:,}/{t:,}",
                ),
            )
            while len(_cache) > 4:   # keep only the newest few snapshots (~MBs each)
                _cache.pop(next(iter(_cache)))
            _bar.empty()
        _asof_metrics = _cache[_key]
        _mdf = pd.DataFrame.from_dict(_asof_metrics, orient="index") \
            .reset_index().rename(columns={"index": "code"})
        work_df = fund_df[["code", "name", "type"]].merge(_mdf, on="code", how="inner")
        st.caption(
            f"📅 快照模式：按 {_asof_iso} 及之前的净值计算收益/夏普/回撤"
            f"（覆盖 {len(work_df):,} 只基金）"
        )
    else:
        work_df = fund_df

    with st.spinner("⏳ 正在筛选…"):
        filtered = work_df.copy()
        if selected_types:
            filtered = filtered[filtered["type"].isin(selected_types)]
        if ret_col in filtered.columns:
            # Only keep funds that actually have a return for the selected period
            # (e.g. newly-launched funds have no 近1年 value); NaN is dropped.
            filtered = filtered[filtered[ret_col] >= min_ret]

        # ── Merge Sharpe/drawdown into display table ──────────────────────────
        # (In as-of mode work_df already carries the snapshot metrics columns.)
        display = filtered.copy()
        if not asof_mode:
            sharpe_df = load_metrics_df(fetcher.last_update_time())
            if sharpe_df is not None:
                display = display.merge(sharpe_df, on="code", how="left")

        # Drawdown filter. Funds without a computed drawdown (e.g. younger than
        # the window) are kept rather than hidden.
        if max_dd < 100 and mdd_col in display.columns:
            dd_pct = pd.to_numeric(display[mdd_col], errors="coerce") * 100
            display = display[dd_pct.isna() | (dd_pct <= max_dd)]

        # Build the presentation table inside the spinner too, so the loading
        # animation covers everything between the click and the rendered rows.
        ret_label = f"{period_label}收益率(%)"
        dd_label = f"{period_label}最大回撤(%)"
        sharpe_label = f"{period_label}夏普比率"
        table = pd.DataFrame()
        table["基金代码"] = display.get("code")
        table["基金名称"] = display.get("name")
        table["类型"] = display.get("type")
        if ret_col in display.columns:
            table[ret_label] = pd.to_numeric(display[ret_col], errors="coerce").round(2)
        if sharpe_col and sharpe_col in display.columns:
            table[sharpe_label] = pd.to_numeric(display[sharpe_col], errors="coerce").round(4)
        if mdd_col in display.columns:
            table[dd_label] = (pd.to_numeric(display[mdd_col], errors="coerce") * 100).round(2)

        # Default order (highest first); click any column header to re-sort.
        default_sort = next(
            (c for c in [sharpe_label, ret_label] if c in table.columns), None
        )
        if default_sort:
            table = table.sort_values(default_sort, ascending=False, na_position="last")
        table = table.reset_index(drop=True)

    _total = len(table)
    table = table.head(100).reset_index(drop=True)
    fetcher.save_filter_result(_fkey, {**_fparams, "total": _total}, table)
    st.caption(
        f"共 {_total:,} 条匹配（基金总量 {len(fund_df):,}）· 显示并缓存前 {len(table)} 条"
    )
    if max_dd < 100 and mdd_col not in display.columns:
        st.caption("⚠️ 暂无回撤数据。请点击「🔄 更新数据（增量+重算）」生成。")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_table, tab_detail, tab_sim = st.tabs(["📋 基金列表", "🔍 基金详情", "💰 模拟盘"])

# ─── Tab 1: Table ────────────────────────────────────────────────────────────
with tab_table:
    if table is None:
        st.info("👆 设置筛选条件后，点击「🔍 开始筛选」生成基金列表。")
    else:
        st.caption("点击表头可排序")
        st.dataframe(
            table,
            use_container_width=True,
            height=560,
        )

        csv = table.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ 下载 CSV",
            data=csv,
            file_name="fund_sharpe.csv",
            mime="text/csv",
        )

# ─── Tab 2: Fund detail ───────────────────────────────────────────────────────
with tab_detail:
    code_input = st.text_input("输入基金代码（6位数字）", placeholder="例如 000001")
    if code_input:
        with st.spinner(f"加载 {code_input} 净值历史…"):
            nav_df = load_nav(code_input.strip().zfill(6))

        if nav_df is None:
            st.error("无法获取该基金净值数据，请检查代码是否正确。")
        else:
            # Show fund info
            info = fund_df[fund_df["code"] == code_input.strip().zfill(6)]
            if not info.empty:
                row = info.iloc[0]
                st.subheader(f"{row.get('name', code_input)}（{code_input}）")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("基金类型", row.get("type", "--"))
                m2.metric("近1年涨跌幅", f"{row.get('ret_1y_pct', '--')}%")
                m3.metric("单位净值", row.get("nav", "--"))
                m4.metric("净值日期", str(row.get("nav_date", "--"))[:10])

            # NAV chart
            if "date" in nav_df.columns and "nav" in nav_df.columns:
                fig_nav = px.line(
                    nav_df,
                    x="date",
                    y="nav",
                    title=f"单位净值走势（{fetcher.NAV_START} 至今）",
                    labels={"date": "日期", "nav": "单位净值"},
                    height=380,
                )
                st.plotly_chart(fig_nav, use_container_width=True)

            # Compute Sharpe on the spot
            result = load_fund_metrics(code_input.strip().zfill(6), rf_rate)
            if result:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("年化收益", f"{result['ann_return']*100:.2f}%")
                s2.metric("年化波动率", f"{result['volatility']*100:.2f}%")
                s3.metric("夏普比率", f"{result['sharpe']:.4f}")
                s4.metric("交易日数据点", result["data_points"])

            # Quarterly top-10 holdings, 2025Q1 → latest disclosed quarter.
            st.markdown("---")
            st.subheader(f"📦 重仓持仓（{fetcher.HOLDINGS_START_YEAR}Q1 至最新）")
            with st.spinner("加载持仓数据…"):
                hold_df = load_holdings(code_input.strip().zfill(6))
            if hold_df is None:
                st.warning("持仓数据获取失败，请稍后重试。")
            elif hold_df.empty:
                st.info("该基金暂无披露的重仓持仓（货币基金、新基金常见）。")
            else:
                quarters = sorted(hold_df["quarter"].unique(), reverse=True)
                for q_tab, q in zip(st.tabs(quarters), quarters):
                    with q_tab:
                        qdf = hold_df[hold_df["quarter"] == q]
                        for kind, label in (("股票", "重仓股票"), ("债券", "重仓债券")):
                            # Annual/semi-annual reports disclose ALL holdings;
                            # 重仓 means the top 10 by weight, so cap at 10.
                            part = qdf[qdf["kind"] == kind].head(10)
                            if part.empty:
                                continue
                            st.markdown(f"**{label}（前十）**")
                            cols = {
                                "代码": part["代码"],
                                "名称": part["名称"],
                                "占净值比例(%)": part["占净值比例"],
                            }
                            if kind == "股票":
                                cols["持股数(万股)"] = part["持股数"]
                            cols["持仓市值(万元)"] = part["持仓市值"]
                            st.dataframe(
                                pd.DataFrame(cols).reset_index(drop=True),
                                use_container_width=True,
                            )
            st.markdown("---")
            st.subheader("📄 净值历史")
            nav_table = nav_df.sort_values("date", ascending=False).reset_index(drop=True)
            nav_table["date"] = pd.to_datetime(nav_table["date"]).dt.strftime("%Y-%m-%d")
            nav_table = nav_table.rename(columns={"date": "净值日期"})
            st.dataframe(
                nav_table,
                use_container_width=True,
                height=560,
            )

# ─── Tab 3: Paper-trading simulator ──────────────────────────────────────────
with tab_sim:
    _code_names = dict(zip(fund_df["code"], fund_df["name"]))
    sim_date = simulator.get_current_date()

    if sim_date is None:
        st.warning("本地还没有净值数据，请先点击侧边栏「🔄 更新数据」。")
    else:
        st.caption(
            f"从 {simulator.SIM_START} 开始 · 初始资金 ¥{simulator.INITIAL_CAPITAL:,.0f} · "
            "按当日单位净值成交，不计手续费 · 回退一天会撤销当天的全部买卖"
        )

        # Flash message from the previous action (survives st.rerun).
        _msg = st.session_state.pop("sim_msg", None)
        if _msg:
            st.success(_msg)

        # ── Day controls — processed before anything below renders ──
        c1, c2, _sp, c4 = st.columns([1.2, 1.2, 3.6, 1])
        if c1.button("▶️ 推进一天", type="primary"):
            sim_date, _moved = simulator.advance_day()
            if not _moved:
                st.toast("已到本地数据的最新日期，无法再推进", icon="⚠️")
        if c2.button("◀️ 回退一天", help="回到上一个交易日，并撤销当前这天的全部买卖"):
            sim_date, _moved = simulator.rollback_day()
            if not _moved:
                st.toast("已在第一个交易日，无法回退", icon="⚠️")
        with c4.popover("🗑️ 重置"):
            st.caption("清空全部模拟交易，回到起点重新开始。")
            if st.button("确认重置", type="primary", key="sim_reset_confirm"):
                simulator.reset()
                st.session_state["sim_msg"] = "模拟盘已重置"
                st.rerun()

        # ── Valuation ──
        pos, cash = simulator.holdings_and_cash(sim_date)
        curve = simulator.equity_curve()
        total = float(curve["value"].iloc[-1]) if not curve.empty \
            else simulator.INITIAL_CAPITAL
        prev_total = float(curve["value"].iloc[-2]) if len(curve) > 1 \
            else simulator.INITIAL_CAPITAL
        day_pnl = total - prev_total
        total_pnl = total - simulator.INITIAL_CAPITAL

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("模拟日期", sim_date)
        m2.metric("总资产", f"¥{total:,.0f}", delta=f"{day_pnl:+,.0f} 当日")
        m3.metric("现金", f"¥{cash:,.0f}")
        m4.metric("总收益", f"¥{total_pnl:+,.0f}")
        m5.metric("总收益率", f"{total_pnl / simulator.INITIAL_CAPITAL * 100:+.2f}%")

        st.markdown("---")
        hold = simulator.holdings_table(sim_date)
        trades = simulator.trades_table(sim_date)

        # Left: chart → holdings → trade log. Right: compact buy/sell panel.
        col_main, col_trade = st.columns([2.8, 1], gap="medium")

        with col_trade:
            with st.form("sim_buy", clear_on_submit=True):
                st.markdown("**🛒 买入**")
                buy_code = st.text_input("基金代码", placeholder="如 000001")
                buy_amt = st.number_input(
                    "金额（元）", min_value=0.0, value=100000.0, step=10000.0)
                if st.form_submit_button("买入", use_container_width=True):
                    if not buy_code.strip():
                        st.error("请输入基金代码")
                    else:
                        _code = buy_code.strip().zfill(6)
                        _err = simulator.buy(_code, buy_amt)
                        if _err:
                            st.error(_err)
                        else:
                            st.session_state["sim_msg"] = (
                                f"已买入 {_code} {_code_names.get(_code, '')} "
                                f"¥{buy_amt:,.0f}")
                            st.rerun()
            with st.form("sim_sell", clear_on_submit=True):
                st.markdown("**📤 卖出**")
                _held = list(pos.keys())
                sell_pick = st.selectbox(
                    "持仓基金", options=_held,
                    format_func=lambda c: (
                        f"{c} {_code_names.get(c, '')}"
                        f"（{pos[c][0]:,.2f} 份）"),
                ) if _held else st.selectbox("持仓基金", options=["（暂无持仓）"])
                sell_all = st.checkbox("全部卖出", value=True)
                sell_shares = st.number_input(
                    "卖出份额（未勾选全部时生效）",
                    min_value=0.0, value=0.0, step=1000.0)
                if st.form_submit_button("卖出", use_container_width=True):
                    if not _held:
                        st.error("当前没有持仓")
                    else:
                        _err = simulator.sell(
                            sell_pick, None if sell_all else sell_shares)
                        if _err:
                            st.error(_err)
                        else:
                            st.session_state["sim_msg"] = (
                                f"已卖出 {sell_pick} "
                                f"{_code_names.get(sell_pick, '')}")
                            st.rerun()
            st.caption("按当日单位净值成交")

        with col_main:
            # ── Held-fund cumulative-return chart (one line per holding) ──
            # Each line starts at that position's entry day at 0% and compounds
            # the fund's NAV relative to the entry NAV. History stops at the
            # simulated date — no peeking at the future.
            if not hold.empty:
                _frames = []
                for _, _h in hold.iterrows():
                    _c = _h["code"]
                    if not _h["open_nav"]:
                        continue
                    _s = simulator.nav_series(_c, _h["open_date"], sim_date)
                    if _s.empty:
                        continue
                    _frames.append(_s.assign(
                        cum_ret=(_s["nav"] / _h["open_nav"] - 1.0) * 100.0,
                        fund=f"{_c} {_code_names.get(_c, '')}",
                    ))
                if _frames:
                    _rets = pd.concat(_frames, ignore_index=True)
                    fig_ret = px.line(
                        _rets, x="date", y="cum_ret", color="fund",
                        title=f"持仓基金累计收益率（自各自买入日起，截至 {sim_date}）",
                        labels={"date": "日期", "cum_ret": "累计收益率（%）",
                                "fund": "基金"},
                        height=380,
                        color_discrete_sequence=[
                            "#4269D0", "#EFB118", "#FF725C", "#6CC5B0",
                            "#3CA951", "#FF8AB7", "#A463F2", "#97BBF5",
                        ],
                    )
                    fig_ret.update_traces(
                        line=dict(width=2),
                        hovertemplate="%{y:+.2f}%<extra>%{fullData.name}</extra>",
                    )
                    fig_ret.add_hline(
                        y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_ret.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_ret, use_container_width=True)

            # ── Holdings ──
            st.markdown(f"#### 📦 当前持仓（{len(hold)} 只）")
            if hold.empty:
                st.info("暂无持仓，全部为现金。")
            else:
                st.dataframe(pd.DataFrame({
                    "代码": hold["code"],
                    "名称": hold["code"].map(_code_names),
                    "成本(¥)": hold["cost"].round(2),
                    "当日收益率(%)": pd.to_numeric(
                        hold["day_ret"], errors="coerce").round(2),
                    "净值日期": hold["nav_date"],
                    "市值(¥)": hold["value"].round(2),
                    "盈亏(¥)": hold["pnl"].round(2),
                    "盈亏(%)": hold["pnl_pct"].round(2),
                }).reset_index(drop=True), use_container_width=True)

            # ── Trade log ──
            with st.expander(f"📜 交易记录（{len(trades)} 笔）"):
                if trades.empty:
                    st.caption("还没有交易。")
                else:
                    st.dataframe(pd.DataFrame({
                        "日期": trades["date"],
                        "操作": trades["action"].map(
                            {"buy": "买入", "sell": "卖出"}),
                        "代码": trades["code"],
                        "名称": trades["code"].map(_code_names),
                        "份额": trades["shares"].round(2),
                        "成交净值": trades["nav"],
                        "金额(¥)": trades["amount"].round(2),
                    }).iloc[::-1].reset_index(drop=True),
                        use_container_width=True)
