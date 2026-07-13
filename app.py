"""Fund Analyzer — Streamlit dashboard."""

import datetime as dt
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import fetcher

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="基金夏普比率分析仪",
    page_icon="📈",
    layout="wide",
)

fetcher.init_db()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 控制面板")

    rf_rate = fetcher.get_risk_free_rate()
    st.metric("无风险利率", f"{rf_rate*100:.2f}%")
    st.caption("1年期国债收益率，自动取、每月刷新")

    st.markdown("---")
    update_btn = st.button("🔄 更新数据（拉当日净值+重算）", type="primary")
    recompute_btn = st.button("♻️ 仅重算（不重新下载）")
    clear_cache_btn = st.button("🧹 清空所有缓存并重算")

    st.markdown("---")
    st.caption("数据来源：天天基金（AKShare）")
    st.caption("夏普比率 = (年化收益 − 无风险利率) / 年化波动率")

# ── Load fund list ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="正在加载基金列表…")
def load_fund_list():
    return fetcher.fetch_fund_list(force_refresh=False)


# Per-fund quarterly holdings; fetcher caches in SQLite, this just skips the
# DB/network round-trip on Streamlit reruns within a session hour.
@st.cache_data(ttl=3600, show_spinner=False)
def load_holdings(code: str):
    return fetcher.fetch_holdings(code)

# Clear-cache button: wipe the in-memory list memo, the SQLite list/NAV/Sharpe
# tables, and any Sharpe results held in this session, then reload fresh. The
# list re-fetches immediately below; metrics are rebuilt by 「🔄 更新数据」.
if clear_cache_btn:
    fetcher.clear_all_caches()
    load_fund_list.clear()
    st.session_state.sharpe_results = {}
    st.success("已清空所有缓存,基金列表已重新加载。请点击「🔄 更新数据（增量+重算）」重新生成指标。")

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
    st.session_state.sharpe_results = fetcher.load_all_precomputed()
    _bar.progress(1.0, text="完成")
    st.success(
        f"更新完成 · 基金 {summary['funds']:,} · 回填 {summary['backfilled']} · "
        f"当日追加 {summary['appended']:,} · 补缺口 {summary['patched']:,}"
        + (f"（失败 {summary['failed']:,}）" if summary["failed"] else "")
        + f" · 重算 {summary['recomputed']:,} · 无风险利率 {summary['rf']*100:.2f}%"
    )

# Recompute-only: rf only feeds the Sharpe formula, so changing it needs no new
# data — just recompute from stored NAV (no network, ~15s) instead of the full
# pipeline. This is the right button after tweaking the risk-free rate.
if recompute_btn:
    _bar = st.progress(0.0, text="重算中…")
    saved = fetcher.recompute_all(
        rf=rf_rate,
        progress_callback=lambda d, t: _bar.progress(
            (d / t) if t else 1.0, text=f"重算指标… {d}/{t}"
        ),
    )
    st.session_state.sharpe_results = fetcher.load_all_precomputed()
    _bar.progress(1.0, text="完成")
    st.success(f"已按无风险利率 {rf_rate*100:.2f}% 重算 {saved:,} 只基金（未联网）")

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
if filter_ready:
    if asof_mode:
        _asof_iso = asof_date.strftime("%Y-%m-%d")
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

    filtered = work_df.copy()
    if selected_types:
        filtered = filtered[filtered["type"].isin(selected_types)]
    if ret_col in filtered.columns:
        # Only keep funds that actually have a return for the selected period
        # (e.g. newly-launched funds have no 近1年 value); NaN is dropped.
        filtered = filtered[filtered[ret_col] >= min_ret]

    # ── Session state for Sharpe results ──────────────────────────────────────
    # Load whatever the pipeline precomputed (daily batch or the 🔄 button);
    # loaded lazily on the first filter run, not on page open.
    if "sharpe_results" not in st.session_state:
        st.session_state.sharpe_results = fetcher.load_all_precomputed()

    # ── Merge Sharpe/drawdown into display table ──────────────────────────────
    # (In as-of mode work_df already carries the snapshot metrics columns.)
    display = filtered.copy()
    if not asof_mode and st.session_state.sharpe_results:
        sharpe_df = pd.DataFrame.from_dict(
            st.session_state.sharpe_results, orient="index"
        ).reset_index().rename(columns={"index": "code"})
        display = display.merge(sharpe_df, on="code", how="left")

    # Drawdown filter. Funds without a computed drawdown (e.g. younger than the
    # window) are kept rather than hidden.
    if max_dd < 100 and mdd_col in display.columns:
        dd_pct = pd.to_numeric(display[mdd_col], errors="coerce") * 100
        display = display[dd_pct.isna() | (dd_pct <= max_dd)]

    st.caption(f"共 {len(display):,} 只基金（总量 {len(fund_df):,}）")
    if max_dd < 100 and mdd_col not in display.columns:
        st.caption("⚠️ 暂无回撤数据。请点击「🔄 更新数据（增量+重算）」生成。")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_table, tab_detail = st.tabs(["📋 基金列表", "🔍 基金详情"])

# ─── Tab 1: Table ────────────────────────────────────────────────────────────
with tab_table:
    if display is None:
        st.info("👆 设置筛选条件后，点击「🔍 开始筛选」生成基金列表。")
    else:
        ret_label = f"{period_label}收益率(%)"
        dd_label = f"{period_label}最大回撤(%)"

        table = pd.DataFrame()
        table["基金代码"] = display.get("code")
        table["基金名称"] = display.get("name")
        table["类型"] = display.get("type")
        if ret_col in display.columns:
            table[ret_label] = pd.to_numeric(display[ret_col], errors="coerce").round(2)

        sharpe_label = f"{period_label}夏普比率"
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

        st.caption("点击表头可排序")
        st.dataframe(
            table.reset_index(drop=True),
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
            nav_df = fetcher.fetch_nav(code_input.strip().zfill(6))

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
            result = fetcher.compute_sharpe_for_fund(code_input.strip().zfill(6), rf=rf_rate)
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
