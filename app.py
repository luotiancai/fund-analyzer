"""Fund Analyzer — Streamlit dashboard."""

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

    with st.spinner("正在更新数据（增量下载当日净值 + 全量重算）…"):
        summary = fetcher.run_pipeline(progress=_on_progress, rf=rf_rate)
    load_fund_list.clear()
    st.session_state.sharpe_results = fetcher.load_all_precomputed()
    _bar.progress(1.0, text="完成")
    st.success(
        f"更新完成 · 基金 {summary['funds']:,} · 回填 {summary['backfilled']} · "
        f"追加 {summary['appended']:,} · 重算 {summary['recomputed']:,} · "
        f"无风险利率 {summary['rf']*100:.2f}%"
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
col_f1, col_f2, col_f3, col_f4 = st.columns([3, 1, 1, 1])
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
    min_ret = st.number_input(f"{period_label}最低收益率 %", value=0.0, step=1.0)
with col_f4:
    max_dd = st.number_input(
        f"{period_label}最大回撤率 %", value=15.0, min_value=0.0, step=1.0,
    )

ret_col, mdd_col, sharpe_col = PERIODS[period_label]

filtered = fund_df.copy()
if selected_types:
    filtered = filtered[filtered["type"].isin(selected_types)]
if ret_col in filtered.columns:
    # Only keep funds that actually have a return for the selected period
    # (e.g. newly-launched funds have no 近1年 value); NaN is dropped.
    filtered = filtered[filtered[ret_col] >= min_ret]

# ── Session state for Sharpe results ─────────────────────────────────────────
# Load whatever the pipeline precomputed (daily batch or the 🔄 button), so the
# table shows Sharpe/drawdown immediately on open — no per-session computation.
if "sharpe_results" not in st.session_state:
    st.session_state.sharpe_results = fetcher.load_all_precomputed()

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

# ── Merge Sharpe/drawdown into display table ─────────────────────────────────
display = filtered.copy()
if st.session_state.sharpe_results:
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
                    title="近一年单位净值走势",
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

            nav_table = nav_df.sort_values("date", ascending=False).reset_index(drop=True)
            st.dataframe(
                nav_table,
                use_container_width=True,
                height=300,
                column_config={
                    "date": st.column_config.DateColumn("净值日期", format="YYYY-MM-DD"),
                },
            )
