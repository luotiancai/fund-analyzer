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

    rf_rate = st.number_input(
        "无风险利率（年化 %）",
        min_value=0.0, max_value=10.0, value=1.13, step=0.01,
    ) / 100

    st.markdown("---")
    force_refresh = st.button("🔄 强制刷新基金列表")
    calc_btn = st.button("⚡ 计算夏普比率与回撤", type="primary")

    st.markdown("---")
    st.caption("数据来源：天天基金（AKShare）")
    st.caption("夏普比率 = (年化收益 − 无风险利率) / 年化波动率")

# ── Load fund list ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="正在加载基金列表…")
def load_fund_list(refresh: bool = False):
    return fetcher.fetch_fund_list(force_refresh=refresh)

with st.spinner("加载基金列表中…"):
    fund_df = load_fund_list(refresh=force_refresh)

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
if "sharpe_results" not in st.session_state:
    st.session_state.sharpe_results = {}

# ── Batch Sharpe calculation ──────────────────────────────────────────────────
if calc_btn:
    codes = filtered["code"].dropna().unique().tolist() if "code" in filtered.columns else []
    total = len(codes)

    if total == 0:
        st.warning("没有可计算的基金。")
    else:
        st.info(
            f"将对 **{total:,}** 只基金计算夏普比率与各区间最大回撤，首次运行需较长时间，结果会实时缓存。"
        )
        progress_bar = st.progress(0, text="准备中…")
        status_text = st.empty()
        start_t = time.time()

        def on_progress(done: int, total_n: int):
            pct = done / total_n
            elapsed = time.time() - start_t
            eta = (elapsed / done * (total_n - done)) if done > 0 else 0
            progress_bar.progress(pct, text=f"{done}/{total_n}  · 已用 {elapsed:.0f}s  · 预计剩余 {eta:.0f}s")
            status_text.caption(f"缓存命中 + 已计算：{done} / {total_n}")

        results = fetcher.batch_compute_sharpe(
            codes,
            rf=rf_rate,
            progress_callback=on_progress,
        )
        st.session_state.sharpe_results = results
        progress_bar.progress(1.0, text=f"完成！共 {len(results):,} 只基金有效。")
        status_text.empty()
        st.success(f"计算完成，有效夏普比率：{len(results):,} 只，耗时 {time.time()-start_t:.1f}s")

# ── Merge Sharpe/drawdown into display table ─────────────────────────────────
display = filtered.copy()
if st.session_state.sharpe_results:
    sharpe_df = pd.DataFrame.from_dict(
        st.session_state.sharpe_results, orient="index"
    ).reset_index().rename(columns={"index": "code"})
    display = display.merge(sharpe_df, on="code", how="left")

# Drawdown filter (only meaningful once drawdowns have been computed; funds
# without a computed drawdown are kept so they aren't hidden before calc).
if max_dd < 100 and mdd_col in display.columns:
    dd_pct = pd.to_numeric(display[mdd_col], errors="coerce") * 100
    display = display[dd_pct.isna() | (dd_pct <= max_dd)]

st.caption(f"共 {len(display):,} 只基金（总量 {len(fund_df):,}）")
if max_dd < 100 and mdd_col not in display.columns:
    st.caption("⚠️ 回撤率筛选需先点击「⚡ 计算夏普比率与回撤」生成回撤数据。")

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

            st.dataframe(nav_df.reset_index(drop=True), use_container_width=True, height=300)
