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
@st.dialog("确认更新数据")
def _confirm_update():
    st.write("将增量拉取最新净值（不重算指标，筛选时按需计算），确定继续？")
    c1, c2 = st.columns(2)
    if c1.button("确定", type="primary", use_container_width=True):
        st.session_state["_run_update"] = True
        st.rerun()
    if c2.button("取消", use_container_width=True):
        st.rerun()


with st.sidebar:
    rf_rate = _get_rf()

    if st.button("🔄 更新数据", type="primary", use_container_width=True,
                 help="增量拉取最新净值；指标在筛选时按需计算"):
        _confirm_update()

    st.divider()
    st.metric("无风险利率", f"{rf_rate*100:.2f}%")
    st.caption("1年期国债收益率，自动取、每月刷新")

    st.divider()
    st.caption("数据来源：天天基金（AKShare）")
    st.caption("夏普比率 = (年化收益 − 无风险利率) / 年化波动率")

update_btn = st.session_state.pop("_run_update", False)

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


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_sse_daily():
    return fetcher.fetch_sse_daily()


# 1y max drawdown as it stood on the buy date (window: buy_date-365d → buy_date),
# on the daily-return growth index so dividend NAV resets don't count as drops.
# Immutable history, so the (code, buy_date) cache never needs invalidating.
@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def mdd_1y_at_buy(code: str, buy_date: str):
    _start = (pd.Timestamp(buy_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    _s = simulator.nav_series(code, _start, buy_date)
    if len(_s) < 2:
        return None
    _adj = (1.0 + pd.to_numeric(_s["daily_ret_pct"], errors="coerce")
            .fillna(0.0) / 100.0).cumprod()
    _peak = _adj.cummax()
    return float(((_peak - _adj) / _peak).max() * 100.0)


# Red bands on trading days where 上证指数 fell over 1% — the user's main
# operating signal, drawn on any date-axis figure that covers [dmin, dmax].
def _add_sse_drop_bands(fig, sse, dmin, dmax):
    if sse is None or sse.empty:
        return
    _d = pd.to_datetime(sse["date"])
    _drop = sse[(sse["pct"] <= -1.0) & (_d >= dmin) & (_d <= dmax)]
    for _, _r in _drop.iterrows():
        _dd = pd.Timestamp(_r["date"])
        fig.add_vrect(
            x0=_dd - pd.Timedelta(hours=12), x1=_dd + pd.Timedelta(hours=12),
            fillcolor="#e0454b", opacity=0.15, line_width=0)
        fig.add_annotation(
            x=_dd, y=1.02, yref="paper", yanchor="bottom",
            text=f"沪指{_r['pct']:.1f}%", showarrow=False, textangle=-40,
            font=dict(size=10, color="#e0454b"))


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

    with st.spinner("正在更新数据（增量补净值）…"):
        summary = fetcher.run_pipeline(progress=_on_progress, rf=rf_rate,
                                       do_recompute=False)
    load_fund_list.clear()
    load_metrics_df.clear()
    _bar.progress(1.0, text="完成")
    st.success(
        f"更新完成 · 基金 {summary['funds']:,} · 回填 {summary['backfilled']} · "
        f"当日追加 {summary['appended']:,} · 补缺口 {summary['patched']:,}"
        + (f"（失败 {summary['failed']:,}）" if summary["failed"] else "")
        + " · 指标将在筛选时按需计算"
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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_table, tab_detail, tab_sim = st.tabs(["📋 基金列表", "🔍 基金详情", "💰 模拟盘"])

# ─── Tab 1: Table ────────────────────────────────────────────────────────────
with tab_table:
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
                min_value=dt.date(2021, 1, 1),
                max_value=dt.date.today(),
                help="按该日及之前的净值历史重算收益/回撤/夏普，还原当天的筛选结果。"
                     "本地净值（仅C类）从 2020-01-01 起，因此最早可选 2021-01-01，"
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
            st.caption("⚠️ 还没有指标数据。请点击「🔄 更新数据」，或运行 `python3 update_daily.py`。")

    display = None
    table = None
    _hit = None
    if filter_ready:
        # Persistent result cache: the top-200 rows of every distinct filter run
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
        table = table.head(200).reset_index(drop=True)
        fetcher.save_filter_result(_fkey, {**_fparams, "total": _total}, table)
        st.caption(
            f"共 {_total:,} 条匹配（基金总量 {len(fund_df):,}）· 显示并缓存前 {len(table)} 条"
        )
        if max_dd < 100 and mdd_col not in display.columns:
            st.caption("⚠️ 暂无回撤数据。请点击「🔄 更新数据」生成。")

    if table is None:
        st.info("👆 设置筛选条件后，点击「🔍 开始筛选」生成基金列表。")
    else:
        st.caption("点击表头可排序 · 行首打勾（可多选）查看基金在截至日期下最新披露的十大持仓")
        _sel = st.dataframe(
            table,
            use_container_width=True,
            height=560,
            on_select="rerun",
            selection_mode="multi-row",
            key="filter_table_sel",
        )

        # ── Selected funds: latest top-10 holdings as of the filter's 截至日期,
        # laid out in a grid — fill each row left to right, then wrap. ──
        _sel_rows = _sel.selection.rows if _sel is not None else []
        _asof_lim = (asof_date if asof_mode
                     else dt.date.today()).strftime("%Y-%m-%d")

        def _render_holdings_cell(frow):
            _fcode = str(frow["基金代码"]).zfill(6)
            _fname = frow.get("基金名称", "")
            _hdf = load_holdings(_fcode)
            st.markdown(f"**📦 {_fcode} {_fname}**")
            if _hdf is None or _hdf.empty:
                st.info("该基金暂无披露的重仓持仓（货币基金、新基金常见）。")
                return
            _qend = {"1": "-03-31", "2": "-06-30", "3": "-09-30", "4": "-12-31"}
            _h = _hdf.dropna(subset=["quarter"]).copy()
            _h["quarter"] = _h["quarter"].astype(str)
            _h["_qend"] = _h["quarter"].str[:4] + _h["quarter"].str[-1].map(_qend)
            _h = _h[_h["_qend"] <= _asof_lim]
            if _h.empty:
                st.info("截至该日期尚无已披露的季度持仓（本地持仓数据从 "
                        f"{fetcher.HOLDINGS_START_Q} 起）。")
                return
            _q = _h["quarter"].max()
            _hq = _h[_h["quarter"] == _q]
            st.caption(f"披露季度：{_q}")
            for _kind, _klabel in (("股票", "重仓股票"), ("债券", "重仓债券")):
                _part = _hq[_hq["kind"] == _kind].head(10)
                if _part.empty:
                    continue
                st.markdown(f"**{_klabel}（前十）**")
                st.dataframe(pd.DataFrame({
                    "代码": _part["代码"],
                    "名称": _part["名称"],
                    "占净值比例(%)": _part["占净值比例"],
                }).reset_index(drop=True), use_container_width=True)

        if _sel_rows:
            st.markdown(f"##### 已选基金的最新十大持仓（截至 {_asof_lim}）")
            st.caption("取季度末 ≤ 截至日期的最新披露季度；实际公告通常滞后季度末约两周")
            _PER_ROW = 3
            with st.spinner("加载持仓…"):
                for _start in range(0, len(_sel_rows), _PER_ROW):
                    _chunk = _sel_rows[_start:_start + _PER_ROW]
                    _cols = st.columns(_PER_ROW, gap="medium")
                    for _col, _ri in zip(_cols, _chunk):
                        with _col, st.container(border=True):
                            _render_holdings_cell(table.iloc[_ri])

        # ── 模拟盘当前持仓的基金：重仓区常驻展示，无需在表格里勾选 ──
        _sim_d = simulator.get_current_date()
        _sim_codes = sorted(simulator.holdings_and_cash(_sim_d)[0]) if _sim_d else []
        if _sim_codes:
            _sim_names = dict(zip(fund_df["code"], fund_df["name"]))
            st.markdown(
                f"##### 💰 模拟盘持仓基金的最新十大持仓（截至 {_asof_lim}）")
            _PER_ROW_SIM = 3
            with st.spinner("加载持仓…"):
                for _start in range(0, len(_sim_codes), _PER_ROW_SIM):
                    _chunk = _sim_codes[_start:_start + _PER_ROW_SIM]
                    _cols = st.columns(_PER_ROW_SIM, gap="medium")
                    for _col, _c in zip(_cols, _chunk):
                        with _col, st.container(border=True):
                            _render_holdings_cell(
                                {"基金代码": _c,
                                 "基金名称": _sim_names.get(_c, "")})

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
                # Hover snaps to the nearest date anywhere on the plot
                # (hoverdistance=-1 removes the proximity cutoff) and a
                # crosshair pinned to the data point highlights the node.
                # Node dots only when sparse enough not to merge into the line.
                if len(nav_df) <= 120:
                    fig_nav.update_traces(mode="lines+markers",
                                          marker=dict(size=8))
                fig_nav.update_layout(
                    hovermode="x unified", hoverdistance=-1, spikedistance=-1)
                _dates = pd.to_datetime(nav_df["date"])
                _span_d = max((_dates.max() - _dates.min()).days, 1)
                fig_nav.update_xaxes(
                    showspikes=True, spikemode="across", spikesnap="data",
                    spikedash="dot", spikethickness=1,
                    hoverformat="%Y-%m-%d", tickformat="%Y-%m-%d",
                    dtick=max(1, _span_d // 8) * 86400000)
                fig_nav.update_yaxes(
                    showspikes=True, spikemode="across", spikesnap="data",
                    spikedash="dot", spikethickness=1)
                st.plotly_chart(fig_nav, use_container_width=True)

            # Compute Sharpe on the spot
            result = load_fund_metrics(code_input.strip().zfill(6), rf_rate)
            if result:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("年化收益", f"{result['ann_return']*100:.2f}%")
                s2.metric("年化波动率", f"{result['volatility']*100:.2f}%")
                s3.metric("夏普比率", f"{result['sharpe']:.4f}")
                s4.metric("交易日数据点", result["data_points"])

            # Quarterly top-10 holdings, HOLDINGS_START_Q → latest disclosed quarter.
            st.markdown("---")
            st.subheader(f"📦 重仓持仓（{fetcher.HOLDINGS_START_Q} 至最新）")
            with st.spinner("加载持仓数据…"):
                hold_df = load_holdings(code_input.strip().zfill(6))
            if hold_df is None:
                st.warning("持仓数据获取失败，请稍后重试。")
            elif hold_df.empty:
                st.info("该基金暂无披露的重仓持仓（货币基金、新基金常见）。")
            else:
                hold_df = hold_df.dropna(subset=["quarter"])
                hold_df["quarter"] = hold_df["quarter"].astype(str)
                quarters = sorted(hold_df["quarter"].unique().tolist(),
                                  reverse=True)
                if not quarters:
                    st.info("该基金暂无披露的重仓持仓（货币基金、新基金常见）。")
                    quarters = []
                for q_tab, q in zip(st.tabs(quarters) if quarters else [], quarters):
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
        # Flash message from the previous action (survives st.rerun).
        _msg = st.session_state.pop("sim_msg", None)
        if _msg:
            st.success(_msg)

        # ── 操作条（先处理动作，再渲染下方状态）──
        c1, c2, _sp, c3, c4 = st.columns([1.2, 1.2, 2.4, 1.2, 1])
        if c1.button("▶️ 推进一天", type="primary", use_container_width=True):
            sim_date, _moved = simulator.advance_day()
            if not _moved:
                st.toast("已到本地数据的最新日期，无法再推进", icon="⚠️")
            else:
                # 主要操作信号：落到沪指跌超1%的日子就提示——空仓时没有
                # 持仓图表可看，这条提示是唯一入口。
                _sse0 = load_sse_daily()
                if _sse0 is not None and not _sse0.empty:
                    _row0 = _sse0[_sse0["date"] == sim_date]
                    if (not _row0.empty and pd.notna(_row0["pct"].iloc[0])
                            and _row0["pct"].iloc[0] <= -1.0):
                        st.toast(f"{sim_date} 沪指下跌 "
                                 f"{_row0['pct'].iloc[0]:.2f}%", icon="🔻")
        if c2.button("◀️ 回退一天", use_container_width=True,
                     help="回到上一个交易日，并撤销当前这天的全部买卖"):
            sim_date, _moved = simulator.rollback_day()
            if not _moved:
                st.toast("已在第一个交易日，无法回退", icon="⚠️")
        with c3.popover("📂 存档管理", use_container_width=True):
            _arch_name = st.text_input(
                "存档名称", placeholder="如：半导体轮动策略", key="arch_name")
            if st.button("💾 保存当前模拟盘", type="primary", key="arch_save",
                         use_container_width=True):
                _err = simulator.save_archive(_arch_name)
                if _err:
                    st.error(_err)
                else:
                    st.session_state["sim_msg"] = (
                        f"已存档「{_arch_name.strip() or '未命名'}」")
                    st.rerun()
            _archs = simulator.list_archives()
            if not _archs.empty:
                st.divider()
                st.caption("⚠️ 载入会覆盖当前模拟盘的全部交易与日期；"
                           "如需保留当前进度，请先存档。")
                for _, _a in _archs.iterrows():
                    _aid = int(_a["id"])
                    # Name is edited in place: change + Enter saves it.
                    _new_name = st.text_input(
                        "策略名称", value=_a["name"],
                        key=f"arch_rename_{_aid}",
                        label_visibility="collapsed",
                        help="直接修改名称，回车保存")
                    if _new_name.strip() and _new_name.strip() != _a["name"]:
                        _err = simulator.rename_archive(_aid, _new_name)
                        st.toast(_err or f"已改名为「{_new_name.strip()}」",
                                 icon="✏️")
                    st.caption(
                        f"{_a['start_date']} ~ {_a['current_date']} · "
                        f"{_a['n_trades']} 笔交易 · 存于 "
                        + time.strftime("%m-%d %H:%M",
                                        time.localtime(_a["saved_at"])))
                    a1, a2, a3 = st.columns(3)
                    if a1.button("载入", key=f"arch_load_{_a['id']}",
                                 use_container_width=True):
                        _err = simulator.load_archive(int(_a["id"]))
                        st.session_state["sim_msg"] = (
                            _err or f"已载入存档「{_a['name']}」")
                        st.rerun()
                    if a2.button("复制", key=f"arch_copy_{_a['id']}",
                                 use_container_width=True,
                                 help="复制一份副本，原方案保持不变，"
                                      "副本可载入后继续修改"):
                        _err = simulator.copy_archive(int(_a["id"]))
                        st.session_state["sim_msg"] = (
                            _err or f"已复制存档「{_a['name']}」")
                        st.rerun()
                    if a3.button("删除", key=f"arch_del_{_a['id']}",
                                 use_container_width=True):
                        simulator.delete_archive(int(_a["id"]))
                        st.rerun()
        with c4.popover("🗑️ 重置", use_container_width=True):
            _min_d = simulator.earliest_nav_day()
            _max_d = simulator.latest_trading_day()
            _new_start = st.date_input(
                "起始日期",
                value=dt.date.fromisoformat(simulator.get_start_date()),
                min_value=dt.date.fromisoformat(_min_d) if _min_d else None,
                max_value=dt.date.fromisoformat(_max_d) if _max_d else None,
                key="sim_start_pick",
                help="选中非交易日会自动顺延到下一个交易日。")
            st.caption("清空全部模拟交易，从上面选定的起始日重新开始。")
            if st.button("确认重置", type="primary", key="sim_reset_confirm"):
                _snapped, _err = simulator.set_start_date(_new_start.isoformat())
                if _err:
                    st.error(_err)
                else:
                    _note = ("" if _snapped == _new_start.isoformat()
                             else f"（{_new_start} 非交易日，顺延至 {_snapped}）")
                    st.session_state["sim_msg"] = f"模拟盘已重置，从 {_snapped} 开始{_note}"
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

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("模拟日期", sim_date)
        m2.metric("总资产", f"¥{total:,.0f}", delta=f"{day_pnl:+,.0f} 当日")
        m3.metric("现金", f"¥{cash:,.0f}")
        m4.metric("总收益", f"¥{total_pnl:+,.0f}")
        m5.metric("总收益率", f"{total_pnl / simulator.INITIAL_CAPITAL * 100:+.2f}%")
        # 沪指 stays visible even with no holdings — it's the operating
        # signal; without it an empty-portfolio day hides whether 上证 fell.
        _sse_all = load_sse_daily()
        if _sse_all is not None and not _sse_all.empty:
            _upto = _sse_all[_sse_all["date"] <= sim_date]
            if not _upto.empty:
                _r = _upto.iloc[-1]
                m6.metric(
                    "上证指数" + ("" if _r["date"] == sim_date
                                  else f"（{_r['date']}）"),
                    f"{_r['close']:,.0f}",
                    delta=(f"{_r['pct']:+.2f}% 当日"
                           if pd.notna(_r["pct"]) else None))
        st.caption(
            f"从 {simulator.get_start_date()} 开始 · "
            f"初始资金 ¥{simulator.INITIAL_CAPITAL:,.0f} · "
            "按当日单位净值成交，不计手续费 · 回退一天会撤销当天的全部买卖 · "
            "起始日期可在「重置」中修改")

        st.divider()
        hold = simulator.holdings_table(sim_date)
        trades = simulator.trades_table(sim_date)

        # ── 图表与持仓表现数据（先构建，再进布局）──
        # Each line starts at that position's entry day at 0% and compounds
        # the fund's official DAILY RETURNS (dividend-adjusted), not the raw
        # unit NAV — a payout day resets the unit NAV (looks like a -10%
        # cliff) while the actual daily return stays ordinary. History stops
        # at the simulated date — no peeking at the future.
        fig_ret, _dd_rows = None, []
        if not hold.empty:
            _frames = []
            for _, _h in hold.iterrows():
                _c = _h["code"]
                if not _h["open_nav"]:
                    continue
                _s = simulator.nav_series(_c, _h["open_date"], sim_date)
                if _s.empty:
                    continue
                # Growth index anchored at 1 on the entry day (the entry
                # day's own return predates the EOD fill, so it's divided
                # out); missing returns count as flat.
                _g = (1.0 + pd.to_numeric(_s["daily_ret_pct"],
                                          errors="coerce").fillna(0.0)
                      / 100.0).cumprod()
                _g = _g / _g.iloc[0]
                _frames.append(_s.assign(
                    adj=_g,
                    cum_ret=(_g - 1.0) * 100.0,
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
                    mode="lines+markers", marker=dict(size=8),
                    hovertemplate="%{y:+.2f}%<extra>%{fullData.name}</extra>",
                )
                fig_ret.add_hline(
                    y=0, line_dash="dot", line_color="gray", opacity=0.5)
                fig_ret.update_layout(
                    hovermode="x unified", hoverdistance=-1,
                    spikedistance=-1)
                # Ticks: at least one day apart (~8 ticks over the span),
                # so a short history never falls back to hour-level ticks
                # that would all render as the same date.
                _rdates = pd.to_datetime(_rets["date"])
                _span_d = max((_rdates.max() - _rdates.min()).days, 1)
                fig_ret.update_xaxes(
                    showspikes=True, spikemode="across", spikesnap="data",
                    spikedash="dot", spikethickness=1,
                    hoverformat="%Y-%m-%d", tickformat="%Y-%m-%d",
                    dtick=max(1, _span_d // 8) * 86400000)

                _add_sse_drop_bands(fig_ret, load_sse_daily(),
                                    _rdates.min(), _rdates.max())

                # Max drawdown since each position's buy date, from the same
                # daily-return growth index the chart uses (so a dividend's
                # NAV reset never counts as a drawdown), plus the fund's 1y
                # max drawdown as it stood on the buy date for comparison.
                _open_dates = dict(zip(hold["code"], hold["open_date"]))
                for _f in _frames:
                    _peak = _f["adj"].cummax()
                    _mdd = ((_peak - _f["adj"]) / _peak).max() * 100.0
                    # Run-up: current rise from the lowest point since the
                    # position was opened, on the same growth index.
                    _low = float(_f["adj"].min())
                    _mru = (float(_f["adj"].iloc[-1]) / _low - 1.0) * 100.0
                    _ret = float(_f["cum_ret"].iloc[-1])
                    _label = _f["fund"].iloc[0]
                    _c = _label.split()[0]
                    _ref = mdd_1y_at_buy(_c, str(_open_dates[_c]))
                    _dd_rows.append((_label, _ret, _mdd, _mru, _ref))
                _dd_rows.sort(key=lambda r: r[2], reverse=True)

        # ── 布局：左 = 图表/持仓/交易记录，右 = 交易面板 + 持仓表现 ──
        col_main, col_side = st.columns([2.8, 1.1], gap="medium")

        with col_side:
            st.markdown("##### 🛒 交易")
            with st.form("sim_buy", clear_on_submit=True):
                st.markdown("**买入**")
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
                st.markdown("**卖出**")
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

        with col_main:
            if fig_ret is not None:
                _chart_col, _dd_col = st.columns([3, 1.4])
                with _chart_col:
                    st.plotly_chart(fig_ret, use_container_width=True)
                    st.caption("🔻 红色竖带 = 上证指数当日下跌超 1%")
                with _dd_col:
                    st.markdown(
                        "**📊 持仓表现（自买入）**",
                        help="总收益率、当前最大前进（当前净值相对买入以来最低点"
                             "的涨幅）、最大回撤均自买入日起算；最后一行为"
                             "买入时点的近1年最大回撤，作为比较基准：回撤或"
                             "亏损幅度超过它时红色提示，当前最大前进超过它时"
                             "绿色标记。")
                    _RED, _GREEN, _GRAY = "#e0454b", "#21a366", "#8a8f98"

                    def _dd_line(label, value, color, bold=False):
                        return (
                            "<div style='display:flex;justify-content:"
                            "space-between;font-size:0.82rem;"
                            "line-height:1.7;'>"
                            f"<span style='color:{_GRAY};'>{label}</span>"
                            f"<span style='color:{color};"
                            f"font-weight:{600 if bold else 400};"
                            f"font-variant-numeric:tabular-nums;'>"
                            f"{value}</span></div>")

                    for _name, _ret, _mdd, _mru, _ref in _dd_rows:
                        _dd_over = _ref is not None and _mdd > _ref
                        _ret_over = (_ref is not None
                                     and _ret < 0 and -_ret > _ref)
                        _mru_over = _ref is not None and _mru > _ref
                        _alarm = _dd_over or _ret_over
                        _border = _RED if _alarm else "rgba(128,128,128,.3)"
                        _rows = (
                            _dd_line("总收益率", f"{_ret:+.2f}%",
                                     _RED if _ret < 0 else _GREEN,
                                     bold=_ret_over)
                            + _dd_line("当前最大前进", f"+{_mru:.2f}%",
                                       _GREEN if _mru_over else "inherit",
                                       bold=_mru_over)
                            + _dd_line("最大回撤", f"-{_mdd:.2f}%",
                                       _RED if _dd_over else "inherit",
                                       bold=_dd_over)
                            + _dd_line("买入时近1年回撤",
                                       (f"-{_ref:.2f}%"
                                        if _ref is not None else "无数据"),
                                       _GRAY))
                        st.markdown(
                            "<div style='border:1px solid "
                            f"{_border};border-radius:8px;"
                            "padding:6px 10px;margin-bottom:8px;'>"
                            "<div style='font-size:0.84rem;"
                            "font-weight:600;margin-bottom:2px;'>"
                            f"{'⚠️ ' if _alarm else ''}{_name}</div>"
                            f"{_rows}</div>",
                            unsafe_allow_html=True)

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
                    _trades_view = pd.DataFrame({
                        "日期": trades["date"],
                        "操作": trades["action"].map(
                            {"buy": "买入", "sell": "卖出"}),
                        "代码": trades["code"],
                        "名称": trades["code"].map(_code_names),
                        "份额": trades["shares"].round(2),
                        "成交净值": trades["nav"],
                        "金额(¥)": trades["amount"].round(2),
                        "卖出盈亏(¥)": pd.to_numeric(
                            trades["pnl"], errors="coerce").round(2),
                        "卖出盈亏(%)": pd.to_numeric(
                            trades["pnl_pct"], errors="coerce").round(2),
                    }).iloc[::-1].reset_index(drop=True)

                    # 卖出行按已实现盈亏上色：绿=盈利卖出，红=亏损卖出
                    # （与持仓表现卡一致：绿好红坏）。
                    def _sell_row_style(row):
                        if row["操作"] == "卖出" and pd.notna(row["卖出盈亏(¥)"]):
                            _c = "#21a366" if row["卖出盈亏(¥)"] >= 0 else "#e0454b"
                            return [f"color: {_c}; font-weight: 600"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        _trades_view.style.apply(_sell_row_style, axis=1)
                        .format(precision=2, na_rep=""),
                        use_container_width=True)
                    # utf-8-sig BOM so Excel opens the CSV with correct 中文.
                    st.download_button(
                        "⬇️ 导出交易记录 CSV",
                        _trades_view.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"模拟盘交易记录_{sim_date}.csv",
                        mime="text/csv",
                    )
