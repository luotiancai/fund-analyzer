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
import streamlit.components.v1 as components

# Streamlit 首个元素入队时会无条件调 env_util.is_repl()(inspect.stack 扫
# 全部已加载模块的源文件),在 /mnt/c 的 9p 文件系统上实测 ~8s,占了首屏
# 卡顿的大半。该检查只为打印「请用 streamlit run」提示,置位其去重标志
# 直接跳过;私有属性,失败则退回原行为。
try:
    import streamlit.delta_generator as _dg
    _dg._use_warning_has_been_displayed = True
except Exception:
    pass

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
    st.caption("夏普比率 = (年化收益 − 无风险利率) / 年化波动率，日频收益计算；"
               "支付宝展示值为周频口径，通常比这里高 0.3 左右")

update_btn = st.session_state.pop("_run_update", False)

# ── Load fund list ────────────────────────────────────────────────────────────
# `cache_key` 是 SQLite 里榜单的 saved_at:数据版本没变就一直命中(TTL 只是
# 兜底),变了(更新按钮/update_daily.py 刷新)立即失效。之前按 1h TTL 过期
# 会在盘中交互时穿透到网络全量重拉,页面卡几十秒且标签页被顶回首页。
@st.cache_data(ttl=24 * 3600, show_spinner="正在加载基金列表…")
def load_fund_list(cache_key):
    return fetcher.fetch_fund_list(force_refresh=False)


# Per-fund detail data. fetcher caches in SQLite, but these wrappers matter on
# reruns: every click anywhere (e.g. 「开始筛选」) re-executes the detail tab,
# and without them each rerun re-read NAV, re-fetched holdings and — worst —
# recomputed + re-WROTE the fund's Sharpe row on the slow /mnt/c disk.
@st.cache_data(ttl=3600, show_spinner=False)
def load_holdings(code: str):
    """(持仓df, 穿透来源) — ETF联接基金的持仓来自同指数场内ETF,
    来源为 (代码, 名称);非联接基金来源为 None。"""
    df = fetcher.fetch_holdings(code)
    return df, fetcher.resolve_target_etf(code)


@st.cache_data(ttl=3600, show_spinner=False)
def load_nav(code: str):
    return fetcher.fetch_nav(code)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fund_metrics(code: str, rf: float):
    return fetcher.compute_sharpe_for_fund(code, rf=rf)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_sse_daily():
    return fetcher.fetch_sse_daily()


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_qvix_daily():
    return fetcher.fetch_qvix_daily()


# 1y max drawdown as it stood on the buy date (window: buy_date-365d → buy_date),
# on the corrected daily-return growth index (nav_series 的 ret 列) so dividend
# NAV resets don't count as drops, while build-up-period fake-zero growth rates
# don't hide real ones. Immutable history, so the (code, buy_date) cache never
# needs invalidating.
@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def mdd_1y_at_buy(code: str, buy_date: str):
    _start = (pd.Timestamp(buy_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    _s = simulator.nav_series(code, _start, buy_date)
    # 买入决策发生在当日盘中，当日净值尚未公布 — 参考窗口止于前一交易日。
    _s = _s[_s["date"] < buy_date]
    if len(_s) < 2:
        return None
    _adj = (1.0 + _s["ret"].fillna(0.0)).cumprod()
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


# Earliest stored NAV date per fund, for the period-matched fund-age exclusion
# in the filter path. Same invalidation contract as load_metrics_df.
@st.cache_data(ttl=3600, show_spinner=False)
def load_first_dates(cache_key):
    df = fetcher.nav_first_dates()
    return None if df.empty else df

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
    load_first_dates.clear()
    _bar.progress(1.0, text="完成")
    st.success(
        f"更新完成 · 基金 {summary['funds']:,} · 回填 {summary['backfilled']} · "
        f"当日追加 {summary['appended']:,} · 补缺口 {summary['patched']:,}"
        + (f"（失败 {summary['failed']:,}）" if summary["failed"] else "")
        + " · 指标将在筛选时按需计算"
    )

with st.spinner("加载基金列表中…"):
    fund_df = load_fund_list(fetcher.fund_list_saved_at())

if fund_df is None or fund_df.empty:
    st.error("无法获取基金列表，请检查网络连接。")
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
# Each time-period maps to its return column (locally recomputed from stored
# NAV where available, rank-list value otherwise — see the merge in the filter
# path), its computed max-drawdown column, and its computed Sharpe column (only
# 6m/1y have Sharpe; shorter windows are too noisy, so None means "no Sharpe").
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
tab_table, tab_detail, tab_sim, tab_sse = st.tabs(
    ["📋 基金列表", "🔍 基金详情", "💰 模拟盘", "📈 上证指数"])

# ─── Tab 1: Table ────────────────────────────────────────────────────────────
with tab_table:
    # 债券/固收类基金整体排除在筛选之外(用户不做债基):类型选项里不出现,
    # 结果里也硬性剔除(见下面 filter 路径)。判定统一在 fetcher.is_bond。
    all_types = sorted(
        t for t in fund_df["type"].dropna().unique().tolist()
        if not fetcher.is_bond(t)
    ) if "type" in fund_df.columns else []
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
            min_ret = st.number_input("所选区间最低收益率 %", value=50.0, step=1.0)
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
                help="还原你在该日进行筛选时能看到的结果：只用该日之前"
                     "（不含当日，当日净值当时尚未公布）的净值历史重算"
                     "收益/回撤/夏普。本地净值（仅C类）从 2020-01-01 起，"
                     "因此最早可选 2021-01-01，保证近1年窗口有完整数据。",
            )
        submitted = st.form_submit_button("🔍 开始筛选", type="primary")

    # ── 截至日期日历上标记上证大跌日 ──────────────────────────────────────────
    # st.date_input 的日历(BaseWeb Datepicker)不支持按日期加样式,从组件
    # iframe 注入脚本到父页面:MutationObserver 监听日历弹层出现,按日期格子
    # 的 aria-label 解析出日期(streamlit 1.50 固定为英文,如 "Choose
    # Wednesday, July 16th 2026. It's available."),给上证跌超1%的交易日画
    # 红色下划线,悬浮显示当日跌幅。日历关闭时选择器查不到任何格子,零开销。
    _sse_marks = load_sse_daily()
    if _sse_marks is not None and not _sse_marks.empty:
        _mk_pct = pd.to_numeric(_sse_marks["pct"], errors="coerce")
        _mk = _sse_marks[_mk_pct <= -1.0]
        _drop_map = dict(zip(_mk["date"].astype(str),
                             _mk_pct[_mk_pct <= -1.0].round(2).astype(float)))
        components.html("""<script>
        const DROPS = __DROPS__;
        const MONTHS = {January:1, February:2, March:3, April:4, May:5, June:6,
                        July:7, August:8, September:9, October:10,
                        November:11, December:12};
        const RE = new RegExp(
          '(' + Object.keys(MONTHS).join('|') + ') (\\\\d+)(?:st|nd|rd|th),? (\\\\d{4})');
        const doc = window.parent.document;
        function mark() {
          doc.querySelectorAll('div[data-baseweb="calendar"] [aria-label]')
            .forEach(el => {
              const m = el.getAttribute('aria-label').match(RE);
              if (!m) return;
              const iso = m[3] + '-' + String(MONTHS[m[1]]).padStart(2, '0')
                        + '-' + String(m[2]).padStart(2, '0');
              // 翻月时 BaseWeb 复用同一批格子 DOM,必须显式清除,否则上个
              // 月的标记会残留在同位置的格子上。
              if (iso in DROPS) {
                el.style.boxShadow = 'inset 0 -3px 0 0 #e0454b';
                el.title = '沪指 ' + DROPS[iso].toFixed(2) + '%';
              } else {
                el.style.boxShadow = '';
                el.removeAttribute('title');
              }
            });
        }
        new MutationObserver(mark).observe(doc.body, {subtree: true, childList: true});
        </script>""".replace("__DROPS__", json.dumps(_drop_map)), height=0)
        st.caption("📅 截至日期的日历中,红色下划线标记 = 上证当日跌超 1%")

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


    # Lazy metrics recompute — the piece the 「🔄 更新数据」 button's
    # do_recompute=False has always been promising ("筛选时按需计算"): if NAV
    # data changed since the last full recompute, the stored Sharpe/drawdown
    # are yesterday's and would let funds past today's thresholds (e.g. a fund
    # whose latest drop pushed 近1年回撤 over the cutoff). Recompute the whole
    # table once, here, before any filtering or cache lookup.
    if filter_ready and not asof_mode and fetcher.metrics_stale():
        _bar = st.progress(0.0, text="🧮 净值已更新，正在重算全市场指标…")
        fetcher.recompute_all(
            rf=rf_rate,
            progress_callback=lambda d, t: _bar.progress(
                (d / t) if t else 1.0,
                text=f"🧮 净值已更新，正在重算全市场指标… {d:,}/{t:,}",
            ),
        )
        load_metrics_df.clear()
        _bar.empty()

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
            # Bumped when the filter rules change (v3: exclude funds younger
            # than the *selected* period window; v4-v6: exclude 债券/固收/偏债
            # types, is_bond 逐步收敛为「含债或固收」; v7: 回撤改按校正收益
            # 复利口径,与模拟盘一致), so stale cached results never get served.
            "rule_ver": 7,
            # Combines the Sharpe/drawdown recompute timestamp with the fund
            # list's own saved_at: the in-app update button refreshes the list
            # (fresh returns) but skips recompute_all, so last_update_time()
            # alone wouldn't invalidate this cache on its own.
            "data_ver": None if asof_mode
                else [fetcher.last_update_time(), fetcher.fund_list_saved_at()],
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
            # 只算所选区间的收益/回撤/夏普三列(其余窗口筛选和展示都用不到,
            # 全算耗时翻倍),缓存键因此带上区间。
            _cols = {ret_col, mdd_col} | ({sharpe_col} if sharpe_col else set())
            _key = (_asof_iso, round(rf_rate, 6), tuple(sorted(_cols)))
            if _key not in _cache:
                _bar = st.progress(0.0, text=f"📅 正在按 {_asof_iso} 重算全市场指标（约半分钟）…")
                _cache[_key] = fetcher.compute_metrics_asof(
                    _asof_iso, rf_rate, cols=_cols,
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
                f"📅 快照模式：按 {_asof_iso} 之前（不含当日）的净值计算"
                f"收益/夏普/回撤，即当天筛选时实际可见的数据"
                f"（覆盖 {len(work_df):,} 只基金）"
            )
        else:
            work_df = fund_df

        with st.spinner("⏳ 正在筛选…"):
            filtered = work_df.copy()
            # 债券/固收类基金硬性剔除(类型选项里也已不出现)。
            if "type" in filtered.columns:
                filtered = filtered[~filtered["type"].map(fetcher.is_bond)]
            if selected_types:
                filtered = filtered[filtered["type"].isin(selected_types)]

            # ── Merge Sharpe/drawdown/period returns BEFORE the return filter ────
            # (In as-of mode work_df already carries the snapshot metrics columns.)
            # 区间收益优先用本地净值重算的值:榜单接口早间常出现净值/日增长率已
            # 更新到最新交易日、而近X收益率列仍是前一窗口旧值的情况(实测 018359
            # 榜单近1年 226.03 = 截至7-15,实际截至7-16 应为 211.39)。无本地净值
            # 历史的基金(非C类)回退榜单值——它们本来也没有夏普/回撤。
            if not asof_mode:
                sharpe_df = load_metrics_df(fetcher.last_update_time())
                if sharpe_df is not None:
                    filtered = filtered.merge(sharpe_df, on="code", how="left",
                                              suffixes=("_list", ""))
                    for _rc in ("ret_1m", "ret_3m", "ret_6m", "ret_1y"):
                        if _rc in filtered.columns and f"{_rc}_list" in filtered.columns:
                            filtered[_rc] = pd.to_numeric(
                                filtered[_rc], errors="coerce"
                            ).fillna(pd.to_numeric(
                                filtered[f"{_rc}_list"], errors="coerce"))

            # ── 剔除历史不满所选区间的基金(与区间匹配) ──────────────────────
            # 选近1年剔除不满1年的,选近6月剔除不满6月的,依此类推。有本地净值
            # 的基金按首日净值日期算真实历史长度——本地重算带 10 天锚点宽限
            # (ANCHOR_GRACE_DAYS),355 天的基金也能算出「近1年」值,这里按
            # 严格天数堵住该口子。无本地净值的基金(非C类)由下面的区间收益率
            # 过滤兜底:榜单对历史不满该区间的基金该列留空,NaN 过不了 >=。
            _min_days = fetcher.RETURN_DAYS.get(ret_col)
            _fd = load_first_dates(fetcher.last_update_time()) if _min_days else None
            if _fd is not None:
                _ref = pd.Timestamp(asof_date) if asof_mode \
                    else pd.Timestamp.today().normalize()
                filtered = filtered.merge(_fd, on="code", how="left")
                _too_young = (
                    filtered["first_nav_date"].notna()
                    & ((_ref - filtered["first_nav_date"]).dt.days < _min_days)
                )
                filtered = filtered[~_too_young].drop(columns=["first_nav_date"])

            if ret_col in filtered.columns:
                # Only keep funds that actually have a return for the selected
                # period; NaN is dropped — this is also what excludes too-young
                # funds that have no local NAV (rank list leaves the column blank).
                filtered = filtered[
                    pd.to_numeric(filtered[ret_col], errors="coerce") >= min_ret]

            display = filtered.copy()

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

        def _latest_holdings(fcode):
            """基金在截至日期下最新披露季度的持仓:(季度, DataFrame, 来源)
            或 (None, 提示文案, 来源)。"""
            _hdf, _src = load_holdings(fcode)
            if _hdf is None or _hdf.empty:
                return (None, "该基金暂无披露的重仓持仓（货币基金、新基金常见）。",
                        _src)
            _qend = {"1": "-03-31", "2": "-06-30", "3": "-09-30", "4": "-12-31"}
            _h = _hdf.dropna(subset=["quarter"]).copy()
            _h["quarter"] = _h["quarter"].astype(str)
            _h["_qend"] = _h["quarter"].str[:4] + _h["quarter"].str[-1].map(_qend)
            _h = _h[_h["_qend"] <= _asof_lim]
            if _h.empty:
                return (None, ("截至该日期尚无已披露的季度持仓（本地持仓数据从 "
                               f"{fetcher.HOLDINGS_START_Q} 起）。"), _src)
            _q = _h["quarter"].max()
            return _q, _h[_h["quarter"] == _q], _src

        def _cell_top10(hq):
            """卡片实际展示的标的代码集合(股票+债券各前十)。"""
            return {
                str(c) for _kind in ("股票", "债券")
                for c in hq[hq["kind"] == _kind].head(10)["代码"]
            }

        # 共同持仓底色:同一标的在所有卡片里同色,便于横向对照。浅色底 +
        # 深色字,深浅主题下都可读;颜色按(出现基金数降序, 代码)分配,
        # 结果稳定;共同标的多于色板时循环复用(代码/名称仍可区分)。
        _SHARED_BG = ["#ffe0b2", "#c8e6c9", "#bbdefb", "#f8bbd0",
                      "#e1bee7", "#fff9c4", "#b2dfdb", "#ffcdd2"]

        def _render_holdings_cell(frow, q, hq, src, shared_colors):
            _fcode = str(frow["基金代码"]).zfill(6)
            st.markdown(f"**📦 {_fcode} {frow.get('基金名称', '')}**")
            if src:
                st.caption(f"⤴ 联接基金，重仓来自同指数场内 ETF："
                           f"{src[0]} {src[1]}")
            if q is None:
                st.info(hq)   # hq 此时是提示文案
                return
            st.caption(f"披露季度：{q}")
            for _kind, _klabel in (("股票", "重仓股票"), ("债券", "重仓债券")):
                _part = hq[hq["kind"] == _kind].head(10)
                if _part.empty:
                    continue
                st.markdown(f"**{_klabel}（前十）**")
                _tbl = pd.DataFrame({
                    "代码": _part["代码"].astype(str),
                    "名称": _part["名称"],
                    "占净值比例(%)": _part["占净值比例"],
                }).reset_index(drop=True)

                def _shared_row_style(row):
                    _bg = shared_colors.get(str(row["代码"]))
                    return ([f"background-color: {_bg}; color: #1a1a1a"]
                            * len(row) if _bg else [""] * len(row))

                st.dataframe(_tbl.style.apply(_shared_row_style, axis=1),
                             use_container_width=True)

        # 模拟盘在持基金常驻网格最前（无需勾选），其后是表格勾选的基金。
        _sim_d = simulator.get_current_date()
        _sim_codes = sorted(simulator.holdings_and_cash(_sim_d)[0]) if _sim_d else []
        _sim_names = dict(zip(fund_df["code"], fund_df["name"]))
        _cells = [{"基金代码": _c, "基金名称": _sim_names.get(_c, "")}
                  for _c in _sim_codes]
        for _ri in _sel_rows:
            # 重新筛选后表格可能变短,session 里残留的旧勾选行号会越界。
            if _ri >= len(table):
                continue
            _frow = table.iloc[_ri]
            if str(_frow["基金代码"]).zfill(6) not in _sim_codes:
                _cells.append(_frow)

        if _cells:
            st.markdown(f"##### 已选基金的最新十大持仓（截至 {_asof_lim}）")
            st.caption("模拟盘在持基金常驻最前；取季度末 ≤ 截至日期的最新披露季度，"
                       "实际公告通常滞后季度末约两周")

            # 两遍渲染:先取全部卡片的持仓,统计出现在 ≥2 只基金里的标的并
            # 分配底色,再带着颜色映射渲染。
            with st.spinner("加载持仓…"):
                _cell_data = []
                for _cell in _cells:
                    _q, _hq, _src = _latest_holdings(
                        str(_cell["基金代码"]).zfill(6))
                    _cell_data.append((_cell, _q, _hq, _src))

                _code_hits = {}
                for _, _q, _hq, _ in _cell_data:
                    if _q is None:
                        continue
                    for _c in _cell_top10(_hq):
                        _code_hits[_c] = _code_hits.get(_c, 0) + 1
                _shared = sorted(
                    (c for c, n in _code_hits.items() if n >= 2),
                    key=lambda c: (-_code_hits[c], c))
                _shared_colors = {
                    c: _SHARED_BG[i % len(_SHARED_BG)]
                    for i, c in enumerate(_shared)}
                if _shared_colors:
                    st.caption("🎨 相同底色 = 多只基金共同持有的标的"
                               f"（共 {len(_shared_colors)} 只）")

                _PER_ROW = 3
                for _start in range(0, len(_cell_data), _PER_ROW):
                    _chunk = _cell_data[_start:_start + _PER_ROW]
                    _cols = st.columns(_PER_ROW, gap="medium")
                    for _col, (_cell, _q, _hq, _src) in zip(_cols, _chunk):
                        with _col, st.container(border=True):
                            _render_holdings_cell(_cell, _q, _hq, _src,
                                                  _shared_colors)

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
                hold_df, hold_src = load_holdings(code_input.strip().zfill(6))
            if hold_src:
                st.caption(f"⤴ 联接基金，重仓来自同指数场内 ETF："
                           f"{hold_src[0]} {hold_src[1]}")
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
        c1, c1m, c2, c2m, _sp, c3, c4 = st.columns(
            [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1])

        def _toast_if_sse_drop(_d):
            # 主要操作信号：落到沪指跌超1%的日子就提示——空仓时没有
            # 持仓图表可看，这条提示是唯一入口。
            _sse0 = load_sse_daily()
            if _sse0 is not None and not _sse0.empty:
                _row0 = _sse0[_sse0["date"] == _d]
                if (not _row0.empty and pd.notna(_row0["pct"].iloc[0])
                        and _row0["pct"].iloc[0] <= -1.0):
                    st.toast(f"{_d} 沪指下跌 "
                             f"{_row0['pct'].iloc[0]:.2f}%", icon="🔻")

        if c1.button("▶️ 推进一天", type="primary", use_container_width=True):
            sim_date, _moved = simulator.advance_day()
            if not _moved:
                st.toast("已到本地数据的最新日期，无法再推进", icon="⚠️")
            else:
                _toast_if_sse_drop(sim_date)
        if c1m.button("⏩ 推进一月", use_container_width=True,
                      help="跳到一个月后的最近交易日（数据不足一个月则到最新日期）"):
            sim_date, _moved = simulator.advance_month()
            if not _moved:
                st.toast("已到本地数据的最新日期，无法再推进", icon="⚠️")
            else:
                _toast_if_sse_drop(sim_date)
        if c2.button("◀️ 回退一天", use_container_width=True,
                     help="回到上一个交易日，并撤销当前这天的全部买卖"):
            sim_date, _moved = simulator.rollback_day()
            if not _moved:
                st.toast("已在第一个交易日，无法回退", icon="⚠️")
        if c2m.button("⏪ 回退一月", use_container_width=True,
                      help="回到一个月前的最近交易日，并撤销中间所有买卖"):
            sim_date, _moved = simulator.rollback_month()
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
            st.divider()
            _up = st.file_uploader(
                "导入交易记录 CSV", type=["csv"], key="trades_csv_up",
                help="接受「⬇️ 导出交易记录 CSV」导出的文件；"
                     "导入会覆盖当前模拟盘（建议先存档），"
                     "起始日期取首笔交易日，模拟日期定位到末笔交易日。")
            if _up is not None and st.button(
                    "📥 导入并覆盖当前模拟盘", key="trades_csv_go",
                    use_container_width=True):
                try:
                    _csv_df = pd.read_csv(_up, encoding="utf-8-sig",
                                          dtype={"代码": str})
                except Exception:
                    _csv_df = None
                if _csv_df is None:
                    st.error("无法读取 CSV 文件")
                else:
                    _sumr, _err = simulator.import_trades_csv(_csv_df)
                    if _err:
                        st.error(_err)
                    else:
                        st.session_state["sim_msg"] = (
                            f"已导入 {_sumr['n']} 笔交易"
                            f"（{_sumr['first']} ~ {_sumr['last']}），"
                            f"模拟日期已定位到 {_sumr['last']}")
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
            # 日期用纯文本输入：st.date_input 的日历浮层在 popover 里会被
            # 盖住/误触收起（portal z-index 问题），文本框没有这些破事。
            _min_d = simulator.earliest_nav_day()
            _max_d = simulator.latest_trading_day()
            _start_txt = st.text_input(
                "起始日期",
                value=simulator.get_start_date(),
                key="sim_start_pick_txt",
                help=f"格式 YYYY-MM-DD，范围 {_min_d} ~ {_max_d}；"
                     "非交易日自动顺延到下一个交易日。")
            st.caption("清空全部模拟交易，从上面填的起始日重新开始。")
            if st.button("确认重置", type="primary", key="sim_reset_confirm"):
                try:
                    _new_start = dt.date.fromisoformat(
                        _start_txt.strip().replace("/", "-"))
                except ValueError:
                    st.error("日期格式不对，请用 YYYY-MM-DD，如 2021-01-05")
                    _new_start = None
                if _new_start:
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
                # out); missing returns count as flat. `ret` is the corrected
                # daily return (see fetcher.effective_daily_ret).
                _g = (1.0 + _s["ret"].fillna(0.0)).cumprod()
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
                    "买入日期": hold["open_date"],
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

# ─── Tab 4: SSE index ────────────────────────────────────────────────────────
with tab_sse:
    sse_df = load_sse_daily()
    if sse_df is None or sse_df.empty:
        st.warning("上证指数数据获取失败，请稍后重试。")
    else:
        sse_all = sse_df.copy()
        sse_all["date"] = pd.to_datetime(sse_all["date"])
        sse_all = sse_all.sort_values("date").reset_index(drop=True)

        _sse_ranges = {"近1月": 30, "近3月": 91, "近6月": 182,
                       "近1年": 365, "近3年": 365 * 3, "近5年": 365 * 5,
                       "近10年": 365 * 10, "全部": None}
        _c_rng, _c_bands, _c_vix = st.columns([4, 1, 1])
        with _c_rng:
            _rng = st.radio("时间区间", list(_sse_ranges.keys()), index=3,
                            horizontal=True, key="sse_range")
        with _c_bands:
            _show_bands = st.checkbox("标记跌超1%的交易日", value=True,
                                      key="sse_bands",
                                      help="长区间下标记较密，可关闭")
        with _c_vix:
            _show_vix = st.checkbox("VIX恐慌指数", value=True,
                                    key="sse_vix",
                                    help="50ETF期权QVIX（中国版VIX），右轴")

        # Window slice keeps the anchor row (last close on/before the window
        # start) so the period change is measured against the true base point —
        # same convention as _window_by_date.
        _days = _sse_ranges[_rng]
        if _days is None:
            view = sse_all
        else:
            _start = sse_all["date"].max() - pd.Timedelta(days=_days)
            _older = sse_all[sse_all["date"] <= _start]
            view = sse_all.loc[_older.index[-1]:] if not _older.empty else sse_all

        _latest = sse_all.iloc[-1]
        _chg = (_latest["close"] / view["close"].iloc[0] - 1.0) * 100.0
        _peak = view["close"].cummax()
        _mdd = float(((_peak - view["close"]) / _peak).max() * 100.0)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("最新收盘", f"{_latest['close']:,.2f}",
                  f"{_latest['pct']:+.2f}%（当日）")
        k2.metric(f"{_rng}涨跌幅", f"{_chg:+.2f}%")
        k3.metric(f"{_rng}最大回撤", f"{_mdd:.2f}%")
        k4.metric("数据日期", _latest["date"].strftime("%Y-%m-%d"))

        fig_sse = px.line(
            view, x="date", y="close",
            title=f"上证指数走势（{_rng}）",
            labels={"date": "日期", "close": "收盘点位"},
            height=420,
        )
        fig_sse.update_traces(
            customdata=view[["pct"]],
            hovertemplate="收盘 %{y:,.2f} · 日涨跌 %{customdata[0]:+.2f}%"
                          "<extra></extra>")

        # VIX恐慌指数（QVIX）on a secondary right axis: levels (~15–40) are
        # incomparable with index points, so it never shares the left scale.
        qvix_view = None
        if _show_vix:
            _qvix = load_qvix_daily()
            if _qvix is not None and not _qvix.empty:
                qvix_view = _qvix.copy()
                qvix_view["date"] = pd.to_datetime(qvix_view["date"])
                qvix_view = qvix_view[
                    (qvix_view["date"] >= view["date"].min())
                    & (qvix_view["date"] <= view["date"].max())]
            if qvix_view is None or qvix_view.empty:
                st.caption("⚠️ VIX恐慌指数数据暂不可用")
                qvix_view = None
        if qvix_view is not None:
            fig_sse.data[0].name = "上证指数"
            fig_sse.data[0].showlegend = True
            fig_sse.add_trace(go.Scatter(
                x=qvix_view["date"], y=qvix_view["close"],
                name="VIX恐慌指数(QVIX)", yaxis="y2",
                line=dict(color="#f28e2b", width=1.3),
                hovertemplate="VIX %{y:.2f}<extra></extra>"))
            # Default (right-side vertical) legend keeps clear of the
            # 跌超1% annotations that sit above the plot area.
            fig_sse.update_layout(
                yaxis2=dict(title="VIX恐慌指数", overlaying="y", side="right",
                            showgrid=False))
            # 25/30 参考虚线:30 是「QVIX>30 且当日下跌 → 大底」的触发线。
            for _lvl, _dash in ((25, "dot"), (30, "dash")):
                fig_sse.add_shape(
                    type="line", xref="paper", x0=0, x1=1,
                    yref="y2", y0=_lvl, y1=_lvl,
                    line=dict(color="#8e44ad", width=1, dash=_dash),
                    opacity=0.6)
                fig_sse.add_annotation(
                    x=1, xref="paper", xanchor="left", y=_lvl, yref="y2",
                    text=str(_lvl), showarrow=False,
                    font=dict(size=10, color="#8e44ad"))

        if _show_bands:
            _add_sse_drop_bands(fig_sse, sse_df,
                                view["date"].min(), view["date"].max())
        fig_sse.update_layout(
            hovermode="x unified", hoverdistance=-1, spikedistance=-1)
        _span_d = max((view["date"].max() - view["date"].min()).days, 1)
        fig_sse.update_xaxes(
            showspikes=True, spikemode="across", spikesnap="data",
            spikedash="dot", spikethickness=1,
            hoverformat="%Y-%m-%d", tickformat="%Y-%m-%d",
            dtick=max(1, _span_d // 8) * 86400000)
        fig_sse.update_yaxes(
            showspikes=True, spikemode="across", spikesnap="data",
            spikedash="dot", spikethickness=1)
        st.plotly_chart(fig_sse, use_container_width=True)

        with st.expander("📄 每日数据（当前区间）"):
            _sse_table = view.sort_values("date", ascending=False).reset_index(drop=True)
            _tbl = pd.DataFrame({
                "日期": _sse_table["date"].dt.strftime("%Y-%m-%d"),
                "收盘点位": _sse_table["close"].round(2),
                "日涨跌(%)": pd.to_numeric(_sse_table["pct"],
                                         errors="coerce").round(2),
            })
            if qvix_view is not None:
                _q = qvix_view.assign(
                    日期=qvix_view["date"].dt.strftime("%Y-%m-%d"),
                    **{"VIX恐慌指数": qvix_view["close"].round(2)})
                _tbl = _tbl.merge(_q[["日期", "VIX恐慌指数"]],
                                  on="日期", how="left")
            st.dataframe(_tbl, use_container_width=True, height=420)

        # ── 策略复盘:买入当时的近3月冠军,大盘回撤>5% 即卖 ────────────────────
        # 结果为离线重建:按东财 lsjz 历史净值把当日全市场(成立满3月的开放
        # 式基金)的近3月涨幅逐只算出复原榜单,并剔除巨额赎回造成的假榜首
        # (纯债基金单日 +8%~+19% 之类,详见 effective_daily_ret 同款口径),
        # 全市场重算无法在页面现算,故硬编码结论。
        with st.expander("📜 策略复盘:买入近3月冠军,大盘回撤超5%卖出"):
            st.dataframe(pd.DataFrame({
                "买入日": ["2018-02-09", "2018-10-29", "2019-02-25",
                          "2020-02-03", "2020-03-12", "2020-03-18",
                          "2020-07-16", "2022-04-25"],
                "当时近3月冠军(剔除赎回假象)": [
                    "工银新趋势灵活配置A (001716)",
                    "博时裕瑞纯债 (001578)",
                    "国泰中证申万证券行业指数A (501016)",
                    "万家经济新动能C (005312)",
                    "万家经济新动能C (005312)",
                    "万家经济新动能C (005312)",
                    "鹏华酒A (160632)",
                    "国泰大宗商品QDII (160216)"],
                "冠军近3月涨幅": ["+8.95%", "+5.20%", "+38.70%", "+35.57%",
                                "+34.36%", "+26.76%", "+45.39%", "+25.38%"],
                "卖出日(回撤>5%)": ["2018-03-23", "2018-11-29", "2019-04-26",
                                  "2020-02-28", "2020-03-18", "2020-07-16",
                                  "2020-09-09", "2022-07-15"],
                "持有收益": ["-4.38%", "+1.25%", "-1.14%", "+22.11%",
                            "-4.69%", "+32.52%", "+7.14%", "+1.33%"],
                "同期上证": ["+0.7%", "+1.0%", "+4.2%", "+4.8%",
                            "-6.6%", "+17.6%", "+1.4%", "+10.2%"],
            }), use_container_width=True, hide_index=True)
            st.caption(
                "案例一买在银行/价值动量顶点,随后风格切换,大盘没跌冠军亏 4.4%;"
                "案例三买在券商涨停潮当天,动量已透支,牛市里跑输大盘 5 个点;"
                "案例二熊市冠军全是债基,趋势延续小赚。案例四是唯一大赚的:买入"
                "日恰为疫情恐慌暴跌日(恐慌买点成立)且半导体动量未完,19 个交易"
                "日 +22%(峰值 +34%)。案例五、六是同一只冠军在熔断周前后的"
                "对照:3/12 买在下跌中段,4 个交易日止损 -4.7%;3/18 买在熔断"
                "尾声,拿满 120 天 +32.5%(峰值 +51.6%),止损线在前者是保护、"
                "在后者全程未触发。结论:追冠军的成败=买入时点的恐慌程度 ×"
                "冠军风格的剩余动量;垂直拉升后追入必败,恐慌尽头买尚有动量的"
                "冠军超额最大,下跌中段被止损打出也认——它同时躲开了更深的坑。"
                "案例七买在白酒单日 -8.9% 的急跌上(冠军自带恐慌折价),55 天"
                "+7.1%、峰值 +17.8%,跑赢同期上证近 6 个点。案例八是新败法:"
                "4/25 是标准的 A 股恐慌日(-5.1%),大盘随后 V 弹 +10%,但冠军"
                "是原油 QDII——动量在海外油价、与 A 股恐慌修复完全脱钩,峰值"
                "+19.6% 随油价见顶全数回吐,仅 +1.3%:恐慌买点要配 A 股冠军"
                "才吃得到反弹。2018 两次榜单原始前几名均为巨额赎回假净值"
                "(单日 +8%~+19% 的\"纯债\"),已剔除;2019/2020/2022 榜首真实。")
