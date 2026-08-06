"""Fund Analyzer — Streamlit dashboard."""

import datetime as dt
import gzip
import hashlib
import json
import math
import os
import shutil
import threading
import time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
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

_CST = ZoneInfo("Asia/Shanghai")


def _fmt_cst(ts: float, fmt: str) -> str:
    """Unix ts → 北京时间字符串,不依赖服务器自身时区(云端容器多半是
    UTC,time.localtime() 直接用会把 UTC 挂钟时间当北京时间显示,差 8
    小时却还显得"合理"——比如 22:45 UTC 写成"指标数据更新于…22:45",
    用户按北京时间读就觉得是昨晚,其实是当天早上6点多)。"""
    return dt.datetime.fromtimestamp(ts, tz=_CST).strftime(fmt)


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


# ── 云端数据引导 ──────────────────────────────────────────────────────────────
# Streamlit Community Cloud 的容器磁盘是临时的:数据库由 GitHub Actions 每日
# 跑批后上传到 Release(tag `data`,见 .github/workflows/update-daily.yml),
# 应用启动时/每小时比对 asset 的版本戳,变了才重新下载(gzip 压缩传输)。
# 本地开发机(库已存在且无 marker 文件)完全跳过,零网络开销。
# marker 同时充当"云端只读模式"开关:存在则隐藏「更新数据」按钮。
#
# 数据按 fetcher.DB_LAYOUT 分成若干个独立的库, 每个库一个 Release 资产
# (<文件名>.gz)。每条发布线互不覆盖 —— 以前所有表挤在一个 400MB 的
# fund_cache.db 里, 发布单元跟更新节奏对不上, 推一次策略结果就能把跑批攒
# 了五天的行情盖回去(2026-08-05)。
#
# 首屏同步拉 rank/market/strategy/sim, 合计 gz 约 2MB, 拉完立刻能画;
# nav(净值 233MB)和 scale(季度数据 13.7MB)首屏一行都不读, 起线程后台拉,
# 落地后 adopt_db 接管。cache 库压根不拉: 它是筛选结果缓存, 丢了自动重算,
# 下一份别人的缓存过来毫无价值。
_DB_BASE = os.environ.get("FUND_ANALYZER_RELEASE_BASE") or (
    "https://github.com/luotiancai/fund-analyzer/releases/download/data/")
_ASSET = {name: _DB_BASE + fn + ".gz"
          for name, fn, _t, _l in fetcher.DB_LAYOUT}
_MARKER = {name: fetcher.DB_PATH[name] + ".from-release"
           for name, _fn, _t, _l in fetcher.DB_LAYOUT}
_EAGER_DBS = ("rank", "market", "strategy", "sim")
_LAZY_DBS = fetcher.LAZY_DBS          # ("nav", "scale")


def _download_gz(url: str, dest: str, marker: str) -> bool:
    """拉 gz 资产解压到 dest,marker 存版本戳;返回 dest 是否就位。

    直连下载地址(302 到对象存储),不走 api.github.com:未认证 API 每 IP
    每小时限 60 次,Streamlit Cloud 出口 IP 共享极易 403,引导一失败就退成
    空库。用响应头 ETag 当版本戳,没变就提前断开不下载。
    """
    have = os.path.exists(dest) and os.path.getsize(dest) > 1024
    try:
        dl = requests.get(url, stream=True, timeout=600)
        dl.raise_for_status()
    except Exception:
        return have                      # 拉不到:有旧快照就先用着
    with dl:
        stamp = (dl.headers.get("ETag")
                 or dl.headers.get("Last-Modified") or "")
        if (have and stamp and os.path.exists(marker)
                and open(marker).read().strip() == stamp):
            return True                  # 版本没变
        tmp = dest + ".tmp"
        try:
            with open(tmp, "wb") as f, gzip.GzipFile(fileobj=dl.raw) as gz:
                shutil.copyfileobj(gz, f)
            os.replace(tmp, dest)
            with open(marker, "w") as m:
                m.write(stamp)
            return True
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            return have                  # 下载失败:继续用旧快照


@st.cache_resource(ttl=3600, show_spinner="正在同步云端数据库…")
def _sync_db_from_release() -> bool:
    """确保各库就位;返回是否运行在 Release 快照上(云端只读模式)。"""
    # 空壳库(引导失败后 _init_db_once 建的,仅几十KB)不算"本地自有",
    # 照常走同步——否则一次引导失败就把应用永久卡在空库本地模式,
    # 页面表现为基金列表空、QVIX 不可用、「更新数据」按钮在云端露出。
    rank_db = fetcher.DB_PATH["rank"]
    have_db = (os.path.exists(rank_db)
               and os.path.getsize(rank_db) > 1024 * 1024)
    if have_db and not os.path.exists(_MARKER["rank"]):
        return False                     # 本地自有数据库,不碰
    ok = False
    for name in _EAGER_DBS:
        got = _download_gz(_ASSET[name], fetcher.DB_PATH[name], _MARKER[name])
        if name == "rank":
            ok = got                     # 榜单拉不到就等于没数据
        elif not got:
            # 其余几个拉不到不影响主流程: _conn() 会 ATTACH 出空库, 对应的
            # 区块自己显示"暂无数据"(如复盘区的"暂无回测明细")。
            fetcher.logger.warning("资产 %s 没拉到, 该库按空处理", name)
    if ok:
        fetcher.LAZY_NAV = True          # 大库另发, 由 _start_lazy_downloads 接
    return ok or have_db


@st.cache_resource(ttl=3600, show_spinner=False)
def _start_lazy_downloads():
    """后台线程拉大库(净值/季度数据),拉完让它接管。首屏不等这些线程。

    起线程而不是同步拉:233MB 挡在首屏前面就白拆了。期间需要净值的三处
    (筛选/详情/模拟盘)由 fetcher.nav_ready() 挡住,显示"加载中"而不是
    拿空表算出一堆看着合理的错数。

    TTL 与 _sync_db_from_release 一致(1h):跑批发新快照后 rank 会换成当天
    的,大库也得跟着换,否则各库的数据日期能差出一天。ETag 没变时
    _download_gz 直接返回、不重下,adopt_db 本身幂等,空跑代价接近零。

    先拉 scale(13.7MB)再拉 nav(233MB):规模数据是基金列表的门槛要用的,
    早几十秒到位就早几十秒不再"查不到规模一律放行"。
    """
    def _work():
        for name in sorted(_LAZY_DBS, key=lambda n: n != "scale"):
            try:
                if _download_gz(_ASSET[name], fetcher.DB_PATH[name],
                                _MARKER[name]):
                    fetcher.adopt_db(name)
            except Exception:
                fetcher.logger.exception("%s 库后台下载失败", name)

    t = threading.Thread(target=_work, name="lazy-db-download", daemon=True)
    t.start()
    return t


_IS_CLOUD = _sync_db_from_release()
_init_db_once()
if fetcher.LAZY_NAV:
    _start_lazy_downloads()


def _nav_notice(what: str):
    """重表还在后台下载时的占位提示。"""
    st.info(f"⏳ 正在后台加载净值数据（约 60MB，仅本次启动需要一次），"
            f"{what}要等它就位。稍等十几秒后再点一次即可。")

# ── 更新数据(仅本地)────────────────────────────────────────────────────────
# 侧边栏已整体移除:云端数据由每日跑批自动更新,无需任何入口;
# 本地入口缩成页面右上角一个小按钮,无风险利率(进夏普计算)折进悬浮提示。
@st.dialog("确认更新数据")
def _confirm_update():
    st.write("将增量拉取最新净值（不重算指标，筛选时按需计算），确定继续？")
    c1, c2 = st.columns(2)
    if c1.button("确定", type="primary", use_container_width=True):
        st.session_state["_run_update"] = True
        st.rerun()
    if c2.button("取消", use_container_width=True):
        st.rerun()


rf_rate = _get_rf()

if not _IS_CLOUD:
    _, _upd_col = st.columns([6, 1])
    if _upd_col.button("🔄 更新数据", use_container_width=True,
                       help="增量拉取最新净值；指标在筛选时按需计算。"
                            f"无风险利率 {rf_rate*100:.2f}%"
                            "（1年期国债收益率，自动取、进夏普计算）"):
        _confirm_update()

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


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def load_scale_hist(code: str):
    """基金季度规模历史(cache-first, 库里没有就现拉一次并入库)。
    列: quarter_end, aum(亿元), publish_date。"""
    return fetcher.fetch_fund_scale_hist(code)


@st.cache_data(ttl=3600, show_spinner=False)
def load_nav(code: str):
    return fetcher.fetch_nav(code)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fund_metrics(code: str, rf: float):
    return fetcher.compute_sharpe_for_fund(code, rf=rf)


@st.cache_data(show_spinner="正在加载上证指数数据…")
def load_sse_daily(cache_key):
    return fetcher.fetch_sse_daily()


_QVIX_THR_COMBOS = [
    ("2年90%", 490, 0.90, "#f28e2b"),
]


@st.cache_data(show_spinner="正在计算候选恐慌阈值…")
def load_qvix_threshold_combos(cache_key):
    """按 backtest_qvix.py 同一套口径(minp_ratio=0.97,含 shift(1):
    第 d 天画的是"截至 d-1 收盘"的分位,与回测信号/实盘用法同口径)现算
    滚动阈值(窗口×分位),供上证指数图叠加展示——与线上生产阈值列
    (490/0.90,minp=475,fetcher.update_qvix_self_daily 里精确维护,
    2026-07-27 起已从720/0.95对齐到这个口径;那列不 shift,存"截至该日
    收盘"的值,次日实盘拿最后一行来比,语义等价)算法一致但
    独立现算,只用于图上对比展示,不影响生产阈值/策略复盘本身。"""
    hist = fetcher.load_qvix_self_history()
    if hist is None or hist.empty:
        return None
    hist = hist.sort_values("date").reset_index(drop=True)
    hist["date"] = pd.to_datetime(hist["date"])
    hist["qvix"] = pd.to_numeric(hist["qvix"], errors="coerce")
    out = hist[["date"]].copy()
    for label, window, pct, _color in _QVIX_THR_COMBOS:
        minp = int(window * 0.97)
        out[label] = (hist["qvix"].rolling(window, min_periods=minp)
                      .quantile(pct).shift(1))
    return out


@st.cache_data(show_spinner=False)
def load_backtest_trades(cache_key):
    """最新一次标准策略跑批的明细(策略库 strategy_runs,backtest_qvix.py
    跑完落库)。cache_key 传数据库文件的 mtime:云端换快照、本地重跑回测
    都会让它变,缓存随即失效,不需要额外的失效逻辑。"""
    return fetcher.load_backtest_trades()


@st.cache_data(show_spinner=False)
def load_strategy_runs(cache_key):
    """历次回测跑批的摘要(不含明细),新的在前。"""
    return fetcher.list_strategy_runs()


@st.cache_data(show_spinner=False)
def load_strategy_run_detail(cache_key, run_id):
    """某一次跑批的明细 → (DataFrame, params, run_at)。"""
    return fetcher.load_strategy_run(run_id)


@st.cache_data(show_spinner="正在加载VIX恐慌指数数据…")
def load_qvix_self(cache_key):
    """自算QVIX历史(qvix_self_history表,上交所官方期权风险指标反推,
    不是 optbbs)。"""
    return fetcher.load_qvix_self_history()


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
tab_sse, tab_table, tab_detail, tab_sim = st.tabs(
    ["📈 上证指数", "📋 基金列表", "🔍 基金详情", "💰 模拟盘"])

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
        col_f1, col_f2, col_f3, col_f4, col_f6, col_f5 = st.columns(
            [3, 1, 1, 1, 1, 1.2])
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
            min_ret = st.number_input(
                "所选区间最低收益率 %", value=None, step=1.0,
                placeholder="不限",
            )
        with col_f4:
            max_dd = st.number_input(
                "所选区间最大回撤率 %", value=None, min_value=0.0, step=1.0,
                placeholder="不限",
            )
        with col_f6:
            min_aum = st.number_input(
                "最低规模（亿）", value=0.5, min_value=0.0, step=0.5,
                help="剔除规模低于此值的基金（默认0.5亿=5000万）。规模太小"
                     "（几十万~几千万）的迷你/僵尸基金净值容易被单笔申赎搅动"
                     "失真、经理也不上心。按截至日期当时已披露的最新季报规模"
                     "判定，且**A/C 等各份额类别合并计算**（它们是同一份基金"
                     "合同下的同一个投资组合，只是收费方式不同；清盘线也按"
                     "整只基金算）。查不到规模的基金保留、在「规模(亿)」列"
                     "显示 —。设 0 = 不限。",
            )
        with col_f5:
            asof_date = st.date_input(
                "截至日期（不选=今天）",
                value=None,
                min_value=dt.date(2019, 1, 1),
                max_value=dt.date.today(),
                help="还原你在该日进行筛选时能看到的结果：只用该日之前"
                     "（不含当日，当日净值当时尚未公布）的净值历史重算"
                     "收益/回撤/夏普。本地净值（仅C类）从 2018-01-01 起，"
                     "因此最早可选 2021-01-01，保证近1年窗口有完整数据。",
            )
        submitted = st.form_submit_button("🔍 开始筛选", type="primary")

    # ── 截至日期日历上标记上证大跌日 ──────────────────────────────────────────
    # st.date_input 的日历(BaseWeb Datepicker)不支持按日期加样式,从组件
    # iframe 注入脚本到父页面:MutationObserver 监听日历弹层出现,按日期格子
    # 的 aria-label 解析出日期(streamlit 1.50 固定为英文,如 "Choose
    # Wednesday, July 16th 2026. It's available."),给上证跌超1%的交易日画
    # 红色下划线,悬浮显示当日跌幅。日历关闭时选择器查不到任何格子,零开销。
    _sse_marks = load_sse_daily(fetcher.index_daily_saved_at("sse"))
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
        # 新一轮筛选:回到第 1 页,并清掉上一轮残留的勾选行号。
        st.session_state.filter_page_no = 1
        st.session_state.filter_picked = set()
    filter_ready = st.session_state.get("filter_applied", False)
    # 筛选要读净值(重算指标、按区间剔除历史不足的基金),云端重表还在后台
    # 下载时先挡住:空表算出来的结果看着像模像样,其实全错。
    if filter_ready and not fetcher.nav_ready():
        _nav_notice("筛选")
        filter_ready = False

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
                f"{_fresh} 指标数据更新于 {_fmt_cst(_last_update, '%Y-%m-%d %H:%M')}"
                f"（{_age_h:.0f} 小时前）"
            )
        else:
            st.caption("⚠️ 还没有指标数据。请点击「🔄 更新数据」，或运行 `python3 update_daily.py`。")

    display = None
    table = None
    _hit = None
    if filter_ready:
        # Persistent result cache: every distinct filter run's full result table
        # is stored in SQLite, so repeating one (even after a restart — notably
        # the ~1min as-of snapshots) is a single read instead of a recompute.
        # Live-mode keys embed the metrics version, so a daily data update
        # naturally starts a fresh entry; as-of snapshots are immutable history.
        _asof_iso = asof_date.strftime("%Y-%m-%d") if asof_mode else None
        _fparams = {
            "types": sorted(selected_types), "period": period_label,
            "min_ret": min_ret, "max_dd": max_dd, "asof": _asof_iso,
            # 规模下限(亿):0/None 视为不限,归一化成 None 免得 0 与不限各占
            # 一个缓存键。
            "min_aum": (min_aum if min_aum and min_aum > 0 else None),
            # Bumped when the filter rules change (v3: exclude funds younger
            # than the *selected* period window; v4-v6: exclude 债券/固收/偏债
            # types, is_bond 逐步收敛为「含债或固收」; v7: 回撤改按校正收益
            # 复利口径,与模拟盘一致; v8: 新增「最低规模」过滤 + 规模(亿)列;
            # v9: 结果不再截断前 200 条,旧条目只有前 200 行、不能再复用),
            # so stale cached results never get served.
            "rule_ver": 9,
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
                f"⚡ 本次筛选命中缓存（{_fmt_cst(_fsaved, '%Y-%m-%d %H:%M')} 计算）"
                f" · 共 {len(table):,} 条匹配"
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

            if min_ret is not None and ret_col in filtered.columns:
                # NaN is dropped when this filter is active — the threshold
                # can't be evaluated for funds with no return in the period.
                filtered = filtered[
                    pd.to_numeric(filtered[ret_col], errors="coerce") >= min_ret]

            display = filtered.copy()

            # Drawdown filter. Funds without a computed drawdown (e.g. younger than
            # the window) are kept rather than hidden.
            if max_dd is not None and mdd_col in display.columns:
                dd_pct = pd.to_numeric(display[mdd_col], errors="coerce") * 100
                display = display[dd_pct.isna() | (dd_pct <= max_dd)]

            # ── 规模过滤 + 规模(亿)列 ─────────────────────────────────────────
            # 规模按「截至日期当时已披露的最新季报」取(as-of, 无未来函数;
            # 快照模式用 asof_date, 实时用今天), 纯读 fund_scale_hist(每晚
            # update_daily 刷新)、不联网。已知规模 < 门槛的剔除; 查不到规模的
            # 保留(在列里显示为空)——避免规模库尚未覆盖到的基金被误杀, 与
            # 回测里「未知即跳过」略有不同(那边是无人值守选基, 这里用户能
            # 自己看着 — 的行取舍)。
            _aum_ref = (asof_date if asof_mode
                        else dt.date.today()).strftime("%Y-%m-%d")
            _aum_map = fetcher.funds_aum_asof(
                display["code"].dropna().tolist(), _aum_ref)
            display["_aum"] = display["code"].map(_aum_map)
            if min_aum and min_aum > 0:
                _known_small = display["_aum"].notna() & (display["_aum"] < min_aum)
                display = display[~_known_small]

            # Build the presentation table inside the spinner too, so the loading
            # animation covers everything between the click and the rendered rows.
            ret_label = f"{period_label}收益率(%)"
            dd_label = f"{period_label}最大回撤(%)"
            sharpe_label = f"{period_label}夏普比率"
            table = pd.DataFrame()
            table["基金代码"] = display.get("code")
            table["基金名称"] = display.get("name")
            table["类型"] = display.get("type")
            table["规模(亿)"] = pd.to_numeric(display.get("_aum"), errors="coerce").round(2)
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
        fetcher.save_filter_result(_fkey, {**_fparams, "total": _total}, table)
        st.caption(
            f"共 {_total:,} 条匹配（基金总量 {len(fund_df):,}）· 已全部缓存，分页展示"
        )
        if max_dd is not None and mdd_col not in display.columns:
            st.caption("⚠️ 暂无回撤数据。请点击「🔄 更新数据」生成。")

    if table is None:
        st.info("👆 设置筛选条件后，点击「🔍 开始筛选」生成基金列表。")
    else:
        # ── 分页 ──────────────────────────────────────────────────────────────
        # 匹配结果全量保留在内存/缓存里,但一次只把当前页切给 st.dataframe——
        # 上万行一次性渲染会明显拖慢页面。切片保留原行号做索引,所以表格左侧
        # 显示的就是该基金在全表中的排名。
        _pc1, _pc2, _pc3 = st.columns([1, 1, 3])
        with _pc1:
            _psize = st.selectbox("每页行数", [50, 100, 200, 500, 1000],
                                  index=2, key="filter_page_size")
        _npages = max(1, math.ceil(len(table) / _psize))
        # 换页大小/重新筛选后,session 里残留的页码可能越界——先夹回范围,
        # 否则 number_input 会因 value 超出 max_value 直接报错。
        if st.session_state.get("filter_page_no", 1) > _npages:
            st.session_state.filter_page_no = _npages
        with _pc2:
            _page = st.number_input("第几页", min_value=1, max_value=_npages,
                                    value=1, step=1, key="filter_page_no")
        _start = (int(_page) - 1) * _psize
        _end = min(_start + _psize, len(table))
        with _pc3:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.caption(
                f"共 {len(table):,} 条 · 第 {_page}/{_npages} 页"
                f"（第 {_start + 1:,}–{_end:,} 条）"
            )

        st.caption("点击表头可排序 · 行首打勾（可多选，跨页保留）查看基金在截至日期下最新披露的十大持仓")
        _sel = st.dataframe(
            table.iloc[_start:_end],
            use_container_width=True,
            height=560,
            on_select="rerun",
            selection_mode="multi-row",
            # 每页一个独立的表格 key:翻页时勾选状态不会按行号错位串到新页。
            key=f"filter_table_sel_{_psize}_{_page}",
        )

        # ── Selected funds: latest top-10 holdings as of the filter's 截至日期,
        # laid out in a grid — fill each row left to right, then wrap. ──
        # 勾选返回的是页内行号;换算成全表行号后存进 session,这样翻到别的页
        # 再回来(或同时勾几页)选中的基金都还在。每次 rerun 只用当前页的
        # 勾选覆盖本页区间,其余页的选择原样保留。
        _page_rows = {_start + _r for _r in
                      (_sel.selection.rows if _sel is not None else [])}
        _picked = {_i for _i in st.session_state.get("filter_picked", set())
                   if not (_start <= _i < _end)} | _page_rows
        st.session_state.filter_picked = _picked
        _sel_rows = sorted(_picked)
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
    # 净值/持仓/规模都在重表里,云端后台下载完之前当作"没输入代码"处理。
    if code_input and not fetcher.nav_ready():
        _nav_notice("基金详情")
        code_input = ""
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
            st.subheader("📊 规模变化（各季度期末净资产）")
            with st.spinner("加载规模数据…"):
                _scale = load_scale_hist(code_input.strip().zfill(6))
            if _scale is None or _scale.empty:
                st.info("该基金暂无规模数据（新基金或数据源缺失）。")
            else:
                _sc = _scale.copy()
                _sc["quarter_end"] = pd.to_datetime(_sc["quarter_end"])
                _sc["aum"] = pd.to_numeric(_sc["aum"], errors="coerce")
                _tbl = _sc.sort_values("quarter_end", ascending=False)
                _tbl["报告日期"] = _tbl["quarter_end"].dt.strftime("%Y-%m-%d")
                _tbl["期末净资产(亿元)"] = _tbl["aum"].round(2)
                st.dataframe(
                    _tbl[["报告日期", "期末净资产(亿元)"]].reset_index(drop=True),
                    use_container_width=True, hide_index=True, height=400)
                st.caption("规模取自基金定期报告的期末净资产（季报须在季末后15个"
                           "工作日内披露；最早一行通常是基金成立日的初始规模）。")

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
    if _IS_CLOUD:
        st.info("☁️ 云端页面是每日数据快照:模拟盘操作请在本地进行,"
                "这里的改动会在次日跑批同步后被覆盖。")
    _code_names = dict(zip(fund_df["code"], fund_df["name"]))
    sim_date = simulator.get_current_date()

    # 模拟盘的交易日历、成交净值、持仓市值全部来自净值表。
    if not fetcher.nav_ready():
        _nav_notice("模拟盘")
    elif sim_date is None:
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
            _sse0 = load_sse_daily(fetcher.index_daily_saved_at("sse"))
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
                        + _fmt_cst(_a["saved_at"], "%m-%d %H:%M"))
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
        _sse_all = load_sse_daily(fetcher.index_daily_saved_at("sse"))
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

                _add_sse_drop_bands(fig_ret, load_sse_daily(fetcher.index_daily_saved_at("sse")),
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
    sse_df = load_sse_daily(fetcher.index_daily_saved_at("sse"))
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
            _show_bands = st.checkbox("标记跌超1%的交易日", value=False,
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

        # VIX恐慌指数（QVIX，自算）on a secondary right axis: levels (~15–40)
        # are incomparable with index points, so it never shares the left
        # scale. 数据源:qvix_self_history(上交所官方期权风险指标反推,
        # 不是 optbbs——见 qvix_calc.py 顶部说明)。阈值线现算
        # (_QVIX_THR_COMBOS,见上方 load_qvix_threshold_combos)——候选
        # 组合已收敛到只剩标准的2年90%一组,不再提供勾选,随VIX直接画。
        qvix_view = None
        if _show_vix:
            _qvix = load_qvix_self(fetcher.qvix_self_history_last_date())
            if _qvix is not None and not _qvix.empty:
                qvix_view = _qvix.dropna(subset=["qvix"]).copy()
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
                x=qvix_view["date"], y=qvix_view["qvix"],
                name="VIX恐慌指数(QVIX,自算)", yaxis="y2",
                line=dict(color="#f28e2b", width=1.3),
                hovertemplate="VIX %{y:.2f}<extra></extra>"))
            # Default (right-side vertical) legend keeps clear of the
            # 跌超1% annotations that sit above the plot area.
            fig_sse.update_layout(
                yaxis2=dict(title="VIX恐慌指数", overlaying="y", side="right",
                            showgrid=False))
            _thr_combos_df = load_qvix_threshold_combos(
                fetcher.qvix_self_history_last_date())
            if _thr_combos_df is not None:
                _thr_view = _thr_combos_df[
                    (_thr_combos_df["date"] >= view["date"].min())
                    & (_thr_combos_df["date"] <= view["date"].max())]
                for _label, _window, _pct, _color in _QVIX_THR_COMBOS:
                    _line = _thr_view.dropna(subset=[_label])
                    if _line.empty:
                        continue
                    fig_sse.add_trace(go.Scatter(
                        x=_line["date"], y=_line[_label],
                        name=f"恐慌阈值({_label})", yaxis="y2",
                        line=dict(color=_color, width=1.2, dash="dash"),
                        hovertemplate=f"阈值({_label}) " +
                                      "%{y:.2f}<extra></extra>"))

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
                    **{"VIX恐慌指数": qvix_view["qvix"].round(2)})
                _tbl = _tbl.merge(_q[["日期", "VIX恐慌指数"]],
                                  on="日期", how="left")
            st.dataframe(_tbl, use_container_width=True, height=420)

        # ── 策略复盘:买入近3月跌幅最大标的,基金/大盘双止损逐日盯盘
        # (backtest_qvix.py) ── 结果为脚本离线跑批后硬编码(全市场近3月
        # 涨跌幅重算无法在页面现算)。
        # 2026-07-24 切换到当前标准参数: QVIX 2年90分位信号(window=490,
        # pct=0.90) + 候选按近3月跌幅从深到浅排(ret_col="ret_3m",
        # pick="bottom", 见 backtest_qvix.find_champion_on_date 同名参数
        # 说明),取跌幅最大的一只——不再是选涨幅最高的"冠军"。起因:
        # 逐信号日统计过候选池里"近3月涨幅前20名"和"跌幅前20名"买入后的
        # 前瞻收益, 11次信号里有7次是跌幅最大的那批反弹更猛(恐慌信号
        # 触发点本身就是刚经历一轮下跌, 涨得最猛的常是提前跑赢、比较
        # 拥挤的仓位, 跌得最惨的反而容易报复性反弹)。
        # 另加两道过滤: ①候选波动率比值(compute_beta)≥1.5(min_vol_ratio,
        # 用户原话"我是来发财的,不是来保本的")——不设这道过滤时"跌幅
        # 最大"选出过长期低波动的保守型基金(全历史年化波动率2.3%~4.6%,
        # 当天排名垫底只是正常噪声、根本谈不上"超跌"); ②候选按跌幅
        # 排完发现已经找不到真正下跌的(第一个非负值)就当天不操作,不
        # 退而求其次买涨幅最小的凑数——这两条一起用之后,79个信号日
        # 只有8次真正满足条件,但质量明显更高。相关系数过滤(min_corr,
        # 中途试过0.5/0.6/0.7)最终判定对这套新逻辑没必要,已关闭。
        # 此前"涨幅冠军+相关系数≥0.6""跌幅最大但不设波动率/跌幅耗尽
        # 判定""跌幅最大但额外排除港股通"的复盘记录均已替换, 历史版本
        # 见 git 历史——港股通/沪港深/恒生系列曾经额外排除(套用QDII的
        # "跟踪境外市场,与大盘脱钩"逻辑), 2026-07-24复核撤销: 实测恒生
        # 指数跟上证的日收益率相关系数常年0.4~0.74(整体0.544), 明显
        # 高于纳斯达克(0.153)、黄金(0.099)这些真正弱相关的境外资产,
        # "脱钩"假设不成立。
        #
        # 候选排名复用 fetcher.compute_metrics_asof——与本页「基金列表」的
        # "截至日期"筛选完全同口径(按日收益率连乘计算区间收益,正确处理
        # 分红除权,自带单日|收益率|>30%异常值过滤),严格早于当日截断(T日
        # 决策只能看到T-1日收盘净值)。
        # 止损逐交易日检查,双线在买入日锁定——基金线=阈值/5×波动率比值,
        # 大盘线=阈值/5;波动率比值=基金日收益率std/大盘日收益率std(纯波动
        # 对比,不按相关系数加权,且剔除单日|收益率|>35%的净值重置/份额折算
        # 脏数据)。候选池剔除:QDII/海外指数型(跟踪境外市场且走额度受限
        # 的QDII通道,与QVIX/大盘恐慌-反弹逻辑脱钩)、有持有期锁定的基金
        # (名称含"N年/N个月/N天持有(期)""滚动持有""定期开放""封闭式"等,
        # 锁定期内实际赎不出来, 回测里的卖出是假的)、"净值僵化-补涨"基金
        # (排名窗口内连续≥5个交易日|日收益率|<0.05%紧接单日|收益率|>8%
        # 补涨/补跌, 判断为净值长期未按市值更新、事后集中补记, 而非真实
        # 动量)。连续接力同一只基金(上一腿卖出日=下一腿买入日且代码相同)
        # 视为未真实离场,中间腿不收手续费,只在链条最后一腿按"链条首次
        # 买入→该腿卖出"的累计持有天数收一次(而非按单腿天数)。数据
        # 2018 年起(2015-2017 期权刚上市流动性薄、QVIX 计算噪声偏大,
        # 已整段剔除, 见 fetcher.py)。
        with st.expander("📜 策略复盘:QVIX 2年90%信号 + 近3月跌幅最大"
                         "(波动率比值≥1.5 + 规模≥2亿;无真跌则改买涨幅最大;"
                         "止损后顺延至标的换掉) + 基金/大盘双止损逐日盯盘",
                          expanded=True):
            # 5% 定线依据(2026-07 校准,别随手改):大盘线 = 恐慌阈值/5,常态
            # 波动率(QVIX 20)下 ≈ 4 倍日σ;历史 12 次触发的固定线网格回测显示
            # 3%~10% 中 5% 最优,4%~6% 为稳健平台。波动率口径自洽(近10年窗口):
            # 线 = 5×近10年日σ = 5×1.03% ≈ 5%,交叉验证 QVIX 近10年中位
            # 18.72÷4 = 4.7%;上证常态σ按 1/3/10 年窗口为 0.81/0.79/0.82,
            # 中枢稳定故窗口不敏感。断裂检测:近1年σ÷近10年σ>2 即时代迁移
            # 需重算。动态线(QVIX/实际σ×入场定死/每日跟随,k=2~6)无稳健
            # 增益——恐慌日按入场波动率定线永远给宽线,反而丢掉恐慌后反弹守
            # 利润的功能。重议条件:QVIX 中枢驻留 35+。
            # 复盘明细读策略库里最新一次**标准策略**跑批(backtest_qvix.py
            # 跑完自动落库, 见 fetcher.save_strategy_run)。以前这段统计
            # 数字全是硬编码在这里的列表字面量:每次重跑回测都得把十几列
            # 数字手抄一遍进页面, 抄错没人发现, 页面上的数和回测真正跑出来
            # 的数也随时可能对不上(实测就对不上过)。现在回测落库、页面读库,
            # 跑批推一次库页面自动跟着变, 不用改代码。
            _bt_df, _bt_params, _bt_at = load_backtest_trades(
                os.path.getmtime(fetcher.DB_PATH["rank"]))
            if _bt_df is None or _bt_df.empty:
                st.info("暂无回测明细——跑一次 `python3 backtest_qvix.py` "
                        "会往策略库追加一条标准跑批, 页面随即显示。")
                _review_df, _review_trigger = None, []
            else:
                # 未平仓的那笔不计入统计(卖出日带"(持仓中)"标记)
                _open = _bt_df["卖出日"].astype(str).str.contains("持仓中")
                _done = _bt_df[~_open]
                _rets = pd.to_numeric(_done["费后收益"], errors="coerce").dropna()
                _n, _wins = len(_rets), int((_rets > 0).sum())
                _cum = ((1 + _rets / 100).prod() - 1) * 100
                _fees = pd.to_numeric(_done["手续费%"], errors="coerce").fillna(0).sum()
                _days = pd.to_numeric(_done["持有天数"], errors="coerce").dropna()
                _best_i = _rets.idxmax()
                _cum_ex = ((1 + _rets.drop(_best_i) / 100).prod() - 1) * 100
                _loss = _rets[_rets <= 0]
                _loss_txt = ("、".join(f"{v:+.2f}%" for v in _loss)
                             if len(_loss) else "无")
                st.caption(
                    f"{_n} 笔已完成:胜率 {_wins / _n * 100:.1f}%({_wins}/{_n}),"
                    f"费后复利 {_cum:+.2f}%,累计手续费 {_fees:.1f}%,"
                    f"平均持有 {_days.mean():.0f} 天,"
                    f"平均费后 {_rets.mean():+.2f}%,"
                    f"最佳 {_rets.max():+.2f}% / 最差 {_rets.min():+.2f}%。"
                    f"最佳那笔是单只小微盘主题基金的极端案例,剔除它其余 "
                    f"{_n - 1} 笔累计仍有 {_cum_ex:+.2f}%——别把它当期望值,"
                    f"不依赖那笔运气结论也成立。亏损笔:{_loss_txt}。"
                    "样本只有个位数、跨度6年,统计上很薄,别当成可靠预期。"
                    "候选须同时满足\"真正下跌 + 波动率比值≥1.5 + 规模≥2亿\";"
                    "当天找不到真跌的合格候选就改买涨幅最大的那只(2026-08-06 加,"
                    "「选向」列标出每笔走的哪条分支)。止损平仓后若又选中同一只,"
                    "就一路跳过信号日直到选出不同标的才建仓(2026-08-06 加)"
                    "——2024-11-18 卖出北证50后它在随后12个连续信号日里一直是"
                    "涨幅冠军,这条规则让策略空仓等到2025-04-07,避开了那波回调。"
                    "规模按信号日当时"
                    "已披露的最新季报口径(无未来函数)、A/C 等份额类别合并计算"
                    "(2026-08-06 改,此前只算 C 类那一个代码,系统性低估),"
                    "排掉了几十万~几千万"
                    "规模、净值易失真的迷你基金。橙色=触发基金回撤线离场,"
                    "蓝色=触发大盘回撤线离场,两格都染色=同日两条线双双触发;"
                    "连续接力同一只基金视为未真实离场,中间腿不收手续费,"
                    "只在链条最后一腿按累计持有天数收一次。"
                    + (f"(明细跑于 {_fmt_cst(_bt_at, '%Y-%m-%d %H:%M')})"
                       if _bt_at else ""))

                # 每笔卖出当天触发了哪条止损线, 直接从回测写下的「卖出原因」
                # 反推(以前是另一份手工维护的列表, 跟表格行数各管各的,
                # 一旦交易笔数变了就会错位染色)。
                def _trig_of(reason: str) -> str:
                    _r = str(reason)
                    _f, _s = "基金" in _r, "大盘" in _r
                    return "both" if _f and _s else ("fund" if _f else
                                                     ("sse" if _s else ""))
                _review_trigger = [_trig_of(r) for r in _bt_df["卖出原因"]]

                # 列名中性化:2026-08-06 加了 fallback_top 之后, 标准策略里
                # 有几笔走的是"跌幅耗尽→改买涨幅最大"分支, 那几行的区间收益
                # 是正的, 再叫"近3月跌幅最大标的/近3月跌幅"就是错的。具体
                # 每笔走的哪条分支看「选向」列。
                _std_tgt = "选中标的(C类全市场,按前一交易日榜单)"
                _review_df = _bt_df.rename(columns={
                    "冠军(C类全市场,按前一交易日榜单)": _std_tgt,
                    "冠军近3月涨幅(前日口径)": "近3月涨跌(前日口径)",
                })
                _review_df = _review_df[[c for c in [
                    "买入日", _std_tgt,
                    "类型", "买入时规模(亿)", "波动率比值(近3月)", "恐慌阈值",
                    "回撤控制线(%)", "大盘回撤线(%)", "近3月涨跌(前日口径)", "选向",
                    "卖出日", "持有收益", "期间最高", "期间最大回撤", "同期上证",
                ] if c in _review_df.columns]].copy()
                # 「备注」是人工点评(backtest_notes 表), 算不出来、只能人写,
                # 每次重跑回测后重写一遍。按买入日 join, 没写的显示空白。
                _notes = fetcher.load_backtest_notes()
                _review_df["备注"] = [_notes.get(str(d), "")
                                     for d in _review_df["买入日"]]

            def _trigger_cell_style(row):
                _styles = [""] * len(row)
                _idx = list(_review_df.columns).index
                _trig = _review_trigger[row.name]
                # both=同日双触发:基金线橙、大盘线蓝两格都染;否则只染触发的那条
                if _trig in ("fund", "both"):
                    _styles[_idx("回撤控制线(%)")] = "background-color: #ffdca8; color: #1a1a1a"
                if _trig in ("sse", "both"):
                    _styles[_idx("大盘回撤线(%)")] = "background-color: #c9e2ff; color: #1a1a1a"
                return _styles

            if _review_df is not None and not _review_df.empty:
                st.dataframe(
                    _review_df.style.apply(_trigger_cell_style, axis=1)
                    .format(precision=2, na_rep=""),
                    use_container_width=True, hide_index=True,
                    height=(len(_review_df) + 1) * 35 + 3,
                    # 除标的名(medium)和备注(large)外一律 small; 按实际列名
                    # 现拼, 免得选基口径一换列名对不上、宽度配置整个失效。
                    column_config={
                        **{c: st.column_config.Column(width="small")
                           for c in _review_df.columns},
                        _std_tgt: st.column_config.Column(width="medium"),
                        "备注": st.column_config.Column(width="large"),
                    })

        # ── 历次回测跑批(2026-08-05):不是线上实盘规则,只挂上来做对比 ──
        # backtest_qvix.py 每跑一次就往策略库追加一条(fetcher.save_strategy_run),
        # 这里把非标准策略的那些按时间倒序列出来,选中哪条就展开哪条的明细。
        # 以前是"一个方案一张固定表、同名整表覆盖",换个参数重跑就把上一次
        # 冲掉了;现在每跑必留痕,回头想比哪两版都翻得到。
        # 策略库是独立文件(fund_strategy.db),跟 400MB 的行情主库彻底分开,
        # 见 fetcher.DB_LAYOUT 说明。
        # 「备注」不给这些跑批挂:那份人工点评(backtest_notes)是对着标准策略
        # 选出的基金写的,对照实验同一个买入日往往选的是另一只,挂过去
        # 就是张冠李戴。
        _runs = [r for r in load_strategy_runs(os.path.getmtime(fetcher.DB_PATH["strategy"]))
                 if not r["is_standard"]]
        if _runs:
            with st.expander(f"🧪 历次对照回测({len(_runs)} 次,非线上规则)",
                             expanded=False):
                def _run_caption(r):
                    _t = _fmt_cst(r["run_at"], "%m-%d %H:%M") or "—"
                    _w = f"{r['win_rate']:.0f}%" if r["win_rate"] is not None else "—"
                    _c = f"{r['cum_return']:+.0f}%" if r["cum_return"] is not None else "—"
                    return f"{_t} · {r['label']} · {r['n_trades']}笔 胜率{_w} 复利{_c}"

                _sel = st.radio(
                    "选一次跑批看明细", [r["id"] for r in _runs],
                    format_func=lambda i: _run_caption(
                        next(r for r in _runs if r["id"] == i)),
                    key="strategy_run_pick")
                _x_df, _x_params, _x_at = load_strategy_run_detail(
                    os.path.getmtime(fetcher.DB_PATH["strategy"]), _sel)
                if _x_df is None or _x_df.empty:
                    st.info("这次跑批没有明细。")
                else:
                    _x_win = "近1月" if _x_params.get("ret_col") == "ret_1m" else "近3月"
                    _x_open = _x_df["卖出日"].astype(str).str.contains("持仓中")
                    _x_done = _x_df[~_x_open]
                    _x_rets = pd.to_numeric(_x_done["费后收益"],
                                            errors="coerce").dropna()
                    _x_days = pd.to_numeric(_x_done["持有天数"],
                                            errors="coerce").dropna()
                    _x_n, _x_wins = len(_x_rets), int((_x_rets > 0).sum())
                    _x_cum = ((1 + _x_rets / 100).prod() - 1) * 100
                    _x_ex = ((1 + _x_rets.drop(_x_rets.idxmax()) / 100).prod() - 1) * 100
                    _x_loss = _x_rets[_x_rets <= 0]
                    _x_pick = str(_x_params.get("pick") or "bottom")
                    if _x_pick == "regime":
                        # regime_basis 区分两版择向依据:早期那版比的是"截至
                        # 前一日的回看窗口",在暴跌当天触发的信号上会把当天
                        # 那根大阴线排除在方向判断外(2025-04-07 单日 -7.34%
                        # 却算出近3月 +2.44% 判成"涨"),已弃用但跑批记录还在,
                        # 规则描述不能拿现行口径去套。
                        if _x_params.get("regime_basis") == "window":
                            _rule = (f"信号日上证比{_x_win}前**涨**(截至前一交易日)"
                                     f"→买{_x_win}涨幅最大的,**跌**→买{_x_win}"
                                     f"跌幅最大的。")
                        else:
                            _rule = (f"信号日**上证当天收涨**→买{_x_win}涨幅最大的,"
                                     f"**当天收跌**→买{_x_win}跌幅最大的。")
                        if _x_params.get("require_drop"):
                            _rule += "走跌幅分支时选中的基金必须真的是负收益,否则当天不买。"
                        else:
                            _rule += "不要求候选自身下跌过。"
                    elif _x_pick == "top":
                        _rule = f"买{_x_win}涨幅最大的(动量)。"
                    else:
                        _rule = f"买{_x_win}跌幅最大的(反转)。"
                    # 过滤条件逐项列出来, 不写"与标准策略一致"——对照实验
                    # 改的可能正是其中某一项(如规模门槛), 那句话会自相矛盾。
                    # 跟标准策略(波动≥1.5/规模≥2亿)不同的值加粗标出来。
                    def _mark(val, std):
                        return f"**{val}**" if val != std else f"{val}"
                    _vr = _x_params.get("min_vol_ratio", 1.5)
                    _aum = _x_params.get("min_aum", 0.5)   # 老跑批没记这项时按旧标准
                    # 规模口径 2026-08-06 从「只算该代码的份额类别」改成
                    # 「A/C 合并」, 同一个 min_aum 数值在两种口径下松紧差很多,
                    # 所以旧口径的跑批必须标出来, 否则会跟标准策略误比。
                    _basis = _x_params.get("aum_basis", "single")
                    st.caption(
                        f"规则:{_rule}"
                        f"候选须满足波动率比值≥{_mark(_vr, 1.5)}、"
                        + (f"规模**{_aum}~{_x_params['max_aum']}亿**"
                           if _x_params.get("max_aum") else
                           f"规模≥{_mark(_aum, 2.0)}亿")
                        + ("(**旧口径:只算该代码的份额类别,未合并 A/C**)"
                           if _basis != "merged" else "")
                        + f",双止损口径同标准策略"
                        f"(粗体=与标准策略不同的参数)。"
                        f"　{_x_n} 笔已完成:胜率 {_x_wins / _x_n * 100:.1f}%"
                        f"({_x_wins}/{_x_n}),费后复利 {_x_cum:+.2f}%,"
                        f"平均持有 {_x_days.mean():.0f} 天,"
                        f"平均费后 {_x_rets.mean():+.2f}%,"
                        f"最佳 {_x_rets.max():+.2f}% / 最差 {_x_rets.min():+.2f}%,"
                        f"剔除最佳那笔其余 {_x_n - 1} 笔仍有 {_x_ex:+.2f}%。"
                        f"亏损笔:"
                        + ("、".join(f"{v:+.2f}%" for v in _x_loss)
                           if len(_x_loss) else "无") +
                        "。⚠️ 对照实验,只用来跟标准策略比,不是线上在用的规则;"
                        "样本十几笔、跨度6年,统计上很薄。"
                        + (f"(跑于 {_fmt_cst(_x_at, '%Y-%m-%d %H:%M')})"
                           if _x_at else ""))

                    # 择向策略逐笔方向不同(有的买涨幅最大、有的买跌幅最大),
                    # 标的那列只能中性叫"选中标的",具体方向看「选向」列。
                    _x_tgt = "选中标的(C类全市场,按前一交易日榜单)"
                    _x_ret = f"{_x_win}涨跌(前日口径)"
                    _x_view = _x_df.rename(columns={
                        "冠军(C类全市场,按前一交易日榜单)": _x_tgt,
                        "冠军近3月涨幅(前日口径)": _x_ret,
                    })
                    _x_view = _x_view[[c for c in [
                        "买入日", _x_tgt, "类型", "买入时规模(亿)",
                        "波动率比值(近3月)", "恐慌阈值", "回撤控制线(%)",
                        "大盘回撤线(%)", _x_ret, "上证当日涨跌", "选向",
                        "卖出日", "持有收益", "期间最高", "期间最大回撤", "同期上证",
                    ] if c in _x_view.columns]].copy()

                    # 不复用上面标准策略块里的 _trig_of:那个定义在"标准策略有
                    # 数据"的分支里, 标准表空着时它根本没被定义过, 这里引用就是
                    # NameError。两行的事, 各管各的。
                    def _x_trig_of(reason):
                        _r = str(reason)
                        _f, _s = "基金" in _r, "大盘" in _r
                        return "both" if _f and _s else ("fund" if _f else
                                                         ("sse" if _s else ""))
                    _x_trig = [_x_trig_of(r) for r in _x_df["卖出原因"]]
                    _x_cols = list(_x_view.columns)

                    def _x_style(row, _trig=_x_trig, _cols=_x_cols):
                        _s = [""] * len(row)
                        if _trig[row.name] in ("fund", "both"):
                            _s[_cols.index("回撤控制线(%)")] = \
                                "background-color: #ffdca8; color: #1a1a1a"
                        if _trig[row.name] in ("sse", "both"):
                            _s[_cols.index("大盘回撤线(%)")] = \
                                "background-color: #c9e2ff; color: #1a1a1a"
                        return _s

                    _x_cfg = {c: st.column_config.Column(width="small")
                              for c in _x_cols}
                    _x_cfg[_x_tgt] = st.column_config.Column(width="medium")
                    st.dataframe(
                        _x_view.style.apply(_x_style, axis=1)
                        .format(precision=2, na_rep=""),
                        use_container_width=True, hide_index=True,
                        height=(len(_x_view) + 1) * 35 + 3,
                        column_config=_x_cfg)
