"""Fund Analyzer — Streamlit dashboard."""

import datetime as dt
import gzip
import hashlib
import json
import math
import re
import os
import shutil
import threading
import time
import numpy as np
import pandas as pd
import today_pick
import plotly.express as px
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
from sse_view import fmt_cst as _fmt_cst



# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="基金分析仪",
    page_icon="📈",
    layout="wide",
)

# init_db runs once per server, not on every rerun — it hits SQLite on /mnt/c
# (slow Windows-disk I/O under WSL), which used to tax every single click.
@st.cache_resource
def _init_db_once():
    fetcher.init_db()
    return True


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
# 首屏同步拉 rank/market/strategy, 合计 gz 约 2MB, 拉完立刻能画;
# nav(净值 gz 61MB)和 scale(季度数据 gz 4MB)首屏一行都不读, **也不预拉** ——
# 见下面 _need_nav: 点了要用净值的功能才开始下。cache 库压根不拉: 它是筛选
# 结果缓存, 丢了自动重算, 下一份别人的缓存过来毫无价值。
#
# ⚠️ 流量优先, 别改回"开页面就后台拉": 用户常开**手机流量**看这个页面, 而且
# 经常 reboot 应用 —— 容器一重建磁盘就清空, marker 跟着没, ETag 条件下载
# 那道防线在这条路径上完全失效, 每次 reboot 都是实打实的 61MB。而他日常只
# 看上证走势和策略回测, 这两块一行净值都不读(回测明细读的是 strategy 库的
# 存档, 不是 nav)。原来的设计等于绝大多数 reboot 白烧 61MB。
_DB_BASE = os.environ.get("FUND_ANALYZER_RELEASE_BASE") or (
    "https://github.com/luotiancai/fund-analyzer/releases/download/data/")
_ASSET = {name: _DB_BASE + fn + ".gz"
          for name, fn, _t, _l in fetcher.DB_LAYOUT}
_MARKER = {name: fetcher.DB_PATH[name] + ".from-release"
           for name, _fn, _t, _l in fetcher.DB_LAYOUT}
_EAGER_DBS = ("rank", "qvix", "market", "strategy")
_LAZY_DBS = fetcher.LAZY_DBS          # ("nav", "scale")
# 惰性库的 gz 大小, 只用于按钮上那句提示("要下多少") —— 会随数据增长慢慢
# 变大, 差个几 MB 无所谓, 不值得为了准确去 HEAD 一次(那本身就是一次请求)。
_LAZY_MB = 65


@st.cache_resource(show_spinner=False)
def _dl_state() -> dict:
    """跨 rerun 存活的下载状态: {"t": 线程, "done"/"total": 压缩字节数}。

    **必须挂在 cache_resource 上**: Streamlit 每次 rerun 都用**全新的模块
    命名空间**重跑整个脚本, 模块级变量活不过一次 rerun。线程句柄放模块级
    的话, 页面一刷新就"忘了自己正在下载", 又把下载按钮摆出来; 进度计数更
    糟 —— 后台线程写的是它启动那一轮的旧字典, 新一轮读到的永远是 0。
    """
    return {"t": None, "done": 0, "total": 0}


class _Counting:
    """套在 response.raw 外面数字节 —— 用户开手机流量, 得让他看见烧了多少。

    数的是**压缩后**的字节(实际走网络的量), 不是 gzip 解出来的 233MB。
    只实现 read: GzipFile 只用这一个方法。
    st 状态字典由调用方传进来, 不在这里调 _dl_state() —— 这个类跑在后台
    线程里, 而 cache_resource 在无 ScriptRunContext 的线程里调会告警。
    """

    def __init__(self, raw, total, prog: dict):
        self._raw, self._n, self._p = raw, 0, prog
        prog["total"], prog["done"] = total, 0

    def read(self, n=-1):
        b = self._raw.read(n)
        self._n += len(b)
        self._p["done"] = self._n
        return b


def _download_gz(url: str, dest: str, marker: str, prog: dict = None) -> bool:
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
            src = dl.raw
            if prog is not None:
                src = _Counting(dl.raw,
                                int(dl.headers.get("Content-Length") or 0), prog)
            with open(tmp, "wb") as f, gzip.GzipFile(fileobj=src) as gz:
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
    """后台线程拉大库(净值/季度数据)。**只由 _need_nav 在用户点按钮时调用。**

    起线程而不是同步拉:解压后 233MB, 挡在页面前面十几秒一动不动很难受;
    起了线程还能一边下一边看上证走势。期间需要净值的三处(筛选/详情/
    今日候选)由 fetcher.nav_ready() 挡住,显示进度而不是拿空表算出一堆看着
    合理的错数。

    TTL 与 _sync_db_from_release 一致(1h):跑批发新快照后 rank 会换成当天
    的,大库也得跟着换,否则各库的数据日期能差出一天。ETag 没变时
    _download_gz 直接返回、不重下,adopt_db 本身幂等,空跑代价接近零。

    先拉 scale(gz 4MB)再拉 nav(gz 61MB):规模数据是基金列表的门槛要用的,
    早几十秒到位就早几十秒不再"查不到规模一律放行"。
    """
    prog = _dl_state()      # 主线程里取好再传进去(见 _Counting 的说明)

    def _work():
        for name in sorted(_LAZY_DBS, key=lambda n: n != "scale"):
            try:
                if _download_gz(_ASSET[name], fetcher.DB_PATH[name],
                                _MARKER[name], prog):
                    fetcher.adopt_db(name)
            except Exception:
                fetcher.logger.exception("%s 库后台下载失败", name)

    t = threading.Thread(target=_work, name="lazy-db-download", daemon=True)
    t.start()
    prog["t"] = t
    return t


_IS_CLOUD = _sync_db_from_release()
_init_db_once()


_PROG_DRAWN = False   # 本轮是否已经画过进度条(模块级变量每轮 rerun 自动归零)


@st.fragment(run_every=2)
def _nav_progress():
    """下载中的进度条; 每 2 秒自己刷新一次, 不重跑整页。下完自动重跑整页。"""
    if fetcher.nav_ready():
        st.rerun(scope="app")           # 库就位了, 让整页重画、功能解锁
        return
    st_ = _dl_state()
    done, total = st_["done"], st_["total"]
    if total:
        st.progress(min(done / total, 1.0),
                    text=f"下载净值库 {done / 1048576:.1f} / "
                         f"{total / 1048576:.0f} MB（压缩后实际流量）")
    else:
        st.progress(0.0, text="正在连接…")


def _need_nav(what: str) -> bool:
    """要读净值库的地方先调这个, 返回 True 才能往下算。

    **不自动下载**: 净值库 gz 61MB, 而用户开手机流量、又经常 reboot(容器一
    重建磁盘就清空, ETag 那道条件下载防线失效)。他日常只看上证和回测, 都不
    碰净值 —— 自动拉等于绝大多数 reboot 白烧 61MB。所以摆个按钮、标明大小,
    点了才下。
    """
    if fetcher.nav_ready():
        return True                      # 本地模式或已下完
    t = _dl_state()["t"]
    if t is not None and t.is_alive():
        # 进度条只画一次: 四个消费点在同一轮里都会走到这儿(st.tabs 会渲染
        # 所有 tab 的内容), 每处都插一个 fragment 就是四个自转的进度条。
        global _PROG_DRAWN
        if not _PROG_DRAWN:
            _PROG_DRAWN = True
            _nav_progress()
        else:
            st.info("⏳ 净值库下载中，完成后本区自动可用。")
        return False
    st.warning(f"「{what}」要读净值库，需下载 **约 {_LAZY_MB}MB**"
               f"（解压后 233MB；容器重启后要重新下）。"
               f"上证走势和策略回测不需要它，可以直接看。")
    if st.button(f"下载并启用{what}", key=f"_navdl_{what}",
                 type="primary"):
        _start_lazy_downloads()
        st.rerun()
    return False

# ── 更新数据(仅本地)────────────────────────────────────────────────────────
# 侧边栏已整体移除:云端数据由每日跑批自动更新,无需任何入口;
# 本地入口缩成页面右上角一个小按钮。
@st.dialog("确认更新数据")
def _confirm_update():
    st.write("将增量拉取最新净值（不重算指标，筛选时按需计算），确定继续？")
    c1, c2 = st.columns(2)
    if c1.button("确定", type="primary", use_container_width=True):
        st.session_state["_run_update"] = True
        st.rerun()
    if c2.button("取消", use_container_width=True):
        st.rerun()


if not _IS_CLOUD:
    _, _upd_col = st.columns([6, 1])
    if _upd_col.button("🔄 更新数据", use_container_width=True,
                       help="增量拉取最新净值；指标在筛选时按需计算。"):
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
# recomputed + re-WROTE the fund's metrics row on the slow /mnt/c disk.
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
def load_fund_metrics(code: str):
    return fetcher.compute_metrics_for_fund(code)


@st.cache_data(show_spinner="正在加载上证指数数据…")
def load_sse_daily(cache_key):
    return fetcher.fetch_sse_daily()


# Precomputed 回撤/收益率 as a merge-ready DataFrame, built once and shared
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
        summary = fetcher.run_pipeline(progress=_on_progress, do_recompute=False)
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
# path) and its computed max-drawdown column.
PERIODS = {
    "近1月": ("ret_1m", "mdd_1m"),
    "近3月": ("ret_3m", "mdd_3m"),
    "近6月": ("ret_6m", "mdd_6m"),
    "近1年": ("ret_1y", "mdd_1y"),
}

# Ensure period return columns are numeric (cache may store them as strings).
for _c in ("ret_1m", "ret_3m", "ret_6m", "ret_1y"):
    if _c in fund_df.columns:
        fund_df[_c] = pd.to_numeric(fund_df[_c], errors="coerce")

# ── 表格上滚不动页面的修复 ────────────────────────────────────────────────────
# 症状:策略复盘和历次回测两个 expander 都展开时,鼠标/触控板在页面上滑不动。
# 原因不在高度(布局是正常的:section.stMain 是 height:100dvh 的滚动容器,
# 内容撑到 2000+px,scrollHeight 也对)。真正的原因是**横向滚动抢手势**:
# st.dataframe 内部是 glide-data-grid,它的 .dvn-scroller 是 overflow:auto 的
# 滚动容器;我们的表列多(复盘表实测内容宽 1725px > 容器 1404px),所以它横向
# 可滚。Chrome 的滚动锁定(scroll latching)会把整段手势钉在指针下这个滚动容器
# 上:纯垂直滚轮它让位给页面,但触控板/鼠标滚轮几乎不可能只有垂直分量,只要
# 带一点横向,整段手势就被表格吃掉——横向挪几十像素,页面一动不动。
#   实测(1776×1288,两个 expander 都展开):指针在复盘表上,deltaX=8/deltaY=100
#   连滚 5 次,页面 scrollTop 纹丝不动(0px),而表格 scrollLeft 从 32→72;
#   同一手势放在表格外的空白处,页面正常滚 401px。
# 只展开一个时,表格盖不满整屏,指针总能落在表格外,所以看起来"没毛病";两个
# 都展开,两张宽表把整个视口占满,就彻底滑不动了——用户报的正是这个条件。
# 修法:捕获阶段拦 wheel,指针在"纵向滚不动"的表格上时,把 deltaY 手动喂给
# section.stMain、deltaX 留给表格,并 preventDefault 掉浏览器自己的锁定逻辑。
# 纵向本来就能滚的表格(高度被截断的)不碰,保持原生行为。
components.html("""<script>
(function () {
  const doc = window.parent.document;
  if (doc.__dfWheelFix) return;          // Streamlit 每次 rerun 都会重放这段
  doc.__dfWheelFix = true;
  doc.addEventListener("wheel", function (e) {
    if (e.ctrlKey) return;               // ctrl+滚轮是浏览器缩放, 别拦
    const sc = e.target.closest && e.target.closest(".dvn-scroller");
    if (!sc) return;
    // 表格自己纵向能滚(内容比可视区高)就让它自己滚, 别抢。
    if (sc.scrollHeight - sc.clientHeight > 1) return;
    const main = doc.querySelector("section.stMain") || doc.scrollingElement;
    if (!main) return;
    e.preventDefault();
    main.scrollTop += e.deltaY;
    sc.scrollLeft += e.deltaX;           // 横向仍然归表格
  }, {capture: true, passive: false});
})();
</script>""", height=0)

# ── Tabs ──────────────────────────────────────────────────────────────────────
# 上证页(行情图/策略复盘/对照回测/每日数据) 2026-10-14 起挪去 BWG 上的静态页
# (build_site.py), 这里只留要净值库的几块。
tab_pick, tab_table, tab_detail = st.tabs(
    ["🎯 今天会买哪只", "📋 基金列表", "🔍 基金详情"])

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
                     "收益/回撤。本地净值（仅C类）从 2018-01-01 起，"
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
    if filter_ready and not _need_nav("筛选"):
        filter_ready = False

    ret_col, mdd_col = PERIODS[period_label]

    # ── As-of snapshot mode ───────────────────────────────────────────────────────
    # A past 截至日期 swaps the live rank-list returns and precomputed metrics for
    # ones recomputed from stored NAV truncated to that date, so the filters below
    # reproduce what the screen would have shown back then. Cached per (date, cols).
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
    # data changed since the last full recompute, the stored 回撤/收益率
    # are yesterday's and would let funds past today's thresholds (e.g. a fund
    # whose latest drop pushed 近1年回撤 over the cutoff). Recompute the whole
    # table once, here, before any filtering or cache lookup.
    if filter_ready and not asof_mode and fetcher.metrics_stale():
        _bar = st.progress(0.0, text="🧮 净值已更新，正在重算全市场指标…")
        fetcher.recompute_all(
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
            # 复利口径; v8: 新增「最低规模」过滤 + 规模(亿)列;
            # v9: 结果不再截断前 200 条,旧条目只有前 200 行、不能再复用),
            # so stale cached results never get served.
            "rule_ver": 9,
            # Combines the metrics recompute timestamp with the fund
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
            # 只算所选区间的收益/回撤两列(其余窗口筛选和展示都用不到,
            # 全算耗时翻倍),缓存键因此带上区间。
            _cols = {ret_col, mdd_col}
            _key = (_asof_iso, tuple(sorted(_cols)))
            if _key not in _cache:
                _bar = st.progress(0.0, text=f"📅 正在按 {_asof_iso} 重算全市场指标（约半分钟）…")
                _cache[_key] = fetcher.compute_metrics_asof(
                    _asof_iso, cols=_cols,
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
                f"收益/回撤，即当天筛选时实际可见的数据"
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

            # ── Merge 回撤/区间收益 BEFORE the return filter ─────────────────────
            # (In as-of mode work_df already carries the snapshot metrics columns.)
            # 区间收益优先用本地净值重算的值:榜单接口早间常出现净值/日增长率已
            # 更新到最新交易日、而近X收益率列仍是前一窗口旧值的情况(实测 018359
            # 榜单近1年 226.03 = 截至7-15,实际截至7-16 应为 211.39)。无本地净值
            # 历史的基金(非C类)回退榜单值——它们本来也没有回撤。
            if not asof_mode:
                metrics_df = load_metrics_df(fetcher.last_update_time())
                if metrics_df is not None:
                    filtered = filtered.merge(metrics_df, on="code", how="left",
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
            table = pd.DataFrame()
            table["基金代码"] = display.get("code")
            table["基金名称"] = display.get("name")
            table["类型"] = display.get("type")
            table["规模(亿)"] = pd.to_numeric(display.get("_aum"), errors="coerce").round(2)
            if ret_col in display.columns:
                table[ret_label] = pd.to_numeric(display[ret_col], errors="coerce").round(2)
            if mdd_col in display.columns:
                table[dd_label] = (pd.to_numeric(display[mdd_col], errors="coerce") * 100).round(2)

            # Default order (highest first); click any column header to re-sort.
            default_sort = ret_label if ret_label in table.columns else None
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

        _cells = []
        for _ri in _sel_rows:
            # 重新筛选后表格可能变短,session 里残留的旧勾选行号会越界。
            if _ri >= len(table):
                continue
            _cells.append(table.iloc[_ri])

        if _cells:
            st.markdown(f"##### 已选基金的最新十大持仓（截至 {_asof_lim}）")
            st.caption("取季度末 ≤ 截至日期的最新披露季度，"
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
            file_name="fund_screen.csv",
            mime="text/csv",
        )

# ─── Tab 2: Fund detail ───────────────────────────────────────────────────────
with tab_detail:
    code_input = st.text_input("输入基金代码（6位数字）", placeholder="例如 000001")
    # 净值/持仓/规模都在重表里,云端后台下载完之前当作"没输入代码"处理。
    if code_input and not _need_nav("基金详情"):
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

            # Compute the fund's metrics on the spot
            result = load_fund_metrics(code_input.strip().zfill(6))
            if result:
                s1, s2, s3 = st.columns(3)
                s1.metric("年化收益", f"{result['ann_return']*100:.2f}%")
                s2.metric("年化波动率", f"{result['volatility']*100:.2f}%")
                s3.metric("交易日数据点", result["data_points"])

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

# ─── Tab: 今天会买哪只(2026-08-13 加, 按钮触发) ─────────────────────────────
# 只在点按钮时算: 要跑一次 compute_metrics_asof(全市场重算区间收益,
# 实测约 9s)加上逐只 compute_beta, 挂在首屏会把整页拖慢, 而这个功能
# 一天最多看一次。
# 不显示"QVIX 是否触发": 页面拿到的是昨收, 盘中实时值在本地跑
# qvix_now.py, 摆一个基于昨收的"未触发"反而误导。
with tab_pick:
    st.caption(
        "从近3月跌幅最深的往下找, 一路上被规模/波动率刷掉的也列出来, "
        "**走到选中那只就截止**——之后的候选策略压根没看过。想看某只"
        "具体基金在这批候选里的位置, 用下面的「查基金代码」单查。"
        "跌幅超上限的那批只报个数(它们被一条跟基金本身无关的规则一刀切掉, "
        "逐只列会把表刷满)。口径与回测完全一致: 全部 T-1(净值当晚才公布)、"
        "规模用当时已披露的最新季报。"
        "「近3月窗口」列标出每只**实际**用的起止日: 终点是该基金早于"
        "决策日的最后一个净值日,起点是终点往回**91个自然日**取最近的"
        "净值日,"
        "所以净值更新慢的基金窗口会整体前移。⚠️ 这跟支付宝/东财的"
        "「近3月」不同——它们按**整3个自然月**取起点,差一个交易日就可能"
        "差零点几个百分点。"
        "默认算今天;把「决策日」拨到历史日期就是**回看那天会买哪只**"
        "——同一套 T-1 口径,连那天有没有 QVIX 信号一起报。")
    if not _need_nav("今日候选"):
        pass
    else:
        # ── 决策日可回拨(2026-08-28 加) ──────────────────────
        # 以前写死 dt.date.today()。scan/probe 本来就收 asof、口径
        # 全是 T-1, 换个日期算的就是那天收盘后该买哪只 —— 跟回测同
        # 一条路径, 但回测只留结果, 这里能看见完整的淘汰路径。
        # 下限 2020-01-01: QVIX 阈值是 490 个交易日的滚动分位, 库从
        # 2018-01-02 起, 头两年热身期算不出阈值(首个有阈值的日子是
        # 2019-12-16)。真选到算不出的日子, 下面那条 st.error 兜底。
        _today = dt.date.today()
        _c1, _c2 = st.columns([1, 2])
        with _c1:
            _asof = st.date_input(
                "决策日", value=_today, min_value=dt.date(2020, 1, 1),
                max_value=_today, key="today_pick_asof",
                help="按这一天收盘后的口径算:只看**早于**这天的净值和"
                     "当时已披露的季报规模,跟回测完全一致。拨到历史"
                     "日期就是回看那天会买哪只、路上刷掉了谁。")
        _is_today = _asof == _today
        _when = "今天" if _is_today else f"{_asof} 那天"
        with _c2:
            st.write("")        # 跟 date_input 的标签行对齐
            _go = st.button(f"算{_when}的候选", key="today_pick_go")
        if _go:
            with st.spinner("全市场重算近3月收益…约10秒"):
                st.session_state["today_pick_res"] = today_pick.scan(_asof)
        _tp = st.session_state.get("today_pick_res")
        # 日期一改就绝不能再渲染上一次的结果: 表里每个数字都绑死在
        # 某个 asof 上, 顶上挂着新日期、身子还是旧数据, 是这个页面
        # 最容易骗到人的一种错。宁可空着让人再点一次。
        if _tp is not None and _tp["asof"].date() != _asof:
            st.info(f"决策日已改成 {_asof},而当前这批结果是按 "
                    f"{_tp['asof'].date()} 算的,两者不能混着看"
                    f"——点「算{_when}的候选」重算。")
            _tp = None
        if _tp is not None:
            # 数据新鲜度校验: 这个功能最容易出的错不是算错, 是拿旧数据
            # 算出一个看起来很正常的答案 —— 必须显式摆出来, 不能默认
            # 它是新的。判据是 missing 不是 behind(见 data_freshness):
            # 要的是**早于决策日**的最后一个交易日的净值, 库到了那天
            # 就够。回看历史日期时库当然"不新", 但对那天的答案毫无影响,
            # 拿 behind 判会天天报一个假警。
            _missing = _tp.get("missing")
            _need = _tp.get("need_date")
            _nm = _tp.get("nav_max")
            _sm = _tp.get("sse_max")
            if _nm is None:
                st.error("净值库是空的,没法算。")
            elif _need is None:
                st.warning(f"⚠️ 上证行情里没有早于 {_asof} 的交易日,"
                           "这天算不出东西——换个日期。")
            elif _missing:
                st.warning(
                    f"⚠️ 数据不够: {_when}要的是截至 {_need.date()} 的净值,"
                    f"而净值库停在 {_nm.date()},**缺 {_missing} 个交易日**"
                    f"(上证已到 {_sm.date()})。下面的结果按旧数据算,仅供"
                    "参考——等每日跑批把净值补上(线上库最迟1小时换一次)"
                    "再点一次。")
            else:
                st.success(
                    f"✅ 数据够: {_when}要的净值截至 {_need.date()},"
                    f"净值库已到 {_nm.date()}")

            # 历史日期才报 QVIX 有没有触发。当天不报: 页面拿到的是昨收,
            # 盘中实时值要跑 qvix_now.py, 摆一个基于昨收的"未触发"反而
            # 误导。历史日期的收盘值是定局, 那天有没有信号恰恰是解读
            # 结果的前提 —— 没信号的日子策略根本不买。
            _thr_ok = (_tp.get("thr") is not None
                       and not pd.isna(_tp["thr"]))
            _q_ok = (_tp.get("qvix") is not None
                     and not pd.isna(_tp["qvix"]))
            if not _is_today and _thr_ok:
                if _tp.get("broken"):
                    # 链失效日: 没有值可报, 但那天恰恰是最恐慌的
                    # 几天之一(见 qvix_calc.CHAIN_BROKEN)。
                    st.success(
                        f"🔔 {_tp['qdate'].date()} QVIX **算不出来**"
                        "——标的跌到跑出期权链的挂牌行权价范围,公式在那天"
                        "失效了。算不出来本身就是极端行情的证据,按**触发**"
                        "处理(与回测口径一致)。")
                elif _q_ok and _tp["hit"]:
                    st.success(
                        f"🔔 {_tp['qdate'].date()} QVIX {_tp['qvix']:.2f} "
                        f">阈值 {_tp['thr']:.2f}——那天**触发了**买入信号。")
                elif _q_ok:
                    st.info(
                        f"😴 {_tp['qdate'].date()} QVIX {_tp['qvix']:.2f} "
                        f"≤阈值 {_tp['thr']:.2f}——那天**没有**信号,策略"
                        "当天不买;下面这批只是「假如买会买哪只」。")
            _std = _tp["std"]
            st.caption(
                f"口径: 近3月跌幅最大 · 跌幅≤{_std['max_drop']:.0f}% · "
                f"波动率比值≥{_std['min_vol_ratio']} · "
                f"规模≥{_std['min_aum']}亿"
                + (f"~{_std['max_aum']}亿" if _std["max_aum"] else "(无上限)")
                + f" · 跌超上限的 {_tp['too_deep']} 只已跳过")
            # 遍历路径和「查代码」共用这一张表: 查一只基金的意义就是
            # 把它摆到今天这批候选里比, 表结构不一样就没法比。数据行
            # 由 today_pick._row 统一产出(见那边说明)。
            def _pick_table(rows):
                return pd.DataFrame([{
                    "": "★" if r["role"].startswith("★") else "",
                    "跌幅榜": f"第{r['rank']}" if r["rank"] else "—",
                    "基金": f"{r['name']} ({r['code']})",
                    "近3月": (f"{r['ret3m']:+.2f}%"
                             if r["ret3m"] is not None else "—"),
                    "近3月窗口": (f"{r['win_start'][5:]}→{r['win_end'][5:]}"
                               if r["win_start"] and r["win_end"] else "—"),
                    "波动率比值": (f"{r['beta']:.2f}"
                               if r["beta"] is not None else "—"),
                    "规模(亿)": (f"{r['aum']:.2f}"
                              if r["aum"] is not None else "—"),
                    "结论": r["why"] or r["role"] or "通过",
                } for r in rows])

            if not _thr_ok:
                st.error(
                    f"{_asof} 这天算不出 QVIX 阈值(库里没有那天,或往前"
                    f"不够 {_tp['std']['window']} 个交易日的历史)"
                    "——换个日期。")
            elif not _tp["rows"]:
                st.info(f"{_when}没有跌幅带内的候选——按策略放弃当天。")
            else:
                _tpd = _pick_table(_tp["rows"])
                st.dataframe(_tpd, use_container_width=True,
                             hide_index=True,
                             height=(len(_tpd) + 1) * 35 + 3)
            if _tp["chosen"]:
                st.markdown(
                    f"**★ {_tp['names'].get(_tp['chosen'], _tp['chosen'])} "
                    f"({_tp['chosen']})**　基金回撤控制线 "
                    f"{_tp['fund_line']:.2f}%　大盘回撤线 "
                    f"{_tp['sse_line']:.2f}%")
            elif _tp["rows"]:
                # 只在真走过一遍跌幅带时才说"走完也没有": rows 空的那
                # 两种情况上面已经各报过一句, 这里再报一次是重复。
                st.info("走完跌幅带也没有合格候选——按策略放弃当天,"
                        "顺延到下一个信号日。")

            # ── 查任意代码(2026-08-18 加) ────────────────────────
            # 用途是跟支付宝/东财对账: 手上看到一只跌得很惨的基金,
            # 想知道这套口径为什么没选它 —— 是规则排掉的(QDII/持有期/
            # 跌幅超上限/规模不够), 还是根本没进池子(数据漏了)。所以
            # 「没进池子」那一栏必须说清是哪一种, 见 today_pick.probe。
            # ⚠️ 近3月口径跟支付宝不同: 这里是终点往回 91 个自然日,
            # 支付宝按整 3 个自然月, 差一个交易日就可能差零点几个点,
            # 「近3月窗口」列标的就是每只实际用的起止日, 对账看这个。
            _probe_in = st.text_input(
                "查基金代码(多个用逗号/空格分隔)", key="today_pick_probe",
                placeholder="例如 026834, 012710 016848",
                help=f"用{_when}这套完全相同的口径算一遍并摆进上面同一"
                     "张表, 看它在跌幅榜第几名、卡在哪道门槛。用来跟"
                     "支付宝的数字对账 —— 对不上先看「近3月窗口」那列。")
            if _probe_in.strip():
                _codes = [c for c in re.split(r"[,，\s]+", _probe_in.strip()) if c]
                with st.spinner("按同一口径现算…"):
                    _pr = today_pick.probe(_asof, _codes, _tp)
                _prd = _pick_table(_pr)
                st.dataframe(_prd, use_container_width=True,
                             hide_index=True,
                             height=(len(_prd) + 1) * 35 + 3)
                st.caption(
                    f"跟上面那张表同一套口径、同为 {_asof} 的数据。跌幅榜"
                    f"名次的分母是那天**真跌**的 {_tp.get('n_drop', '—')} 只"
                    "(涨的不在策略的遍历路径上, 混进分母只会让名次虚高)。")
