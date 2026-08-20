"""AKShare data fetching with SQLite caching."""

import io
import os
import sqlite3
import json
import re
import threading
import time
import logging
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import socket
import urllib3.util.connection

# 强制所有 HTTP 走 IPv4。
#
# 2026-07-30 起,GitHub Actions / Streamlit Cloud 上访问 hq.sinajs.cn 突然开始
# 报 [Errno 101] Network is unreachable。这不是被墙——被墙通常表现为超时或
# 连接被重置;Errno 101 的含义是"目标地址族没有路由",包根本没发出去。
# 查 DNS 发现新浪给 hq.sinajs.cn 加了 AAAA 记录:
#     A     125.94.246.104
#     AAAA  240e:97d:2000:a00::11:105     (240e::/20 = 中国电信 IPv6)
# 按 RFC 6724 客户端优先走 IPv6,而这些境外容器没有到该网络的 IPv6 路由,
# 于是秒失败。时间点吻合:07-29 那次 Actions 还跑通(邮件已发),07-30 开始断。
# 旁证:stock.finance.sina.com.cn(拿合约代码那个接口)只有 A 记录、没有
# AAAA,它就从没出过这个错。
#
# urllib3 官方留了 allowed_gai_family 钩子,改成只返回 AF_INET 就只解析
# IPv4。放模块级是有意的:akshare 底层也是 requests/urllib3,这样一处生效、
# 全链路覆盖,不用去改每个调用点(很多调用点在 akshare 内部,我们也改不到)。
# 副作用是本进程所有 HTTP 都只走 IPv4——GitHub/天天基金/上交所的 IPv4 都
# 一直可达,风险很低。
urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET


class _LazyAkshare:
    """akshare 导入要 ~3s(/mnt/c 上尤甚),而缓存命中的日常路径用不到它;
    首次真正访问属性时才导入,并把模块回填进全局名 ak。"""
    def __getattr__(self, name):
        import akshare
        globals()["ak"] = akshare
        return getattr(akshare, name)


ak = _LazyAkshare()

logger = logging.getLogger(__name__)
_CST = ZoneInfo("Asia/Shanghai")

# The DB lives on the WSL-native filesystem (ext4), NOT the project dir: the
# project sits on /mnt/c where every SQLite I/O crosses the 9p protocol —
# 10-100x slower, which made each Streamlit rerun spend ~10s just opening and
# reading the cache. Override with FUND_ANALYZER_DATA if needed.
_DATA_DIR = os.environ.get("FUND_ANALYZER_DATA") or os.path.join(
    os.path.expanduser("~"), ".local", "share", "fund-analyzer")
os.makedirs(_DATA_DIR, exist_ok=True)
CACHE_DB = os.path.join(_DATA_DIR, "fund_cache.db")
_LEGACY_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fund_cache.db")

# ── 分库布局 ─────────────────────────────────────────────────────────────────
# 一个 400MB 的 fund_cache.db 装下所有表, 是这个项目历史上大部分数据事故的
# 根源: 发布单元(整个文件)跟更新节奏(每张表各不相同)对不上。
#   · 想更新几 KB 的回测结果, 要整库上传 —— 2026-08-05 就这么把云端攒了
#     5 天的行情盖回了 07-30;
#   · 想更新 9MB 的规模表, 同样得连 233MB 净值一起推, 于是干脆不推;
#   · 本地想同步一点新数据, 最小粒度是 62MB 的重表段;
#   · 云端那份库是每日跑批"下载→改→传回"一路继承的, 本地 DROP 掉的表在它
#     里面永远删不掉(线上至今留着 4 张策略拆库前的僵尸表)。
#
# 所以按「谁写它 + 多久变一次 + 首屏要不要等」把表分到独立文件里, 每个文件
# 是一条独立的发布线, 互相盖不到:
#
#   库          文件               权威写入方      更新节奏   首屏
#   rank        fund_rank.db       每日跑批        日          要
#   market      market.db          跑批+本地QVIX   日          要
#   strategy    fund_strategy.db   本地回测        手动        要
#   cache       cache.db           线上 app        随时        丢了自动重建
#   nav         fund_nav.db        每日跑批        日          不要(233MB)
#   scale       fund_scale.db      每日跑批        季          不要(13.7MB)
#
# lazy=True 的两个是"点进基金详情才用"的大表, 云端首屏之后才后台拉。
DB_LAYOUT = (
    # (库名, 文件名, 表, 是否惰性加载)
    ("rank", "fund_rank.db",
     ("fund_list", "fund_sharpe", "fund_nav_meta", "app_meta",
      "fund_index_code", "etf_target_map"), False),
    ("market", "market.db",
     ("index_daily_cache", "qvix_self_history"), False),
    ("strategy", "fund_strategy.db",
     ("strategy_runs", "backtest_notes"), False),
    ("cache", "cache.db",
     ("filter_results",), False),
    ("nav", "fund_nav.db",
     ("fund_nav_daily",), True),
    ("scale", "fund_scale.db",
     ("fund_scale_hist", "fund_scale_miss", "fund_holdings"), True),
)
DB_PATH = {name: os.path.join(_DATA_DIR, fn) for name, fn, _t, _l in DB_LAYOUT}
LAZY_DBS = tuple(name for name, _fn, _t, lazy in DB_LAYOUT if lazy)

# 云端两段式:app.py 置 True 后, 惰性库在下载落地前不 ATTACH 真文件, 而是挂
# 一个同名的 :memory: 空库(见 _conn)。为什么不直接 ATTACH 那个还不存在的
# 路径:SQLite 会就地建一个空文件, 之后 os.path.exists 恒为真, "下载好了没"
# 就再也判断不出来了。也不能在 main 里建空占位表——SQLite 解析不带库名的表
# 名是 temp→main→attached, main 里留个同名空表会把 ATTACH 上来的真表整个
# 遮蔽掉(线上那 4 张僵尸表正在这么遮蔽 backtest_notes)。挂内存空库两头都
# 避开了: 查得到表(返回空结果, 页面不炸), 又不留下任何文件痕迹。
LAZY_NAV = False       # app.py 在云端两段式模式下置 True
_ready_dbs = set()     # 已落地并接管的惰性库名(仅 LAZY_NAV 下有意义)

_LEGACY_SINGLE_DB = CACHE_DB   # 分库前的 fund_cache.db, 由 split_dbs.py 迁移
_IN_MIGRATION = False          # split_dbs.py 置 True, 免得它自己触发迁移提示


def nav_ready() -> bool:
    """惰性库(净值/规模持仓)是否都已可用。本地恒为 True。"""
    return not LAZY_NAV or set(LAZY_DBS) <= _ready_dbs


def db_ready(name: str) -> bool:
    """单个库是否可用。非惰性库恒为 True。"""
    return name not in LAZY_DBS or not LAZY_NAV or name in _ready_dbs


def _migrate_db_location():
    """One-time move of an existing fund_cache.db out of the project dir.

    Uses the SQLite backup API (not a file copy) so pending WAL content is
    carried over intact; the legacy file is then removed.
    """
    if os.path.exists(CACHE_DB) or not os.path.exists(_LEGACY_DB):
        return
    logger.warning("migrating fund_cache.db to %s (one-time, ~373MB)", _DATA_DIR)
    src = sqlite3.connect(_LEGACY_DB)
    dst = sqlite3.connect(CACHE_DB)
    with dst:
        src.backup(dst)
    dst.close()
    src.close()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(_LEGACY_DB + suffix)
        except OSError:
            pass
# 26h:榜单本质是每日数据,由 update_daily.py/「更新数据」按钮强制刷新;
# 留 2h 余量给每日节奏。此前 1h 会让盘中任意交互触发的 rerun 穿透缓存、
# 现场全量重拉榜单卡住页面几十秒,还把标签页顶回首页。
FUND_LIST_TTL = 26 * 3600
NAV_TTL = 86400         # 24 hours
NAV_START = "2018-01-01"  # NAV history is kept from this date onward
MAX_WORKERS = 8
# 无风险利率。夏普 2026-08-07 移除后, 唯一的消费者是自算 QVIX: qvix_core 用
# Black-Scholes 反推期权理论价, 优先按 SHIBOR 期限结构插值, 取不到 SHIBOR 时
# 才退到这里的 1 年期国债收益率(见 qvix_calc.compute_qvix 的 fallback_rate)。
# 别再当成夏普的残留删掉。
RISK_FREE_RATE = 0.0113  # fallback 1-year China gov bond yield (see get_risk_free_rate)
RF_TTL = 30 * 86400      # auto-refresh the risk-free rate ~monthly
HOLDINGS_START_YEAR = 2020     # first year fetched for quarterly holdings
HOLDINGS_START_Q = "2020Q4"    # earliest quarter kept ("YYYYQn" strings compare fine)
HOLDINGS_TTL = 7 * 86400       # current year re-checked weekly for new quarterly reports
HOLDINGS_TTL_PAST = 30 * 86400 # past years' disclosures barely change


# ── DB helpers ───────────────────────────────────────────────────────────────

_schema_ready = False      # 进程内只建一次表(CREATE IF NOT EXISTS 也不是免费的)


def _conn():
    """连上全部分库。返回的连接里, 不带库名的表名照常能解析到对应的库,
    所以那七十来处 "FROM fund_nav_daily" 一个字都不用改。

    main 固定是 :memory: —— 所有真实的表都在 ATTACH 上来的库里, main 永远
    是空的。这是刻意的: SQLite 解析不带库名的表名是 temp→main→attached,
    只要 main 里有一张同名表就会把 attached 上来的真表整个遮蔽掉。以前
    main 是 fund_cache.db, 这个坑踩过两次(nav 的占位表, 以及线上至今留着
    的 4 张策略拆库前的僵尸表正在遮蔽 backtest_notes)。main 空着, 这类
    bug 在结构上就不可能发生。

    timeout 给并发跑批留出等锁时间, 而不是直接 "database is locked"。
    """
    global _schema_ready
    conn = sqlite3.connect(":memory:", check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    placeholders = []
    for name, _fn, _tables, lazy in DB_LAYOUT:
        if lazy and LAZY_NAV and name not in _ready_dbs:
            # 云端首屏: 大库还没下下来。挂一个同名的内存空库并建空表, 查询
            # 返回空结果而不是 "no such table" 把页面炸掉。不去 ATTACH 那个
            # 还不存在的路径 —— SQLite 会就地建出空文件, 之后
            # os.path.exists 恒为真, "下载好了没" 就再也判断不出来了。
            conn.execute(f"ATTACH DATABASE ':memory:' AS {name}")
            placeholders.append(name)
            continue
        conn.execute(f"ATTACH DATABASE ? AS {name}", (DB_PATH[name],))
    # 内存占位库每次都是新的, 必须每次建表; 真实文件库只在进程内首次建。
    for name in placeholders:
        conn.executescript(_DDL[name])
    if not _schema_ready:
        for name, _fn, _tables, lazy in DB_LAYOUT:
            if name not in placeholders:
                conn.executescript(_DDL[name])
        conn.commit()
        _schema_ready = True
    return conn


def adopt_db(name: str):
    """惰性库下载落地后让它接管, 并把该库的派生索引建回去。

    idx_nav_date 不随快照下发(它占 120MB, 是净值库的三成, 却是纯派生数据):
    4.9M 行现建实测 1~2 秒, 比多下 17MB(gz)划算。

    不再需要"删掉 main 里的占位表"那一步了 —— 占位表现在挂在一个每次连接
    都重建的内存库里, 换成真文件后自然就没了(见 _conn)。
    """
    _ready_dbs.add(name)
    conn = _conn()
    try:
        if name == "nav":
            # 必须写成 nav.idx_nav_date:CREATE INDEX 不带库名一律建在 main,
            # 而 SQLite 要求索引和表同库,表在 nav 库里的话会直接报 no such
            # table(读表名的 temp→main→attached 解析顺序在这儿不适用)。
            conn.execute("CREATE INDEX IF NOT EXISTS nav.idx_nav_date "
                         "ON fund_nav_daily(date)")
            conn.commit()
    finally:
        conn.close()
    logger.info("db adopted: %s → %s", name, DB_PATH[name])


def adopt_nav_db():
    """兼容旧名字(app.py 老版本会调它)。"""
    adopt_db("nav")


# ── 建表语句(按库归位)────────────────────────────────────────────────────────
# 每个库一段脚本, 表名一律带库名限定。必须带: CREATE TABLE 不写库名一律建在
# main, 而 main 是 :memory:(见 _conn), 建出来的空表会把 ATTACH 上来的真表
# 整个遮蔽掉——正是线上那 4 张僵尸表在干的事。
#
# 加新表时要同时改 DB_LAYOUT, 否则它不属于任何库、也不会被快照切分带上。
_DDL = {

"rank": """
    -- 榜单快照:两万来只基金的全部字段压成一整块 JSON 存在一行里, 每天跑批
    -- 全量重写。改一个字段要重写 7MB, 但它本来就是整读整写的。
    CREATE TABLE IF NOT EXISTS rank.fund_list (
        id       INTEGER PRIMARY KEY,
        data     TEXT    NOT NULL,
        saved_at REAL    NOT NULL
    );
    -- Per-fund freshness + newest stored date, so gap detection and TTL
    -- checks never have to scan fund_nav_daily.
    CREATE TABLE IF NOT EXISTS rank.fund_nav_meta (
        code      TEXT PRIMARY KEY,
        saved_at  REAL NOT NULL,
        last_date TEXT
    );
    -- 表名是历史遗留: 建表时它只存夏普, 后来长出了 mdd/ret 各四列, 2026-08-07
    -- 夏普三列(sharpe/sharpe_6m/sharpe_1y)又被整个拿掉。现在装的是年化收益、
    -- 年化波动率、分区间最大回撤和分区间收益率。没改名是为了不动云端 release
    -- 里那份 fund_rank.db。
    CREATE TABLE IF NOT EXISTS rank.fund_sharpe (
        code        TEXT PRIMARY KEY,
        ann_return  REAL,
        volatility  REAL,
        data_points INTEGER,
        saved_at    REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS rank.app_meta (
        key      TEXT PRIMARY KEY,
        value    REAL,
        saved_at REAL NOT NULL
    );
    -- ETF联接基金 → 目标场内ETF 的映射(重仓穿透用)。target_code 为空
    -- 串表示解析失败,按 TTL 重试;成功的映射视为永久。
    CREATE TABLE IF NOT EXISTS rank.etf_target_map (
        code        TEXT PRIMARY KEY,
        target_code TEXT NOT NULL,
        target_name TEXT NOT NULL,
        saved_at    REAL NOT NULL
    );
    -- 基金跟踪指数代码缓存(mobapi FundMNBasicInformation.INDEXCODE),
    -- 供联接基金与候选 ETF 做指数一致性验证。
    CREATE TABLE IF NOT EXISTS rank.fund_index_code (
        code       TEXT PRIMARY KEY,
        index_code TEXT NOT NULL,
        saved_at   REAL NOT NULL
    );
""",

"market": """
    -- 外部指数/波动率日线缓存, 一个 key 一整块 JSON。key: sse(上证)、
    -- hsi/vhsi(恒生及其波动率)、ixic/vix、au9999/gvz、qvix(外部源)。
    -- sse/hsi/vhsi 每日跑批刷新, 其余按 TTL 惰性拉。
    CREATE TABLE IF NOT EXISTS market.index_daily_cache (
        key TEXT PRIMARY KEY, data TEXT, saved_at REAL);
    -- 自算 QVIX 日频历史 + 当日阈值(490交易日90分位)。策略的信号判定读它。
    -- ⚠️ QVIX 不分方向: 暴跌暴涨都推高它(与上证日涨跌相关系数仅 -0.19,
    -- 97个信号日里 55% 发生在上涨日)。2026-08 试过加一列 skew(虚值认沽IV
    -- − 虚值认购IV, |delta|≈0.25)来补方向, 判别力确实强(相关 -0.463), 但
    -- **拿它当买入信号回测下来是负的**, 已整套删除, 别再走一遍: 单独跑
    -- 98分位 15 笔 +579%, 其中赚钱的 4 笔本就是 QVIX 也会开的仓, 剩下 10
    -- 笔"新增信号"复利 -6.08%、胜率 3/10 —— 平静期(QVIX 12~13)skew 冲高多
    -- 半是期权卖方结构(备兑压低认购IV)造成的, 不是真有人抢买保护, 进场后
    -- 没有恐慌就没有反弹可吃, 只能原地漂到止损。
    CREATE TABLE IF NOT EXISTS market.qvix_self_history (
        date TEXT PRIMARY KEY, qvix REAL, note TEXT, threshold REAL);
""",

"strategy": """
    -- 每跑一次回测追加一条(见 save_strategy_run)。is_standard 标记线上
    -- 标准策略那次, 主复盘表读最新的标准跑批。
    CREATE TABLE IF NOT EXISTS strategy.strategy_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_at REAL, label TEXT,
        params TEXT, trades TEXT, is_standard INTEGER DEFAULT 0,
        n_trades INTEGER, win_rate REAL, cum_return REAL);
    -- 人工写的复盘点评, 按买入日索引。
    CREATE TABLE IF NOT EXISTS strategy.backtest_notes (
        buy_date TEXT PRIMARY KEY, note TEXT, saved_at REAL);
""",


"cache": """
    -- Top rows (200) of each distinct filter run, keyed by a hash of the
    -- filter params (+ metrics version in live mode), so repeating a
    -- filter is a read instead of a recompute — across restarts too.
    -- 纯缓存: 丢了自动重算, 所以这个库不参与"别覆盖"的那套小心翼翼。
    CREATE TABLE IF NOT EXISTS cache.filter_results (
        key      TEXT PRIMARY KEY,
        params   TEXT NOT NULL,
        data     TEXT NOT NULL,
        saved_at REAL NOT NULL
    );
""",

"nav": """
    -- One row per fund per day: appends are single-row INSERTs instead of
    -- rewriting a whole per-fund JSON blob (the old fund_nav design, which
    -- churned ~20KB of freelist pages per fund per update).
    CREATE TABLE IF NOT EXISTS nav.fund_nav_daily (
        code          TEXT NOT NULL,
        date          TEXT NOT NULL,    -- ISO yyyy-mm-dd
        nav           REAL,
        daily_ret_pct REAL,
        acc_nav       REAL,
        PRIMARY KEY (code, date)
    ) WITHOUT ROWID;
""",

"scale": """
    -- 基金季度规模(期末净资产)历史, 来自东财 F10 规模变动(gmbd)。一支基金
    -- 一季一行, aum 单位亿元, publish_date 是该季报的估算披露日(季末后第15
    -- 个交易日), 供回测/选基按"信号日当时能看到的最新一期"取规模、避免未来
    -- 函数。注意规模门槛用的是 A/C 合并口径, 见 fund_aum_asof。
    CREATE TABLE IF NOT EXISTS scale.fund_scale_hist (
        code         TEXT NOT NULL,
        quarter_end  TEXT NOT NULL,   -- ISO yyyy-mm-dd(季末)
        aum          REAL,            -- 期末净资产,亿元
        publish_date TEXT NOT NULL,   -- 估算披露日 ISO yyyy-mm-dd
        saved_at     REAL NOT NULL,
        PRIMARY KEY (code, quarter_end)
    ) WITHOUT ROWID;
    -- 规模抓取"请求成功但页面里确实没有规模变动表"的记录(场内份额、刚成立、
    -- 已清盘)。没有这张表的话它们在 fund_scale_hist 里永远留不下痕迹, 每天
    -- 跑批都要重抓一遍。只在请求确实成功时才记 —— 见 fetch_fund_scale_hist。
    CREATE TABLE IF NOT EXISTS scale.fund_scale_miss (
        code     TEXT PRIMARY KEY,
        tried_at REAL NOT NULL
    ) WITHOUT ROWID;
    -- Quarterly top-10 holdings (stocks + bonds) per fund, one year of
    -- quarters per row, stored as normalized JSON records. 跟规模同为季度
    -- 数据、同一个更新节奏, 所以同库。
    CREATE TABLE IF NOT EXISTS scale.fund_holdings (
        code     TEXT NOT NULL,
        year     TEXT NOT NULL,
        data     TEXT NOT NULL,
        saved_at REAL NOT NULL,
        PRIMARY KEY (code, year)
    ) WITHOUT ROWID;
""",
}


def init_db():
    """建好全部分库的表结构, 并做历史迁移。

    建表本身已经由 _conn() 按 _DDL 完成(每个进程首次连接时), 这里只剩
    分库无关的收尾: WAL、列迁移、旧 JSON 净值迁移。
    """
    if os.path.exists(_LEGACY_SINGLE_DB) and not _IN_MIGRATION:
        # 分库前的单库还在: 说明这台机器没跑过 split_dbs.py。不自动迁移——
        # 401MB 的库拆一次要几分钟, 悄悄在 import 时做会让人以为程序卡死。
        logger.warning(
            "检测到分库前的 %s, 新代码不会读它。跑 `python3 split_dbs.py` 迁移。",
            _LEGACY_SINGLE_DB)
    conn = _conn()
    # WAL lets the app keep reading while the pipeline writes (and vice versa).
    # 每个 attached 库各自设置——journal_mode 是 per-database 的 pragma。
    for name, _fn, _t, _l in DB_LAYOUT:
        if db_ready(name):
            try:
                conn.execute(f"PRAGMA {name}.journal_mode=WAL")
            except sqlite3.DatabaseError as e:
                logger.debug("WAL on %s failed: %s", name, e)
    # Add per-period max-drawdown / return columns (migration for existing DBs).
    # Returns are recomputed locally from stored NAV because the EastMoney rank
    # list's 近X收益率 columns lag its own nav_date in the morning (nav/日增长率
    # updated, period returns still the prior window's).
    for col in ("mdd_1m", "mdd_3m", "mdd_6m", "mdd_1y",
                "ret_1m", "ret_3m", "ret_6m", "ret_1y"):
        try:
            conn.execute(f"ALTER TABLE rank.fund_sharpe ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass  # column already exists
    # 反向迁移(2026-08-07): 夏普不再看, 把存量库里的三列丢掉。新建的库压根
    # 没有这些列, 所以 OperationalError 是正常路径而非异常。DROP COLUMN 需要
    # SQLite 3.35+(Python 3.12 自带的远高于此); 真跑在老 SQLite 上就跳过,
    # 残留列没人读也没人写, 留着只占地方不出错。
    for col in ("sharpe", "sharpe_6m", "sharpe_1y"):
        try:
            conn.execute(f"ALTER TABLE rank.fund_sharpe DROP COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # 列不存在(新库)或 SQLite 太老
    _migrate_nav_blobs(conn)
    conn.commit()
    conn.close()


def _migrate_nav_blobs(conn):
    """One-time migration: legacy fund_nav JSON blobs → fund_nav_daily rows."""
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fund_nav'"
    ).fetchone():
        return
    for r in conn.execute("SELECT code, data, saved_at FROM fund_nav").fetchall():
        df = _nav_from_json(r["data"])
        if not df.empty and "date" in df.columns:
            _write_nav_rows(conn, r["code"], df, saved_at=r["saved_at"])
    conn.execute("DROP TABLE fund_nav")
    logger.info("migrated legacy fund_nav JSON blobs to fund_nav_daily")


# ── Risk-free rate ───────────────────────────────────────────────────────────
# The 1-year China government bond yield, fetched automatically and cached for a
# month, so nobody has to keep a number up to date by hand.

def _get_meta(key: str):
    conn = _conn()
    row = conn.execute("SELECT value, saved_at FROM app_meta WHERE key=?", (key,)).fetchone()
    conn.close()
    return (row["value"], row["saved_at"]) if row else (None, None)


def _set_meta(key: str, value: float):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO app_meta (key, value, saved_at) VALUES (?, ?, ?)",
        (key, value, time.time()),
    )
    conn.commit()
    conn.close()


def _fetch_treasury_1y() -> Optional[float]:
    """Latest 1-year China government bond yield as a decimal (e.g. 0.0113)."""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    df = ak.bond_china_yield(start_date=start, end_date=end)
    df = df[df["曲线名称"] == "中债国债收益率曲线"].sort_values("日期")
    val = pd.to_numeric(df["1年"], errors="coerce").dropna()
    return float(val.iloc[-1]) / 100.0 if not val.empty else None


def get_risk_free_rate(force_refresh: bool = False) -> float:
    """1-year China treasury yield as the risk-free rate, cached ~monthly.

    Falls back to the last cached value, then RISK_FREE_RATE, if the fetch fails.
    自算 QVIX 的 Black-Scholes 兜底利率(SHIBOR 取不到时)——见 RISK_FREE_RATE
    上面那段注释。
    """
    value, saved_at = _get_meta("rf_rate")
    if not force_refresh and value is not None and (time.time() - saved_at) < RF_TTL:
        return value
    try:
        rf = _fetch_treasury_1y()
        if rf is not None and 0 < rf < 0.2:   # sanity bound
            _set_meta("rf_rate", rf)
            return rf
    except Exception as e:
        logger.debug("risk-free rate fetch failed: %s", e)
    return value if value is not None else RISK_FREE_RATE


def clear_all_caches():
    """Wipe every cache table: fund list, NAV history, computed 指标.

    Used by the sidebar "清空所有缓存" button so a code/口径 change can be picked
    up cleanly — afterwards the list re-fetches and 指标 recompute on the next
    ⚡ run rather than being served stale from cache.
    """
    conn = _conn()
    conn.execute("DELETE FROM fund_list")
    conn.execute("DELETE FROM fund_nav_daily")
    conn.execute("DELETE FROM fund_nav_meta")
    conn.execute("DELETE FROM fund_sharpe")
    conn.execute("DELETE FROM fund_holdings")
    conn.commit()
    conn.close()


# Look-back windows in CALENDAR days, matching how EastMoney defines 近1月/3月/
# 6月/1年 (date-to-date from the latest NAV date), so the computed drawdown
# covers the same period as the 近X 收益率 columns shown alongside it.
DRAWDOWN_DAYS = {"mdd_1m": 30, "mdd_3m": 91, "mdd_6m": 182, "mdd_1y": 365}

# Period returns (%), matching the rank list's 近1月/3月/6月/1年 columns; used
# when metrics are recomputed as of a past date and the list values don't apply.
RETURN_DAYS = {"ret_1m": 30, "ret_3m": 91, "ret_6m": 182, "ret_1y": 365}


def effective_daily_ret(df: pd.DataFrame) -> pd.Series:
    """每日实际收益率(小数),供收益/回撤/分红检测统一使用。

    优先取官方日增长率(daily_ret_pct);但当它与净值环比矛盾(差>0.3pp)
    且累计净值也同步变动(说明不是分红/拆分)时,回退为净值环比——修复
    定开/建仓期基金按周披露净值却把日增长率报成 0 的数据问题(全库约
    1.7 万行、3 千余只基金,如 008092 的 2020 年初,直接累乘会把当时的
    股灾算成 0 波动)。真正的分红日(累计净值走势与日增长率一致、单位
    净值跳水)仍信官方日增长率。缺失值回退净值环比。
    """
    nav = pd.to_numeric(df["nav"], errors="coerce")
    acc = pd.to_numeric(df["acc_nav"], errors="coerce").fillna(nav) \
        if "acc_nav" in df.columns else nav
    r = pd.to_numeric(df["daily_ret_pct"], errors="coerce") / 100.0 \
        if "daily_ret_pct" in df.columns \
        else pd.Series(np.nan, index=df.index)
    implied_nav = nav / nav.shift(1) - 1.0
    implied_acc = acc / acc.shift(1) - 1.0
    conflict = (r - implied_nav).abs() > 0.003
    dividendish = (r - implied_acc).abs() <= 0.003
    out = r.copy()
    use_implied = (conflict & ~dividendish & implied_nav.notna()) | r.isna()
    out[use_implied] = implied_nav[use_implied]
    # 巨额赎回:惩罚性赎回费摊入剩余净值时,净值/累计净值/官方日增长率
    # 三者一致地单日暴涨(014939 2025-03-31 +68.7%、018658 +38.2%、
    # 005297 +31.3%),上面的分红判别拦不住,只能按幅度判无效。阈值取
    # ±30%:全库实测最大真实行情日是 +25.6%(2024-10-08 北交所基金),
    # 满仓 30cm 涨停的理论上限也不到 30%。
    out[out.abs() > 0.30] = 0.0
    return out


# A window anchor may miss by a few days when the ideal start lands in a
# holiday gap or just before the stored history begins (data starts NAV_START,
# but 01-01 itself is a holiday). Accept the earliest NAV as anchor when it is
# at most this many days late; beyond that the fund is genuinely too young.
ANCHOR_GRACE_DAYS = 10


def _window_by_date(df: pd.DataFrame, days_back: int) -> Optional[pd.DataFrame]:
    """Rows from the anchor through the latest NAV, for a trailing date window.

    The anchor is the last NAV on or before (latest_date - days_back); it is the
    base point one period ago (e.g. the NAV "one year ago"). It is kept in the
    slice so it can serve as the drawdown peak candidate and as the base for the
    first in-window daily return. Returns None when the fund has no NAV old
    enough to anchor the window (e.g. a fund younger than the period).

    `df` must be sorted ascending by `date` with a 0..n-1 RangeIndex.
    """
    end_date = df["date"].max()
    start_date = end_date - timedelta(days=days_back)
    older = df[df["date"] <= start_date]
    if older.empty:
        first = df["date"].iloc[0]
        if (first - start_date).days <= ANCHOR_GRACE_DAYS:
            return df
        return None
    return df.loc[older.index[-1]:]


def _max_drawdown(nav: pd.Series) -> Optional[float]:
    """Max drawdown magnitude (positive fraction) for an ascending NAV series."""
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    if len(nav) < 2:
        return None
    cummax = nav.cummax()
    dd = nav / cummax - 1.0
    return float(-dd.min())


def _annualized(r: pd.Series, span_days: int):
    """(annual_return, annual_vol) from a daily-return series.

    Everything is derived from the window itself — no fixed trading-day constant
    — so it adapts to A-shares, QDII/US funds, HK, etc. automatically:
      • return: the actual compounded return over the window, annualized by its
        real calendar span (a full year keeps its real return, no inflation);
      • volatility: daily σ × √(observations per year), where observations-per-
        year is *measured* from how many NAV points actually fell in the window
        rather than assumed to be 252.
    None if degenerate.
    """
    r = r.dropna()
    n = len(r)
    if n < 2:
        return None
    std_daily = r.std(ddof=1)
    if std_daily == 0 or np.isnan(std_daily):
        return None
    span = max(span_days, 1)
    total_growth = float((1.0 + r).prod())          # e.g. 4.38 = +338%
    ann_return = total_growth ** (365.0 / span) - 1.0
    obs_per_year = n * 365.0 / span                  # measured, not the 252 convention
    ann_vol = std_daily * np.sqrt(obs_per_year)
    return ann_return, ann_vol


def _period_return(df: pd.DataFrame, days_back: int) -> Optional[float]:
    """Compounded % return over the trailing `days_back` calendar-day window
    (daily growth rates multiplied up, so dividends are handled). None when
    the fund lacks history old enough to anchor the window."""
    window = _window_by_date(df, days_back)
    if window is None:
        return None
    r = window["r"].iloc[1:].dropna()
    if r.empty:
        return None
    return float(((1.0 + r).prod() - 1.0) * 100.0)


def _period_mdd(df: pd.DataFrame, days_back: int) -> Optional[float]:
    """Max drawdown over the trailing `days_back` window, 按校正日收益率的
    复利增长指数计算(分红视作再投)。

    此前用累计净值:它把历史分红单利加回,对分红基金会低估回撤
    (002583 近1年 14.51% vs 复利口径 16.45%)。
    需要 df 已带校正收益列 r(_metrics_from_nav 保证)。"""
    window = _window_by_date(df, days_back)
    if window is None:
        return None
    r = window["r"].iloc[1:].fillna(0.0)   # 首行是窗口锚点,其收益属窗口外
    if r.empty:
        return None
    idx = pd.concat([pd.Series([1.0]), (1.0 + r).cumprod()],
                    ignore_index=True)
    return _max_drawdown(idx)


# ── C-class share detection ──────────────────────────────────────────────────
# Only C-class shares get NAV history stored/backfilled (the user only buys C).
# A name counts as C-class when C is followed by end-of-string, a parenthesis,
# 类, or a currency suffix — e.g. 「XX混合C」「XXC(QDII)」「XX(QDII)C人民币」.
# The C inside abbreviations like CES/MSCI/CAC40 never matches.
_C_CLASS_RE = re.compile(r"[Cc](类|\(|（|人民币|美元|$)")


def is_c_class(name) -> bool:
    if not name or pd.isna(name):
        return False
    return _C_CLASS_RE.search(str(name).strip()) is not None


# Overseas-equity funds are excluded from storage/backfill entirely: QDII
# quota limits keep roughly half of them capped at a few hundred yuan per day
# (median cap ~500 CNY as of 2026-07), so they can't actually be bought in
# meaningful size and would only pollute screening results.
OVERSEAS_EQUITY_TYPES = {"指数型-海外股票", "QDII-普通股票", "QDII-混合偏股"}


def is_overseas_equity(fund_type) -> bool:
    return fund_type in OVERSEAS_EQUITY_TYPES


# 债券/固收/偏债类基金一律不入库、不参与筛选(用户不做债基)。类型名含
# 「债」或「固收」即命中:债券型全部子类、指数型-固收、混合型-偏债、
# QDII-纯债、QDII-混合债。偏股/混合偏股等不含这两词,不会误伤。
def is_bond(fund_type) -> bool:
    t = str(fund_type)
    return "债" in t or "固收" in t


# ── Fund list ────────────────────────────────────────────────────────────────

# The fund list is a single cached snapshot (always read/written whole), so it
# lives in one fixed row (id=1) and is upserted in place.
def _load_fund_list_cache() -> Optional[pd.DataFrame]:
    conn = _conn()
    row = conn.execute(
        "SELECT data, saved_at FROM fund_list WHERE id = 1"
    ).fetchone()
    conn.close()
    if row and (time.time() - row["saved_at"]) < FUND_LIST_TTL:
        return pd.DataFrame(json.loads(row["data"]))
    return None


def _save_fund_list_cache(df: pd.DataFrame):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO fund_list (id, data, saved_at) VALUES (1, ?, ?)",
        (df.to_json(orient="records", force_ascii=False), time.time()),
    )
    conn.commit()
    conn.close()


def fetch_fund_list(force_refresh: bool = False) -> pd.DataFrame:
    """Return all open-end funds with basic performance from EastMoney.

    Uses ak.fund_open_fund_rank_em(symbol='全部') which returns:
    序号, 基金代码, 基金简称, 日期, 单位净值, 累计净值, 日增长率,
    近1周, 近1月, 近3月, 近6月, 近1年, 近2年, 近3年, 今年来, 成立来, 手续费
    """
    if not force_refresh:
        cached = _load_fund_list_cache()
        if cached is not None:
            return cached

    df = ak.fund_open_fund_rank_em(symbol="全部")
    df = df.rename(columns={
        "基金代码": "code",
        "基金简称": "name",
        "日期": "nav_date",
        "单位净值": "nav",
        "累计净值": "acc_nav",
        "日增长率": "daily_ret",
        "近1周": "ret_1w",
        "近1月": "ret_1m",
        "近3月": "ret_3m",
        "近6月": "ret_6m",
        "近1年": "ret_1y",
        "近2年": "ret_2y",
        "近3年": "ret_3y",
        "今年来": "ret_ytd",
        "成立来": "ret_inception",
        "手续费": "fee",
    })

    # fund_open_fund_rank_em doesn't include type; merge with fund_name_em
    try:
        name_df = ak.fund_name_em()[["基金代码", "基金类型"]].rename(
            columns={"基金代码": "code", "基金类型": "type"}
        )
        df = df.merge(name_df, on="code", how="left")
    except Exception:
        df["type"] = "未知"

    df["ret_1y_pct"] = pd.to_numeric(df["ret_1y"], errors="coerce")

    _save_fund_list_cache(df)
    return df


# ── NAV history ──────────────────────────────────────────────────────────────

def _nav_from_json(blob: str) -> pd.DataFrame:
    """Rebuild a NAV DataFrame from a legacy fund_nav JSON blob (migration only).

    df.to_json serialized datetimes as epoch-millisecond ints, which
    pd.to_datetime would otherwise misread as nanoseconds (everything → 1970).
    Newer rows were stored ISO-formatted; handle both shapes.
    """
    df = pd.DataFrame(json.loads(blob))
    if not df.empty and "date" in df.columns:
        if pd.api.types.is_numeric_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], unit="ms")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _write_nav_rows(conn, code: str, df: pd.DataFrame,
                    saved_at: Optional[float] = None):
    """Upsert NAV rows for one fund and refresh its meta row. No commit —
    the caller owns the transaction."""
    df = df.dropna(subset=["date"])
    if df.empty:
        return
    dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    nav = pd.to_numeric(df["nav"], errors="coerce")
    ret = pd.to_numeric(df["daily_ret_pct"], errors="coerce") \
        if "daily_ret_pct" in df.columns else pd.Series(np.nan, index=df.index)
    acc = pd.to_numeric(df["acc_nav"], errors="coerce").fillna(nav) \
        if "acc_nav" in df.columns else nav
    rows = [
        (code, d,
         None if pd.isna(n) else float(n),
         None if pd.isna(r) else float(r),
         None if pd.isna(a) else float(a))
        for d, n, r, a in zip(dates, nav, ret, acc)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO fund_nav_daily "
        "(code, date, nav, daily_ret_pct, acc_nav) VALUES (?, ?, ?, ?, ?)", rows)
    conn.execute(
        "INSERT OR REPLACE INTO fund_nav_meta (code, saved_at, last_date) "
        "VALUES (?, ?, (SELECT MAX(date) FROM fund_nav_daily WHERE code=?))",
        (code, saved_at if saved_at is not None else time.time(), code))


def _load_nav_df(code: str, conn=None) -> pd.DataFrame:
    """Stored NAV history for one fund, ascending by date.

    Columns: date (datetime64), nav, daily_ret_pct, acc_nav.
    """
    own = conn is None
    if own:
        conn = _conn()
    df = pd.read_sql_query(
        "SELECT date, nav, daily_ret_pct, acc_nav FROM fund_nav_daily "
        "WHERE code = ? ORDER BY date", conn, params=(code,))
    if own:
        conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


_EM_PZD_URL = "https://fund.eastmoney.com/pingzhongdata/{code}.js"


def _fetch_nav_full(code: str) -> Optional[pd.DataFrame]:
    """NAV history since NAV_START in a single request via pingzhongdata.

    The one js blob carries unit NAV, daily growth AND accumulated NAV, so this
    replaces akshare's fund_open_fund_info_em (which downloads the same blob
    once per indicator and parses it with a JS engine) — the two series are
    pulled out with a regex + json.loads instead.
    Columns: date, nav, daily_ret_pct, acc_nav (ascending). None on failure.
    """
    def _dates(ms):
        # timestamps are midnight Beijing time; naive UTC parse would land on
        # the previous day, so convert before dropping the timezone
        s = pd.to_datetime(ms, unit="ms", utc=True)
        return s.dt.tz_convert("Asia/Shanghai").dt.normalize().dt.tz_localize(None)

    try:
        r = requests.get(_EM_PZD_URL.format(code=code),
                         headers=_EM_LSJZ_HEADERS, timeout=20)
        m = re.search(r"var Data_netWorthTrend\s*=\s*(\[.*?\])\s*;", r.text)
        if not m:
            return None
        unit = pd.DataFrame(json.loads(m.group(1)))
        if unit.empty:
            return None
        df = pd.DataFrame({
            "date": _dates(unit["x"]),
            "nav": pd.to_numeric(unit["y"], errors="coerce"),
            "daily_ret_pct": pd.to_numeric(
                unit.get("equityReturn"), errors="coerce"),
        })

        # Accumulated (dividend-reinvested) NAV for drawdown; fall back to unit
        # NAV where the accumulated series is missing.
        m = re.search(r"var Data_ACWorthTrend\s*=\s*(\[.*?\])\s*;", r.text)
        acc_raw = json.loads(m.group(1)) if m else []
        if acc_raw:
            acc = pd.DataFrame(acc_raw, columns=["x", "acc_nav"])
            acc["date"] = _dates(acc["x"])
            df = df.merge(acc[["date", "acc_nav"]], on="date", how="left")
        else:
            df["acc_nav"] = df["nav"]
        df["acc_nav"] = pd.to_numeric(df["acc_nav"], errors="coerce").fillna(df["nav"])

        df = df[df["date"] >= pd.Timestamp(NAV_START)]
        return df.sort_values("date").reset_index(drop=True)

    except Exception as e:
        logger.debug("full NAV fetch failed for %s: %s", code, e)
        return None


# ── 基金季度规模(AUM)历史 ────────────────────────────────────────────────────
# 期末净资产(亿元)来自基金「季度报告」——四个季度都有, 披露截止是季末后
# 「15个工作日」内(《基金信息披露管理办法》), 四个季度同一口径。A股交易日
# ≈法定工作日, 所以用交易日历数季末后第15个交易日当披露日, 能自动把国庆/
# 春节顺延算进去(Q4→次年1月下旬、Q3→10月底)。存 publish_date, 这样按信号日
# 取"当时真正能看到的最新一期", 避免用到尚未披露的数据(未来函数)。
# (半年报6月末/年报12月末是另外更详细的披露, 但期末净资产在季报里就有, 所以
# 规模不用等到8月底/次年3月底。)
SCALE_TTL = 7 * 24 * 3600  # fetch_fund_scale_hist 的 cache-first 兜底(app按需取数用)
_TRADE_CAL = None           # 交易日历缓存(升序 DatetimeIndex/Series), 进程内只拉一次


def _trade_calendar():
    global _TRADE_CAL
    if _TRADE_CAL is None:
        cal = ak.tool_trade_date_hist_sina()
        _TRADE_CAL = pd.to_datetime(cal["trade_date"]).dt.normalize() \
            .sort_values().reset_index(drop=True)
    return _TRADE_CAL


def _scale_publish_date(period_end: str) -> str:
    """定期报告披露截止 = 期末后第15个交易日(≈15个工作日, 含节假日顺延)。
    取不到交易日历时退回 +25 自然日兜底(不影响正确性方向: 只会略偏晚)。"""
    d = pd.Timestamp(period_end)
    try:
        after = _trade_calendar()
        after = after[after > d]
        if len(after) >= 15:
            return after.iloc[14].strftime("%Y-%m-%d")
    except Exception:
        pass
    return (d + pd.Timedelta(days=25)).strftime("%Y-%m-%d")


def load_fund_scale_hist(code: str) -> pd.DataFrame:
    """基金季度规模历史(升序),列: quarter_end, aum(亿元), publish_date。
    库里没有则返回空 DataFrame(不触发网络,调用方决定是否 fetch)。"""
    conn = _conn()
    df = pd.read_sql_query(
        "SELECT quarter_end, aum, publish_date FROM fund_scale_hist "
        "WHERE code=? ORDER BY quarter_end", conn, params=(code,))
    conn.close()
    return df


def fetch_fund_scale_hist(code: str, force_refresh: bool = False) -> pd.DataFrame:
    """基金季度规模历史(cache-first)。库里有且未过期(<SCALE_TTL)直接用,
    否则从东财 F10「规模变动」接口(type=gmbd)抓全历史、整段 upsert 进
    fund_scale_hist。用 gmbd 而非 pingzhongdata 的 Data_fluctuationScale:
    后者只给最近约5个季度, 回测到2022-2024年的老基金会整段查不到规模。
    返回列: quarter_end, aum(亿元), publish_date(升序)。"""
    conn = _conn()
    if not force_refresh:
        meta = conn.execute(
            "SELECT MAX(saved_at) AS s FROM fund_scale_hist WHERE code=?",
            (code,)).fetchone()
        if meta and meta["s"] and (time.time() - meta["s"]) < SCALE_TTL:
            conn.close()
            return load_fund_scale_hist(code)
    conn.close()

    rows = []
    answered = False   # 请求确实拿到了可解析的 content 字段(区分「没有规模表」
                       # 和「请求失败/被限流」——见下面 fund_scale_miss 处说明)
    try:
        r = requests.get(
            _EM_F10_URL,
            params={"type": "gmbd", "code": code, "page": 1, "rt": "0.1"},
            headers={"Referer": f"https://fundf10.eastmoney.com/gmbd_{code}.html",
                     "User-Agent": "Mozilla/5.0"},
            timeout=20)
        # content 是双引号包起来的表格 HTML(内部只用单引号,无转义双引号),
        # 后面还跟别的字段, 所以非贪婪匹配到第一个闭合双引号即为整张表。
        m = re.search(r'content:"(.*?)"', r.text, re.DOTALL)
        html = m.group(1) if m else ""
        answered = r.status_code == 200 and m is not None
        # 表列: 日期 | 期间申购 | 期间赎回 | 期末总份额 | 期末净资产(亿元) | 净资产变动率
        for tr in re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL):
            tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
            if len(tds) < 5:
                continue
            qe = tds[0].strip()
            if not re.match(r"\d{4}-\d{2}-\d{2}", qe):
                continue
            try:
                aum = float(tds[4].strip())
            except ValueError:
                continue  # "---" 等占位(尚未披露该期净资产)
            rows.append((code, qe, aum, _scale_publish_date(qe), time.time()))
        time.sleep(0.15)  # 只在真·网络取数(缓存未命中)时限速, 防批量抓被封
    except Exception as e:
        logger.debug("scale fetch failed for %s: %s", code, e)

    conn = _conn()
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO fund_scale_hist "
            "(code, quarter_end, aum, publish_date, saved_at) VALUES (?, ?, ?, ?, ?)",
            rows)
        conn.execute("DELETE FROM fund_scale_miss WHERE code=?", (code,))
    elif answered:
        # 请求成功、页面里确实没有规模变动表(场内份额/刚成立/已清盘): 记一笔,
        # 让 refresh_scale_hist 的 recheck 窗口能跳过它(否则这只基金在库里毫无
        # 痕迹, 每次跑批都要重抓)。
        #
        # 只在 answered 时记。2026-08-04 那次全榜单扩量跑批(d649f8b)就是栽在
        # 这里: 上万次 0.15s 间隔的请求触发东财限流, 返回体里没有 content 字段,
        # 当时的代码把这 5524 只一律记成「没有规模表」, 20 天 recheck 窗口又让
        # 它们一直不被重抓——基金列表的规模门槛因此对其中大半基金形同虚设,
        # A/C 合并口径下这些兄弟份额还会被当成 0 规模。抽检 20 只重抓, 20 只
        # 全都有数据。所以失败一律不留痕, 宁可下次重抓。
        conn.execute("INSERT OR REPLACE INTO fund_scale_miss (code, tried_at) "
                     "VALUES (?, ?)", (code, time.time()))
    conn.commit()
    conn.close()
    return load_fund_scale_hist(code)


# ── A/C 份额合并 ────────────────────────────────────────────────────────────
# 东财 gmbd 接口按**基金代码**给期末净资产, 而 A 类/C 类是两个独立代码, 拿到
# 的是各自份额类别的净资产, 不是整只基金的。但 A/C 只是同一份基金合同下的
# 不同收费方式(A 前端申购费、C 销售服务费), 共用一个投资组合、一个托管账户:
#   · 「规模太小、净值容易被单笔申赎搅动」这个风险来自整个组合, 不分份额类别;
#   · 清盘线(《运作管理办法》第41条: 连续60个工作日基金资产净值低于5000万)
#     和发起式基金满3年不足2亿自动终止, 条文主体都是「基金」, 按合并口径算。
# 所以规模门槛要用合并规模。份额类别之间没有公开的关联字段, 只能按名称配对:
# 去掉名称末尾的类别字母即为同一只基金(已核对榜单 20072 只, 分组后 0 组出现
# 同名不同 type 的撞车, 且回测选中过的 9 只 C 类都能正确配到 A 类)。
_SHARE_SUFFIX_RE = re.compile(r"[ABCDEFHIOQRYZ]$")
_SIBLING_MAP = None   # {code: (同一只基金的全部份额代码, 含自身)}, 进程内缓存


def _build_sibling_map() -> dict:
    """从榜单缓存建「代码 → 同基金全部份额代码」映射。读不到榜单时返回空 dict
    (调用方退化成单份额口径, 不报错)。"""
    try:
        conn = _conn()
        row = conn.execute("SELECT data FROM fund_list WHERE id = 1").fetchone()
        conn.close()
        if not row:
            return {}
        df = pd.DataFrame(json.loads(row["data"]))
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["base"] = df["name"].astype(str).map(lambda n: _SHARE_SUFFIX_RE.sub("", n))
        out = {}
        for _, sub in df.groupby("base"):
            fam = tuple(sorted(sub["code"]))
            for c in fam:
                out[c] = fam
        return out
    except Exception as e:
        logger.warning("sibling map build failed: %s", e)
        return {}


def share_class_siblings(code: str) -> tuple:
    """同一只基金的全部份额类别代码(含自身)。配不上时只返回自身。"""
    global _SIBLING_MAP
    if _SIBLING_MAP is None:
        _SIBLING_MAP = _build_sibling_map()
    return _SIBLING_MAP.get(str(code).zfill(6), (str(code).zfill(6),))


def _aum_asof_one(code: str, date, fetch_if_missing: bool = True) -> Optional[float]:
    """单个代码(单一份额类别)在 date 当时可见的最新一期期末净资产(亿元)。"""
    df = load_fund_scale_hist(code)
    if df.empty and fetch_if_missing:
        df = fetch_fund_scale_hist(code)
    if df.empty:
        return None
    dstr = pd.Timestamp(date).strftime("%Y-%m-%d")
    avail = df[df["publish_date"] <= dstr]
    return float(avail["aum"].iloc[-1]) if not avail.empty else None


def fund_aum_asof(code: str, date, fetch_if_missing: bool = True,
                  merge_classes: bool = True) -> Optional[float]:
    """信号日 date 当时能看到的最新一期基金规模(亿元);无可用数据返回 None。
    只用 publish_date <= date 的季度,避免用到尚未披露的规模(未来函数)。

    merge_classes=True(默认, 2026-08-06 起): 把同一只基金的各份额类别(A/C/E…)
    的期末净资产**加总**, 见上面 A/C 份额合并那段说明。各类别取「各自当时可见
    的最新一期」再相加(C 类往往比 A 类晚成立、季报期数不齐, 按期数对齐会白丢
    数据)。查得到规模的类别才计入, 整只基金一个类别都查不到才返回 None——所以
    兄弟份额缺数据时是**偏小**(保守), 不会凭空放大。"""
    codes = share_class_siblings(code) if merge_classes else (code,)
    total = None
    for c in codes:
        # 兄弟份额只读库不触网: 它们本来就在 scale_universe(榜单全部非债)的
        # 每日刷新范围内, 为凑一个门槛值去逐只补抓会把回测拖成小时级。
        v = _aum_asof_one(c, date, fetch_if_missing=(c == code and fetch_if_missing))
        if v is not None:
            total = v if total is None else total + v
    return total


def funds_aum_asof(codes: list, date, merge_classes: bool = True) -> dict:
    """批量版 fund_aum_asof: 一次 SQL 取多只基金在 date 当时可见的最新一期
    规模(亿元)。返回 {code: aum 或 None}(codes 里查不到的也给 None)。纯读
    库、不触发网络, 供 app 列表按需给整页基金标规模。

    merge_classes 同 fund_aum_asof, 默认按 A/C 合并口径加总, 保证列表上显示的
    规模跟回测门槛用的是同一个数。"""
    if not codes:
        return {}
    dstr = pd.Timestamp(date).strftime("%Y-%m-%d")
    conn = _conn()
    # 每只取 publish_date<=date 里 quarter_end 最大的那期(= 当时最新已披露)。
    # 先 GROUP BY 求每只的最新季末, 再 join 回取该期 aum(比逐行相关子查询快)。
    rows = conn.execute(
        "SELECT f.code, f.aum FROM fund_scale_hist f "
        "JOIN (SELECT code, MAX(quarter_end) AS mq FROM fund_scale_hist "
        "      WHERE publish_date <= ? GROUP BY code) t "
        "  ON f.code = t.code AND f.quarter_end = t.mq", (dstr,)).fetchall()
    conn.close()
    got = {r["code"]: r["aum"] for r in rows}
    if not merge_classes:
        return {c: got.get(c) for c in codes}
    out = {}
    for c in codes:
        vals = [got[s] for s in share_class_siblings(c) if got.get(s) is not None]
        out[c] = sum(vals) if vals else None
    return out


def latest_published_quarter(today=None) -> Optional[str]:
    """截至 today, 已披露(publish_date<=today)的最新一期季末(ISO)。规模季报
    约季末后1个月披露, 所以这个值大约在 1月底/4月底/7月底/10月底 各跳一档。"""
    today = today or datetime.now(_CST).date()
    tstr = today.isoformat()
    cands = [f"{y}-{md}"
             for y in (today.year, today.year - 1, today.year - 2)
             for md in ("03-31", "06-30", "09-30", "12-31")
             if _scale_publish_date(f"{y}-{md}") <= tstr]
    return max(cands) if cands else None


def scale_universe() -> list:
    """需要维护规模历史的基金:榜单里全部非债基金。

    以前只覆盖「本地有净值历史的基金」(list_nav_codes(),约 5.7k 只),
    但基金列表的规模门槛是对整个榜单(非债约 13.5k 只)生效的,没抓到规模
    的一律按「未知」放行——门槛因此对大半基金形同虚设,快照模式尤其明显
    (2022 那会儿有规模的只有 2.5k 只)。所以范围按榜单来。
    榜单缓存读不到时退回本地净值那批(至少不比以前差)。
    """
    try:
        conn = _conn()
        row = conn.execute("SELECT data FROM fund_list WHERE id = 1").fetchone()
        conn.close()
        if row:
            df = pd.DataFrame(json.loads(row["data"]))
            if "type" in df.columns:
                df = df[~df["type"].map(is_bond)]
            codes = {str(c).zfill(6) for c in df["code"].dropna()}
            if codes:
                return sorted(codes | list_nav_codes())
    except Exception as e:
        logger.warning("scale_universe fell back to NAV codes: %s", e)
    return sorted(list_nav_codes())


def refresh_scale_hist(codes: Optional[list] = None,
                       progress: Optional[Callable] = None) -> int:
    """批量刷新基金季度规模。规模是季度数据(一年4次, 各季末后约1个月披露),
    所以只在"某只基金还缺最新一期已披露季报"时才真抓——效果是每年约4次
    (1月底/4月底/7月底/10月底 新季报出来后)各刷一轮, 其余日子几乎全跳过、
    零网络。避免了按固定TTL会导致的"全体同日过期→每周2小时尖峰"。

    额外用 recheck 窗口(20天)兜底极少数"永远缺最新季报"的基金(如刚成立还
    没出过季报的), 免得每天都去重抓它们。返回实际发生网络抓取的只数。由
    update_daily.py 每日调用(不放进 run_pipeline, 免得拖慢 in-app 更新按钮)。"""
    if codes is None:
        codes = sorted(scale_universe())
    # 自愈:披露滞后口径变过(如 _SCALE_PUB_LAG 调整)时, 库里存量行的
    # publish_date 会过时。按 quarter_end 批量重算对齐(几十条, 秒级), 免得
    # 靠重抓才更新——否则 have_latest 已命中的基金不会再抓、旧披露日就一直
    # 留着, app/回测的 as-of 口径也就跟着错。
    conn = _conn()
    for (qe,) in conn.execute("SELECT DISTINCT quarter_end FROM fund_scale_hist").fetchall():
        conn.execute("UPDATE fund_scale_hist SET publish_date=? "
                     "WHERE quarter_end=? AND publish_date<>?",
                     (_scale_publish_date(qe), qe, _scale_publish_date(qe)))
    conn.commit()
    due = latest_published_quarter()
    have_latest = set()
    if due:
        have_latest = {r["code"] for r in conn.execute(
            "SELECT code FROM fund_scale_hist GROUP BY code "
            "HAVING MAX(quarter_end) >= ?", (due,))}
    _cut = time.time() - 20 * 24 * 3600
    recent = {r["code"] for r in conn.execute(
        "SELECT code FROM fund_scale_hist GROUP BY code "
        "HAVING MAX(saved_at) > ?", (_cut,))}
    # 抓过但一行都没有的(F10 无规模变动表)同样进 recheck 窗口,不然每天重抓。
    recent |= {r["code"] for r in conn.execute(
        "SELECT code FROM fund_scale_miss WHERE tried_at > ?", (_cut,))}
    conn.close()
    fetched = 0
    total = len(codes)
    for i, code in enumerate(codes):
        # 已有最新已披露季报 → 跳过; 没有但20天内刚抓过(还是没有)→ 也先跳过
        if code not in have_latest and code not in recent:
            fetch_fund_scale_hist(code, force_refresh=True)  # upsert + 0.15s 限速
            fetched += 1
        if progress:
            progress("刷新规模", i + 1, total)
    return fetched


def fetch_nav(code: str) -> Optional[pd.DataFrame]:
    """Return NAV history since NAV_START for a single fund (cache-first).

    Serves stored rows while fresh (< NAV_TTL); otherwise downloads the whole
    history in one pingzhongdata request and stores it row-per-day.
    Columns: date, nav, daily_ret_pct, acc_nav
    """
    conn = _conn()
    meta = conn.execute(
        "SELECT saved_at FROM fund_nav_meta WHERE code = ?", (code,)).fetchone()
    if meta and (time.time() - meta["saved_at"]) < NAV_TTL:
        df = _load_nav_df(code, conn)
        conn.close()
        return df if len(df) >= 20 else None
    conn.close()

    df = _fetch_nav_full(code)
    if df is None or len(df) < 20:
        return None
    conn = _conn()
    _write_nav_rows(conn, code, df)
    conn.commit()
    conn.close()
    return df


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def _save_metrics_row(code: str, ann_return: float, volatility: float,
                 n: int, mdd: dict, rets: dict):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO fund_sharpe "
        "(code, ann_return, volatility, data_points, "
        " mdd_1m, mdd_3m, mdd_6m, mdd_1y, "
        " ret_1m, ret_3m, ret_6m, ret_1y, saved_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (code, ann_return, volatility, n,
         mdd.get("mdd_1m"), mdd.get("mdd_3m"), mdd.get("mdd_6m"), mdd.get("mdd_1y"),
         rets.get("ret_1m"), rets.get("ret_3m"), rets.get("ret_6m"), rets.get("ret_1y"),
         time.time()),
    )
    conn.commit()
    conn.close()


def _metrics_from_nav(nav_df: pd.DataFrame,
                      cols: Optional[set] = None) -> Optional[dict]:
    """Compute annualized return / volatility + per-period max-drawdown and
    return from an already-fetched NAV DataFrame. Pure (no I/O).

    `cols` 给出时只算这些区间列(如 {"ret_1y","mdd_1y"}),并跳过全期年化——
    快照筛选只用所选区间,砍掉无关窗口能省近半耗时;None 保持全量
    (recompute_all 落库用)。"""
    df = nav_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    # daily_ret_pct is percentage; convert to decimal
    if "acc_nav" in df.columns:
        df["acc_nav"] = pd.to_numeric(df["acc_nav"], errors="coerce").fillna(df["nav"])
    else:
        df["acc_nav"] = df["nav"]
    df = df.sort_values("date").reset_index(drop=True)
    # 官方日增长率经 effective_daily_ret 校正(建仓期假 0 → 净值环比)。
    df["r"] = effective_daily_ret(df)

    returns = df["r"].dropna()
    if len(returns) < 20:
        return None

    # Per-period max drawdown over trailing calendar-day windows, date-aligned
    # with EastMoney's 近X 收益率. A fund younger than a window gets None for it
    # (no anchor) rather than a misleadingly short reading.
    # Drawdown runs on accumulated NAV so dividends aren't mistaken for drops.
    mdd = {key: _period_mdd(df, days)
           for key, days in DRAWDOWN_DAYS.items()
           if cols is None or key in cols}
    rets = {key: _period_return(df, days)
            for key, days in RETURN_DAYS.items()
            if cols is None or key in cols}

    if cols is not None:
        return {**mdd, **rets}

    n = len(returns)
    span = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    res = _annualized(returns, span)
    if res is None:
        return None
    ann_return, ann_vol = res

    return {
        "ann_return": ann_return, "volatility": ann_vol,
        "data_points": n, **mdd, **rets,
    }


def _save_metrics(code: str, m: dict):
    _save_metrics_row(
        code, m["ann_return"], m["volatility"], m["data_points"],
        {k: m[k] for k in ("mdd_1m", "mdd_3m", "mdd_6m", "mdd_1y")},
        {k: m[k] for k in ("ret_1m", "ret_3m", "ret_6m", "ret_1y")},
    )


def compute_metrics_for_fund(code: str) -> Optional[dict]:
    """Fetch NAV (network/cache), compute metrics, persist, and return them."""
    nav_df = fetch_nav(code)
    if nav_df is None:
        return None
    m = _metrics_from_nav(nav_df)
    if m is not None:
        _save_metrics(code, m)
    return m


# ── Persistent filter-result cache ───────────────────────────────────────────

# 全部筛选结果加起来最多占这么多字节(JSON 原文,约等于建库体积)。
FILTER_CACHE_BYTES = 64 * 1024 * 1024


def save_filter_result(key: str, meta: dict, df: pd.DataFrame):
    """Store one filter run's rows + its params/total under `key`."""
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO filter_results (key, params, data, saved_at) "
        "VALUES (?, ?, ?, ?)",
        (key, json.dumps(meta, ensure_ascii=False),
         df.to_json(orient="split", force_ascii=False), time.time()))
    # Keep the table bounded. 一条结果现在存的是全部匹配行(动辄近万行、
    # 几百 KB),只按条数封顶会把库撑大(整库还要 gzip 传到 Release),
    # 所以再加一道总字节上限:按时间倒序累计,超过 FILTER_CACHE_BYTES 的
    # 老结果丢掉。最新一条(rn = 1)无论多大都保留,否则刚存的会被自己删掉。
    conn.execute(
        "DELETE FROM filter_results WHERE key NOT IN ("
        "  SELECT key FROM ("
        "    SELECT key,"
        "      ROW_NUMBER() OVER (ORDER BY saved_at DESC) AS rn,"
        "      SUM(LENGTH(data)) OVER (ORDER BY saved_at DESC"
        "        ROWS UNBOUNDED PRECEDING) AS cum"
        "    FROM filter_results"
        "  ) WHERE rn = 1 OR (rn <= 200 AND cum <= ?)"
        ")", (FILTER_CACHE_BYTES,))
    conn.commit()
    conn.close()


def load_filter_result(key: str):
    """(df, meta, saved_at) for a stored filter run, or None.

    dtype inference is disabled on read so fund codes keep leading zeros.
    """
    conn = _conn()
    row = conn.execute(
        "SELECT params, data, saved_at FROM filter_results WHERE key = ?",
        (key,)).fetchone()
    conn.close()
    if not row:
        return None
    df = pd.read_json(io.StringIO(row["data"]), orient="split",
                      dtype=False, convert_dates=False)
    return df, json.loads(row["params"]), row["saved_at"]


# ── Quarterly top holdings ───────────────────────────────────────────────────
# EastMoney F10 discloses each fund's top-10 stock/bond holdings per quarter.
# Fetched one year at a time (the API's granularity) and cached per (code, year).

_HOLDINGS_COLS = ["quarter", "kind", "代码", "名称", "占净值比例", "持股数", "持仓市值"]

_EM_F10_URL = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"


def _em_f10_holdings_raw(code: str, year: str, typ: str) -> Optional[pd.DataFrame]:
    """One year's raw holdings tables from EastMoney F10, with a 季度 label
    column. `typ`: "jjcc" = 股票, "zqcc" = 债券.

    Replaces ak.fund_portfolio_hold_em / fund_portfolio_bond_hold_em, whose
    requests carry no Referer — the endpoint started answering those with a
    404 page (which akshare then fails to parse as JSON).
    Returns None on failure, an empty DataFrame when the year disclosed
    nothing.
    """
    from bs4 import BeautifulSoup
    from akshare.utils import demjson
    try:
        r = requests.get(
            _EM_F10_URL,
            params={"type": typ, "code": code, "topline": "10000",
                    "year": year, "month": "", "rt": "0.9"},
            headers={"Referer": f"https://fundf10.eastmoney.com/ccmx_{code}.html"},
            timeout=20)
        data = demjson.decode(r.text[r.text.find("{"):-1])
        html = data["content"]
        soup = BeautifulSoup(html, "lxml")
        labels = [h.text.split("\xa0\xa0")[1]
                  for h in soup.find_all("h4", attrs={"class": "t"})]
        if not labels:
            return pd.DataFrame()
        tables = pd.read_html(
            io.StringIO(html), converters={"股票代码": str, "债券代码": str})
    except Exception as e:
        logger.debug("F10 holdings fetch failed %s %s %s: %s", code, year, typ, e)
        return None
    frames = []
    for lbl, t in zip(labels, tables):
        t = t.copy()
        # Header cells wrap, e.g. "占净值 比例" / "持股数 （万股）" — normalize.
        t.columns = [str(c).replace(" ", "") for c in t.columns]
        t = t.rename(columns={"持股数（万股）": "持股数",
                              "持仓市值（万元）": "持仓市值",
                              "持仓市值（万元人民币）": "持仓市值"})
        if "占净值比例" in t.columns:
            t["占净值比例"] = t["占净值比例"].astype(str).str.rstrip("%")
        t["季度"] = lbl
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _fetch_holdings_year(code: str, year: str) -> Optional[pd.DataFrame]:
    """One year's quarterly top holdings (stocks + bonds), normalized.

    Returns an empty DataFrame when the fund disclosed nothing that year, or
    None when both requests failed (network error — caller keeps stale cache).
    """
    frames, failures = [], 0
    for kind, typ, code_col, name_col in (
        ("股票", "jjcc", "股票代码", "股票名称"),
        ("债券", "zqcc", "债券代码", "债券名称"),
    ):
        raw = _em_f10_holdings_raw(code, year, typ)
        if raw is None:
            failures += 1
            continue
        if raw.empty:
            continue
        # 季度 looks like "2025年1季度股票投资明细" → "2025Q1"
        q = raw["季度"].astype(str).str.extract(r"(\d{4})年(\d)季度")
        df = pd.DataFrame({
            "quarter": q[0] + "Q" + q[1],
            "kind": kind,
            "代码": raw[code_col].astype(str),
            "名称": raw[name_col].astype(str),
            "占净值比例": pd.to_numeric(raw["占净值比例"], errors="coerce"),
            "持股数": pd.to_numeric(raw["持股数"], errors="coerce")
                if "持股数" in raw.columns else np.nan,
            "持仓市值": pd.to_numeric(raw["持仓市值"], errors="coerce"),
        })
        frames.append(df.dropna(subset=["quarter"]))
    if failures == 2:
        return None
    if not frames:
        return pd.DataFrame(columns=_HOLDINGS_COLS)
    return pd.concat(frames, ignore_index=True)


# ── ETF联接基金重仓穿透 ──────────────────────────────────────────────────────
# 联接基金 90%+ 仓位是目标 ETF 本身,季报直接持股占净值不到 1%,展示无意义。
# 解析出目标场内 ETF 后改拉它的重仓。目标 ETF 代码没有直接接口可查:
# 名称粗排(全量代码表里的场内 ETF 简称)+ 跟踪指数代码精确验证。

_EM_BASIC_INFO_URL = ("https://fundmobapi.eastmoney.com/FundMNewApi/"
                      "FundMNBasicInformation")
_EM_FUNDCODE_JS = "https://fund.eastmoney.com/js/fundcode_search.js"
_ETF_MAP_FAIL_TTL = 30 * 86400   # 解析失败一个月后重试

_fund_names: Optional[dict] = None
_etf_cands: Optional[list] = None


def _fund_name(code: str) -> Optional[str]:
    """基金名称(来自缓存的基金列表),进程内建一次字典。"""
    global _fund_names
    if _fund_names is None:
        try:
            df = fetch_fund_list()
            _fund_names = dict(zip(df["code"], df["name"]))
        except Exception as e:
            logger.debug("fund name lookup failed: %s", e)
            return None
    return _fund_names.get(code)


def _fund_index_code(code: str, conn) -> Optional[str]:
    """基金跟踪指数代码,DB 缓存;网络失败返回 None 且不落缓存。"""
    row = conn.execute(
        "SELECT index_code FROM fund_index_code WHERE code=?", (code,)).fetchone()
    if row:
        return row["index_code"] or None
    try:
        r = requests.get(
            _EM_BASIC_INFO_URL,
            params={"FCODE": code, "deviceid": "Wap", "plat": "Wap",
                    "product": "EFund", "version": "2.0.0"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        idx = (r.json().get("Datas") or {}).get("INDEXCODE") or ""
    except Exception as e:
        logger.debug("index code fetch failed %s: %s", code, e)
        return None
    conn.execute("INSERT OR REPLACE INTO fund_index_code VALUES (?, ?, ?)",
                 (code, idx, time.time()))
    conn.commit()
    return idx or None


def _etf_candidates() -> list:
    """场内指数 ETF 候选 [(code, name)],来自全量基金代码表,进程内缓存。"""
    global _etf_cands
    if _etf_cands is None:
        try:
            r = requests.get(_EM_FUNDCODE_JS,
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            data = json.loads(r.text[r.text.find("["):r.text.rfind("]") + 1])
            _etf_cands = [(d[0], d[2]) for d in data
                          if "ETF" in d[2] and "联接" not in d[2]
                          and "指数型" in (d[3] or "")]
        except Exception as e:
            logger.debug("fundcode list fetch failed: %s", e)
            return []
    return _etf_cands


def _lcs_len(a: str, b: str) -> int:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).find_longest_match(
        0, len(a), 0, len(b)).size


def resolve_target_etf(code: str, name: Optional[str] = None
                       ) -> Optional[Tuple[str, str]]:
    """ETF联接基金 → (目标ETF代码, 名称);非联接基金或解析失败返回 None。

    成功映射永久缓存;确认无匹配按 _ETF_MAP_FAIL_TTL 重试;
    网络故障不落缓存,下次再试。
    """
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT target_code, target_name, saved_at FROM etf_target_map "
            "WHERE code=?", (code,)).fetchone()
        if row:
            if row["target_code"]:
                return row["target_code"], row["target_name"]
            if time.time() - row["saved_at"] < _ETF_MAP_FAIL_TTL:
                return None
        if name is None:
            name = _fund_name(code)
        if not name or "ETF" not in name or "联接" not in name:
            return None
        feeder_idx = _fund_index_code(code, conn)
        if not feeder_idx:
            return None                        # 网络失败,不缓存,下次重试
        # 名称粗排,两梯队依次做指数验证:
        # ① 同管理人(场内简称尾部的管理人简称出现在联接名里,如
        #    "创业板ETF易方达")按相似度排——核心词被缩写时(纳斯达克100
        #    →纳指)纯名称匹配会错配到其他管理人的同指数 ETF,先验自家;
        # ② 纯名称相似(最长公共子串)——覆盖场内简称不带管理人的情形
        #    (如"上证50ETF")。错配指数的候选会被 INDEXCODE 验证拒绝。
        base = name.split("联接")[0]           # "易方达创业板ETF"
        from collections import Counter
        base_chars = Counter(base)
        def _sim(cand_name: str) -> int:
            common = sum((Counter(cand_name) & base_chars).values())
            return _lcs_len(base, cand_name) * 2 + common
        cands = _etf_candidates()
        mgr = [c for c in cands
               if (t := c[1].rsplit("ETF", 1)[-1]) and t in base]
        mgr.sort(key=lambda c: _sim(c[1]), reverse=True)
        by_lcs = sorted(cands, key=lambda c: _lcs_len(base, c[1]),
                        reverse=True)
        seen, ranked = set(), []
        for c in mgr[:8] + by_lcs[:8]:
            if c[0] not in seen:
                seen.add(c[0])
                ranked.append(c)
        target = None
        for c_code, c_name in ranked:
            if _fund_index_code(c_code, conn) == feeder_idx:
                target = (c_code, c_name)
                break
        conn.execute(
            "INSERT OR REPLACE INTO etf_target_map VALUES (?, ?, ?, ?)",
            (code, target[0] if target else "",
             target[1] if target else "", time.time()))
        conn.commit()
        return target
    finally:
        conn.close()


def fetch_holdings(code: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Quarterly top holdings from HOLDINGS_START_Q (2020Q4) to the latest
    disclosed quarter, cache-first.

    ETF联接基金自动穿透:改拉目标场内 ETF 的重仓(联接自身季报的直接持股
    占净值不足 1%,无参考价值)。来源标注可用 resolve_target_etf 查询。

    Columns: quarter ("2025Q1"), kind (股票/债券), 代码, 名称, 占净值比例(%),
    持股数(万股, stocks only), 持仓市值(万元). Sorted newest quarter first,
    biggest position first within a quarter. Returns None only when nothing
    could be fetched and no cache exists.
    """
    target = resolve_target_etf(code)
    if target:
        return fetch_holdings(target[0], force_refresh)
    years = [str(y) for y in range(HOLDINGS_START_YEAR, datetime.now().year + 1)]
    frames, any_data = [], False
    conn = _conn()
    for year in years:
        row = conn.execute(
            "SELECT data, saved_at FROM fund_holdings WHERE code=? AND year=?",
            (code, year)).fetchone()
        # Rows written by an older version stored the raw akshare frame
        # (股票代码/季度 columns); treat those as a cache miss so they are
        # refetched and rewritten in the normalized format.
        cached = None
        if row:
            recs = json.loads(row["data"])
            if not recs or "quarter" in recs[0]:
                cached = pd.DataFrame(recs, columns=_HOLDINGS_COLS)
        ttl = HOLDINGS_TTL if year == str(datetime.now().year) else HOLDINGS_TTL_PAST
        if cached is not None and not force_refresh \
                and (time.time() - row["saved_at"]) < ttl:
            df = cached
        else:
            df = _fetch_holdings_year(code, year)
            if df is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO fund_holdings "
                    "(code, year, data, saved_at) VALUES (?, ?, ?, ?)",
                    (code, year, df.to_json(orient="records", force_ascii=False),
                     time.time()))
                conn.commit()
            elif cached is not None:   # fetch failed → serve stale cache
                df = cached
        if df is not None:
            any_data = True
            if not df.empty:
                frames.append(df)
    conn.close()
    if not any_data:
        return None
    if not frames:
        return pd.DataFrame(columns=_HOLDINGS_COLS)
    out = pd.concat(frames, ignore_index=True)
    # The start year is fetched whole (the API's granularity is a year); only
    # quarters from HOLDINGS_START_Q onward are surfaced.
    out = out[out["quarter"] >= HOLDINGS_START_Q]
    return out.sort_values(
        ["quarter", "kind", "占净值比例"], ascending=[False, True, False]
    ).reset_index(drop=True)


# ── Index daily history (上证指数) ────────────────────────────────────────────

# 指数/QVIX 历史起点,独立于基金净值的 NAV_START(2020):图表要看近10年,
# 取 2015 起同时覆盖 QVIX 全史(2015-02 起)和 2015 股灾样本。
INDEX_START = "2015-01-01"


def _fetch_with_timeout(fn, timeout=8):
    """akshare 这两个指数源(pd.read_csv 直连 http)不带超时参数,源站一旦
    卡住会挂起几分钟不报错,把整个 Streamlit rerun 一起拖死。用后台线程
    强制掐表,超时立刻放弃转走"陈旧缓存兜底"这条已有路径。

    用 threading.Thread(daemon=True) 而非 ThreadPoolExecutor:后者的 worker
    是非 daemon 线程,会被 concurrent.futures 注册的 atexit 钩子 join,一次性
    脚本(notify_qvix.py)进程退出时会被晾着的慢线程拖到它读完为止——实测
    把 8 秒超时拖成了 2 分钟。daemon 线程不受这个 atexit 影响,解释器可以
    直接退出、不等它。"""
    result: list = []
    error: list = []

    def _run():
        try:
            result.append(fn())
        except Exception as e:  # noqa: BLE001 — 转交给等待方
            error.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"fetch exceeded {timeout}s")
    if error:
        raise error[0]
    return result[0]


def fetch_sse_daily(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """上证指数 daily history from NAV_START, cache-first.

    No time-based expiry — the cache is only refreshed when
    force_refresh=True, which update_daily.py's 06:00 跑批 does every
    trading day. Columns: date (ISO str), close, pct (daily % change).
    Serves stale cache when the refresh fails; None only with no cache at all.
    """
    conn = _conn()
    row = conn.execute(
        "SELECT data, saved_at FROM index_daily_cache WHERE key='sse'"
    ).fetchone()

    def _from_row(r):
        return pd.read_json(io.StringIO(r["data"]), orient="split",
                            dtype=False, convert_dates=False)

    if row and not force_refresh:
        conn.close()
        return _from_row(row)

    df = None
    try:
        # Sina source (stock_zh_index_daily): the EastMoney push2 host is
        # blocked by some proxies. No 涨跌幅 column — derive from closes
        # over the full history, then cut to NAV_START.
        raw = _fetch_with_timeout(lambda: ak.stock_zh_index_daily(symbol="sh000001"))
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
        }).dropna(subset=["date", "close"])
        df["pct"] = df["close"].pct_change() * 100.0
        df = df[df["date"] >= INDEX_START].reset_index(drop=True)
    except Exception as e:
        logger.debug("SSE index fetch failed: %s", e)

    if df is not None and not df.empty:
        conn.execute(
            "INSERT OR REPLACE INTO index_daily_cache (key, data, saved_at) "
            "VALUES ('sse', ?, ?)",
            (df.to_json(orient="split", force_ascii=False), time.time()))
        conn.commit()
    elif row:   # refresh failed → stale cache beats nothing
        df = _from_row(row)
    conn.close()
    return df


def fetch_qvix_daily(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """VIX恐慌指数（50ETF期权QVIX，中国版VIX）daily close, cache-first.

    No time-based expiry — same contract as fetch_sse_daily: only refreshed
    when force_refresh=True (the daily 跑批). Columns: date (ISO str), close.
    CBOE VIX has no akshare source here, so the A股 analog (optbbs QVIX) is
    used.
    """
    conn = _conn()
    row = conn.execute(
        "SELECT data, saved_at FROM index_daily_cache WHERE key='qvix'"
    ).fetchone()

    def _from_row(r):
        return pd.read_json(io.StringIO(r["data"]), orient="split",
                            dtype=False, convert_dates=False)

    if row and not force_refresh:
        conn.close()
        return _from_row(row)

    df = None
    try:
        raw = _fetch_with_timeout(ak.index_option_50etf_qvix)
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
        }).dropna(subset=["date", "close"])
        df = df[df["date"] >= INDEX_START].reset_index(drop=True)
    except Exception as e:
        logger.debug("QVIX fetch failed: %s", e)

    if df is not None and not df.empty:
        conn.execute(
            "INSERT OR REPLACE INTO index_daily_cache (key, data, saved_at) "
            "VALUES ('qvix', ?, ?)",
            (df.to_json(orient="split", force_ascii=False), time.time()))
        conn.commit()
    elif row:   # refresh failed → stale cache beats nothing
        df = _from_row(row)
    conn.close()
    return df


# VHSI(恒指波幅指数,港股版VIX)历史数据有个大坑:新浪源2014-09-24之后
# 整段空白到2018-04-30(仅1个孤立数据点),再空白到2021-03-18才恢复连续
# 每日更新——2015-2020这6年基本没有可用数据,不是偶发缺失,是整段没有。
# 用之前必须先砍掉这段空窗,否则滚动阈值窗口会横跨一大截空数据。HSI
# (恒生指数)本身没有这个问题(2013-08起逐年完整,已核实)。
_VHSI_START = "2021-03-18"


def fetch_hsi_daily(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """恒生指数 daily history, cache-first——跟 fetch_sse_daily 同一套
    契约(index_daily_cache 表, key='hsi', 无过期时间, 只在 force_refresh
    时刷新)。数据源新浪 stock_hk_index_daily_sina(symbol='HSI'),2013-08
    起逐年完整、无缺口(已核实)。"""
    conn = _conn()
    row = conn.execute(
        "SELECT data, saved_at FROM index_daily_cache WHERE key='hsi'"
    ).fetchone()

    def _from_row(r):
        return pd.read_json(io.StringIO(r["data"]), orient="split",
                            dtype=False, convert_dates=False)

    if row and not force_refresh:
        conn.close()
        return _from_row(row)

    df = None
    try:
        raw = _fetch_with_timeout(lambda: ak.stock_hk_index_daily_sina(symbol="HSI"))
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
        }).dropna(subset=["date", "close"])
        df["pct"] = df["close"].pct_change() * 100.0
        df = df[df["date"] >= INDEX_START].reset_index(drop=True)
    except Exception as e:
        logger.debug("HSI index fetch failed: %s", e)

    if df is not None and not df.empty:
        conn.execute(
            "INSERT OR REPLACE INTO index_daily_cache (key, data, saved_at) "
            "VALUES ('hsi', ?, ?)",
            (df.to_json(orient="split", force_ascii=False), time.time()))
        conn.commit()
    elif row:
        df = _from_row(row)
    conn.close()
    return df


def fetch_vhsi_daily(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """VHSI(恒指波幅指数,港股版VIX)daily close, cache-first——同上契约,
    key='vhsi'。数据源新浪 stock_hk_index_daily_sina(symbol='VHSI'),
    只保留 _VHSI_START(2021-03-18)起的数据(见该常量说明,更早的整段
    缺失砍掉)。"""
    conn = _conn()
    row = conn.execute(
        "SELECT data, saved_at FROM index_daily_cache WHERE key='vhsi'"
    ).fetchone()

    def _from_row(r):
        return pd.read_json(io.StringIO(r["data"]), orient="split",
                            dtype=False, convert_dates=False)

    if row and not force_refresh:
        conn.close()
        return _from_row(row)

    df = None
    try:
        raw = _fetch_with_timeout(lambda: ak.stock_hk_index_daily_sina(symbol="VHSI"))
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
        }).dropna(subset=["date", "close"])
        df = df[df["date"] >= _VHSI_START].reset_index(drop=True)
    except Exception as e:
        logger.debug("VHSI fetch failed: %s", e)

    if df is not None and not df.empty:
        conn.execute(
            "INSERT OR REPLACE INTO index_daily_cache (key, data, saved_at) "
            "VALUES ('vhsi', ?, ?)",
            (df.to_json(orient="split", force_ascii=False), time.time()))
        conn.commit()
    elif row:
        df = _from_row(row)
    conn.close()
    return df


def index_daily_saved_at(key: str) -> Optional[float]:
    """Unix time index_daily_cache[key] ('sse' or 'qvix') was last written,
    or None if never fetched. Used as an st.cache_data cache-busting key so
    the app picks up a fresh 跑批 write immediately rather than on a timer."""
    conn = _conn()
    row = conn.execute(
        "SELECT saved_at FROM index_daily_cache WHERE key = ?", (key,)
    ).fetchone()
    conn.close()
    return row["saved_at"] if row else None




_TRADE_CAL_CACHE = {"at": 0.0, "days": None}


def _trading_days() -> Optional[set]:
    """A股交易日集合,进程内缓存12小时(日历一年只变几次,没必要每次现拉)。
    拉不到返回 None,调用方退化成只按星期几判断。"""
    now = time.time()
    if (_TRADE_CAL_CACHE["days"] is not None
            and now - _TRADE_CAL_CACHE["at"] < 12 * 3600):
        return _TRADE_CAL_CACHE["days"]
    try:
        cal = ak.tool_trade_date_hist_sina()
        days = set(pd.to_datetime(cal["trade_date"]).dt.date)
    except Exception as e:
        logger.debug("交易日历拉取失败,退化成按星期判断: %s", e)
        return _TRADE_CAL_CACHE["days"]
    _TRADE_CAL_CACHE.update(at=now, days=days)
    return days


# 开盘到收盘(北京时间),中午 11:30~13:00 的休市**包含在内**,不切成两段。
# 休市时新浪返回的是 11:30 的静态快照,照常现算得到的就是"上午收盘"那个
# 值——这正是午休时该显示的最新值。要是把午休排除掉,页面会在 11:30~13:00
# 退回去显示**昨天**的收盘值,而上午明明已经交易了两小时,那才是错的。
# 集合竞价 9:15~9:25 不算:那时只有申报、没有连续成交,期权买卖盘还没铺开,
# 现算出来的值噪声大且不可比。
# 盘中QVIX的"取哪个数"分档(北京时间,交易日)。交易所连续竞价是
# 9:30~11:30 和 13:00~15:00,中午休市和收盘后新浪返回的都是休市/收盘那一刻
# 的静态报价,所以那两段不是"没有数",而是"数已经定格了":
#   00:00~09:30  昨天的收盘值(今天还没开始交易)
#   09:30~11:30  实时现算
#   11:30~13:00  定格在 11:30(上午收盘值)
#   13:00~15:00  实时现算
#   15:00~24:00  定格在 15:00(今日收盘值)
# 定格靠给 qvix_calc.compute_qvix 传 as_of 实现:报价本来就是静态的,只要
# 把"剩余到期时间"也钉在同一刻,同一批报价就永远算出同一个数,不会出现
# 下午2点看和晚上10点看不一致。非交易日全天走"昨天的收盘值"。
_OPEN = dt_time(9, 30)
_NOON_CLOSE = dt_time(11, 30)
_NOON_OPEN = dt_time(13, 0)
_CLOSE = dt_time(15, 0)


def is_trading_day(day=None) -> bool:
    """是不是A股交易日。节假日靠交易日历排除;日历拉不到时退化成只看
    星期几——那种情况下节假日会被误判成交易日,但届时新浪返回的是上一
    交易日的静态报价,算出来等于上一收盘值,不会凭空造出一个错的实时值。"""
    day = day or datetime.now(_CST).date()
    days = _trading_days()
    if days is not None:
        return day in days
    return day.weekday() < 5


def qvix_phase(now: Optional[datetime] = None) -> tuple:
    """当下该取哪个数 → (档位, as_of)。

    档位∈{"live","noon","close","prev"};as_of 是要钉住的时刻
    ("live" 为 None=用当下,"prev" 也是 None=不现算、直接读库)。"""
    now = now or datetime.now(_CST)
    if not is_trading_day(now.date()):
        return "prev", None
    t = now.time()
    if t < _OPEN:
        return "prev", None
    if t < _NOON_CLOSE:
        return "live", None
    if t < _NOON_OPEN:
        return "noon", now.replace(hour=11, minute=30, second=0, microsecond=0)
    if t < _CLOSE:
        return "live", None
    return "close", now.replace(hour=15, minute=0, second=0, microsecond=0)




def save_qvix_self_history(rows: list) -> None:
    """写入/覆盖自算QVIX历史(backfill_qvix_history.py 用)。跟 optbbs 的
    index_daily_cache 是两张独立的表,互不覆盖——这样optbbs的值和自算值
    可以并排比对,而不是自算结果把optbbs历史顶替掉。rows 是
    [{"date","qvix","note"}, ...]。"""
    # 用 UPSERT 而不是 INSERT OR REPLACE: 后者是**整行替换**, 会把同一行的
    # threshold 抹成 NULL(它每次跑批都重算, 所以平时看不出问题 —— 但只要
    # 以后再往这张表加别的列, 整行替换就会静默清空它)。这里只动 qvix/note。
    conn = _conn()
    conn.executemany(
        "INSERT INTO qvix_self_history (date, qvix, note) VALUES (?, ?, ?) "
        "ON CONFLICT(date) DO UPDATE SET qvix=excluded.qvix, note=excluded.note",
        [(r["date"], r.get("qvix"), r.get("note")) for r in rows])
    conn.commit()
    conn.close()


def qvix_self_history_last_date() -> Optional[str]:
    """qvix_self_history 里最新一条的日期(字符串),给 app.py 当
    st.cache_data 的缓存key用——新一天数据写进来,这个值就变,缓存自然
    失效,不用等TTL。没有数据时返回 None。"""
    conn = _conn()
    row = conn.execute(
        "SELECT MAX(date) AS d FROM qvix_self_history").fetchone()
    conn.close()
    return row["d"] if row and row["d"] else None


def load_qvix_self_history() -> Optional[pd.DataFrame]:
    """读取自算QVIX历史,按日期排序;没有数据时返回 None。"""
    conn = _conn()
    df = pd.read_sql("SELECT * FROM qvix_self_history ORDER BY date", conn)
    conn.close()
    return df if not df.empty else None


def save_qvix_self_threshold(dates: list, thresholds: list) -> None:
    """把滚动2年90分位恐慌阈值写回 qvix_self_history 的 threshold 列
    (按自算QVIX序列现算的,不是套用optbbs那条历史算出来的阈值)。"""
    conn = _conn()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(qvix_self_history)")]
    if "threshold" not in cols:
        conn.execute("ALTER TABLE qvix_self_history ADD COLUMN threshold REAL")
    conn.executemany(
        "UPDATE qvix_self_history SET threshold = ? WHERE date = ?",
        [(t, d) for d, t in zip(dates, thresholds)])
    conn.commit()
    conn.close()


def _run_summary(trades: list) -> tuple:
    """(笔数, 胜率%, 累计费后复利%) —— 只统计已平仓的笔。

    未平仓那笔的标记在「卖出日」上("YYYY-MM-DD(持仓中)"), 不在「卖出原因」
    里(那里写"未触发"), 把浮盈当已实现会同时污染胜率和累计收益。
    """
    done = [t for t in trades if "持仓中" not in str(t.get("卖出日", ""))]
    rets = [t.get("费后收益") for t in done]
    rets = [float(r) for r in rets if r is not None]
    if not rets:
        return len(done), None, None
    wins = sum(1 for r in rets if r > 0)
    cum = 1.0
    for r in rets:
        cum *= (1 + r / 100)
    return len(rets), wins / len(rets) * 100, (cum - 1) * 100


def describe_run(params: dict) -> str:
    """按参数自动拼一个人看得懂的方案名(没给 --label 时用)。"""
    p = params or {}
    win = "近1月" if p.get("ret_col") == "ret_1m" else "近3月"
    pick = p.get("pick")
    if pick == "regime":
        head = f"大盘当日涨跌择向·{win}排名"
    elif pick == "top":
        head = f"{win}涨幅最大"
    else:
        head = f"{win}跌幅最大"
    bits = [head]
    if p.get("require_drop"):
        bits.append("跌向须真跌")
    if p.get("min_vol_ratio"):
        bits.append(f"波动≥{p['min_vol_ratio']}")
    # 跌幅深度上限: 只在跌向分支有意义
    if p.get("max_drop") and pick != "top":
        bits.append(f"跌幅≤{p['max_drop']}%")
    # 2026-08-12 前的跑批还带着 fallback_top / defer_until_different 两项,
    # 那两条规则已删除。历史跑批的 label 是存库时就定死的字符串, 不会因为
    # 这里不再拼这两段而改变, 所以直接不认这两个键。
    if p.get("no_same_day_rebuy"):
        bits.append("止损当天禁买回")
    if p.get("min_aum") and p.get("max_aum"):
        bits.append(f"规模{p['min_aum']}~{p['max_aum']}亿")
    elif p.get("min_aum"):
        bits.append(f"规模≥{p['min_aum']}亿")
    elif p.get("max_aum"):
        bits.append(f"规模≤{p['max_aum']}亿")
    # 规模口径只在非标准(旧单份额)时点出来: 合并是 2026-08-06 起的标准动作,
    # 每条标签都缀一句反而是噪声; 旧口径不标出来则会被误当成同口径比较。
    if p.get("min_aum") or p.get("max_aum"):
        if p.get("aum_basis", "single") != "merged":
            bits.append("规模单份额口径")
    if p.get("dd_divisor") and p["dd_divisor"] != 5.0:
        bits.append(f"止损除数{p['dd_divisor']}")
    return " · ".join(bits)


def save_strategy_run(trades: list, params: Optional[dict] = None,
                      label: Optional[str] = None,
                      is_standard: bool = False) -> int:
    """把**一次**回测的结果追加进 strategy.strategy_runs, 返回 run id。

    每跑一次存一条, 不覆盖历史——以前是"一个方案一张表、同名整表覆盖",
    换个参数重跑就把上一次的结果冲掉了, 想回头对比只能重跑。现在每次
    都留痕, 页面上按时间列出来随便翻。

    trades 原样存 JSON(回测多加一列也不用动表结构), params 记录这次跑的
    参数, 汇总指标(笔数/胜率/累计收益)顺手算好存下来, 免得页面为了画
    一个列表把每条的明细全解一遍。

    is_standard=True 标记这条是"线上标准策略"那一跑, 主复盘表读最新的
    那条标准跑批(见 load_backtest_trades)。"""
    n, win, cum = _run_summary(trades)
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO strategy.strategy_runs "
        "(run_at, label, params, trades, is_standard, n_trades, win_rate, cum_return) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (time.time(), label or describe_run(params),
         json.dumps(params or {}, ensure_ascii=False, default=str),
         json.dumps(trades, ensure_ascii=False, default=str),
         1 if is_standard else 0, n, win, cum))
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    return run_id


def list_strategy_runs(limit: int = 50) -> list:
    """→ [{id, run_at, label, params, is_standard, n_trades, win_rate,
    cum_return}, ...],新的在前。不带 trades(明细按需再取, 见
    load_strategy_run), 免得列一页把几十份明细全读进内存。"""
    conn = _conn()
    rows = conn.execute(
        "SELECT id, run_at, label, params, is_standard, n_trades, win_rate,"
        " cum_return FROM strategy.strategy_runs "
        "ORDER BY run_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        try:
            p = json.loads(r["params"]) if r["params"] else {}
        except Exception:
            p = {}
        out.append({"id": r["id"], "run_at": r["run_at"], "label": r["label"],
                    "params": p, "is_standard": bool(r["is_standard"]),
                    "n_trades": r["n_trades"], "win_rate": r["win_rate"],
                    "cum_return": r["cum_return"]})
    return out


def load_strategy_run(run_id: int) -> tuple:
    """→ (DataFrame, params dict, run_at unix秒);取不到返回 (None, {}, None)。"""
    conn = _conn()
    row = conn.execute(
        "SELECT trades, params, run_at FROM strategy.strategy_runs WHERE id=?",
        (run_id,)).fetchone()
    conn.close()
    if not row or not row["trades"]:
        return None, {}, None
    try:
        trades = json.loads(row["trades"])
        params = json.loads(row["params"]) if row["params"] else {}
    except Exception as e:
        logger.warning("回测明细解析失败(run %s): %s", run_id, e)
        return None, {}, None
    if not trades:
        return None, params, row["run_at"]
    return pd.DataFrame(trades), params, row["run_at"]


def load_backtest_trades() -> tuple:
    """最新一次**标准策略**跑批的明细 → (DataFrame, params, run_at, run_id)。
    没跑过返回 (None, {}, None, None)——调用方(app)据此决定是否隐藏复盘区。

    run_id 是 2026-08-13 加的: 页面上原来只显示"明细跑于 <时间>", 而这些
    跑批的 label 是按参数自动拼的、前缀高度雷同(一串"近3月跌幅最大 · 跌向须
    真跌 · 波动≥…"), 光靠时间戳对不上是哪一跑, 讨论时说"#35/#37"在页面上
    根本找不到对应。"""
    conn = _conn()
    row = conn.execute(
        "SELECT id FROM strategy.strategy_runs WHERE is_standard=1 "
        "ORDER BY run_at DESC, id DESC LIMIT 1").fetchone()
    conn.close()
    if not row:
        return None, {}, None, None
    return load_strategy_run(row["id"]) + (row["id"],)


def save_backtest_notes(notes: dict) -> None:
    """回测逐笔点评落库(strategy.backtest_notes,按买入日索引)。

    单独一张表而不是塞进 strategy_runs 的 JSON:点评是人写的、跨多次
    跑批复用(同一个买入日换了参数还是那天的行情), 而 strategy_runs
    每跑一次追加一条, 塞进去就得每次重抄一遍。
    upsert 语义:只覆盖传进来的买入日。"""
    conn = _conn()
    now = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO strategy.backtest_notes "
        "(buy_date, note, saved_at) VALUES (?,?,?)",
        [(str(d), str(n), now) for d, n in notes.items()])
    conn.commit()
    conn.close()


def load_backtest_notes() -> dict:
    """{买入日: 点评};没有则空 dict。"""
    conn = _conn()
    rows = conn.execute(
        "SELECT buy_date, note FROM strategy.backtest_notes").fetchall()
    conn.close()
    return {r["buy_date"]: r["note"] for r in rows}


def update_qvix_self_daily() -> tuple:
    """收盘后重算"最近一个已收盘交易日"的自算QVIX,写入 qvix_self_history
    并重算滚动2年90分位阈值。update_daily.py(06:00)和 notify_qvix.py
    (14:40)都调这个,取代原来基于 optbbs 的 fetch_qvix_daily(force_refresh=True)。

    目标日期不是"今天"——06:00 时今天还没开盘,14:40 时今天还没收盘,
    两边其实都是在补"上一个交易日"的数据,天然幂等(INSERT OR REPLACE),
    重复调用无副作用。上交所官方接口本身也有发布延迟(实测过收盘3小时后
    仍未发布),这天万一还没发布,下一次调用(次日/当天晚些)自然会
    补上,调用方不需要关心这件事。

    返回 (qvix_value_or_None, note)。"""
    import qvix_calc
    today = datetime.now(_CST).date()
    try:
        cal = ak.tool_trade_date_hist_sina()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"]).dt.date
        prior = cal[cal["trade_date"] < today]["trade_date"]
        target = prior.max()
    except Exception as e:
        logger.warning("QVIX 每日自算:拉交易日历失败 %s", e)
        return None, f"交易日历拉取失败: {e}"

    vix, note = qvix_calc.compute_qvix_for_date(target)
    save_qvix_self_history([{"date": target.isoformat(), "qvix": vix, "note": note}])
    if vix is None:
        logger.warning("QVIX 每日自算(%s)失败: %s", target, note)
    else:
        logger.info("QVIX 每日自算(%s) = %.2f", target, vix)

    # 自愈:补最近15天里"接口临时失败"留下的空值(qvix IS NULL)。update 每次
    # 只算"上一个收盘日", 某天上交所接口抽风留的空值之后不会再被重算、会成
    # 永久空洞(2026-07-28/29 就这么丢过)。这里顺带把近15天的空值重试一遍,
    # 接口恢复后自动补上。失败不影响主流程。
    try:
        _h = load_qvix_self_history()
        if _h is not None:
            _cut = (today - timedelta(days=15)).isoformat()
            _nulls = _h[_h["qvix"].isna() & (_h["date"].astype(str) >= _cut)
                        & (_h["date"].astype(str) != target.isoformat())]["date"].tolist()
            for _d in _nulls:
                _v, _n = qvix_calc.compute_qvix_for_date(datetime.fromisoformat(_d).date())
                if _v is not None:
                    save_qvix_self_history([{"date": _d, "qvix": _v, "note": None}])
                    logger.info("QVIX 空值自愈回补(%s) = %.2f", _d, _v)
    except Exception as e:
        logger.warning("QVIX 空值自愈失败(不影响主流程): %s", e)

    hist = load_qvix_self_history()
    if hist is not None:
        hist = hist.sort_values("date").reset_index(drop=True)
        # 2026-07-27 起生产口径从 720/0.95(3年95分位)对齐到 490/0.90
        # (2年90分位)——与 2026-07-24 定档的标准策略(backtest_qvix.py
        # 头部快照)及 app.py 展示线同口径,不然 notify 邮件报的阈值偏高
        # ~1.6个点,QVIX 落在两阈值之间时会漏信号。
        # min_periods=475(=490×0.97):阈值要接近满窗口才给值,不足宁可
        # 空着也不用不完整窗口凑数——早期用240(约1年)会让阈值在头
        # 两年里被少量样本撑出来的分位数带偏,不够稳。不用严格490:历史
        # 里偶发接口失败/数据缺失(约1.6%的交易日),真设成490会导致
        # 窗口只要出现过一天缺失就整个失真成NaN,475留了约15天的
        # 容错,仍然远比240严格(同 backtest_qvix.py 的 minp_ratio=0.97)。
        # 这列**故意不 shift**:存的是"截至该日收盘"的分位。实盘用法是次日
        # 盘中拿最后一行来比(qvix_now.py), 今天的数据天然不在窗口里;
        # 回测(backtest_qvix.py)在自己那边 shift(1) 达成同一口径。别在
        # 这里加 shift——那会让实盘比的阈值凭空旧一天。
        hist["threshold"] = hist["qvix"].rolling(490, min_periods=475).quantile(0.90)
        save_qvix_self_threshold(hist["date"].tolist(), hist["threshold"].tolist())
    return vix, note


# ── Daily-batch pipeline ──────────────────────────────────────────────────────
# Used by update_daily.py: backfill once, then each day append the latest NAV
# point (from the bulk fund-list call) and recompute 年化/回撤/收益率 for all.


def list_nav_codes() -> set:
    """Codes that already have a stored NAV history."""
    conn = _conn()
    rows = conn.execute("SELECT code FROM fund_nav_meta").fetchall()
    conn.close()
    return {r["code"] for r in rows}


def nav_first_dates() -> pd.DataFrame:
    """Earliest stored NAV date per fund, as columns (code, first_nav_date).

    Only funds with stored NAV (C-class) appear. Backfill starts at 2020-01-01,
    so for older funds the value is that floor, not the true inception — it
    reads as "at least this old", which is all the ≤1y filter windows need.
    """
    conn = _conn()
    try:
        df = pd.read_sql_query(
            "SELECT code, MIN(date) AS first_nav_date "
            "FROM fund_nav_daily GROUP BY code", conn)
    finally:
        conn.close()
    df["first_nav_date"] = pd.to_datetime(df["first_nav_date"], errors="coerce")
    return df


_EM_LSJZ_URL = "https://api.fund.eastmoney.com/f10/lsjz"
_EM_LSJZ_HEADERS = {"Referer": "https://fundf10.eastmoney.com/"}


def _fetch_nav_range(code: str, start: datetime,
                     end: datetime) -> Optional[pd.DataFrame]:
    """NAV rows in [start, end] via EastMoney's date-ranged 历史净值 API.

    One paged JSON request (~KB) carries unit NAV, accumulated NAV and daily
    growth together — unlike ak.fund_open_fund_info_em, which downloads the
    fund's entire since-inception pingzhongdata blob (~MB) once per indicator.
    Returns an empty DataFrame when the range has no rows, None on failure.
    Columns: date, nav, daily_ret_pct, acc_nav (ascending by date).
    """
    rows, page = [], 1
    try:
        while True:
            resp = requests.get(_EM_LSJZ_URL, headers=_EM_LSJZ_HEADERS, timeout=15,
                                params={
                                    "fundCode": code,
                                    "pageIndex": page,
                                    "pageSize": 49,
                                    "startDate": start.strftime("%Y-%m-%d"),
                                    "endDate": end.strftime("%Y-%m-%d"),
                                })
            payload = resp.json()
            data = payload.get("Data")
            if not isinstance(data, dict):   # ErrCode -999 etc.
                return None
            batch = data.get("LSJZList") or []
            rows.extend(batch)
            if not batch or len(rows) >= (payload.get("TotalCount") or 0):
                break
            page += 1
    except Exception as e:
        logger.debug("ranged NAV fetch failed for %s: %s", code, e)
        return None

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = pd.DataFrame({
        "date": pd.to_datetime(df["FSRQ"], errors="coerce"),
        "nav": pd.to_numeric(df["DWJZ"], errors="coerce"),
        "daily_ret_pct": pd.to_numeric(df["JZZZL"], errors="coerce"),
        "acc_nav": pd.to_numeric(df["LJJZ"], errors="coerce"),
    })
    df = df.dropna(subset=["date", "nav"])
    df["acc_nav"] = df["acc_nav"].fillna(df["nav"])
    return df.sort_values("date").reset_index(drop=True)


def _fetch_nav_incremental(code: str, after_date: datetime) -> Optional[pd.DataFrame]:
    """Fetch NAV rows newer than `after_date` for a single fund.

    Asks the ranged API for just the missing span; falls back to the full
    single-request download only if the ranged API fails.
    """
    start = max(after_date + timedelta(days=1), pd.Timestamp(NAV_START))
    df = _fetch_nav_range(code, start, datetime.now())
    if df is None:
        df = _fetch_nav_full(code)
    if df is None:
        return None
    df = df[df["date"] > after_date].reset_index(drop=True)
    return df if not df.empty else None


def _backfill_incremental(codes: list, workers: int = MAX_WORKERS,
                          progress: Optional[Callable] = None) -> int:
    """Incrementally fill NAV gaps for `codes` (threaded).

    For each code, reads the last stored date, fetches only newer rows, and
    inserts them.  Returns the count of codes that got new data.
    """
    total, done, patched = len(codes), 0, 0
    if not codes:
        return 0

    def _patch_one(code: str) -> bool:
        conn = _conn()
        meta = conn.execute(
            "SELECT last_date FROM fund_nav_meta WHERE code=?", (code,)).fetchone()
        conn.close()
        if not meta or not meta["last_date"]:
            return False

        new_rows = _fetch_nav_incremental(code, pd.to_datetime(meta["last_date"]))
        if new_rows is None or new_rows.empty:
            return False

        conn = _conn()
        _write_nav_rows(conn, code, new_rows)
        conn.commit()
        conn.close()
        return True

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_patch_one, c): c for c in codes}
        for fut in as_completed(futures):
            done += 1
            try:
                if fut.result():
                    patched += 1
            except Exception:
                pass
            if progress and (done % 50 == 0 or done == total):
                progress(done, total)
    return patched


def _only_weekends_between(a: datetime, b: datetime) -> bool:
    """True if every calendar day strictly between a and b is a Sat/Sun.

    Used to tell "the fund list's latest NAV is the only missing point"
    (consecutive trading days, possibly across a weekend) from a real
    multi-day gap. Holidays make this return False, which safely falls
    through to the ranged fetch.
    """
    d = pd.Timestamp(a).normalize() + timedelta(days=1)
    end = pd.Timestamp(b).normalize()
    while d < end:
        if d.weekday() < 5:
            return False
        d += timedelta(days=1)
    return True


def _append_nav_point(conn, code: str, date, nav: float,
                      acc_nav: Optional[float], ret_pct: Optional[float]) -> bool:
    """Append one NAV row (from the bulk fund-list call) — a single INSERT,
    zero network. The caller owns the transaction (no commit here)."""
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    if ret_pct is None:
        prev = conn.execute(
            "SELECT nav FROM fund_nav_daily WHERE code=? AND date<? "
            "ORDER BY date DESC LIMIT 1", (code, d)).fetchone()
        if prev and prev["nav"]:
            ret_pct = (nav / prev["nav"] - 1.0) * 100.0
    conn.execute(
        "INSERT OR REPLACE INTO fund_nav_daily "
        "(code, date, nav, daily_ret_pct, acc_nav) VALUES (?, ?, ?, ?, ?)",
        (code, d, nav, ret_pct, acc_nav if acc_nav is not None else nav))
    conn.execute(
        "INSERT OR REPLACE INTO fund_nav_meta (code, saved_at, last_date) "
        "VALUES (?, ?, ?)", (code, time.time(), d))
    return True


def append_incremental(list_df: pd.DataFrame,
                       progress: Optional[Callable] = None) -> dict:
    """Bring every stored NAV history up to the fund list's latest date.

    Two tiers, cheapest first:
      • gap of exactly one trading day (only weekends in between) — the bulk
        fund-list call already carries that day's nav/acc_nav/daily_ret, so
        the point is appended directly, zero extra network;
      • bigger gap (holidays, many days since last run) — fetch just the
        missing date span via the ranged 历史净值 API (threaded).

    Gap detection reads fund_nav_meta.last_date — one small table scan.
    `progress(phase, done, total)` as in run_pipeline. Returns a summary dict
    with counts (failed = gapped funds whose ranged fetch returned nothing).
    """
    def _p(phase, done, total):
        if progress:
            progress(phase, done, total)

    latest = list_df.dropna(subset=["code"]).drop_duplicates("code").set_index("code")

    conn = _conn()
    stored = {r["code"]: r["last_date"]
              for r in conn.execute("SELECT code, last_date FROM fund_nav_meta")}

    appended = skipped = 0
    gapped = []
    total = len(stored)
    for i, (code, last_iso) in enumerate(stored.items()):
        if i % 500 == 0 or i == total - 1:
            _p("追加当日净值", i + 1, total)
        if not last_iso or code not in latest.index:
            skipped += 1
            continue
        row = latest.loc[code]
        new_date = pd.to_datetime(row["nav_date"], errors="coerce")
        nav = pd.to_numeric(row["nav"], errors="coerce")
        if pd.isna(new_date) or pd.isna(nav):
            skipped += 1
            continue
        last_date = pd.to_datetime(last_iso)
        if new_date <= last_date:
            skipped += 1                       # already up-to-date
            continue

        if _only_weekends_between(last_date, new_date):
            acc = pd.to_numeric(row.get("acc_nav"), errors="coerce")
            ret = pd.to_numeric(row.get("daily_ret"), errors="coerce")
            if _append_nav_point(conn, code, new_date, float(nav),
                                 None if pd.isna(acc) else float(acc),
                                 None if pd.isna(ret) else float(ret)):
                appended += 1
                continue
        gapped.append(code)                    # real gap → ranged fetch
    conn.commit()
    conn.close()

    patched = 0
    if gapped:
        patched = _backfill_incremental(
            gapped, MAX_WORKERS, lambda d, t: _p("增量补缺口", d, t)
        )

    return {"appended": appended, "patched": patched, "skipped": skipped,
            "gap_codes": len(gapped), "failed": len(gapped) - patched}


def recompute_all(progress_callback: Optional[Callable] = None) -> int:
    """Recompute 年化/回撤/区间收益 for every stored fund from its stored NAV.

    No network for NAV — pure CPU over cached NAV. All reads/writes share one
    connection and a single commit, so 20k funds finish in seconds.
    """
    conn = _conn()
    codes = [r["code"] for r in conn.execute("SELECT code FROM fund_nav_meta").fetchall()]
    total, done, saved = len(codes), 0, 0
    for code in codes:
        done += 1
        try:
            nav_df = _load_nav_df(code, conn)
            m = _metrics_from_nav(nav_df) if not nav_df.empty else None
        except Exception as e:
            logger.debug("recompute parse error %s: %s", code, e)
            m = None
        if m is not None:
            conn.execute(
                "INSERT OR REPLACE INTO fund_sharpe "
                "(code, ann_return, volatility, data_points, "
                " mdd_1m, mdd_3m, mdd_6m, mdd_1y, "
                " ret_1m, ret_3m, ret_6m, ret_1y, saved_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (code, m["ann_return"], m["volatility"], m["data_points"],
                 m["mdd_1m"], m["mdd_3m"], m["mdd_6m"], m["mdd_1y"],
                 m["ret_1m"], m["ret_3m"], m["ret_6m"], m["ret_1y"], time.time()),
            )
            saved += 1
        if progress_callback and (done % 500 == 0 or done == total):
            progress_callback(done, total)
    conn.commit()
    # Record WHICH data these metrics were computed from — the newest NAV
    # trading date and how many funds carry it — so metrics_stale() compares
    # data versions, not wall-clock timestamps ("今天的数据算今天的,明天的
    # 数据来了才失效"). The timestamp marker is kept separately for display.
    # Deliberately NOT derived from MAX(fund_sharpe.saved_at): the detail tab
    # persists single rows via compute_metrics_for_fund, which would make the
    # whole table look fresh after viewing one fund.
    date_num, at_date = _nav_data_version(conn)
    conn.close()
    if date_num is not None:
        _set_meta("metrics_nav_date", date_num)
        _set_meta("metrics_nav_date_rows", at_date)
    _set_meta("metrics_recomputed_at", time.time())
    return saved


def _asof_returns_vectorized(conn, asof: str, cols: set):
    """compute_metrics_asof 的向量化快路: 只算区间收益列(cols ⊆ RETURN_DAYS)。

    逐只路径(下面那个 for 循环)的耗时**不在 SQL 上**: 实测 5722 只共 13.0s,
    其中一次性把数据读出来只要 1s, 剩下 9~12s 全花在"每只基金构造一个
    DataFrame 再跑 _metrics_from_nav"上, 光 effective_daily_ret 就占一半。
    所以这里把整段计算搬到**一整张表**上做, 5722 次 pandas 调用变成十几次。

    口径与逐只路径**完全一致**(切换前逐只比对过 5722 只, 见下面几条保证):
      · 切片起点取 min(每只最后净值日) - 窗口天数 - 宽限, 保证每只基金的
        区间窗口(锚点在内)整段落在切片里 —— 锚点是"最后净值日往回 days_back
        天之前的最后一个净值日", 切片截短了会让锚点前移/落空, 算出来的收益
        就变了(实测直接按 asof-130天 切会有 8 只对不上、2 只消失);
      · 切片内行数 < 60 的稀疏基金(定开/按周披露)一律退回逐只路径 —— 那两道
        "至少 20 个点"的门槛在全历史上判和在切片上判不是一回事;
      · effective_daily_ret 的第一行拿不到环比(shift 出 NaN), 但切片第一行
        必然早于任何一只的窗口锚点, 影响不到结果。
    """
    ends = conn.execute(
        "SELECT code, MAX(date) AS e, COUNT(*) AS n FROM fund_nav_daily "
        "WHERE date < ? GROUP BY code", (asof,)).fetchall()
    if not ends:
        return {}, set()
    need = max(RETURN_DAYS[c] for c in cols) + ANCHOR_GRACE_DAYS + 5
    slice_start = (pd.Timestamp(min(r["e"] for r in ends))
                   - timedelta(days=need)).strftime("%Y-%m-%d")
    full_n = {r["code"]: r["n"] for r in ends}
    big = pd.read_sql_query(
        "SELECT code, date, nav, daily_ret_pct, acc_nav FROM fund_nav_daily "
        "WHERE date < ? AND date >= ? ORDER BY code, date",
        conn, params=(asof, slice_start))
    if big.empty:
        return {}, set(full_n)

    code = big["code"]
    n_slice = code.value_counts()
    # 稀疏/历史被切片截短的退回逐只路径(见 docstring 第二条)
    fallback = {c for c, n in full_n.items()
                if n >= 20 and n_slice.get(c, 0) < 60}
    if fallback:
        keep = ~code.isin(fallback)
        big = big[keep].reset_index(drop=True)
        code = big["code"]
        if big.empty:
            return {}, fallback
    # 全历史点数不够 20 的, 逐只路径也会跳过 —— 这里一并排除, 结果一致
    thin = {c for c, n in full_n.items() if n < 20}
    if thin:
        big = big[~code.isin(thin)].reset_index(drop=True)
        code = big["code"]
        if big.empty:
            return {}, fallback

    dates = pd.to_datetime(big["date"])
    nav = pd.to_numeric(big["nav"], errors="coerce")
    acc = pd.to_numeric(big["acc_nav"], errors="coerce").fillna(nav)
    r = pd.to_numeric(big["daily_ret_pct"], errors="coerce") / 100.0
    implied_nav = nav / nav.groupby(code, sort=False).shift(1) - 1.0
    implied_acc = acc / acc.groupby(code, sort=False).shift(1) - 1.0
    # 下面五行是 effective_daily_ret 的逐字向量化版, 改那边记得改这里
    conflict = (r - implied_nav).abs() > 0.003
    dividendish = (r - implied_acc).abs() <= 0.003
    use_implied = (conflict & ~dividendish & implied_nav.notna()) | r.isna()
    rr = r.mask(use_implied, implied_nav)
    rr = rr.mask(rr.abs() > 0.30, 0.0)      # mask 不碰 NaN, 与逐只版一致

    pos = pd.Series(np.arange(len(big)), index=big.index)
    last_pos = pos.groupby(code, sort=False).max()
    first_pos = pos.groupby(code, sort=False).min()
    end_f = dates.groupby(code, sort=False).transform("max")
    first_date = dates.groupby(code, sort=False).transform("min")
    # (1+r) 的组内累乘: NaN 当因子 1(等价于逐只版先 dropna 再连乘),
    # 另开一列数非 NaN 个数, 用来复现"窗口内一个有效收益都没有→None"。
    fac = (1.0 + rr).fillna(1.0)
    cum = fac.groupby(code, sort=False).cumprod()
    nn = rr.notna().astype(int).groupby(code, sort=False).cumsum()

    out = {}
    cum_v, nn_v = cum.to_numpy(), nn.to_numpy()
    for key in cols:
        days = RETURN_DAYS[key]
        start_f = end_f - pd.Timedelta(days=days)
        older = dates <= start_f
        anchor = pos.where(older).groupby(code, sort=False).max()   # NaN=没有
        # 没有锚点时的宽限: 全历史第一条比理想起点晚不超过 ANCHOR_GRACE_DAYS
        # 就拿第一条当锚点(逐只版 _window_by_date 的同一条规则), 否则该只无值
        grace_ok = ((first_date.groupby(code, sort=False).first()
                     - start_f.groupby(code, sort=False).first()).dt.days
                    <= ANCHOR_GRACE_DAYS)
        anchor = anchor.fillna(first_pos.where(grace_ok))
        for c, a in anchor.items():
            if pd.isna(a):
                out.setdefault(c, {})[key] = None
                continue
            a, e = int(a), int(last_pos[c])
            if e <= a or nn_v[e] - nn_v[a] <= 0:
                out.setdefault(c, {})[key] = None     # 窗口里没有有效收益
                continue
            out.setdefault(c, {})[key] = float(
                (cum_v[e] / cum_v[a] - 1.0) * 100.0)
    return out, fallback


def compute_metrics_asof(asof: str,
                         progress_callback: Optional[Callable] = None,
                         cols: Optional[set] = None) -> dict:
    """Every fund's metrics as an observer ON `asof` could have seen them.

    Truncates each fund's history to rows STRICTLY BEFORE `asof` (ISO
    yyyy-mm-dd): on day D the day's own NAV is published only after close, so
    a screen run on D can only be based on data through D-1. Recomputes
    period returns and drawdown over the same trailing windows — no network,
    nothing persisted. Funds with under 20 NAV points by that date are
    omitted. Returns {code: metrics-dict}.

    `cols` 只算指定区间列(见 _metrics_from_nav),筛选快照用。
    """
    conn = _conn()
    codes = [r["code"] for r in conn.execute("SELECT code FROM fund_nav_meta")]
    # 只要区间收益列(今日候选/回测/pct_rank 都是 cols={"ret_3m"})时走向量化
    # 快路, 实测 13.0s → 见 _asof_returns_vectorized。回撤列还没向量化(窗口内
    # cummax 麻烦), 带 mdd_* 的调用(筛选页的「截至日期」)照旧逐只算。
    if cols and all(c in RETURN_DAYS for c in cols):
        fast, fallback = _asof_returns_vectorized(conn, asof, set(cols))
        codes = [c for c in codes if c in fallback]     # 剩这些逐只补算
        out = dict(fast)
        for i, code in enumerate(codes):
            df = pd.read_sql_query(
                "SELECT date, nav, daily_ret_pct, acc_nav FROM fund_nav_daily "
                "WHERE code = ? AND date < ? ORDER BY date",
                conn, params=(code, asof))
            if len(df) >= 20:
                df["date"] = pd.to_datetime(df["date"])
                try:
                    m = _metrics_from_nav(df, cols=cols)
                except Exception as e:
                    logger.debug("asof metrics failed %s: %s", code, e)
                    m = None
                if m is not None:
                    out[code] = m
        if progress_callback:
            progress_callback(len(out), len(out))
        conn.close()
        return out

    out, total = {}, len(codes)
    for i, code in enumerate(codes):
        df = pd.read_sql_query(
            "SELECT date, nav, daily_ret_pct, acc_nav FROM fund_nav_daily "
            "WHERE code = ? AND date < ? ORDER BY date",
            conn, params=(code, asof))
        if len(df) >= 20:
            df["date"] = pd.to_datetime(df["date"])
            try:
                m = _metrics_from_nav(df, cols=cols)
            except Exception as e:
                logger.debug("asof metrics failed %s: %s", code, e)
                m = None
            if m is not None:
                out[code] = m
        if progress_callback and ((i + 1) % 500 == 0 or i + 1 == total):
            progress_callback(i + 1, total)
    conn.close()
    return out


def load_all_precomputed() -> dict:
    """Every precomputed 年化/回撤/区间收益 row, keyed by code, ignoring TTL.

    Lets the app show metrics instantly on load; freshness is the daily
    pipeline's responsibility (surfaced via last_update_time()).
    """
    conn = _conn()
    rows = conn.execute(
        "SELECT code, ann_return, volatility, data_points, "
        "mdd_1m, mdd_3m, mdd_6m, mdd_1y, "
        "ret_1m, ret_3m, ret_6m, ret_1y FROM fund_sharpe"
    ).fetchall()
    conn.close()
    return {r["code"]: {k: r[k] for k in r.keys() if k != "code"} for r in rows}


def last_update_time() -> Optional[float]:
    """Unix time of the last FULL metrics recompute, or None if never.

    Prefers the recompute_all marker; MAX(fund_sharpe.saved_at) is only a
    legacy fallback (pre-marker DBs) — it overstates freshness because the
    detail tab persists individual rows, bumping the max after one lookup.
    """
    t, _ = _get_meta("metrics_recomputed_at")
    if t is not None:
        return t
    conn = _conn()
    row = conn.execute("SELECT MAX(saved_at) AS t FROM fund_sharpe").fetchone()
    conn.close()
    return row["t"] if row and row["t"] else None


def _nav_data_version(conn) -> tuple:
    """(newest NAV trading date as a yyyymmdd float, #funds already carrying
    that date) — the identity of "which day's data is in the store". Appending
    a new trading day flips the date; late stragglers for the same day bump
    the count. Re-running the pipeline on a weekend/holiday (no new NAV)
    changes neither, so metrics keyed to this never recompute for nothing.
    (None, 0) with no NAV data at all.
    """
    row = conn.execute(
        "SELECT MAX(last_date) AS d FROM fund_nav_meta").fetchone()
    if not row or not row["d"]:
        return None, 0
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM fund_nav_meta WHERE last_date = ?",
        (row["d"],)).fetchone()["n"]
    return float(row["d"].replace("-", "")), n


def metrics_stale() -> bool:
    """True when the stored NAV holds data the last full metrics recompute
    hasn't seen — a newer trading date, or more funds reporting the same
    newest date — i.e. filtering on the precomputed 回撤/收益率 would use the
    previous day's values.

    A missing marker (pre-marker DBs) counts as stale, so the first filter
    after upgrading recomputes once and plants the markers.
    """
    conn = _conn()
    date_num, at_date = _nav_data_version(conn)
    conn.close()
    if date_num is None:
        return False   # no NAV data at all — nothing to recompute from
    met_date, _ = _get_meta("metrics_nav_date")
    met_rows, _ = _get_meta("metrics_nav_date_rows")
    return met_date != date_num or met_rows != at_date


def fund_list_saved_at() -> Optional[float]:
    """Unix time the fund list snapshot (returns, incl. ret_1m/3m/6m/1y) was
    last saved, or None if never fetched.

    The in-app update button refreshes the fund list every run but skips
    recompute_all() (do_recompute=False), so last_update_time() alone doesn't
    move — callers needing a cache key that reflects *any* data refresh
    (not just a metrics recompute) should combine this with last_update_time().
    """
    conn = _conn()
    row = conn.execute("SELECT saved_at FROM fund_list WHERE id = 1").fetchone()
    conn.close()
    return row["saved_at"] if row else None


def _backfill_codes(codes: list, workers: int = MAX_WORKERS,
                    progress: Optional[Callable] = None) -> int:
    """Download full ~1y NAV history for `codes` (threaded). Returns success count."""
    total, done, ok = len(codes), 0, 0
    if not codes:
        return 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_nav, c): c for c in codes}
        for fut in as_completed(futures):
            done += 1
            try:
                if fut.result() is not None:
                    ok += 1
            except Exception:
                pass
            if progress and (done % 50 == 0 or done == total):
                progress(done, total)
    return ok


def run_pipeline(progress: Optional[Callable] = None, do_backfill: bool = True,
                 workers: int = MAX_WORKERS,
                 do_recompute: bool = True) -> dict:
    """Full daily pipeline shared by the CLI script and the in-app button:
    ① refresh fund list → ② backfill funds missing history → ③ append the
    list's latest NAV point / range-fetch bigger gaps → ④ recompute
    年化/回撤/区间收益 for all (skippable via do_recompute=False when metrics
    are computed lazily at filter time instead).

    `progress(phase, done, total)` is invoked throughout for a UI bar / logging.
    Returns a summary dict.
    """
    def _p(phase, done, total):
        if progress:
            progress(phase, done, total)

    _p("拉取基金列表", 0, 1)
    list_df = fetch_fund_list(force_refresh=True)
    all_codes = list_df["code"].dropna().unique().tolist()
    _p("拉取基金列表", 1, 1)

    backfilled = 0
    if do_backfill:
        have = list_nav_codes()
        # Only C-class, non-overseas-equity shares get a NAV history; without
        # this filter every fund cleaned out of the DB (non-C, QDII/海外股)
        # would be re-downloaded daily.
        meta = list_df.dropna(subset=["code"]).drop_duplicates("code") \
            .set_index("code")[["name", "type"]]
        names = meta["name"]
        types = meta["type"]
        todo = [c for c in all_codes
                if c not in have and is_c_class(names.get(c))
                and not is_overseas_equity(types.get(c))
                and not is_bond(types.get(c))]
        backfilled = _backfill_codes(
            todo, workers, lambda d, t: _p("回填缺失历史", d, t)
        )

    res = append_incremental(list_df, progress=_p)
    # No staleness marker needed here: metrics_stale() derives the data
    # version straight from fund_nav_meta, so whatever this run appended is
    # visible to the next filter's check automatically (and a run that
    # appended nothing new leaves metrics valid).

    saved = 0
    if do_recompute:
        saved = recompute_all(progress_callback=lambda d, t: _p("重算指标", d, t))

    return {
        "funds": len(all_codes),
        "backfilled": backfilled,
        "appended": res["appended"],
        "patched": res["patched"],
        "failed": res["failed"],
        "recomputed": saved,
    }
