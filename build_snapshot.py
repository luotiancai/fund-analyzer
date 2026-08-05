#!/usr/bin/env python3
"""把完整的 fund_cache.db 切成云端用的两段式快照。

为什么要切:Streamlit Community Cloud 的容器磁盘是临时的,每次冷启动(含
休眠唤醒)都要重新拉一遍快照,而应用要等它下完才画得出第一屏。整库 415MB
/ gz 82MB,这一等就是几十秒。

但首屏(上证指数 + QVIX + 榜单)读的表加起来只有 9MB:净值 250MB、它的
date 索引 120MB、规模 10MB、持仓 4MB 这几样,要等用户点进筛选/详情/模拟盘
才用得上。于是拆成:

    fund_cache_core.db.gz   ~2MB    启动时同步拉,拉完首屏立刻能画
    fund_cache_nav.db.gz    ~63MB   首屏之后后台拉,落地后 ATTACH 接管

nav 库里不带 idx_nav_date:它是纯派生数据,占整库三成,应用侧
fetcher.adopt_nav_db() 现建只要 1~2 秒,比多下 17MB(gz)划算。

整库快照 fund_cache.db.gz 照常保留:每日跑批要拿它当上一轮的底子(见
.github/workflows/update-daily.yml),同时也是应用侧的回退路径——两段式
资产还没发布出来时,app.py 会自动退回去拉它。

用法:
    python3 build_snapshot.py <源库> <core 输出> <nav 输出>
不给参数时按默认路径(FUND_ANALYZER_DATA/fund_cache.db → 当前目录)。
"""

import os
import sqlite3
import sys
import time

# 与 fetcher.HEAVY_TABLES 保持一致(这里不 import fetcher:跑批环境装了它的
# 依赖,但这个脚本也可能在更干净的环境里单跑)。
HEAVY_TABLES = ("fund_nav_daily", "fund_holdings", "fund_scale_hist")


def _mb(path):
    return os.path.getsize(path) / 1048576


def _carve(src, dst, keep_heavy):
    """VACUUM INTO 出一份副本,再删掉不要的那一半并回收空间。

    先整库拷贝再删,而不是建新库逐表 INSERT:前者一条 SQL 搞定、不用复刻
    建表语句(WITHOUT ROWID、主键这些漏一个就是隐性劣化),代价只是中间
    多占一份磁盘。
    """
    if os.path.exists(dst):
        os.unlink(dst)
    t0 = time.perf_counter()
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        conn.execute("VACUUM INTO ?", (dst,))
    finally:
        conn.close()

    conn = sqlite3.connect(dst)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'")]
        drop = [n for n in names
                if (n in HEAVY_TABLES) != bool(keep_heavy)]
        for n in drop:
            conn.execute(f"DROP TABLE IF EXISTS [{n}]")
        if keep_heavy:
            # 派生数据,应用侧 adopt 时重建。
            conn.execute("DROP INDEX IF EXISTS idx_nav_date")
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    return time.perf_counter() - t0


def build(src, core_out, nav_out):
    if not os.path.exists(src):
        raise SystemExit(f"找不到源库: {src}")
    print(f"源库 {src}  {_mb(src):.1f} MB")
    el = _carve(src, core_out, keep_heavy=False)
    print(f"  core → {core_out}  {_mb(core_out):.2f} MB  ({el:.1f}s)")
    el = _carve(src, nav_out, keep_heavy=True)
    print(f"  nav  → {nav_out}   {_mb(nav_out):.1f} MB  ({el:.1f}s)")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        _src, _core, _nav = sys.argv[1:4]
    elif len(sys.argv) == 1:
        _dir = os.environ.get("FUND_ANALYZER_DATA") or os.path.join(
            os.path.expanduser("~"), ".local", "share", "fund-analyzer")
        _src = os.path.join(_dir, "fund_cache.db")
        _core, _nav = "fund_cache_core.db", "fund_cache_nav.db"
    else:
        raise SystemExit(__doc__)
    build(_src, _core, _nav)
