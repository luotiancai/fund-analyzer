#!/usr/bin/env python3
"""把云端的库同步到本地(按库、按需、远端没变就不下)。

    python3 sync_down.py                 # 同步日常那几个小库(约 2MB)
    python3 sync_down.py market          # 只同步上证/QVIX(约 0.2MB)
    python3 sync_down.py --all           # 连净值/规模一起(约 65MB)
    python3 sync_down.py --status        # 只看本地和云端各是什么版本

分库之前, 本地想要一点新数据的最小粒度是 62MB 的重表段 —— 想同步今天的
QVIX 也得整段拉, 于是实际上从来不同步, 本地库一路飘。现在按库拉:

    market.db     ~0.2MB   上证/QVIX/阈值   每天要
    fund_rank.db  ~2MB     榜单/指标        每天要
    fund_scale.db ~3MB     季度规模/持仓    一季度一次
    fund_nav.db   ~60MB    净值             跑回测前偶尔一次

日常(默认那组)从 62MB 降到约 2.2MB。再加条件下载: 记住上次拿到的版本戳,
远端没变直接跳过、零流量。

注意本地和云端**权威方不同的库**不在默认组里:
  · strategy 是你本地跑回测写的 —— 拉下来会盖掉你还没推的跑批;
  · sim 的权威方是线上 app;
  · cache 是缓存, 拉了没意义。
要拉这些得显式点名。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_assets   # noqa: E402
import fetcher        # noqa: E402

# 日常同步组: 只有"云端是权威、本地纯消费"且小的库。
DAILY = ("market", "rank")
# --all 再加上大库。strategy/sim/cache 任何时候都不自动拉(见 docstring)。
ALL = DAILY + ("scale", "nav")


def status():
    remote = cloud_assets.asset_times()
    stamps = cloud_assets._stamps()
    print(f"{'库':10}{'本地文件':>12}{'云端资产':>12}  云端更新于        状态")
    for name, fn, _t, _l in fetcher.DB_LAYOUT:
        p = fetcher.DB_PATH[name]
        loc = f"{os.path.getsize(p)/1048576:.1f}MB" if os.path.exists(p) else "—"
        r = remote.get(cloud_assets.ASSET_OF[name])
        rem = f"{r['size']/1048576:.1f}MB" if r else "—"
        upd = r["updated"][:16].replace("T", " ") if r else ""
        key = f"{cloud_assets.ASSET_OF[name]}@{p}"
        if not r:
            st = "云端没有"
        elif stamps.get(key) == r["updated"]:
            st = "已是最新"
        elif os.path.exists(p):
            st = "⬇ 有更新"
        else:
            st = "⬇ 本地没有"
        print(f"{name:10}{loc:>12}{rem:>12}  {upd:18}{st}")


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--status" in sys.argv:
        status()
        return
    names = list(ALL) if "--all" in sys.argv else (argv or list(DAILY))
    bad = [n for n in names if n not in fetcher.DB_PATH]
    if bad:
        raise SystemExit(f"未知的库: {', '.join(bad)}\n"
                         f"可选: {', '.join(fetcher.DB_PATH)}")
    remote = cloud_assets.asset_times()
    print(f"同步 {', '.join(names)} → {fetcher._DATA_DIR}")
    for n in names:
        try:
            # 直接落到真实数据目录(不是 peek 的缓存): 这是"用", 不是"看"。
            cloud_assets.fetch(n, dest=fetcher.DB_PATH[n], remote_times=remote,
                               refresh="--refresh" in sys.argv)
        except FileNotFoundError:
            print(f"  {n}: 云端没有这个资产, 跳过")
    if "nav" in names:
        # 索引不随资产下发(纯派生数据, 占净值库三成), 落地后现建。
        print("建 nav.idx_nav_date …", end="", flush=True)
        conn = fetcher._conn()
        conn.execute("CREATE INDEX IF NOT EXISTS nav.idx_nav_date "
                     "ON fund_nav_daily(date)")
        conn.commit()
        conn.close()
        print(" 好了")
    print("✅ 同步完成")


if __name__ == "__main__":
    main()
