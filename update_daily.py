#!/usr/bin/env python3
"""Daily batch: keep NAV history fresh and recompute 年化/回撤/收益率 for all funds.

The Streamlit app only *reads* the precomputed metrics, so all the slow network
work lives here and runs out of band. The same pipeline is also exposed as the
in-app「🔄 更新数据」button via fetcher.run_pipeline().

Pipeline (fetcher.run_pipeline):
  ① 拉取基金列表(1 次批量调用,带回全部基金的最新净值点)
  ② 历史回填:对还没有净值历史的 C 类基金,每只 1 次请求下载 2020-01-01 至今
     的序列(基本一次性;非 C 类不存净值,见 fetcher.is_c_class)
  ③ 增量补净值:只差一个交易日的基金直接追加基金列表带回的当日净值点(零请求);
     缺口更大的用天天基金历史净值接口按日期段拉取(每只一次几 KB 的请求)
  ④ 重算:用存好的净值对全部基金重算年化 + 最大回撤 + 区间收益(纯 CPU,几秒)

用法
  手动:     python3 update_daily.py
  仅重算:    python3 update_daily.py --recompute-only
  仅补QVIX:  python3 update_daily.py --qvix-only   (VPS 专用, 见 run_qvix_only)
  仅刷规模:   python3 update_daily.py --scales-only (规模覆盖范围扩大后补全用)
  cron:      0 18 * * 1-5  cd /path/to/fund-analyzer && python3 update_daily.py >> update.log 2>&1
"""

import argparse
import logging
import os
import time

import fetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("update_daily")

# Throttle per-phase logging so a 20k-fund phase doesn't spam one line per 50.
_last_log = {}


def _log_progress(phase, done, total):
    if done == total or time.time() - _last_log.get(phase, 0) > 2:
        log.info("   %s %d/%d", phase, done, total)
        _last_log[phase] = time.time()


def _qvix_dates_with_value() -> set:
    """qvix_self_history 里**已经有值**(qvix 非空)的日期集合。
    用来判断一次补跑到底补上了什么:没补上任何东西就不必推一次库。"""
    hist = fetcher.load_qvix_self_history()
    if hist is None:
        return set()
    return set(hist[hist["qvix"].notna()]["date"].astype(str))


def _emit_gha_output(**kv) -> None:
    """把结果写给 GitHub Actions 的 step outputs。本地跑时 GITHUB_OUTPUT
    不存在,什么也不做——脚本本身不该因为"没在CI里"就报错。"""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    try:
        with open(path, "a") as f:
            for k, v in kv.items():
                f.write("%s=%s\n" % (k, v))
    except Exception as e:      # 写不进去不该让整个补跑算失败
        log.warning("写 GITHUB_OUTPUT 失败(不影响补跑结果): %s", e)


def run_qvix_only() -> list:
    """只补自算QVIX,不碰基金净值、不刷指数缓存。返回本次补上的日期列表。

    **这条路径现在是 QVIX 的唯一生产者, 跑在国内那台 VPS 上**(2026-08-28
    改)。原因是上交所的期权风险指标接口挡 GitHub runner 的境外 IP:那天同一
    时刻本机和 VPS 都拿得到 656 条合约, runner 上三次重试全失败(报文分别是
    非 JSON 错误页和 connection reset)。云端跑批从此不碰 QVIX, 它写的
    qvix.db 也不由跑批发布 —— 一张表一个写入方, 没有互相覆盖的机会。

    配套的 qvix-retry.yml(原来北京早上6点在 GitHub 上补跑一次)同时删掉了:
    换个时间点再试一遍解决不了 IP 被挡, 只是每天多一条绿色的空跑记录。

    幂等:update_qvix_self_daily 是 INSERT OR REPLACE, 重复跑算出来是同一个
    数, 没有副作用。它自带"补最近15天空值"的自愈逻辑, 所以 VPS 关机几天再
    开机, 一次就把断档补齐。返回的日期列表用来决定要不要推库 —— 什么都没
    补上就不必为一个没变的库走一趟 0.3MB 的上传和回读验证。"""
    before = _qvix_dates_with_value()
    vix, note = fetcher.update_qvix_self_daily()
    filled = sorted(_qvix_dates_with_value() - before)

    if vix is not None:
        log.info("   QVIX(自算) %.2f", vix)
    else:
        log.warning("   QVIX 自算仍未成功: %s", note)

    if filled:
        log.info("   本次补上 %d 天: %s", len(filled), ", ".join(filled))
    else:
        log.info("   没有新补上的值(凌晨那次已经算好了,或接口这会儿还是不通)")

    _emit_gha_output(changed=str(bool(filled)).lower(),
                     filled=",".join(filled))
    return filled


def main():
    parser = argparse.ArgumentParser(description="基金净值每日跑批")
    parser.add_argument("--recompute-only", action="store_true",
                        help="跳过下载,只用已存净值重算年化/回撤/收益率")
    parser.add_argument("--qvix-only", action="store_true",
                        help="只补自算QVIX,不跑基金净值/指数(VPS 上每日跑这条)")
    parser.add_argument("--scales-only", action="store_true",
                        help="只刷基金季度规模,不跑净值/指数/QVIX"
                             "(规模覆盖范围扩大后的一次性补全用)")
    args = parser.parse_args()

    t0 = time.time()
    fetcher.init_db()

    if args.scales_only:
        # 规模覆盖范围从「本地有净值的基金」扩到「榜单全部非债基金」后,
        # 首轮要补约 8000 只、一个多小时。日常跑批里顺带补也行,但那样得等
        # 到下一次凌晨跑批;单独一条路径可以手动/workflow_dispatch 立刻补。
        log.info("仅刷新基金季度规模(不跑净值/指数/QVIX)…")
        n = fetcher.refresh_scale_hist(
            progress=lambda p, d, t: _log_progress(p, d, t))
        log.info("✅ 规模刷新完成(实抓 %d 只),耗时 %.0f 秒", n, time.time() - t0)
        return

    if args.qvix_only:
        log.info("仅补 QVIX(不跑基金净值/指数)…")
        run_qvix_only()
        log.info("✅ 完成,耗时 %.0f 秒", time.time() - t0)
        return

    if args.recompute_only:
        log.info("仅重算:用已存净值重算年化 + 回撤 + 区间收益…")
        saved = fetcher.recompute_all(progress_callback=lambda d, t: _log_progress("重算", d, t))
        log.info("   写入 %d 只指标", saved)
    else:
        summary = fetcher.run_pipeline(progress=_log_progress)
        log.info("基金 %d · 回填 %d · 当日追加 %d · 补缺口 %d（失败 %d）· 重算 %d",
                 summary["funds"], summary["backfilled"], summary["appended"],
                 summary["patched"], summary["failed"], summary["recomputed"])

        # 基金季度规模刷新(供选基/回测的规模门槛用)。cache-first, 7天内
        # 已抓的跳过, 平时几乎零网络; 只有新基金/过期的才真抓。
        try:
            n = fetcher.refresh_scale_hist(
                progress=lambda p, d, t: _log_progress(p, d, t))
            log.info("   基金规模刷新完成(本次实抓 %d 只)", n)
        except Exception as e:
            log.warning("   基金规模刷新失败(不影响主流程): %s", e)

    # ── 指数刷新 ────────────────────────────────────────────────────────────
    # 上证指数刷新照旧(app 侧无过期时间,只在这里force_refresh才变)。
    #
    # **QVIX 不在这里算了**(2026-08-28 移走)。上交所的期权风险指标接口挡
    # GitHub runner 的境外 IP:同一时刻本机和国内 VPS 都拿得到 656 条合约,
    # runner 上三次重试全失败。QVIX 因此拆成独立的 qvix.db、权威写入方改成
    # 国内那台 VPS(见 fetcher.DB_LAYOUT)。跑批既不下载 qvix.db 也不发布它,
    # 在这儿硬算只有害处:每天往表里留一行空值,而那行空值又会被 VPS 那侧
    # "补最近15天空值"的自愈逻辑当成活儿反复重试。
    log.info("刷新指数缓存(上证/恒生/VHSI)…")
    try:
        sse = fetcher.fetch_sse_daily(force_refresh=True)
        fetcher.fetch_hsi_daily(force_refresh=True)
        fetcher.fetch_vhsi_daily(force_refresh=True)
        if sse is not None:
            s_last = sse.iloc[-1]
            log.info("   上证 %s 收 %.0f(%+.2f%%)",
                     s_last["date"], s_last["close"], s_last["pct"])
    except Exception as e:
        log.warning("   指数刷新失败(不影响基金跑批): %s", e)

    log.info("✅ 完成,总耗时 %.1f 分钟", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
