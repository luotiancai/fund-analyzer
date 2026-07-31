#!/usr/bin/env python3
"""在**本机**算当前 QVIX 并打印。纯本地工具, 不上传任何东西。

为什么必须在本机算:
  自算 QVIX 依赖新浪实时行情(hq.sinajs.cn), 而新浪按 IP 段限流——实测
    本机家用宽带      单请求成功率 ~100%
    Streamlit Cloud   ~50%
    腾讯云 SCF(上海)  ~50%
  两个不同云厂商、境内境外各一个都是一半, 基本排除网络路由问题, 指向机房
  IP 被整体降级(自算 QVIX 的人多了, 新浪这么干很合理)。而新浪只认
  *.sina.com.cn 的 Referer、浏览器无法伪造, 所以"让手机直接抓"也走不通。
  结论: 唯一稳定的出口就是这台机器。

线上不再有实时 QVIX:页面只保留日度自算历史和恐慌阈值(由 GitHub Actions
每天凌晨跑批更新), 实时值一律在本机看。这样线上就完全不依赖新浪那条被限流
的链路, 也不会出现"页面显示的数跟阈值不是同一把尺子"的问题。

怎么用:
  · 自动: launchd 在交易日 14:40 触发(见 ~/Library/LaunchAgents/com.fundanalyzer.qvix.plist), 失败则在 14:45/14:50/14:55 及
    15:10/16:30/19:00/22:00 补跑, 当天成功过一次就不再跑(--daily 模式)。
    这套多时点设计是因为这台机器**不是一直有网**: launchd 到点就触发、
    不管有没有网, 也不会自动重试, 只挂一个时点的话那天断网就永久错过。
    收盘后补跑仍然是对的——15:00 后档位变成"今日收盘"、计算时刻钉死在
    15:00, 晚上补跑算出来的就是当天收盘值, 跟准点跑完全一致。
  · 手动: 双击 qvix.command(或 `python3 publish_qvix.py`), 不受"当天已跑过"
    限制, 每次都现算。
  跑完会把当前 QVIX、恐慌阈值、距触发还差多少直接打在屏幕上, 同时发布到
  Release, 手机/网页刷新就能看到同一个数。
  没有定时任务——页面在境外, 没法反向叫本机算, 所以只能你想看时点一下。

只认自算值:
  算不出来就如实报错, 绝不退回 optbbs——它系统性偏高约0.18、极端日差2以上,
  而恐慌阈值是拿自算序列算的, 混用等于拿两把尺子量同一件事。
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetcher      # noqa: E402  (要先 insert path)
import qvix_core    # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%m-%d %H:%M:%S")
log = logging.getLogger("publish_qvix")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qvix_now.json")


def _published_today(date_str: str) -> bool:
    """今天是否已经成功发布过。--daily 模式用来跳过重复跑。"""
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f).get("date") == date_str
    except Exception:
        return False


def main() -> int:
    # --daily: 定时任务用。当天已经成功发布过就直接退出——launchd 挂了好几个
    # 时点(见模块说明), 目的是"当天第一次有网的时点跑一次", 不是每个时点都跑。
    daily = "--daily" in sys.argv
    if daily and _published_today(
            fetcher.datetime.now(fetcher._CST).strftime("%Y-%m-%d")):
        log.info("今天已发布过, 跳过(--daily)")
        return 0

    # 先把自算历史和恐慌阈值刷到最新。
    # 本机**没有**任何定时任务(云端那份跑批只更新 Release 里的快照, 不会回流
    # 到本地库), 所以不在这里刷, 下面打印的阈值就会一直停在最后一次手动跑批
    # 那天的值, 而且看不出来是旧的。这一步只打一次上交所官方接口(境外也通的
    # 那个)+一次滚动分位, 几秒钟, 顺带自愈最近15天的空值。
    try:
        v, note = fetcher.update_qvix_self_daily()
        if v is None:
            log.warning("自算历史刷新未成功(%s), 阈值可能不是最新", note)
    except Exception as e:
        log.warning("自算历史刷新失败(%s), 阈值可能不是最新", e)

    phase, as_of = fetcher.qvix_phase()
    now = fetcher.datetime.now(fetcher._CST)
    date_str = now.strftime("%Y-%m-%d")

    if phase == "prev":
        log.info("非交易时段(档位=prev), 跳过")
        return 0
    try:
        r = qvix_core.compute_qvix(as_of=as_of,
                                   fallback_rate=fetcher.get_risk_free_rate())
    except Exception as e:
        log.error("自算异常: %s", e)
        return 1
    if r is None or r[0] is None:
        log.error("自算失败(合约或报价拿不全), 未记录")
        return 1

    payload = {"qvix": r[0], "time": r[1], "date": date_str,
               "phase": phase, "source": "自算"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    log.info("自算 QVIX = %.2f (档位=%s, 时刻=%s)", r[0], phase, r[1])

    # 给人看的摘要:点一次就把该知道的都摆出来, 省得再去翻页面
    _PHASE_CN = {"live": "实时", "noon": "上午收盘", "close": "今日收盘"}
    thr = thr_date = None
    try:
        hist = fetcher.load_qvix_self_history()
        if hist is not None:
            row = hist.dropna(subset=["threshold"])
            if not row.empty:
                thr = float(row["threshold"].iloc[-1])
                thr_date = str(row["date"].iloc[-1])
    except Exception:
        pass
    print()
    print("  " + "─" * 42)
    print(f"    QVIX      {r[0]:>6.2f}   ({_PHASE_CN.get(phase, phase)} {r[1]})")
    if thr is not None:
        gap = r[0] - thr
        state = "🔔 已破阈值" if gap >= 0 else f"距触发还差 {-gap:.2f}"
        print(f"    恐慌阈值  {thr:>6.2f}   {state}")
        print(f"              (2年90分位, 截至 {thr_date})")
    print("  " + "─" * 42)
    print("    (本机结果, 不上传; 线上页面只有日度历史和阈值)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
