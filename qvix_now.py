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
  · 自动: launchd 在工作日 14:40 跑一次, 就这一次(见
    ~/Library/LaunchAgents/com.fundanalyzer.qvix.plist), 加 --notify 把结果
    推成 macOS 通知。成功和失败**都**弹一个不会自动消失的对话框(必须点
    "知道了"才关掉)——14:40 跑的时候人多半不在电脑前, 横幅几秒就滑走了,
    等回来什么都看不到。不自动补跑: 这台机器不是一直有网, 但补跑要么打扰、
    要么无声无息, 不如让人知道"今天这次没成", 回来双击 qvix.command
    手动跑一次即可。
    节假日不用管: 脚本自己查交易日历(qvix_phase), 非交易日静默退出。
  · 手动: 双击 qvix.command, 每次都现算, 不加 --notify(终端里本来就看得见)。
  跑完把当前 QVIX、恐慌阈值、距触发还差多少直接打在屏幕上, 并写到本地
  qvix_now.json。**不上传**——线上页面只有日度自算历史和阈值。

注意 plist 里的路径是写死的绝对路径, 挪动本项目目录后必须同步改 plist 并
重新 bootstrap, 否则定时任务会静默失效(launchd 只往日志里写一行找不到文件)。
另外别把项目放在 ~/Desktop 下: launchd 起的 python3 没有 Desktop 的 TCC
权限, 会报 "Operation not permitted"。

只认自算值:
  算不出来就如实报错, 绝不退回 optbbs——它系统性偏高约0.18、极端日差2以上,
  而恐慌阈值是拿自算序列算的, 混用等于拿两把尺子量同一件事。
"""

import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetcher      # noqa: E402  (要先 insert path)
import qvix_core    # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%m-%d %H:%M:%S")
log = logging.getLogger("qvix_now")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qvix_now.json")
CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qvix_notify.conf")


def _osa(script: str, timeout: float = 20) -> None:
    """跑一段 AppleScript。失败静默——通知不出来不该影响主流程。

    timeout=None 用于 display dialog 这种要一直等人点的脚本: subprocess 的
    timeout 到点会**杀掉 osascript**, 弹窗跟着消失, 那"必须点击才能关闭"就
    成了空话(这条之前踩过——失败弹窗写着不会自动消失, 实际 20 秒就没了)。
    阻塞的代价是这个进程一直挂着等人点, 对一天只跑一次的定时任务可以接受。"""
    try:
        subprocess.run(["osascript", "-e", script], timeout=timeout,
                       capture_output=True)
    except Exception as e:
        log.debug("通知失败: %s", e)


def _as_lit(s: str) -> str:
    """把 Python 字符串转成 AppleScript 字符串字面量。
    反斜杠和双引号要转义(AppleScript 里 \\ 是转义符);裸换行它反倒是接受的,
    但拼在一起容易看不出来, 统一走这个函数省得每处各想一遍。"""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _imessage_to():
    """iMessage 收件人。刻意不写死在代码里——手机号不该进 git。
    取值顺序: 环境变量 QVIX_IMESSAGE_TO > 同目录 qvix_notify.conf(已 gitignore)
    里的 imessage_to=xxx。没配就返回 None, 表示"不发", 不是错误。"""
    to = os.environ.get("QVIX_IMESSAGE_TO", "").strip()
    if to:
        return to
    try:
        with open(CONF, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                k, _, v = line.partition("=")
                if k.strip() == "imessage_to":
                    return v.strip() or None
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("读 %s 失败: %s", os.path.basename(CONF), e)
    return None


def _send_imessage(msg: str) -> bool:
    """把结果发到自己手机。
    横幅/对话框只有人在电脑前才看得见, 而这任务 14:40 跑的时候人多半不在,
    iMessage 是唯一能追到手机上的一条。
    失败**不静默**(跟上面两个通知不同): 远程通知没送到、而你以为送到了, 比
    压根没有更糟, 所以照实记 warning。"""
    to = _imessage_to()
    if not to:
        return False
    script = ('tell application "Messages"\n'
              '  set svc to 1st service whose service type = iMessage\n'
              f'  send {_as_lit(msg)} to participant {_as_lit(to)} of svc\n'
              'end tell')
    try:
        p = subprocess.run(["osascript", "-e", script], timeout=30,
                           capture_output=True, text=True)
    except Exception as e:
        log.warning("iMessage 发送异常: %s", e)
        return False
    if p.returncode != 0:
        # 常见原因: Messages 没登录 iCloud; 或首次运行时"自动化"权限没批准
        # (系统设置 > 隐私与安全性 > 自动化, 允许 python3 控制"信息")。
        log.warning("iMessage 未发出: %s", (p.stderr or "").strip())
        return False
    # 措辞留神: 返回 0 只说明 Messages **收下**了这条, 不代表已送达。真断网时
    # 它会转成待发/发送失败, 退出码照样是 0。这里没法从 osascript 拿到投递结果。
    log.info("iMessage 已交给 Messages 发送 -> %s", to)
    return True


def _notify_ok(title: str, msg: str) -> None:
    """成功: 跟失败一样弹**不会自动消失**的对话框, 必须点"知道了"才关掉。

    原来成功走的是横幅(display notification), 几秒就滑走了——14:40 跑的时候
    人多半不在电脑前, 等回来横幅早没了, 于是"今天到底跑没跑、QVIX 是多少"
    只能自己去翻 qvix_now.json。改成对话框后, 不管人什么时候回到电脑前, 那
    一天的结果都还挂在屏幕上等着。
    iMessage 照发, 且必须在弹窗**之前**发(理由同 _notify_fail: dialog 会一直
    阻塞)。"""
    t, m = title.replace('"', "'"), msg.replace('"', "'")
    _send_imessage(f"{t}\n{m}")
    _osa(f'display dialog "{m}" with title "{t}" '
         'buttons {"知道了"} default button 1', timeout=None)


def _notify_fail(msg: str) -> None:
    """失败: 对话框, **不会自动消失**, 人回到电脑前必然看见。
    横幅几秒就没了, 而这个任务一天只跑一次, 错过就是错过。
    也发一条 iMessage, 但别指望它兜底: 失败最常见的原因就是没网, 那 iMessage
    同样发不出去(会记 warning)。对话框才是失败路径上真正可靠的那一条。"""
    m = msg.replace('"', "'")
    # 必须发在弹窗**之前**: display dialog 会一直阻塞到有人点"知道了", 放后面
    # 的话短信要等你回到电脑前才发得出去, 正好把这条通道的意义抵消掉。
    _send_imessage("QVIX 定时任务失败\n" + msg)
    _osa('display dialog "' + m + '" with title "QVIX 定时任务失败" '
         'buttons {"知道了"} default button 1 with icon caution', timeout=None)


def main() -> int:
    # --notify: 定时任务用, 把结果推成 macOS 通知。手动跑时不加——终端里
    # 本来就看得见, 再弹一次是噪音。
    notify = "--notify" in sys.argv

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
        # 周末/节假日/开盘前:本来就没有实时值可算, 静默退出, 不打扰。
        log.info("非交易时段(档位=prev), 跳过")
        return 0
    try:
        r = qvix_core.compute_qvix(as_of=as_of,
                                   fallback_rate=fetcher.get_risk_free_rate())
    except Exception as e:
        log.error("自算异常: %s", e)
        if notify:
            _notify_fail(f"算不出来: {type(e).__name__}\n\n"
                         "多半是当时没网。回到电脑前双击 qvix.command 手动跑一次即可。")
        return 1
    if r is None or r[0] is None:
        log.error("自算失败(合约或报价拿不全), 未记录")
        if notify:
            _notify_fail("算不出来: 合约或报价没拿全。\n\n"
                         "多半是当时没网, 或新浪限流。回到电脑前双击 "
                         "qvix.command 手动跑一次即可。")
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

    if notify:
        gap_txt = ""
        if thr is not None:
            gap = r[0] - thr
            gap_txt = ("  🔔 已破阈值" if gap >= 0 else f"  距触发 {-gap:.2f}")
        _notify_ok(f"QVIX {r[0]:.2f}",
                   f"{_PHASE_CN.get(phase, phase)} {r[1]}"
                   + (f"   阈值 {thr:.2f}{gap_txt}" if thr is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
