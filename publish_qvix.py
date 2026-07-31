#!/usr/bin/env python3
"""在**本机**算盘中 QVIX,把结果发布成 Release 上的一个小 JSON,供云端页面读取。

为什么必须在本机算:
  自算 QVIX 依赖新浪实时行情(hq.sinajs.cn), 而新浪按 IP 段限流——实测
    本机家用宽带      单请求成功率 ~100%
    Streamlit Cloud   ~50%
    腾讯云 SCF(上海)  ~50%
  两个不同云厂商、境内境外各一个都是一半, 基本排除网络路由问题, 指向机房
  IP 被整体降级(自算 QVIX 的人多了, 新浪这么干很合理)。而新浪只认
  *.sina.com.cn 的 Referer、浏览器无法伪造, 所以"让手机直接抓"也走不通。
  结论: 唯一稳定的出口就是这台机器。

为什么发小文件而不是塞进数据库快照:
  数据库快照 81MB, 盘中每几分钟传一次不现实; 这个 JSON 才几百字节。

怎么用:
  双击项目目录下的 qvix.command(或命令行 `python3 publish_qvix.py`)。
  跑完会把当前 QVIX、恐慌阈值、距触发还差多少直接打在屏幕上, 同时发布到
  Release, 手机/网页刷新就能看到同一个数。
  没有定时任务——页面在境外, 没法反向叫本机算, 所以只能你想看时点一下。

只发布"真·自算值":
  算不出来就什么都不写, 保留上一次的值, 绝不把 optbbs 的数写进这个文件——
  这个文件存在的全部意义就是"跟自算阈值同一把尺子", 混进别的源就没意义了。
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
log = logging.getLogger("publish_qvix")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qvix_now.json")
RELEASE_TAG = "data"
ASSET = "qvix_now.json"


def main() -> int:
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
        log.error("自算异常, 不发布(保留上一次的值): %s", e)
        return 1
    if r is None or r[0] is None:
        log.error("自算失败, 不发布(保留上一次的值)")
        return 1

    payload = {"qvix": r[0], "time": r[1], "date": date_str,
               "phase": phase, "source": "自算"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    log.info("自算 QVIX = %.2f (档位=%s, 时刻=%s)", r[0], phase, r[1])

    try:
        subprocess.run(["gh", "release", "upload", RELEASE_TAG, OUT, "--clobber"],
                       check=True, capture_output=True, timeout=120)
    except Exception as e:
        log.error("上传失败: %s", getattr(e, "stderr", e))
        return 1
    log.info("已发布到 Release(tag=%s, 资产=%s)", RELEASE_TAG, ASSET)

    # 给人看的摘要:点一次就把该知道的都摆出来, 省得再去翻页面
    _PHASE_CN = {"live": "实时", "noon": "上午收盘", "close": "今日收盘"}
    thr = None
    try:
        hist = fetcher.load_qvix_self_history()
        if hist is not None:
            row = hist.dropna(subset=["threshold"])
            if not row.empty:
                thr = float(row["threshold"].iloc[-1])
    except Exception:
        pass
    print()
    print("  " + "─" * 42)
    print(f"    QVIX      {r[0]:>6.2f}   ({_PHASE_CN.get(phase, phase)} {r[1]})")
    if thr is not None:
        gap = r[0] - thr
        state = "🔔 已破阈值" if gap >= 0 else f"距触发还差 {-gap:.2f}"
        print(f"    恐慌阈值  {thr:>6.2f}   {state}")
    print("  " + "─" * 42)
    print("    已发布, 手机/网页刷新即可看到同一个数")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
