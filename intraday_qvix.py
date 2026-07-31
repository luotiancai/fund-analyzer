#!/usr/bin/env python3
"""盘中自算 QVIX → 发布成 Release 上的一个小 JSON,供云端页面读取。

为什么要这么绕:
  自算 QVIX 依赖新浪实时行情(hq.sinajs.cn),而 Streamlit Cloud(GCP)到新浪
  这条跨境链路只有约 50% 的建连成功率(实测:同一次现算里近月链拿到22个报价、
  次月链却 connect timeout;下一次反过来)。多半是共享出口 IP 被限流,加重试
  只是拿页面加载时间赌运气。
  而 GitHub Actions(Azure)上这条链路是稳的——打完 IPv4 补丁后实测自算成功、
  几秒出结果。所以让 Actions 算好、写成一个几百字节的小文件挂在 Release 上,
  页面直接读那个文件,彻底绕开云端那条抖动的链路。

为什么值得单独发一个小文件而不是塞进数据库快照:
  数据库快照 81MB,盘中每几分钟传一次不现实;这个 JSON 才几百字节。

触发方式:
  只挂 workflow_dispatch,不用 GitHub 的 schedule——那个触发器高峰期会迟到
  几十分钟到数小时甚至整次丢弃(见 notify-qvix.yml 顶部实测记录)。由外部
  定时服务(cron-job.org)在交易时段每 5~10 分钟调一次 GitHub API 触发,
  这条路径实测秒级启动。

只发布"真·自算值":
  算不出来就什么都不写(保留上一次的值), 绝不把 optbbs 的数写进这个文件。
  这个文件存在的全部意义就是"跟自算阈值同一把尺子", 混进别的源就没意义了。
"""

import json
import logging
import os
import subprocess
import sys

import fetcher
import qvix_calc

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%m-%d %H:%M:%S")
log = logging.getLogger("intraday_qvix")

OUT = "qvix_now.json"
RELEASE_TAG = "data"


def main():
    # 建一个空库就够了:这个任务不下载 81MB 的数据快照(算盘中值只需要网络),
    # 但 compute_qvix 会调 fetcher.get_risk_free_rate() 取"SHIBOR拿不到时的
    # 兜底利率", 那个函数要读 app_meta 表。没有库就是 no such table: app_meta,
    # 整个自算被这一步带挂(实测踩过)。init_db() 会在空目录里建好表结构,
    # 利率取不到时函数自己有常量兜底, 不影响结果。
    fetcher.init_db()

    phase, as_of = fetcher.qvix_phase()
    if phase == "prev":
        log.info("当前非交易时段(档位=%s),无需发布", phase)
        return 0

    r = qvix_calc.compute_qvix(as_of=as_of)
    if r is None or r[0] is None:
        log.error("自算失败,不发布(保留上一次的值)")
        return 1
    qvix, ts = r

    payload = {
        "qvix": qvix,
        # 值对应的时刻:live 档是现在, noon/close 档是被钉住的 11:30 / 15:00
        "time": ts,
        "date": fetcher.datetime.now(fetcher._CST).strftime("%Y-%m-%d"),
        "phase": phase,
        "source": "自算",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    log.info("自算 QVIX = %.2f (档位=%s, 时刻=%s)", qvix, phase, ts)

    if not os.environ.get("GH_TOKEN"):
        log.warning("没有 GH_TOKEN,只写本地文件不上传:\n%s", payload)
        return 0
    subprocess.run(["gh", "release", "upload", RELEASE_TAG, OUT, "--clobber"],
                   check=True)
    log.info("已发布到 Release(tag=%s)", RELEASE_TAG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
