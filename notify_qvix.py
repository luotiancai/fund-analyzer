#!/usr/bin/env python3
"""每个交易日 14:40 微信推送:盘中 QVIX vs 恐慌阈值(滚动3年95分位)+ 触发状态。

设计给收盘前的决策窗口用:QVIX 取 optbbs 分钟接口的最新一笔(实时),
阈值用日线缓存(截至昨日)算,大盘当日涨跌取新浪实时行情——三者拼出
「QVIX>阈值 且 当日下跌,或单日≤-5%」的 B 点触发判定,15:00 前来得及下单。

SendKey 读取顺序:环境变量 SERVERCHAN_KEY → 同目录 .serverchan_key 文件。
非交易日(新浪行情日期不是今天)静默退出,cron 直接排工作日即可:
  40 14 * * 1-5  cd /path/to/fund-analyzer && .venv/bin/python notify_qvix.py >> notify.log 2>&1
"""

import datetime as dt
import logging
import os
import sys

import requests

import fetcher

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%m-%d %H:%M:%S")
log = logging.getLogger("notify_qvix")

_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".serverchan_key")


def _sendkey():
    key = os.environ.get("SERVERCHAN_KEY", "").strip()
    if not key and os.path.exists(_KEY_FILE):
        key = open(_KEY_FILE).read().strip()
    return key


def _sse_realtime():
    """(日期, 当日涨跌%) from 新浪实时;日期非今天 → 非交易日。"""
    r = requests.get("https://hq.sinajs.cn/list=sh000001", timeout=10,
                     headers={"Referer": "https://finance.sina.com.cn",
                              "User-Agent": "Mozilla/5.0"})
    f = r.text.split('"')[1].split(",")
    prev_close, current, quote_date = float(f[2]), float(f[3]), f[30]
    return quote_date, current, (current / prev_close - 1) * 100


def _qvix_now():
    """盘中最新 QVIX(optbbs 分钟接口最后一笔)。"""
    d = fetcher.ak.index_option_50etf_min_qvix()
    d = d.dropna(subset=["qvix"])
    last = d.iloc[-1]
    return float(last["qvix"]), str(last["time"])


def _threshold():
    """滚动3年95分位阈值(日线缓存截至昨日)。"""
    q = fetcher.fetch_qvix_daily()
    if q is None or len(q) < 240:
        return None
    return float(q["close"].rolling(720, min_periods=240)
                 .quantile(0.95).iloc[-1])


def main():
    today = dt.date.today().strftime("%Y-%m-%d")
    quote_date, sse_now, sse_pct = _sse_realtime()
    if quote_date != today:
        log.info("非交易日(行情日期 %s),跳过", quote_date)
        return

    qvix, qtime = _qvix_now()
    thr = _threshold()
    if thr is None:
        log.error("阈值计算失败")
        sys.exit(1)

    triggered = (qvix > thr and sse_pct < 0) or sse_pct <= -5.0
    status = "🔔 B点触发!" if triggered else "未触发"
    title = f"QVIX {qvix:.2f} / 阈值 {thr:.2f} · {status}"
    body = (f"**{today} {qtime}**\n\n"
            f"- 盘中 QVIX:**{qvix:.2f}**\n"
            f"- 恐慌阈值(3年95分位):**{thr:.2f}**\n"
            f"- 上证:{sse_now:.0f}({sse_pct:+.2f}%)\n"
            f"- 判定:QVIX{'>' if qvix > thr else '≤'}阈值,"
            f"大盘{'下跌' if sse_pct < 0 else '上涨'} → **{status}**\n\n"
            + ("触发条件满足:按规则看前一日榜单选国内 C 类冠军,15:00 前下单。"
               if triggered else "不满足触发条件,继续等待。"))

    key = _sendkey()
    if not key:
        log.warning("未配置 SendKey(环境变量 SERVERCHAN_KEY 或 .serverchan_key 文件),只打日志:\n%s\n%s", title, body)
        return
    r = requests.post(f"https://sctapi.ftqq.com/{key}.send",
                      data={"title": title, "desp": body}, timeout=15)
    ok = r.json().get("code") == 0
    log.info("推送%s: %s", "成功" if ok else f"失败 {r.text[:100]}", title)


if __name__ == "__main__":
    main()
