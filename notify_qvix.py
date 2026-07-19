#!/usr/bin/env python3
"""每个交易日 14:40 邮件推送:盘中 QVIX 与恐慌阈值(滚动3年95分位)。

只报数不做判定:QVIX 取 optbbs 分钟接口的最新一笔(实时),阈值用日线
缓存(截至昨日)算,是否触发由收件人自己看。新浪实时行情仅用于判断
当天是否交易日。

跑在 GitHub Actions(见 .github/workflows/notify-qvix.yml),邮件经
QQ 邮箱 SMTP 直发(自发自收,手机 QQ 邮箱 App 即时提醒),凭据从环境变量读:
  SMTP_USER  发件 QQ 邮箱地址
  SMTP_PASS  QQ 邮箱 SMTP 授权码(设置→账号→开启SMTP服务→生成授权码)
  MAIL_TO    收件人,缺省同 SMTP_USER
  SMTP_HOST/SMTP_PORT  缺省 smtp.qq.com / 465
非交易日(新浪行情日期不是当天)静默退出。
"""

import datetime as dt
import logging
import os
import sys
import time
from zoneinfo import ZoneInfo

import requests

import fetcher

_CST = ZoneInfo("Asia/Shanghai")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%m-%d %H:%M:%S")
log = logging.getLogger("notify_qvix")

def _send_mail(subject: str, body: str) -> bool:
    import smtplib
    from email.mime.text import MIMEText
    from email.header import Header

    user = os.environ.get("SMTP_USER", "").strip()
    pw = os.environ.get("SMTP_PASS", "").strip()
    to = os.environ.get("MAIL_TO", "").strip() or user
    host = os.environ.get("SMTP_HOST", "smtp.qq.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    if not user or not pw:
        return False
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = user
    msg["To"] = to
    with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
        smtp.login(user, pw)
        smtp.sendmail(user, [to], msg.as_string())
    return True


def _sse_quote_date():
    """新浪实时行情的日期;非今天 → 非交易日。"""
    r = requests.get("https://hq.sinajs.cn/list=sh000001", timeout=10,
                     headers={"Referer": "https://finance.sina.com.cn",
                              "User-Agent": "Mozilla/5.0"})
    return r.text.split('"')[1].split(",")[30]


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


def _wait_until_cst():
    """WAIT_UNTIL_CST=HH:MM 时睡到北京时间该时刻(已过则立即继续)。
    给 GitHub Actions 用:cron 有分钟级抖动,提前启动、脚本内精确对时。"""
    target = os.environ.get("WAIT_UNTIL_CST", "").strip()
    if not target:
        return
    hh, mm = map(int, target.split(":"))
    now = dt.datetime.now(_CST)
    goal = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    delta = (goal - now).total_seconds()
    if delta > 0:
        log.info("等待至北京时间 %s(%.0f 秒)…", target, delta)
        time.sleep(delta)


def main():
    _wait_until_cst()
    today = dt.datetime.now(_CST).strftime("%Y-%m-%d")
    quote_date = _sse_quote_date()
    if quote_date != today:
        log.info("非交易日(行情日期 %s),跳过", quote_date)
        return

    qvix, qtime = _qvix_now()
    thr = _threshold()
    if thr is None:
        log.error("阈值计算失败")
        sys.exit(1)

    title = f"QVIX {qvix:.2f} / 阈值 {thr:.2f}"
    body = (f"{today} {qtime}\n\n"
            f"盘中 QVIX:{qvix:.2f}\n"
            f"恐慌阈值(3年95分位):{thr:.2f}\n")

    try:
        sent = _send_mail(title, body)
    except Exception as e:
        log.error("邮件发送失败: %s", e)
        sys.exit(1)
    if sent:
        log.info("邮件已发: %s", title)
    else:
        log.warning("未配置 SMTP_USER/SMTP_PASS,只打日志:\n%s\n%s", title, body)


if __name__ == "__main__":
    main()
