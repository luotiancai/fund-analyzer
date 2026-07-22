#!/usr/bin/env python3
"""每个交易日 14:40 邮件推送:盘中 QVIX 与恐慌阈值(滚动3年95分位)。

只报数不做判定:QVIX 取 optbbs 分钟接口的最新一笔(实时),阈值用日线
缓存(截至昨日)算,是否触发由收件人自己看。新浪实时行情仅用于判断
当天是否交易日。

跑在 GitHub Actions(见 .github/workflows/notify-qvix.yml),由外部定时
服务(如 cron-job.org)在北京 14:40 直接调 GitHub API 触发
workflow_dispatch——不用 GitHub 自己的 schedule 触发器,那个高峰期能
迟到几小时(实测撞过),脚本这边也就不用再自带"睡到点"的补偿逻辑。
邮件经 QQ 邮箱 SMTP 直发(自发自收,手机 QQ 邮箱 App 即时提醒),凭据从
环境变量读:
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


def _threshold(today: str):
    """滚动3年95分位阈值,窗口截至昨日——显式剔除今天的行,
    防止 optbbs 日线接口盘中吐出当天数据混进分位窗口。

    force_refresh=True:QVIX 收盘价源发布常常晚于 06:00 跑批(实测某次
    到 06:45 都还没出前一日收盘,得等到 9 点多),不强制刷新就一直用
    跑批时抓到的旧值,直到次日跑批才补上。这里 14:40 运行,早已过了
    发布延迟,顺带当天内自愈,不用等第二天。"""
    q = fetcher.fetch_qvix_daily(force_refresh=True)
    if q is None:
        return None
    q = q[q["date"] < today]
    if len(q) < 240:
        return None
    return float(q["close"].rolling(720, min_periods=240)
                 .quantile(0.95).iloc[-1])


def main():
    today = dt.datetime.now(_CST).strftime("%Y-%m-%d")
    quote_date = _sse_quote_date()
    if quote_date != today:
        log.info("非交易日(行情日期 %s),跳过", quote_date)
        return

    qvix, qtime = _qvix_now()
    thr = _threshold(today)
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
