#!/usr/bin/env python3
"""每个交易日 14:40 邮件推送:盘中 QVIX 与恐慌阈值(滚动3年95分位)。

只报数不做判定:QVIX 取自 fetcher.fetch_qvix_now()(上交所期权实时
行情自算,不再用 optbbs,见 qvix_calc.py 顶部说明),阈值取自
fetcher.qvix_self_history 表(同样是自算历史,截至昨日),是否触发由
收件人自己看。新浪实时行情仅用于判断当天是否交易日。

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


def _threshold():
    """滚动3年95分位阈值,从自算历史(qvix_self_history)现取最新一条。

    先调 update_qvix_self_daily() 补一次"最近一个已收盘交易日":上交所
    官方数据源发布也有延迟(实测过收盘3小时后仍未发布),06:00 跑批时
    大概率还没发布,这里 14:40 运行,早已过了发布延迟,顺带当天内自愈,
    不用等次日跑批。"""
    fetcher.update_qvix_self_daily()
    hist = fetcher.load_qvix_self_history()
    if hist is None:
        return None
    row = hist.dropna(subset=["threshold"])
    if row.empty:
        return None
    return float(row["threshold"].iloc[-1])


def main():
    today = dt.datetime.now(_CST).strftime("%Y-%m-%d")
    quote_date = _sse_quote_date()
    if quote_date != today:
        log.info("非交易日(行情日期 %s),跳过", quote_date)
        return

    qvix, qtime = fetcher.fetch_qvix_now()
    if qvix is None:
        log.error("QVIX 自算失败,拿不到值")
        sys.exit(1)
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
