#!/usr/bin/env python3
"""每个交易日 14:40 邮件推送:盘中 QVIX 与恐慌阈值(滚动2年90分位)。

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
非交易日静默退出(优先看新浪行情日期,新浪不可达时退回交易日历)。
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


def _is_trading_today() -> bool:
    """今天是不是交易日。优先看新浪实时行情的日期(它只在交易日更新到当天,
    最直接);新浪不可达时退回交易日历。

    以前这里裸调新浪、没有任何异常捕获,而 hq.sinajs.cn 从境外机房是 IP 层
    不可达的([Errno 101] Network is unreachable)。它又是 main() 的第一步,
    所以新浪一断整个推送任务就崩:2026-07-30 实际发生过,那天没发出任何
    QVIX 提醒,收件箱里只有 GitHub 的构建失败通知——看着像技术故障、很容易
    划走,可万一那天恰好破了阈值就完全错过了。
    QVIX 取值本身有 optbbs 兜底、境外拿得到,不该被这道坎卡死。"""
    today = dt.datetime.now(_CST).strftime("%Y-%m-%d")
    try:
        r = requests.get("https://hq.sinajs.cn/list=sh000001", timeout=10,
                         headers={"Referer": "https://finance.sina.com.cn",
                                  "User-Agent": "Mozilla/5.0"})
        return r.text.split('"')[1].split(",")[30] == today
    except Exception as e:
        log.warning("新浪行情日期取不到(%s),改用交易日历判断", e)
        return fetcher.is_trading_day()


def _threshold():
    """滚动2年90分位阈值,从自算历史(qvix_self_history)现取最新一条。

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
    if not _is_trading_today():
        log.info("非交易日,跳过")
        return

    qvix, qtime, qsrc = fetcher.fetch_qvix_now()
    if qvix is None:
        log.error("QVIX 盘中取值失败(自算+optbbs 都没拿到)")
        sys.exit(1)
    thr = _threshold()
    if thr is None:
        log.error("阈值计算失败")
        sys.exit(1)

    title = f"QVIX {qvix:.2f} / 阈值 {thr:.2f}"
    body = (f"{today} {qtime}\n\n"
            f"盘中 QVIX:{qvix:.2f}（{qsrc}）\n"
            f"恐慌阈值(2年90分位):{thr:.2f}\n")

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
