"""
回测: QVIX恐慌信号买入 + 双止损(基金回撤控制线 / 大盘回撤线)
- 买入: QVIX > 3年95分位阈值, 且资金可用(空仓或当天恰好卖出)
- 标的: 前一交易日近3月冠军(C类全市场), 冠军排名复用 fetcher.compute_
  metrics_asof——与 app.py「基金列表」页"截至日期"筛选完全同口径(按日
  收益率连乘, 正确处理分红除权, 自带单日|收益率|>30%异常值过滤), 而非
  简化的 end_nav/anchor_nav-1(曾把 2020-07-16 算错成广发医疗保健夺冠,
  实际应为汇丰晋信智造先锋, 已交叉验证修正)
- 卖出: 基金回撤控制线(买入日阈值/5×波动率比值) 或 大盘回撤线(买入日阈值/5),
  逐交易日检查, 先到先卖(双线在买入日锁定, 与 app.py 复盘口径一致)
  波动率比值 = 基金日收益率std / 大盘日收益率std(纯波动对比, 不按相关系数加权)
"""
import sys, os, time, sqlite3, io
import numpy as np
import pandas as pd
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetcher

DB = os.path.expanduser("~/.local/share/fund-analyzer/fund_cache.db")

# 已核实的净值异常(直连东财源头核对过, 不是本地缓存问题, 但明显非真实
# 市场收益): 014939 同泰产业升级混合C 2025-03-31 单位净值单日 +68.7%
# (0.9630→1.6249), 累计净值同步跳升(排除分红除权), 前后走势平稳无回撤/
# 打回迹象(判断为永久性净值断层, 而非孤立坏点), 但一只普通混合型偏股
# 基金不可能真实单日上涨 68.7%。原因未知(净值更正公告或数据错误均有
# 可能), 回测中把该日跳变的比例从该日起从整条净值序列剔除, 避免虚假
# 拉高 2025-04-07 的冠军排名(未剔除时该基金显示 +88.76%近3月涨幅夺冠,
# 剔除后仅 +11.87%, 真实冠军应为鹏华碳中和主题混合C +46.64%)。
NAV_ANOMALIES = {
    "014939": [(pd.Timestamp("2025-03-31"), 1.6249 / 0.9630)],
}

# ── 阈值组合回测参考快照(2026-07-24, 数据/规则变了要重跑, 别当成结论)──
# QVIX 只用2018年起数据(2015-2017判定为期权刚上市时流动性薄、噪声偏
# 大, 已从 qvix_self_history 整段删除, 见 fetcher.py); 信号池下限按
# fund_nav_daily 实测起点(2020-01-02)+91天动态算, 不再借宽限期凑数
# (见 run_backtest 里 _signal_floor 那段)。已排除海外/QDII基金。
# 命令: python3 backtest_qvix.py --window 490 --pct 0.90 (2年90%类推,
#       1年=250天/2年=490天/3年=720天)
#
#   窗口  分位   笔数  胜率        累计收益(费后复利)  平均持有  平均收益(费后)
#   2年   90%    11   7/11=63.6%   +139.68%           59天      +9.54%
#   2年   95%    7    5/7=71.4%    +78.41%            77天      +9.59%
#   3年   90%    9    6/9=66.7%    +105.11%           59天      +9.79%
#   3年   95%    4    4/4=100.0%   +99.89%            88天      +19.79%
#
# ─ 2年90%(11笔) ─ 买入日/冠军/类型/波动率比值/阈值/回撤线/大盘线/近3月涨幅/卖出日/收益(费后)/手续费%/期间最高/最大回撤/同期上证/卖出原因
# 2020-07-06 广发医疗保健股票C(009163)      股票型      2.12 29.11 12.34 5.82  +48.39%  2020-07-16 +4.05%(费后+3.55%)  0.5 +10.9%  6.2% -3.7%  大盘7.0%>=5.8%
# 2020-07-16 汇丰晋信智造先锋股票C(001644)    股票型      1.85 29.60 10.95 5.92  +57.94%  2020-09-09 -3.20%              0.0 +10.5% 12.4% +1.4%  基金12.4%>=11.0%
# 2022-04-25 国泰中证煤炭ETF联接C(008280)     指数型-股票  1.84 26.33  9.69 5.27  +14.33%  2022-06-20 +12.88%             0.0 +25.9% 10.4% +13.2% 基金10.4%>=9.7%
# 2022-10-24 创金合信专精特新股票发起C(014737) 股票型      2.48 23.24 11.53 4.65  +23.74%  2022-12-16 -10.00%             0.0  +4.0% 13.4% +6.4%  基金13.4%>=11.5%
# 2023-08-28 易方达中证全指证券公司ETF联接C(012700) 指数型-股票 2.05 22.28 9.13 4.46 +10.51% 2023-10-19 -7.70%            0.0  +0.0%  8.2% -3.0%  大盘5.4%>=4.5%
# 2024-02-05 万家精选混合C(015566)           混合型-偏股  1.61 22.28  7.17 4.46  +11.95%  2024-03-18 +8.66%              0.0 +17.7%  7.7% +14.2% 基金7.7%>=7.2%
# 2024-09-26 方正富邦中证保险C(018099)        指数型-股票  1.51 20.45  6.18 4.09  +18.21%  2024-10-09 +7.11%(费后+6.61%)  0.5 +17.9%  9.1% +8.6%  基金9.1%>=6.2%
# 2024-10-09 广发北证50成份指数C(017513)      指数型-股票  3.11 20.51 12.75 4.10  +63.61%  2024-11-18 +36.76%             0.0 +46.8% 11.6% +2.0%  大盘4.2%>=4.1%
# 2024-11-18 中欧北证50成份指数发起C(021299)   指数型-股票  2.86 21.77 12.45 4.35 +120.08%  2024-12-16 -8.90%(费后-9.40%)  0.5  +5.1% 13.3% +1.9%  基金13.3%>=12.5%
# 2025-04-07 鹏华碳中和主题混合C(016531)       混合型-偏股  4.88 21.42 20.91 4.28  +54.70%  2025-11-20 +27.72%             0.0 +62.5% 21.4% +26.9% 基金21.4%>=20.9%
# 2026-03-23 广发远见智选混合C(016874)        混合型-偏股  3.82 22.54 17.22 4.51  +57.21%  2026-06-05 +39.10%             0.0 +50.6% 12.6% +5.6%  大盘5.1%>=4.5%
#
# ─ 2年95%(7笔) ─
# 2020-07-06 广发医疗保健股票C(009163)      股票型      2.12 30.89 13.10 6.18  +48.39%  2020-07-16 +4.05%(费后+3.55%)  0.5 +10.9%  6.2% -3.7%  大盘7.0%>=6.2%
# 2020-07-16 汇丰晋信智造先锋股票C(001644)    股票型      1.85 31.20 11.54 6.24  +57.94%  2020-09-09 -3.20%              0.0 +10.5% 12.4% +1.4%  基金12.4%>=11.5%
# 2022-10-31 创金合信专精特新股票发起C(014737) 股票型      2.31 24.12 11.14 4.82  +21.70%  2022-11-30 -7.61%              0.0  +4.1% 11.3% +8.9%  基金11.3%>=11.1%
# 2024-02-05 万家精选混合C(015566)           混合型-偏股  1.61 23.29  7.50 4.66  +11.95%  2024-03-18 +8.66%              0.0 +17.7%  7.7% +14.2% 基金7.7%>=7.5%
# 2024-09-27 西部利得新动力混合C(673073)      混合型-灵活  2.35 22.01 10.34 4.40  +25.77%  2024-10-09 +3.03%(费后+2.53%)  0.5  +9.5%  5.9% +5.5%  大盘6.6%>=4.4%
# 2024-10-09 广发北证50成份指数C(017513)      指数型-股票  3.11 22.22 13.82 4.44  +63.61%  2024-11-22 +36.90%             0.0 +46.8% 11.6% +0.3%  大盘5.9%>=4.4%
# 2025-04-07 鹏华碳中和主题混合C(016531)       混合型-偏股  4.88 25.93 25.31 5.19  +54.70%  2026-03-20 +26.31%             0.0 +68.6% 25.1% +27.8% 大盘5.4%>=5.2%
#
# ─ 3年90%(9笔) ─
# 2022-04-25 国泰中证煤炭ETF联接C(008280)     指数型-股票  1.84 26.34  9.69 5.27  +14.33%  2022-06-20 +12.88%             0.0 +25.9% 10.4% +13.2% 基金10.4%>=9.7%
# 2024-02-05 万家精选混合C(015566)           混合型-偏股  1.61 22.28  7.17 4.46  +11.95%  2024-03-18 +8.66%              0.0 +17.7%  7.7% +14.2% 基金7.7%>=7.2%
# 2024-09-27 西部利得新动力混合C(673073)      混合型-灵活  2.35 21.82 10.25 4.36  +25.77%  2024-10-09 +3.03%(费后+2.53%)  0.5  +9.5%  5.9% +5.5%  大盘6.6%>=4.4%
# 2024-10-09 广发北证50成份指数C(017513)      指数型-股票  3.11 21.83 13.58 4.37  +63.61%  2024-11-22 +36.90%             0.0 +46.8% 11.6% +0.3%  大盘5.9%>=4.4%
# 2024-11-22 中欧北证50成份指数发起C(021299)   指数型-股票  2.87 22.75 13.06 4.55 +130.89%  2024-12-17 -11.61%(费后-12.11%) 0.5  +1.9% 13.3% +2.9% 基金13.3%>=13.1%
# 2025-04-07 鹏华碳中和主题混合C(016531)       混合型-偏股  4.88 22.54 22.00 4.51  +54.70%  2025-11-21 +28.04%             0.0 +62.5% 21.4% +23.8% 大盘4.8%>=4.5%
# 2026-02-02 永赢高端装备智选混合发起C(015790) 混合型-偏股  5.02 20.84 20.92 4.17  +73.87%  2026-03-19 -9.77%              0.0  +4.2% 14.9% -0.2%  大盘4.2%>=4.2%
# 2026-03-23 广发远见智选混合C(016874)        混合型-偏股  3.82 20.86 15.94 4.17  +57.21%  2026-06-01 +32.34%             0.0 +50.6% 12.6% +6.4%  大盘4.4%>=4.2%
# 2026-06-23 财通集成电路产业股票C(006503)     股票型      3.56 20.89 14.87 4.18 +150.19%  2026-07-06 -10.84%(费后-11.34%) 0.5  +7.3% 16.9% -1.6% 基金16.9%>=14.9%
#
# ─ 3年95%(4笔) ─
# 2024-02-05 万家精选混合C(015566)           混合型-偏股  1.61 23.25  7.49 4.65  +11.95%  2024-03-18 +8.66%              0.0 +17.7%  7.7% +14.2% 基金7.7%>=7.5%
# 2024-09-27 西部利得新动力混合C(673073)      混合型-灵活  2.35 22.77 10.70 4.55  +25.77%  2024-10-09 +3.03%(费后+2.53%)  0.5  +9.5%  5.9% +5.5%  大盘6.6%>=4.6%
# 2024-10-09 广发北证50成份指数C(017513)      指数型-股票  3.11 22.84 14.21 4.57  +63.61%  2024-11-22 +36.90%             0.0 +46.8% 11.6% +0.3%  大盘5.9%>=4.6%
# 2025-04-07 鹏华碳中和主题混合C(016531)       混合型-偏股  4.88 24.28 23.70 4.86  +54.70%  2025-12-16 +31.06%             0.0 +62.5% 21.4% +23.5% 大盘5.1%>=4.9%


def _apply_nav_anomaly(code, date, nav):
    """对已知异常基金, 剔除 date 当天起的净值跳变(返回修正后的 nav)."""
    for anomaly_date, ratio in NAV_ANOMALIES.get(code, []):
        if pd.Timestamp(date) >= anomaly_date:
            nav = nav / ratio
    return nav


def get_conn():
    return sqlite3.connect(DB, check_same_thread=False, timeout=30)


def load_cached_json(conn, key):
    row = conn.execute(
        "SELECT data FROM index_daily_cache WHERE key=?", (key,)).fetchone()
    if not row:
        return pd.DataFrame()
    df = pd.read_json(io.StringIO(row[0]), orient="split",
                      dtype=False, convert_dates=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


# 净值僵化-补涨检测参数: 排名窗口内连续 STALE_MIN_RUN 天|日收益率|<
# STALE_FLAT_EPS(净值近乎不动, 不像有股票仓位的基金该有的波动), 紧接着
# 单日|收益率|>STALE_JUMP_THRESH 的补涨/补跌跳变——判断为净值长期未按
# 市值更新、事后集中补记(如 002631 2024-01-30 前连续9个交易日涨跌幅
# 全部<0.05%, 随后单日+15.66%补涨, 直连东财源头核对非缓存问题, 但该
# 基金全历史波动率其实正常, 只在这段窗口反常, 原因未知)。命中的基金
# 从冠军候选池整体剔除, 而不是像 014939 那样单点修正——这类模式此前
# 未必只出现过一次, 与 effective_daily_ret 的单日>30%硬过滤是两种不同
# 场景, 互不替代。
STALE_FLAT_EPS = 0.0005
STALE_MIN_RUN = 5
STALE_JUMP_THRESH = 0.08


def _has_stale_catchup(conn, code, window_start, window_end):
    """排名窗口 [window_start, window_end] 内是否出现净值僵化-补涨模式."""
    df = pd.read_sql_query(
        "SELECT date, nav FROM fund_nav_daily WHERE code=? AND date>=? AND date<=? ORDER BY date",
        conn, params=(code, window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
    if len(df) < STALE_MIN_RUN + 1:
        return False
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    ret = df["nav"].pct_change().dropna().values
    run = 0
    for r in ret:
        if abs(r) < STALE_FLAT_EPS:
            run += 1
            continue
        if run >= STALE_MIN_RUN and abs(r) > STALE_JUMP_THRESH:
            return True
        run = 0
    return False


def find_champion_on_date(conn, asof_date, exclude_codes=None):
    """找 asof_date 当天视角下的近3月冠军(排除 exclude_codes). 返回 (code, ret_3m).

    复用 fetcher.compute_metrics_asof——按日收益率连乘计算区间收益(正确
    处理分红除权,不会像 end_nav/anchor_nav-1 那样被除权日的净值跳水拉低),
    且自带单日|收益率|>30%异常值过滤(effective_daily_ret, 已覆盖 014939
    2025-03-31 断层等已知案例)。该函数按 asof 严格早于当日截断数据
    (T日决策只能看到T-1日收盘净值), 与 app.py「基金列表」页"截至日期"
    筛选完全同口径, 已用 2020-07-16 001644 vs 009163 交叉验证过。

    exclude_codes 用于剔除 QDII 等跟踪境外市场、与大盘弱相关的基金——
    策略买入逻辑建立在"大盘恐慌信号→买入国内动量最强标的"上, QDII 收益
    与 QVIX/大盘走势脱钩, 选入冠军池会削弱大盘回撤线对该笔仓位的意义。
    命中净值僵化-补涨模式(见 _has_stale_catchup)的基金也一并跳过, 逐个
    往下找下一名, 直到选出一个数据正常的真实冠军。
    """
    metrics = fetcher.compute_metrics_asof(asof_date, cols={"ret_3m"})
    if not metrics:
        return None, 0
    candidates = {
        c: m["ret_3m"] for c, m in metrics.items()
        if m.get("ret_3m") is not None
        and (not exclude_codes or c not in exclude_codes)
    }
    if not candidates:
        return None, 0

    end = pd.Timestamp(asof_date)
    window_end = end - timedelta(days=1)
    window_start = end - timedelta(days=101)

    for code in sorted(candidates, key=candidates.get, reverse=True):
        if _has_stale_catchup(conn, code, window_start, window_end):
            continue
        return code, round(candidates[code], 2)
    return None, 0


def compute_beta(conn, sse_df, code, buy_date):
    """买入日前91天窗口的波动率比值(基金日收益率std / 大盘日收益率std).

    纯波动对比,不按相关系数加权——目的是衡量基金相对大盘的振幅倍数,
    而非系统性风险敞口(标准 Beta 会被低相关性拉低,弱化真实波动)。
    """
    end = pd.Timestamp(buy_date)
    start = end - timedelta(days=91)

    nav_df = pd.read_sql_query(
        "SELECT date, nav FROM fund_nav_daily WHERE code=? AND date>=? AND date<? ORDER BY date",
        conn, params=(code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    if len(nav_df) < 20:
        return 1.0
    nav_df["nav"] = pd.to_numeric(nav_df["nav"], errors="coerce")
    f_ret = nav_df["nav"].pct_change().dropna().values
    # 剔除净值重置/份额折算等技术性跳变(单日|收益率|>35%非真实市场波动)。
    # 阈值定在35%: 北交所个股涨跌幅上限±30%, 021299 2024-09-30/10-08 曾真实
    # 单日+21.9%/+25.2%(924行情后北证50补涨, 广发017513同日+22.5%/+25.6%
    # 交叉验证非脏数据), 20%阈值曾把这两天真实波动误判成技术性跳变, 把该
    # 笔波动率比值从~2.6压到1.19。35%仍能拦住014939那种+68.7%的净值断层。
    f_ret = f_ret[np.abs(f_ret) <= 0.35]

    sse_w = sse_df[(sse_df["date"] >= start) & (sse_df["date"] < end)]
    m_ret = sse_w["close"].pct_change().dropna().values

    if len(f_ret) < 20 or len(m_ret) < 20:
        return 1.0

    m_std = np.std(m_ret)
    if m_std == 0 or np.isnan(m_std):
        return 1.0
    return round(float(np.std(f_ret) / m_std), 2)


def get_fund_nav_after(conn, code, from_date):
    """获取基金从 from_date 起的净值序列 [(date, nav), ...] (已按 NAV_ANOMALIES 修正)"""
    rows = conn.execute(
        "SELECT date, nav FROM fund_nav_daily WHERE code=? AND date>=? ORDER BY date",
        (code, from_date.strftime("%Y-%m-%d"))).fetchall()
    return [(pd.Timestamp(r[0]), _apply_nav_anomaly(code, r[0], float(r[1])))
            for r in rows if r[1]]


def run_backtest(window: int = 720, pct: float = 0.95, minp_ratio: float = 0.97):
    """window=滚动窗口(交易日), pct=分位数, minp_ratio=窗口内至少要有
    多大比例的有效数据才出阈值(容错缺失日,同 fetcher.update_qvix_self_daily
    的 700/720 那套道理)。默认 720/0.95 是当前线上在用的参数。"""
    conn = get_conn()

    # Load fund names and types from JSON cache
    fund_names = {}
    fund_types = {}
    raw = conn.execute("SELECT data FROM fund_list").fetchone()
    if raw and raw[0]:
        import json as _json
        items = _json.loads(raw[0])
        for item in items:
            c = item.get("code", "")
            fund_names[c] = item.get("name", c)
            fund_types[c] = item.get("type", "")

    # QDII/海外指数型跟踪境外市场, 与 QVIX/大盘恐慌-反弹逻辑脱钩, 排除出
    # 冠军候选池(如"指数型-海外股票"的广发道琼斯石油指数C, 之前只过滤
    # "QDII"字样漏掉了这类, 类型字符串里没有QDII三个字但同样跟踪境外)
    qdii_codes = {c for c, t in fund_types.items() if ("QDII" in t or "海外" in t)}

    # Load QVIX —— 自算(qvix_self_history,上交所官方期权风险指标反推,
    # 不再是 optbbs 的 index_daily_cache)。阈值按传入的 window/pct 现算,
    # 不用表里预存的那一列(那一列固定是线上用的720/0.95)。
    print(f"加载数据... (窗口={window}天, 分位={pct})")
    qvix = fetcher.load_qvix_self_history()
    qvix = qvix.rename(columns={"qvix": "close"})
    qvix["date"] = pd.to_datetime(qvix["date"])
    qvix["close"] = pd.to_numeric(qvix["close"], errors="coerce")
    qvix = qvix.sort_values("date").reset_index(drop=True)
    minp = int(window * minp_ratio)
    qvix["thr"] = qvix["close"].rolling(window, min_periods=minp).quantile(pct)

    sse = load_cached_json(conn, "sse")
    sse["close"] = pd.to_numeric(sse["close"], errors="coerce")
    sse = sse.sort_values("date").reset_index(drop=True)

    # 冠军排名要看"近3月涨幅",净值库(fund_nav_daily)实测最早只到
    # 2020-01-02(不是曾经以为的2018-01,那是旧注释,已过时)——早于
    # "库起点+91天"的信号日虽然 fetcher.ANCHOR_GRACE_DAYS=10 的宽限期
    # 偶尔能让某天擦边通过(2020-03-24 精确压线10天),但那是"近3月"窗口
    # 被截断到只有81天的假数据,不能真实代表冠军排名,直接从信号池剔除,
    # 不指望宽限期兜底(宽限期本是为容忍个别基金上市日不对齐设计的,不
    # 该被整个数据库的起点边界借用)。
    _nav_floor = conn.execute("SELECT MIN(date) FROM fund_nav_daily").fetchone()[0]
    _signal_floor = pd.Timestamp(_nav_floor) + pd.Timedelta(days=91)
    print(f"净值库起点 {_nav_floor}, 近3月冠军窗口最早可信信号日 {_signal_floor.date()}")

    # Signal days: QVIX > threshold
    signals = qvix[(qvix["close"] > qvix["thr"]) & (qvix["thr"].notna())]
    signals = signals[signals["date"] >= _signal_floor]
    print(f"信号日(QVIX > 阈值): {len(signals)} 天")
    signal_map = {row["date"]: row["thr"] for _, row in signals.iterrows()}

    trades = []
    position = None

    # 逐交易日走: 持仓时每天检查双止损线, 空仓(或当天刚卖出)遇信号日则买入
    all_days = sse[sse["date"] >= _signal_floor]

    for _, day_row in all_days.iterrows():
        day = day_row["date"]
        day_str = day.strftime("%Y-%m-%d")
        sse_close = float(day_row["close"])

        # ── Step 1: 持仓时逐日检查双止损 ──
        if position is not None:
            nav_series = position["nav_map"]
            current_nav = nav_series.get(day, position.get("last_nav"))
            if current_nav is None:
                continue
            position["last_nav"] = current_nav

            position["peak_nav"] = max(position["peak_nav"], current_nav)
            position["min_nav"] = min(position["min_nav"], current_nav)
            position["peak_sse"] = max(position["peak_sse"], sse_close)
            sse_dd = (position["peak_sse"] - sse_close) / position["peak_sse"] * 100
            fund_dd = (position["peak_nav"] - current_nav) / position["peak_nav"] * 100
            position["max_dd"] = max(position.get("max_dd", 0.0), fund_dd)

            sell_reason = None
            if fund_dd >= position["fund_dd_limit"]:
                sell_reason = f"基金{fund_dd:.1f}%>={position['fund_dd_limit']:.1f}%"
            elif sse_dd >= position["sse_dd_limit"]:
                sell_reason = f"大盘{sse_dd:.1f}%>={position['sse_dd_limit']:.1f}%"

            if sell_reason:
                ret_pct = (current_nav / position["buy_nav"] - 1) * 100
                hold_days = (day - position["buy_date"]).days
                # 同期上证
                sse_ret = (sse_close / position["buy_sse"] - 1) * 100 \
                    if position["buy_sse"] else 0
                # 期间最大回撤(逐日沿途峰值口径)
                max_dd = position.get("max_dd", 0.0)
                code = position["code"]
                name = fund_names.get(code, code)
                trades.append({
                    "买入日": position["buy_date"].strftime("%Y-%m-%d"),
                    "冠军(C类全市场,按前一交易日榜单)": f"{name} ({code})",
                    "类型": fund_types.get(code, ""),
                    "波动率比值(近3月)": position["beta"],
                    "恐慌阈值": round(position["threshold"], 2),
                    "回撤控制线(%)": round(position["fund_dd_limit"], 2),
                    "大盘回撤线(%)": round(position["sse_dd_limit"], 2),
                    "冠军近3月涨幅(前日口径)": f"+{position['ret_3m']:.2f}%",
                    "卖出日": day.strftime("%Y-%m-%d"),
                    "期间最高": f"+{(position['peak_nav']/position['buy_nav']-1)*100:.1f}%",
                    "期间最大回撤": f"{max_dd:.1f}%",
                    "同期上证": f"{sse_ret:+.1f}%",
                    "持有天数": hold_days,
                    "卖出原因": sell_reason,
                    "_code": code,
                    "_buy_date": position["buy_date"],
                    "_sell_date": day,
                    "_ret_pct": ret_pct,
                })
                position = None

        # ── Step 2: 空仓(含当天刚卖出)且为信号日时买入 ──
        if position is None and day in signal_map:
            threshold = signal_map[day]
            code, ret_3m = find_champion_on_date(conn, day_str, qdii_codes)
            if code is None:
                continue

            # 获取当天买入净值
            row = conn.execute(
                "SELECT nav FROM fund_nav_daily WHERE code=? AND date=?",
                (code, day_str)).fetchone()
            if not row or not row[0]:
                # 取之后最近的
                row2 = conn.execute(
                    "SELECT date, nav FROM fund_nav_daily WHERE code=? AND date>=? ORDER BY date LIMIT 1",
                    (code, day_str)).fetchone()
                if not row2:
                    continue
                buy_nav = _apply_nav_anomaly(code, row2[0], float(row2[1]))
                actual_buy_date = pd.Timestamp(row2[0])
            else:
                buy_nav = _apply_nav_anomaly(code, day, float(row[0]))
                actual_buy_date = day

            beta = compute_beta(conn, sse, code, day_str)
            fund_dd_limit = threshold / 5.0 * beta

            # SSE peak at buy (从买入日开始追踪, 不是历史最高)
            sse_on_buy = sse[sse["date"] <= actual_buy_date]
            sse_peak = float(sse_on_buy["close"].iloc[-1]) if not sse_on_buy.empty else 3000.0

            # 预载基金净值序列, 供逐日止损检查
            nav_map = dict(get_fund_nav_after(conn, code, actual_buy_date))

            position = {
                "code": code,
                "buy_date": actual_buy_date,
                "buy_nav": buy_nav,
                "peak_nav": buy_nav,
                "min_nav": buy_nav,
                "last_nav": buy_nav,
                "peak_sse": sse_peak,
                "buy_sse": sse_peak,
                "ret_3m": ret_3m,
                "beta": beta,
                "fund_dd_limit": fund_dd_limit,
                "sse_dd_limit": threshold / 5.0,
                "threshold": threshold,
                "nav_map": nav_map,
            }

    # Close open position
    if position is not None:
        row = conn.execute(
            "SELECT date, nav FROM fund_nav_daily WHERE code=? ORDER BY date DESC LIMIT 1",
            (position["code"],)).fetchone()
        if row and row[1]:
            last_date = pd.Timestamp(row[0])
            last_nav = _apply_nav_anomaly(position["code"], row[0], float(row[1]))
            ret_pct = (last_nav / position["buy_nav"] - 1) * 100
            hold_days = (last_date - position["buy_date"]).days
            # 同期上证
            sse_last = sse.iloc[-1]["close"] if not sse.empty else 3000
            sse_ret = (float(sse_last) / position["buy_sse"] - 1) * 100 \
                if position["buy_sse"] else 0
            max_dd = position.get("max_dd", 0.0)
            code = position["code"]
            name = fund_names.get(code, code)
            trades.append({
                "买入日": position["buy_date"].strftime("%Y-%m-%d"),
                "冠军(C类全市场,按前一交易日榜单)": f"{name} ({code})",
                "类型": fund_types.get(code, ""),
                "波动率比值(近3月)": position["beta"],
                "恐慌阈值": round(position["threshold"], 2),
                "回撤控制线(%)": round(position["fund_dd_limit"], 2),
                "大盘回撤线(%)": round(position["sse_dd_limit"], 2),
                "冠军近3月涨幅(前日口径)": f"+{position['ret_3m']:.2f}%",
                "卖出日": f"{last_date.strftime('%Y-%m-%d')}(持仓中)",
                "期间最高": f"+{(position['peak_nav']/position['buy_nav']-1)*100:.1f}%",
                "期间最大回撤": f"{max_dd:.1f}%",
                "同期上证": f"{sse_ret:+.1f}%",
                "持有天数": hold_days,
                "卖出原因": "未触发",
                "_code": code,
                "_buy_date": position["buy_date"],
                "_sell_date": last_date,
                "_ret_pct": ret_pct,
            })

    _apply_chain_fees(trades)
    conn.close()
    return trades


def _apply_chain_fees(trades):
    """连续接力同一只基金(上一笔卖出日=下一笔买入日且代码相同)不算真实
    离场, 中间腿不收手续费; 只有链条最后一腿按"链条首次买入→该腿卖出"的
    累计持有天数收一次手续费(按实际持有时长计, 而非单腿天数)。"""
    n = len(trades)
    chain_start = None
    for i, t in enumerate(trades):
        prev = trades[i - 1] if i > 0 else None
        is_continuation = (prev is not None and
                           prev["_code"] == t["_code"] and
                           prev["_sell_date"] == t["_buy_date"])
        chain_start = t["_buy_date"] if not is_continuation else chain_start
        nxt = trades[i + 1] if i + 1 < n else None
        is_last_of_chain = not (nxt is not None and
                                nxt["_code"] == t["_code"] and
                                nxt["_buy_date"] == t["_sell_date"])

        if is_last_of_chain:
            total_days = (t["_sell_date"] - chain_start).days
            fee = 1.5 if total_days < 7 else (0.5 if total_days < 30 else 0)
        else:
            fee = 0.0

        ret_pct = t["_ret_pct"]
        ret_after_fee = ret_pct - fee
        ret_str = (f"{ret_pct:+.2f}% (费后{ret_after_fee:+.2f}%)"
                   if fee > 0 else f"{ret_pct:+.2f}%")
        t["手续费%"] = fee
        t["费后收益"] = round(ret_after_fee, 2)
        t["持有收益"] = ret_str

    for t in trades:
        for k in ("_code", "_buy_date", "_sell_date", "_ret_pct"):
            del t[k]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QVIX恐慌信号回测")
    parser.add_argument("--window", type=int, default=720, help="滚动窗口(交易日),默认720(约3年)")
    parser.add_argument("--pct", type=float, default=0.95, help="分位数,默认0.95")
    args = parser.parse_args()

    t0 = time.time()
    trades = run_backtest(window=args.window, pct=args.pct)
    elapsed = time.time() - t0

    if not trades:
        print("无交易记录")
        return

    df = pd.DataFrame(trades)
    print(f"\n{'='*110}")
    print(f"回测结果(窗口={args.window} 分位={args.pct}): {len(df)} 笔交易, 耗时 {elapsed:.0f}s")
    print(f"{'='*110}\n")

    completed = df[~df["卖出原因"].str.contains("持仓中")]
    if not completed.empty:
        # 用费后收益算累计
        rets = completed["费后收益"]
        days = completed["持有天数"]
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        total_ret = ((1 + rets / 100).prod() - 1) * 100
        total_fee = completed["手续费%"].sum()
        print(f"已完成: {len(completed)} 笔")
        print(f"  胜率: {len(wins)}/{len(completed)} = {len(wins)/len(completed)*100:.1f}%")
        print(f"  累计收益(费后复利): {total_ret:+.2f}%")
        print(f"  累计手续费: {total_fee:.1f}%")
        print(f"  平均持有: {days.mean():.0f} 天")
        print(f"  平均收益(费后): {rets.mean():+.2f}%")
        print(f"  最佳: {rets.max():+.2f}%")
        print(f"  最差: {rets.min():+.2f}%")

    # 输出表格
    display_cols = ["买入日", "冠军(C类全市场,按前一交易日榜单)", "类型",
                    "波动率比值(近3月)", "恐慌阈值", "回撤控制线(%)", "大盘回撤线(%)",
                    "冠军近3月涨幅(前日口径)", "卖出日", "持有收益",
                    "手续费%", "期间最高", "期间最大回撤", "同期上证", "卖出原因"]
    print(f"\n{df[display_cols].to_string(index=False)}")


if __name__ == "__main__":
    main()
