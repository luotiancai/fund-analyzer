"""
回测: QVIX恐慌信号买入 + 双止损(基金回撤控制线 / 大盘回撤线)
- 买入: QVIX > 2年90分位阈值, 且资金可用(空仓或当天恰好卖出)
- 标的: 前一交易日近3月冠军(C类全市场, 规模≥5000万), 冠军排名复用 fetcher.compute_
  metrics_asof——与 app.py「基金列表」页"截至日期"筛选完全同口径(按日
  收益率连乘, 正确处理分红除权, 自带单日|收益率|>30%异常值过滤), 而非
  简化的 end_nav/anchor_nav-1(曾把 2020-07-16 算错成广发医疗保健夺冠,
  实际应为汇丰晋信智造先锋, 已交叉验证修正)
- 卖出: 基金回撤控制线(买入日阈值/5×波动率比值) 或 大盘回撤线(买入日阈值/5),
  逐交易日检查, 先到先卖(双线在买入日锁定, 与 app.py 复盘口径一致)
  波动率比值 = 基金日收益率std / 大盘日收益率std(纯波动对比, 不按相关系数加权)
"""
import sys, os, time, sqlite3, io, re
import numpy as np
import pandas as pd
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetcher

DB = os.path.expanduser("~/.local/share/fund-analyzer/fund_cache.db")

# 基金规模(AUM)过滤:规模数据(季度期末净资产,亿元)存在 fund_cache.db 的
# fund_scale_hist 表, 取数/按信号日取值的逻辑都在 fetcher(见
# fetcher.fund_aum_asof, 无未来函数——只用已披露的季报)。这里直接调它。

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

# ── 标准策略回测参考快照(2026-08-03, 数据/规则变了要重跑, 别当成结论)──
# 上面这段窗口×分位对比、以及后面出现过的"涨幅冠军+相关系数过滤"都是
# 中途淘汰的旧方案, 历史版本见 git 历史, 不再列在这里。当前标准策略是
# QVIX 2年90分位信号(window=490,pct=0.90) + 近3月跌幅最大(ret_col=
# "ret_3m",pick="bottom") + 候选波动率比值≥1.5(min_vol_ratio=1.5) +
# 候选规模≥5000万(min_aum=0.5, 2026-07-28 新增, 见 min_aum 说明) +
# 跌幅耗尽(候选按跌幅从深到浅排, 找到第一个非负值就当天不操作, 不退而
# 求其次买涨幅最小的) + 排除QDII/海外/持有期锁定基金(港股通/沪港深/
# 恒生系列不再排除, 见 _HK_RE 定义处说明), 命令:
#   python3 backtest_qvix.py
# (以上全部是当前 run_backtest 的默认值, 不用额外传参——含 window=490/
# pct=0.90, 2026-07-27 起生产阈值列/通知邮件也已对齐这个口径,
# 默认值随之从 720/0.95 改过来; min_aum=0.5 于 2026-07-28 加入)
#
#   笔数  胜率        累计收益(费后复利)  平均持有  平均收益(费后)
#   9    8/9=88.9%   +570.36%           71天      +28.51%
#   (剔除单笔运气021528财通成长优选混合C+140.47%后, 剩8笔仍有+178.77%,
#   只有一笔小亏-0.06%——9笔里唯一的亏损, 且亏得极小)
#   样本只有9笔、跨6年, 统计上很薄, 别当成可靠预期。
#
#   本次(2026-07-31)相对上一版快照(2026-07-28: 8笔/6-8=75%/+490.48%)的
#   全部差异, 都来自当天两处修复, 策略规则本身一个字没动:
#     ① QVIX自算换月规则改用现行版白皮书口径 + 修K0选取(见 qvix_calc
#        模块docstring)。带来: 新增2022-03-15嘉实港股通新经济指数C
#        (+12.50%)——那天QVIX旧数据是空值、信号整个被吞掉; 2026-03-23
#        那笔从-0.37%翻成+0.44%、2025-04-07从+140.23%到+140.47%(阈值
#        微调使止损线跟着动, 卖点各提前2天/4天)。
#     ② 信号日下限护栏修好(见 _signal_floor 处注释)。上一版快照记录时
#        护栏是好的(+490.48%正是剔除2020-03-24后的值), 之后因005696
#        华安睿明两年定开混合C的2018年历史进库、把全表MIN(date)拖到
#        2018-04-23而失效, 中间一度让2020-03-24那笔假信号(近3月窗口
#        只有81天)混进结果。修好后仍然剔除, 与上一版快照口径一致。
#
#   2026-08-03: 信号阈值加 .shift(1)(比"截至昨日收盘"的分位, 见
#   run_backtest 里注释)——对齐实盘"盘中实时值比昨日阈值"的用法, 之前
#   窗口含信号日当天收盘、口径比实盘严一天。重跑后9笔交易的买卖日期和
#   收益**全部不变**, 只有记录的阈值/止损线小数位微调(最大是2026-03-23
#   那笔阈值22.51→22.02); 新增的2025-09-02信号日落在2025-04-07持仓期内,
#   不产生交易。
#
#   加规模门槛前(min_aum=None)是 +470.41%: 门槛把2026-03那笔几十万规模的
#   东方城镇消费主题C(+3.10%)换成了正常规模的广发医药创新发起式C(跌幅
#   更深、被止损兜住), 另外2022-10/2023-08两笔标的也一并升级。5000万
#   vs 1亿量级差别不大, 选5000万即可。注: 这个 +470.41% 是2026-07-28
#   旧口径下测的, 本次没有重测 min_aum=None, 只该用来看"门槛换掉了哪些
#   标的", 别拿它跟上面的 +570.36% 直接相减。
#   注: 2026-07-29 修正规模披露日口径(季报四季度都是季末后~1个月披露, 之前
#   误按半年报/年报的62/92天), 使2026-03那笔从国融融盛(+30.83%)校正为广发
#   医药创新(跌得更深且当天规模已够), 累计从+675.40%回落——这是修 bug 后的
#   真实值, 之前偏高是用了尚不该可见的更旧季度规模所致。
#
# 买入日/冠军/类型/波动率比值/阈值/回撤线/大盘线/近3月涨幅/卖出日/收益(费后)/手续费%/期间最高/最大回撤/同期上证/卖出原因
# 2022-03-15 嘉实港股通新经济指数C(006614)     指数型-股票  2.26 27.47 12.42 5.49  -37.47%  2022-04-21 +12.50%   0.0 +26.8% 11.3%  +0.5% 大盘6.2%>=5.5%
# 2022-04-25 诺安创新驱动混合C(002051)         混合型-灵活  1.73 26.33  9.11 5.27  -37.80%  2022-07-15 +14.19%   0.0 +23.2%  7.3% +10.2% 大盘5.3%>=5.3%
# 2022-10-24 富国中证港股通互联网ETF发起式联接C(014674) 指数型-股票 2.19 23.25 10.18 4.65 -26.93% 2022-12-22 +49.70% 0.0 +54.7% 8.6% +2.6% 大盘4.9%>=4.7%
# 2023-08-28 诺安积极回报混合C(012847)         混合型-灵活  3.20 22.27 14.25 4.45  -27.95%  2023-10-19 -0.06%    0.0  +8.5% 12.5% -3.0%  大盘5.4%>=4.5%
# 2024-02-05 汇丰晋信时代先锋混合C(014918)     混合型-偏股  2.11 22.28  9.40 4.46  -34.29%  2024-04-12 +15.18%   0.0 +27.6%  9.8% +11.7% 基金9.8%>=9.4%
# 2024-09-26 华富健康文娱灵活配置混合C(019200)  混合型-灵活  1.93 20.26  7.82 4.05  -22.41%  2024-10-09 +18.05%(费后+17.55%) 0.5 +32.8% 11.1% +8.6% 基金11.1%>=7.8%+大盘6.6%>=4.1%(同日双触发)
# 2024-10-10 海富通科技创新混合C(009024)       混合型-偏股  1.81 20.46  7.41 4.09   -2.42%  2024-11-14 +6.66%    0.0 +15.9%  8.0% +2.4%  基金8.0%>=7.4%
# 2025-04-07 财通成长优选混合C(021528)         混合型-灵活  3.41 21.25 14.50 4.25  -22.35%  2025-11-14 +140.47%  0.0 +181.4% 14.5% +28.9% 基金14.5%>=14.5%
# 2026-03-23 广发医药创新混合发起式C(017963)   混合型-偏股  2.68 22.02 11.80 4.40  -19.69%  2026-06-03 +0.44%    0.0 +14.2% 12.1%  +7.1% 基金12.1%>=11.8%


# 有持有期限制的基金(买入后锁定期内不能赎回, 名称里常见"N年/N个月/N天
# 持有(期)"、"滚动持有"、"定期开放"、"封闭式/封闭运作")——策略的止损
# 逻辑要求随时能在触发回撤线那天卖出, 选到这类基金实际上赎不出来,
# 回测里的"卖出"是假的, 必须整批排除出冠军候选池。没有专门的 type
# 字段能区分, 只能从基金名称正则匹配(已核对覆盖 fund_list 全部3258条
# 含"持有"字样的基金, 0条漏网)。
_HOLD_PERIOD_RE = re.compile(
    r"(\d+|[一二两三四五六七八九十]+)\s*(年|个月|月|天)(滚动)?持有"
    r"|持有期|定期开放|封闭式|封闭运作|滚动持有")

# 港股/沪港深/恒生指数系列基金一度被排除(理由是"跟踪境外市场,与
# QVIX/大盘恐慌-反弹逻辑脱钩",套用QDII基金的排除逻辑),2026-07-24
# 复核后撤销: 实测恒生指数与上证指数的日收益率相关系数常年在0.4~0.74
# (整体0.544),明显高于纳斯达克(0.153)、黄金(0.099)这些真正弱相关的
# 境外资产,"脱钩"这个假设本身站不住脚,不该跟QDII一视同仁。_HK_RE
# 定义留着(以后想单独分析这批基金还用得上),但不再计入 exclude_codes。
_HK_RE = re.compile(r"港股|沪港深|恒生|香港")


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


def _corr_with_market(conn, sse_df, code, window_start, window_end):
    """基金近3月窗口日收益率与上证指数日收益率的相关系数(Pearson).

    数据不够(<20个共同交易日)返回 None, 由调用方决定怎么处理(通常当
    "过不了筛选"处理, 而不是当0分——0是"完全不相关", 数据不够是"不
    知道", 两者含义不同)。
    """
    df = pd.read_sql_query(
        "SELECT date, nav FROM fund_nav_daily WHERE code=? AND date>=? AND date<=? ORDER BY date",
        conn, params=(code, window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
    if len(df) < 20:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["ret"] = df["nav"].pct_change()
    sse_w = sse_df[(sse_df["date"] >= window_start) & (sse_df["date"] <= window_end)].copy()
    sse_w["ret"] = pd.to_numeric(sse_w["close"], errors="coerce").pct_change()
    m = df.merge(sse_w[["date", "ret"]], on="date", suffixes=("_f", "_m")).dropna()
    if len(m) < 20:
        return None
    c = m["ret_f"].corr(m["ret_m"])
    return None if pd.isna(c) else c


def find_champion_on_date(conn, asof_date, exclude_codes=None,
                          sse_df=None, min_corr=None,
                          ret_col="ret_3m", pick="top", min_vol_ratio=None,
                          min_aum=None, require_drop=True):
    """找 asof_date 当天视角下排名第一的标的(排除 exclude_codes). 返回 (code, ret).

    复用 fetcher.compute_metrics_asof——按日收益率连乘计算区间收益(正确
    处理分红除权,不会像 end_nav/anchor_nav-1 那样被除权日的净值跳水拉低),
    且自带单日|收益率|>30%异常值过滤(effective_daily_ret, 已覆盖 014939
    2025-03-31 断层等已知案例)。该函数按 asof 严格早于当日截断数据
    (T日决策只能看到T-1日收盘净值), 与 app.py「基金列表」页"截至日期"
    筛选完全同口径, 已用 2020-07-16 001644 vs 009163 交叉验证过。

    ret_col: 排名依据的区间收益列, "ret_1m"(近1月,30天)或"ret_3m"
    (近3月,91天), 见 fetcher.RETURN_DAYS。pick="top" 选该列最高的
    (动量/冠军), pick="bottom" 选最低的(跌幅最大/反转候选)——起因:
    实测11次QVIX恐慌信号里有7次是"近3月跌幅最大的20只"反弹幅度反而
    超过"近3月涨幅最大的20只"(2026-07-24分析, 恐慌信号触发点本身就是
    刚经历过一轮下跌, 涨得最猛的常是提前跑赢、比较拥挤的仓位, 跌得
    最惨的反而容易报复性反弹), pick="bottom" 就是把这个反转假设直接
    做成候选规则去回测验证。pick="bottom" 时要求候选的 ret_col 必须
    是负数才算数(真正下跌过), 不是单纯"这批候选里排名最后一个"——
    候选按跌幅从深到浅排, 走到第一个非负值就直接判定当天没有真正
    下跌的标的, 不操作(而不是退而求其次买涨幅最小的那个凑数), 见
    下面 min_vol_ratio 说明附近的循环逻辑。

    exclude_codes 用于剔除 QDII 等跟踪境外市场的基金——策略买入逻辑
    建立在"大盘恐慌信号→买入国内标的"上, QDII 收益与 QVIX/大盘走势
    脱钩, 选入会削弱大盘回撤线对该笔仓位的意义。命中净值僵化-补涨
    模式(见 _has_stale_catchup)的基金也一并跳过, 逐个往下找下一名,
    直到选出一个数据正常的真实候选。

    sse_df/min_corr 是可选的"共振"过滤(默认不启用, 2026-07-24实测
    对当前策略已无必要, 见 run_backtest 说明): 给了 min_corr 才生效,
    要求候选与上证近3月日收益率相关系数 >= min_corr, 不够格的跳过、
    接着往下一名找。

    min_vol_ratio: 可选的振幅过滤, 给了才生效——要求候选的波动率比值
    (compute_beta, 基金日收益率std/大盘日收益率std)>= min_vol_ratio,
    不够格的跳过、接着往下一名找。起因: "跌幅最大"选出的候选有时是
    长期低波动的保守型基金(如2020-07那两笔, 全历史年化波动率只有
    2.3%~4.6%, 接近债券基金水平, 当天排名垫底只是正常噪声、根本谈不上
    "超跌"), 买这种基金既吃不到反弹弹性、回撤线又窄得动不动就止损,
    跟"跌幅最大→反转候选"的策略初衷脱节。2026-07-24 定为标准参数
    min_vol_ratio=1(候选振幅至少要跟大盘同量级, 用户原话"我是来发财
    的,不是来保本的")。
    """
    metrics = fetcher.compute_metrics_asof(asof_date, cols={ret_col})
    if not metrics:
        return None, 0
    candidates = {
        c: m[ret_col] for c, m in metrics.items()
        if m.get(ret_col) is not None
        and (not exclude_codes or c not in exclude_codes)
    }
    if not candidates:
        return None, 0

    end = pd.Timestamp(asof_date)
    window_end = end - timedelta(days=1)
    window_start = end - timedelta(days=fetcher.RETURN_DAYS[ret_col] + 10)

    for code in sorted(candidates, key=candidates.get, reverse=(pick == "top")):
        if require_drop and pick == "bottom" and candidates[code] >= 0:
            # 候选按跌幅从深到浅排, 走到这里说明真正下跌的候选(连同
            # 能同时满足 min_vol_ratio 等其他条件的)已经找完了, 剩下
            # 全是涨的——不是超跌反转的局, 今天不操作, 不退而求其次去
            # 买涨幅最小的那个(那样就不是"跌幅最大"策略了, 是变相的
            # "涨幅倒数第一", 意义不一样)。require_drop=False 时关掉这条,
            # 照买跌幅最大(涨幅最小)那个, 用于回测「去掉不操作规则」的效果。
            break
        if _has_stale_catchup(conn, code, window_start, window_end):
            continue
        if min_corr is not None:
            c = _corr_with_market(conn, sse_df, code, window_start, window_end)
            if c is None or c < min_corr:
                continue
        if min_vol_ratio is not None:
            vr = compute_beta(conn, sse_df, code, asof_date)
            if vr < min_vol_ratio:
                continue
        if min_aum is not None:
            # 规模过滤放在最后:前面便宜的DB过滤先淘汰掉大部分候选, 减少
            # 首轮抓规模的网络请求(之后走 fund_scale_hist 缓存)。规模数据
            # 缺失(None)按不达标处理——宁可跳过也不买一只连规模都查不到的
            # 基金(多是极小/新基金)。
            aum = fetcher.fund_aum_asof(code, asof_date)
            if aum is None or aum < min_aum:
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


def run_backtest(window: int = 490, pct: float = 0.90, minp_ratio: float = 0.97,
                 min_corr: float = None, ret_col: str = "ret_3m", pick: str = "bottom",
                 min_vol_ratio: float = 1.5, dd_divisor: float = 5.0,
                 min_aum: float = 0.5, require_drop: bool = True):
    """window=滚动窗口(交易日), pct=分位数, minp_ratio=窗口内至少要有
    多大比例的有效数据才出阈值(容错缺失日,同 fetcher.update_qvix_self_daily
    的 475/490 那套道理)。默认 490/0.90 是当前线上在用的参数
    (2026-07-27 起生产阈值列/通知邮件从 720/0.95 对齐到这个口径)。

    min_corr: 冠军候选与上证指数近3月相关系数门槛, 见
    find_champion_on_date 的 sse_df/min_corr 说明。2026-07-24 起默认
    改回 None(不过滤)——之前用2年90%实测0.5/0.6/0.7三档定过0.6, 但
    后面被 ret_col/pick 这条反转策略取代, 相关系数过滤对新逻辑没有
    实测过、也没有继续保留的理由, 显式关掉。仍保留参数只是为了不删掉
    这条已验证过有效的机制, 以后想重新启用可以传 0.6。

    ret_col/pick: 排名依据("ret_1m"近1月/"ret_3m"近3月)和方向
    ("top"=选最高即冠军/动量, "bottom"=选最低即跌幅最大/反转候选)。
    2026-07-24 起标准策略定为 ret_col="ret_3m"+pick="bottom"(近3月
    跌幅最大)——用2年90%信号实测: 冠军/动量(相关系数≥0.6版本)11笔
    最大亏损-9.40%/平均亏损-5.37%; 近3月跌幅最大12笔最大亏损仅
    -4.81%/平均亏损-1.72%, 亏损明显更可控, 不依赖某一笔运气好的极端
    案例(该案例是021528财通成长优选混合C+140.23%, 剔除它后累计收益
    仍不差, 亏损可控这个结论不受影响)。

    min_vol_ratio: 候选波动率比值(compute_beta)下限, 见
    find_champion_on_date 同名参数说明。2026-07-24 定为标准参数=1.5
    (用户原话"我是来发财的,不是来保本的",要求候选振幅要明显强于大盘)
    ——之前不设这道过滤时, "跌幅最大"选出过两只长期低波动的保守型
    基金(如2020-07那两笔, 全历史年化波动率只有2.3%~4.6%, 当天排名
    垫底只是正常噪声、根本谈不上"超跌"); 只提高门槛到1.0/1.5、仍退而
    求其次买涨幅最小(甚至微涨)的候选时胜率反而下降(1.0档11笔跌到
    46.2%); 配合 pick="bottom" 的"跌幅耗尽即不操作"逻辑(候选按跌幅从
    深到浅排, 走到第一个非负值就直接放弃当天, 不买涨幅最小的那个凑
    数)之后, min_vol_ratio=1.5 才真正起效: 2年90%信号下从79个信号日
    里只挑出8笔真正"深跌+高波动"的交易, 胜率87.5%(7/8), 累计收益(费后
    复利)+470.41%, 剔除单笔运气(021528财通成长优选混合C+140.23%)后
    仍有+137.45%, 明显优于不设"跌幅耗尽即放弃"时的版本。另外这批候选
    只排除 QDII/海外(见 exclude_codes 里的 qdii_codes)——曾经额外
    排除过港股通/沪港深/恒生系列(理由是"跟踪境外市场,跟QVIX/大盘
    恐慌-反弹逻辑脱钩", 套用QDII的逻辑), 2026-07-24复核后撤销: 实测
    恒生指数跟上证指数的日收益率相关系数常年0.4~0.74(整体0.544), 明显
    高于纳斯达克(0.153)、黄金(0.099)这些真正弱相关的境外资产,"脱钩"
    假设不成立。之前排除港股通时漏选(其实是漏排除)的"国泰中证港股通
    科技ETF发起联接C"(2022年10-12月港股互联网超跌反弹段选中, +31.54%)
    现在正常保留在候选池里, 也是这版收益明显更好的原因之一。
    见 find_champion_on_date 同名参数说明。

    dd_divisor: 回撤线除数——基金回撤控制线=阈值/dd_divisor×波动率比值,
    大盘回撤线=阈值/dd_divisor。默认5.0(线上在用口径), 可传4/6等回测
    更宽/更窄的止损带对效果的影响。

    min_aum: 候选基金规模下限(亿元), 默认0.5(=5000万, 2026-07-28定为标准
    参数)。规模来自 fetcher.fund_aum_asof(信号日当时能看到的最新一期季报,
    无未来函数——季报有披露滞后, 只用已披露的)。低于门槛、或那天还查不到
    任何已披露季报(多是成立太新的迷你基金)的候选跳过、往下一名找。起因:
    "跌幅最大"排在最前面的常是几十万~几千万规模的迷你/僵尸基金, 净值容易
    被单笔申赎搅动失真, 选出的"冠军"是噪声(如2026-03原本选中东方城镇消费
    主题C, 规模仅几万)。实测加0.5亿门槛后累计收益(费后复利)从+470.41%变到
    +490.48%(2026-07-29修正规模披露日口径后的值; 修正前一度显示+675.40%,
    是错用了更旧季度规模所致), 主要是把那笔垃圾换成正常规模标的; 5000万 vs
    1亿量级差别不大, 选5000万即可。传 None 关掉过滤。"""
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
            # 用 `or` 而不是 .get 的默认值:新上市基金的 type/name 会以
            # null 出现在榜单里(键存在、值为 None),.get(k, "") 这时返回的
            # 是 None 不是 "",下面 "QDII" in t 直接 TypeError 整个回测挂掉。
            fund_names[c] = item.get("name") or c
            fund_types[c] = item.get("type") or ""

    # QDII/海外指数型跟踪境外市场, 与 QVIX/大盘恐慌-反弹逻辑脱钩, 排除出
    # 冠军候选池(如"指数型-海外股票"的广发道琼斯石油指数C, 之前只过滤
    # "QDII"字样漏掉了这类, 类型字符串里没有QDII三个字但同样跟踪境外)
    qdii_codes = {c for c, t in fund_types.items() if ("QDII" in t or "海外" in t)}
    # 港股通/沪港深/恒生系列不再排除(见 _HK_RE 定义处说明: 实测恒生指数
    # 跟上证相关系数常年0.4~0.74,"脱钩"假设不成立,2026-07-24撤销这条)
    # 有持有期锁定的基金也排除(见 _HOLD_PERIOD_RE 定义处说明)
    hold_codes = {c for c, n in fund_names.items() if _HOLD_PERIOD_RE.search(n)}
    exclude_codes = qdii_codes | hold_codes

    # Load QVIX —— 自算(qvix_self_history,上交所官方期权风险指标反推,
    # 不再是 optbbs 的 index_daily_cache)。阈值按传入的 window/pct 现算,
    # 不用表里预存的那一列(那一列固定是线上用的490/0.90)。
    print(f"加载数据... (窗口={window}天, 分位={pct}, 排名依据={ret_col}, "
          f"方向={pick}, 相关系数门槛={min_corr}, 波动率比值下限={min_vol_ratio}, "
          f"回撤线除数={dd_divisor})")
    qvix = fetcher.load_qvix_self_history()
    qvix = qvix.rename(columns={"qvix": "close"})
    qvix["date"] = pd.to_datetime(qvix["date"])
    qvix["close"] = pd.to_numeric(qvix["close"], errors="coerce")
    qvix = qvix.sort_values("date").reset_index(drop=True)
    minp = int(window * minp_ratio)
    # .shift(1):信号日 d 比的是"截至 d-1 收盘"的分位, 不含 d 当天——实盘
    # 就是拿盘中实时值比昨日阈值(qvix_now.py), 回测不 shift 的话暴涨日
    # 自己会把当日阈值抬高一点, 口径比实盘严一天。生产阈值列(fetcher.
    # update_qvix_self_daily)不 shift, 因为那列存的是"截至该日收盘"的值,
    # 次日实盘拿最后一行来比, 语义正好等价于这里的 shift(1)。
    qvix["thr"] = qvix["close"].rolling(window, min_periods=minp).quantile(pct).shift(1)

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
    # 起点取"每只基金最早净值日"的1%分位, 而不是全表 MIN(date)。全表 MIN
    # 会被极个别离群基金拉到很早: 实测 005696 华安睿明两年定开混合C 有
    # 2018-04-23 的数据(定开基金, 全程仅488条), 而其余5681只**全部**始于
    # 2020-01-02。一只基金就把下限拖前近两年(2018-07-23), 这条护栏彻底
    # 失效, 上面那段注释里点名要剔除的 2020-03-24 一直照常混在回测里,
    # 用的是近3月窗口只有81天的假排名。分位数对这种离群点免疫(0.05%~2%
    # 分位算出来都是 2020-01-02), 取1%留足余量。
    _firsts = pd.to_datetime(pd.read_sql(
        "SELECT MIN(date) AS f FROM fund_nav_daily GROUP BY code", conn)["f"])
    _nav_floor = _firsts.quantile(0.01)
    _signal_floor = _nav_floor + pd.Timedelta(days=91)
    print(f"净值库起点 {_nav_floor.date()}(1%分位; 全表最早 {_firsts.min().date()}, "
          f"离群不采信), 近3月冠军窗口最早可信信号日 {_signal_floor.date()}")

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

            # 两条线各自独立判断:同一天可能都破(此前用 if/elif 只记基金那条,
            # 会漏掉"大盘线同日也触发"的事实)。both=同日双触发。
            _fund_hit = fund_dd >= position["fund_dd_limit"]
            _sse_hit = sse_dd >= position["sse_dd_limit"]
            _fund_txt = f"基金{fund_dd:.1f}%>={position['fund_dd_limit']:.1f}%"
            _sse_txt = f"大盘{sse_dd:.1f}%>={position['sse_dd_limit']:.1f}%"
            sell_reason = None
            if _fund_hit and _sse_hit:
                sell_reason = f"{_fund_txt}+{_sse_txt}(同日双触发)"
            elif _fund_hit:
                sell_reason = _fund_txt
            elif _sse_hit:
                sell_reason = _sse_txt

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
                    "买入时规模(亿)": position.get("buy_aum"),
                    "波动率比值(近3月)": position["beta"],
                    "恐慌阈值": round(position["threshold"], 2),
                    "回撤控制线(%)": round(position["fund_dd_limit"], 2),
                    "大盘回撤线(%)": round(position["sse_dd_limit"], 2),
                    "冠军近3月涨幅(前日口径)": f"{position['ret_3m']:+.2f}%",
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
            code, ret_3m = find_champion_on_date(conn, day_str, exclude_codes,
                                                 sse_df=sse, min_corr=min_corr,
                                                 ret_col=ret_col, pick=pick,
                                                 min_vol_ratio=min_vol_ratio,
                                                 min_aum=min_aum,
                                                 require_drop=require_drop)
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
            fund_dd_limit = threshold / dd_divisor * beta

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
                "sse_dd_limit": threshold / dd_divisor,
                "threshold": threshold,
                "nav_map": nav_map,
                # 买入日当时已披露的最新一期季报规模(as-of, 无未来函数),
                # 即策略选基那天实际能看到的规模。选基环节已经查过一次,
                # 这里走 fund_scale_hist 缓存, 不产生额外网络请求。
                "buy_aum": fetcher.fund_aum_asof(code, day_str),
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
                "买入时规模(亿)": position.get("buy_aum"),
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
    parser.add_argument("--window", type=int, default=490, help="滚动窗口(交易日),默认490(约2年,线上口径)")
    parser.add_argument("--pct", type=float, default=0.90, help="分位数,默认0.90(线上口径)")
    parser.add_argument("--min-corr", type=lambda s: None if s.lower() == "none" else float(s),
                        default=None,
                        help="冠军候选与上证指数近3月相关系数门槛,默认不启用"
                             "(2026-07-24起改回默认关闭,见 run_backtest 说明);"
                             "传具体数值(如0.6)启用")
    parser.add_argument("--lookback", choices=["1m", "3m"], default="3m",
                        help="排名依据的区间,默认3m(近3月,91天);"
                             "1m=近1月(30天)")
    parser.add_argument("--pick", choices=["top", "bottom"], default="bottom",
                        help="默认bottom=选区间跌幅最大(反转候选,"
                             "2026-07-24定为标准策略);"
                             "top=选区间涨幅最高(冠军/动量,旧逻辑)")
    parser.add_argument("--min-vol-ratio",
                        type=lambda s: None if s.lower() == "none" else float(s),
                        default=1.5,
                        help="候选波动率比值下限,默认1.5(2026-07-24定档,"
                             "见 run_backtest 说明);传 none 关掉过滤")
    parser.add_argument("--dd-divisor", type=float, default=5.0,
                        help="回撤线除数,默认5.0(线上口径:基金线=阈值/除数×"
                             "波动率比值,大盘线=阈值/除数);传4=更宽止损带,"
                             "6=更窄止损带")
    parser.add_argument("--min-aum",
                        type=lambda s: None if s.lower() == "none" else float(s),
                        default=0.5,
                        help="基金规模下限(亿元,信号日当时能看到的最新季报口径),"
                             "默认0.5=5000万(2026-07-28标准);1=1亿;传none关闭。"
                             "低于门槛或查不到规模的候选跳过、往下一名找")
    parser.add_argument("--no-require-drop", action="store_true",
                        help="关掉「跌幅耗尽即不操作」规则:信号日即便所有候选都"
                             "是正收益也照买跌幅最大(涨幅最小)那个。默认保留该"
                             "规则(标准策略:没有真正下跌的标的当天就不买)")
    args = parser.parse_args()

    _ret_col = "ret_1m" if args.lookback == "1m" else "ret_3m"
    t0 = time.time()
    trades = run_backtest(window=args.window, pct=args.pct, min_corr=args.min_corr,
                          ret_col=_ret_col, pick=args.pick,
                          min_vol_ratio=args.min_vol_ratio,
                          dd_divisor=args.dd_divisor,
                          min_aum=args.min_aum,
                          require_drop=not args.no_require_drop)
    elapsed = time.time() - t0

    if not trades:
        print("无交易记录")
        return

    # 落库供 app「策略复盘」表读取(页面不再硬编码这张表, 见
    # fetcher.save_backtest_trades 说明)。只有跑默认参数(=线上标准策略)
    # 时才写, 免得随手跑一次 --pick top 之类的对照实验就把页面上的
    # 标准策略明细顶掉。
    _is_standard = (args.window == 490 and args.pct == 0.90
                    and args.min_corr is None and args.lookback == "3m"
                    and args.pick == "bottom" and args.min_vol_ratio == 1.5
                    and args.dd_divisor == 5.0 and args.min_aum == 0.5
                    and not args.no_require_drop)
    if _is_standard:
        fetcher.save_backtest_trades(trades, {
            "window": args.window, "pct": args.pct, "ret_col": _ret_col,
            "pick": args.pick, "min_vol_ratio": args.min_vol_ratio,
            "dd_divisor": args.dd_divisor, "min_aum": args.min_aum,
        })
        print("已写入 backtest_trades 表(app 复盘表读这里)")
    else:
        print("非默认参数, 不写库(页面只展示标准策略明细)")

    df = pd.DataFrame(trades)
    print(f"\n{'='*110}")
    print(f"回测结果(窗口={args.window} 分位={args.pct}): {len(df)} 笔交易, 耗时 {elapsed:.0f}s")
    print(f"{'='*110}\n")

    # 未平仓那笔的标记在「卖出日」上("YYYY-MM-DD(持仓中)"), 不在「卖出原因」
    # 里(那里写的是"未触发")——原来按卖出原因过滤等于没过滤, 会把浮盈浮亏
    # 当成已实现收益算进胜率和累计收益。当前区间恰好没有未平仓的笔, 所以
    # 一直没暴露。
    completed = df[~df["卖出日"].astype(str).str.contains("持仓中")]
    if not completed.empty:
        rets = completed["费后收益"]
        days = completed["持有天数"]
        wins = rets[rets > 0]
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
