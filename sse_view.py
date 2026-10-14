"""上证页(行情图 + 策略复盘 + 对照回测)的数据整理, 不依赖 streamlit。

app.py(Streamlit)和 build_site.py(BWG 上的静态页)共用这一份: 复盘那段
统计说明、列顺序、止损染色都在这里只写一次, 两个前端只管画。以前这些全写
在 app.py 的页面代码里, 再抄一份给静态页, 两边迟早对不上。

返回的说明文字是 markdown 片段(只用 **粗体** 和 `代码`), Streamlit 直接
渲染, 静态页用 md_inline() 转成 HTML。
"""

import datetime as dt
import html
import re
from zoneinfo import ZoneInfo

import pandas as pd

import fetcher
import qvix_calc

_CST = ZoneInfo("Asia/Shanghai")

# 「可买池分位」列名。这一列不是回测产出的, 是 pct_rank.py 事后算好写回
# trades 的(见那个脚本的 docstring), 所以列可能不在——标准跑批换了而
# pct_rank.py 没补跑时就是这样。名字在两处出现过, 写死成常量免得改一处漏
# 一处; 跟 pct_rank.COL 必须一致。
COL_POOL_PCT = "可买池分位"

QVIX_COLOR = "#f28e2b"

# 时间区间 → 自然日数(None=全部)。静态页的 JS 按同一张表切窗口。
SSE_RANGES = {"近1月": 30, "近3月": 91, "近6月": 182,
              "近1年": 365, "近2年": 365 * 2, "近3年": 365 * 3,
              "近5年": 365 * 5, "近10年": 365 * 10, "全部": None}
SSE_RANGE_DEFAULT = "近1年"

# 止损染色: 触发哪条线就给哪一格上色。both=同日双触发, 两格都染。
FUND_LINE_COL, SSE_LINE_COL = "回撤控制线(%)", "大盘回撤线(%)"
TRIG_BG = {FUND_LINE_COL: "#ffdca8", SSE_LINE_COL: "#c9e2ff"}

TGT_COL = "选中标的(C类全市场,按前一交易日榜单)"
_RAW_TGT = "冠军(C类全市场,按前一交易日榜单)"
_RAW_RET = "冠军近3月涨幅(前日口径)"


def fmt_cst(ts: float, fmt: str) -> str:
    """Unix ts → 北京时间字符串,不依赖服务器自身时区(云端容器多半是
    UTC,time.localtime() 直接用会把 UTC 挂钟时间当北京时间显示,差 8
    小时却还显得"合理"——比如 22:45 UTC 写成"指标数据更新于…22:45",
    用户按北京时间读就觉得是昨晚,其实是当天早上6点多)。"""
    return dt.datetime.fromtimestamp(ts, tz=_CST).strftime(fmt)


def md_inline(s: str) -> str:
    """说明文字里那点 markdown(**粗体** / `代码`)→ HTML, 其余原样转义。"""
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def window(sse_all: pd.DataFrame, days):
    """按自然日切窗口, 保留锚点行(窗口起点当天或之前最后一个收盘), 区间
    涨跌按真正的基点算 —— 与 _window_by_date 同口径。"""
    if days is None:
        return sse_all
    start = sse_all["date"].max() - pd.Timedelta(days=days)
    older = sse_all[sse_all["date"] <= start]
    return sse_all.loc[older.index[-1]:] if not older.empty else sse_all


def qvix_thresholds(hist: pd.DataFrame):
    """按 backtest_qvix.py 同一套口径(含 shift(1):第 d 天画的是"截至
    d-1 收盘"的分位,与回测信号/实盘用法同口径)现算滚动阈值,供上证指数图
    叠加展示——与线上生产阈值列(fetcher.update_qvix_self_daily 里精确
    维护;那列不 shift,存"截至该日收盘"的值,次日实盘拿最后一行来比,
    语义等价)算法一致但独立现算,只用于图上对比展示,不影响生产阈值/
    策略复盘本身。窗口/最小样本/分位一律取 fetcher 的常量,别在这里另写
    一份数字——图上画的和策略判的必须是同一条线。"""
    if hist is None or hist.empty:
        return None
    hist = hist.sort_values("date").reset_index(drop=True)
    return pd.DataFrame({
        "date": pd.to_datetime(hist["date"]),
        "thr": (pd.to_numeric(hist["qvix"], errors="coerce")
                .rolling(fetcher.QVIX_THR_WINDOW,
                         min_periods=fetcher.QVIX_THR_MINP)
                .quantile(fetcher.QVIX_THR_PCT).shift(1)),
    })


def chain_broken_mask(qvix_hist: pd.DataFrame) -> pd.Series:
    """链失效日(标的跌出期权链挂牌行权价范围, 公式失效; 见
    qvix_calc.CHAIN_BROKEN)。那天没有数值, 但恰恰是最恐慌的那几天。"""
    return qvix_hist["note"].map(qvix_calc.is_chain_broken)


# ── 策略复盘(标准策略) ────────────────────────────────────────────────────

REVIEW_INTRO = (
    "**QVIX 2年90%信号 + 近3月跌幅最大**"
    "(波动率比值≥1.5 + 规模≥2亿 + 近3月跌幅≤30%;没有真跌的合格候选就"
    "放弃当天、顺延到下一个信号日)**+ 基金/大盘双止损逐日盯盘**"
    "。卖出后隔 2 个交易日才允许下一次买入(赎回款 T+2 到账,"
    "卖出当天和次日买不进去)")


def trig_of(reason) -> str:
    """每笔卖出当天触发了哪条止损线, 直接从回测写下的「卖出原因」反推
    (以前是另一份手工维护的列表, 跟表格行数各管各的, 一旦交易笔数变了
    就会错位染色)。→ "fund" / "sse" / "both" / ""。"""
    _r = str(reason)
    _f, _s = "基金" in _r, "大盘" in _r
    return "both" if _f and _s else ("fund" if _f else ("sse" if _s else ""))


def trig_cols(trig: str) -> list:
    """该行要染色的列名。"""
    return ([FUND_LINE_COL] if trig in ("fund", "both") else []) + \
           ([SSE_LINE_COL] if trig in ("sse", "both") else [])


def _done_rets(df: pd.DataFrame):
    """未平仓的那笔不计入统计(卖出日带"(持仓中)"标记)。"""
    done = df[~df["卖出日"].astype(str).str.contains("持仓中")]
    rets = pd.to_numeric(done["费后收益"], errors="coerce").dropna()
    days = pd.to_numeric(done["持有天数"], errors="coerce").dropna()
    return done, rets, days


def review_summary(bt_df: pd.DataFrame, bt_id, bt_at) -> str:
    """标准策略那张表上方的统计说明(markdown)。"""
    _done, _rets, _days = _done_rets(bt_df)
    _n, _wins = len(_rets), int((_rets > 0).sum())
    _cum = ((1 + _rets / 100).prod() - 1) * 100
    _fees = pd.to_numeric(_done["手续费%"], errors="coerce").fillna(0).sum()
    _best_i = _rets.idxmax()
    _cum_ex = ((1 + _rets.drop(_best_i) / 100).prod() - 1) * 100
    # 平均可买池分位现算, 别写死: 这列由 pct_rank.py 事后写回
    # trades, 换一次标准策略笔数和数值都会变(2026-09-04 加
    # sell_cooldown 那次就从 8 笔的旧数变成 9 笔)。列不在(标准
    # 跑批还没补跑 pct_rank.py)时整句不显示, 而不是显示一个假数。
    _pcts = pd.to_numeric(
        bt_df.get(COL_POOL_PCT, pd.Series(dtype=str))
        .astype(str).str.rstrip("%"), errors="coerce").dropna()
    _pct_txt = (f";{len(_pcts)}笔平均{_pcts.mean():.1f}%"
                if len(_pcts) else "")
    _loss = _rets[_rets <= 0]
    _loss_txt = ("、".join(f"{v:+.2f}%" for v in _loss)
                 if len(_loss) else "无")
    return (
        f"{_n} 笔已完成:胜率 {_wins / _n * 100:.1f}%({_wins}/{_n}),"
        f"费后复利 {_cum:+.2f}%,累计手续费 {_fees:.1f}%,"
        f"平均持有 {_days.mean():.0f} 天,"
        f"平均费后 {_rets.mean():+.2f}%,"
        f"最佳 {_rets.max():+.2f}% / 最差 {_rets.min():+.2f}%。"
        f"最佳那笔是单只小微盘主题基金的极端案例,剔除它其余 "
        f"{_n - 1} 笔累计仍有 {_cum_ex:+.2f}%——别把它当期望值,"
        f"不依赖那笔运气结论也成立。亏损笔:{_loss_txt}。"
        "样本只有个位数、跨度6年,统计上很薄,别当成可靠预期。"
        "「可买池分位」是这一笔在当天**本来可以买的所有基金**(规模≥2亿、"
        "排除QDII/海外和持有期锁定和净值脏数据,一两千只)里排第几——分母"
        "**不含**策略自己那三道筛选(真跌/跌幅≤30%/波动≥1.5),否则等于给"
        "筛选免票;每只都按同一套双止损跑过,50%就是闭眼瞎选"
        f"{_pct_txt}。"
        "候选须同时满足\"真正下跌但跌幅≤30% + 波动率比值≥1.5 + 规模≥2亿\""
        "(2026-08-12 在修好除权 bug 的数据上重验后定档:波动门槛低于1.5会让"
        "跌幅榜被\"没真跌的低波动标的\"占据;10亿上限的依据是事后最优解"
        "全部落在7亿以下——但那个观察是在2~10亿的池子里得出的、属于循环论证,"
        "2026-08-13 已撤掉上限:去掉后 8 笔里有 3 笔换成 19~24 亿的大基金,"
        "两胜一负净赚 40 个点(复利 +160 个点);30%跌幅上限的依据——事后最优解的近3月跌幅"
        "没有一只超过-27%,而跌得最深的那笔往往是\"跌的市场和反弹的市场"
        "不是同一个\");"
        "当天找不到真跌的合格候选就放弃当天、顺延到下一个信号日"
        "——2026-08-12 之前这里是\"改买涨幅最大的那只\","
        "已删除:那是动量逻辑,跟\"恐慌到位→超跌反弹\"的入场前提"
        "自相矛盾,而且跌幅耗尽本身就说明前提不成立。三笔兜底交易"
        "是 +3.55%/-3.20%/+36.76%,除北证50那笔外接近于零,而那笔"
        "同一只基金在止损后连续12个信号日仍是涨幅冠军、照规则买回去"
        "是 -10.47%,说明 +36.76% 是买入时点的运气而非规则的能力。"
        "规模按信号日当时"
        "已披露的最新季报口径(无未来函数)、A/C 等份额类别合并计算"
        "(2026-08-06 改,此前只算 C 类那一个代码,系统性低估),"
        "排掉了几十万~几千万"
        "规模、净值易失真的迷你基金。橙色=触发基金回撤线离场,"
        "蓝色=触发大盘回撤线离场,两格都染色=同日两条线双双触发;"
        "连续接力同一只基金视为未真实离场,中间腿不收手续费,"
        "只在链条最后一腿按累计持有天数收一次。"
        + (f"(跑批 #{bt_id},明细跑于 {fmt_cst(bt_at, '%Y-%m-%d %H:%M')})"
           if bt_at else ""))


def review_table(bt_df: pd.DataFrame, notes: dict):
    """→ (展示用 DataFrame, 逐行触发标记)。"""
    triggers = [trig_of(r) for r in bt_df["卖出原因"]]
    # 列名中性化:2026-08-06 加了 fallback_top 之后, 标准策略里
    # 有几笔走的是"跌幅耗尽→改买涨幅最大"分支, 那几行的区间收益
    # 是正的, 再叫"近3月跌幅最大标的/近3月跌幅"就是错的。具体
    # 每笔走的哪条分支看「选向」列。
    df = bt_df.rename(columns={_RAW_TGT: TGT_COL,
                               _RAW_RET: "近3月涨跌(前日口径)"})
    df = df[[c for c in [
        "买入日", TGT_COL,
        "类型", "买入时规模(亿)", "波动率比值(近3月)", "恐慌阈值",
        FUND_LINE_COL, SSE_LINE_COL, "近3月涨跌(前日口径)", "选向",
        # 「可买池分位」= 这一笔在**当天本来可以买的所有基金**里排第几
        # (打败了百分之多少)。分母跟「可达天花板」那条对照跑批一致:
        # 规模≥2亿 + 排除 QDII/海外、持有期锁定、净值脏数据, 其余不设
        # 条件; 每只候选都按同一套双止损跑出实现收益, 跟本笔可比;
        # 50% 就等于闭眼瞎选。
        # 它**不是**策略自己的候选池(策略还要过"真跌+跌幅≤30%+波动
        # ≥1.5"三道筛选) —— 这是故意的: 分母若用策略筛完的名单, 衡量
        # 的就只剩"在选好的候选里排第几", 等于给那几道筛选免票, 而它们
        # 恰恰是策略的一部分、也最该被检验。
        # 分母为什么要卡 2亿: 不卡的话混进的是策略根本不会买的迷你
        # 基金, 而且那批净值本身不可信(015111 在 2024-03-28 单日
        # +20.18%, 当季末净资产只剩100万 —— 那是93%份额被短期赎回、
        # 赎回费分摊给剩下7%造成的, 不是投资收益)。
        # 由 pct_rank.py 单独算好写进 trades —— 每个信号日要给一两千
        # 只基金逐日跑止损, 约15分钟, 不适合塞进每次回测。
        # ⚠️ 换了标准策略要重跑 pct_rank.py, 否则这列是上一版的旧数。
        "卖出日", "持有收益", COL_POOL_PCT, "期间最高", "期间最大回撤",
        "同期上证",
    ] if c in df.columns]].copy()
    # 「备注」是人工点评(backtest_notes 表), 算不出来、只能人写,
    # 每次重跑回测后重写一遍。按买入日 join, 没写的显示空白。
    df["备注"] = [notes.get(str(d), "") for d in df["买入日"]]
    return df, triggers


# ── 历次对照回测 ─────────────────────────────────────────────────────────

def run_label(r: dict) -> str:
    """下拉框里每条跑批的一行摘要。"""
    _t = fmt_cst(r["run_at"], "%m-%d %H:%M") or "—"
    _w = f"{r['win_rate']:.0f}%" if r["win_rate"] is not None else "—"
    _c = f"{r['cum_return']:+.0f}%" if r["cum_return"] is not None else "—"
    # 编号放最前面: label 是按参数自动拼的, 十几条跑批前缀
    # 高度雷同, 只有 #id 能一眼对上(讨论里说的也是这个号)。
    return (f"#{r['id']} · {_t} · {r['label']} · "
            f"{r['n_trades']}笔 胜率{_w} 复利{_c}")


def run_detail(x_df: pd.DataFrame, x_params: dict, run_id, x_at):
    """某次对照跑批 → (说明 markdown, 展示用 DataFrame, 逐行触发标记)。"""
    _x_win = "近1月" if x_params.get("ret_col") == "ret_1m" else "近3月"
    _x_done, _x_rets, _x_days = _done_rets(x_df)
    _x_n, _x_wins = len(_x_rets), int((_x_rets > 0).sum())
    _x_cum = ((1 + _x_rets / 100).prod() - 1) * 100
    _x_ex = ((1 + _x_rets.drop(_x_rets.idxmax()) / 100).prod() - 1) * 100
    _x_loss = _x_rets[_x_rets <= 0]
    _x_pick = str(x_params.get("pick") or "bottom")
    if _x_pick == "regime":
        # regime_basis 区分两版择向依据:早期那版比的是"截至
        # 前一日的回看窗口",在暴跌当天触发的信号上会把当天
        # 那根大阴线排除在方向判断外(2025-04-07 单日 -7.34%
        # 却算出近3月 +2.44% 判成"涨"),已弃用但跑批记录还在,
        # 规则描述不能拿现行口径去套。
        if x_params.get("regime_basis") == "window":
            _rule = (f"信号日上证比{_x_win}前**涨**(截至前一交易日)"
                     f"→买{_x_win}涨幅最大的,**跌**→买{_x_win}"
                     f"跌幅最大的。")
        else:
            _rule = (f"信号日**上证当天收涨**→买{_x_win}涨幅最大的,"
                     f"**当天收跌**→买{_x_win}跌幅最大的。")
        if x_params.get("require_drop"):
            _rule += "走跌幅分支时选中的基金必须真的是负收益,否则当天不买。"
        else:
            _rule += "不要求候选自身下跌过。"
    elif _x_pick == "top":
        _rule = f"买{_x_win}涨幅最大的(动量)。"
    else:
        _rule = f"买{_x_win}跌幅最大的(反转)。"
    # 有些跑批的选法根本不是 pick 那几种(如"事后最佳上限":
    # 每笔取当天候选池里自己跑完双止损后实现收益最高的那只),
    # 拿 pick 去套会把规则描述得完全不对。这类跑批自己在
    # params 里带 rule_note/exit_note, 有就顶掉自动拼的那句。
    _rule = x_params.get("rule_note") or _rule

    # 过滤条件逐项列出来, 不写"与标准策略一致"——对照实验
    # 改的可能正是其中某一项(如规模门槛), 那句话会自相矛盾。
    # 跟标准策略(波动≥1.5/规模≥2亿)不同的值加粗标出来。
    def _mark(val, std):
        return f"**{val}**" if val != std else f"{val}"
    _vr = x_params.get("min_vol_ratio", 1.5)
    # 涨幅分支门槛: 老跑批没记这项, 那时两个分支同用一个数
    _tvr = x_params.get("top_min_vol_ratio", "same")
    _mdrop = x_params.get("max_drop")   # 老跑批没记这项
    _aum = x_params.get("min_aum", 0.5)   # 老跑批没记这项时按旧标准
    # 规模口径 2026-08-06 从「只算该代码的份额类别」改成
    # 「A/C 合并」, 同一个 min_aum 数值在两种口径下松紧差很多,
    # 所以旧口径的跑批必须标出来, 否则会跟标准策略误比。
    _basis = x_params.get("aum_basis", "single")
    # 候选池/费率口径: 场内 ETF 版跟场外是两个市场, 页面上
    # 必须一眼看出来, 否则跟场外跑批并排列着会被当成同类比较。
    _uni = x_params.get("universe", "all")
    _fee = x_params.get("fee_model", "otc")
    _mkt = ("**场内 ETF**(费率口径:"
            + ("场内佣金双边" if _fee == "exchange" else "场外赎回费")
            + ")" if _uni == "etf" else "")
    caption = (
        f"规则:{_rule}"
        f"候选须满足波动率比值≥{_mark(_vr, 1.5)}"
        + ("(仅跌幅分支, 涨幅分支不设门槛)" if _tvr is None
           else "(涨跌两个分支同一门槛)" if _tvr == "same"
           else f"(跌幅分支; 涨幅分支≥**{_tvr}**)")
        + "、"
        + (f"近3月跌幅≤{_mdrop}%、" if _mdrop == 30.0
           else f"近3月跌幅≤**{_mdrop}%**、" if _mdrop
           else "**跌幅不设上限**、")
        + (f"规模**{_aum}~{x_params['max_aum']}亿**"
           if x_params.get("max_aum") else
           f"规模≥{_mark(_aum, 2.0)}亿")
        + ("(**旧口径:只算该代码的份额类别,未合并 A/C**)"
           if _basis != "merged" else "")
        + (f"。候选池:{_mkt}" if _mkt else "")
        + (f",{x_params['exit_note']}"
           if x_params.get("exit_note") else ",双止损口径同标准策略")
        + f"(粗体=与标准策略不同的参数)。"
        f"　{_x_n} 笔已完成:胜率 {_x_wins / _x_n * 100:.1f}%"
        f"({_x_wins}/{_x_n}),费后复利 {_x_cum:+.2f}%,"
        f"平均持有 {_x_days.mean():.0f} 天,"
        f"平均费后 {_x_rets.mean():+.2f}%,"
        f"最佳 {_x_rets.max():+.2f}% / 最差 {_x_rets.min():+.2f}%,"
        f"剔除最佳那笔其余 {_x_n - 1} 笔仍有 {_x_ex:+.2f}%。"
        f"亏损笔:"
        + ("、".join(f"{v:+.2f}%" for v in _x_loss)
           if len(_x_loss) else "无") +
        "。⚠️ 对照实验,只用来跟标准策略比,不是线上在用的规则;"
        "样本十几笔、跨度6年,统计上很薄。"
        + (f"(跑批 #{run_id},跑于 {fmt_cst(x_at, '%Y-%m-%d %H:%M')})"
           if x_at else ""))

    # 择向策略逐笔方向不同(有的买涨幅最大、有的买跌幅最大),
    # 标的那列只能中性叫"选中标的",具体方向看「选向」列。
    _x_ret = f"{_x_win}涨跌(前日口径)"
    view = x_df.rename(columns={_RAW_TGT: TGT_COL, _RAW_RET: _x_ret})
    view = view[[c for c in [
        "买入日", TGT_COL, "类型", "买入时规模(亿)",
        "波动率比值(近3月)", "恐慌阈值", FUND_LINE_COL,
        SSE_LINE_COL, _x_ret, "上证当日涨跌", "选向",
        "卖出日", "持有收益", "期间最高", "期间最大回撤", "同期上证",
    ] if c in view.columns]].copy()
    return caption, view, [trig_of(r) for r in x_df["卖出原因"]]
