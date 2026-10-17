#!/usr/bin/env python3
"""把上证页(行情图 + 策略复盘 + 对照回测 + 每日数据)生成成一张静态网页。

    python3 build_site.py --out /opt/fund-analyzer/site

跑在 BWG 上, 由 vps_site_build.sh 每小时调一次, nginx 直接发这个目录。
为什么不用 Streamlit: 那边每点一下都把整份 app.py 重跑一遍、再经 websocket
把整张图传回来, 手机上又卡又慢; 而这几块数据一天只变一次、只看不改, 预先
生成好就够了 —— 切时间区间这类交互全在浏览器里做, 不回服务器。

只读 market / qvix / strategy 三个小库(加起来 2MB 不到), 1GB 的 BWG 跑得动。
「今天会买哪只」和基金详情要净值库(解开 385MB), 不在这里, 留在 Streamlit。

统计说明、列顺序、止损染色都来自 sse_view.py, 跟 Streamlit 版共用一份。
"""

import argparse
import json
import math
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetcher    # noqa: E402
import sse_view   # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "site", "template.html")

# echarts 只要折线/散点 + markArea/tooltip/legend, common 版就够(gz ~220KB,
# 完整版 ~340KB)。文件名带版本号, nginx 给它一年缓存, 手机只下一次。
ECHARTS_VER = "5.5.1"
ECHARTS_FILE = f"echarts-{ECHARTS_VER}.common.min.js"
ECHARTS_URL = (f"https://cdn.jsdelivr.net/npm/echarts@{ECHARTS_VER}"
               "/dist/echarts.common.min.js")


def _cell(v):
    """跟 Streamlit 版 Styler.format(precision=2, na_rep="") 同口径。"""
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.2f}"
    return str(v)


def _table(df, triggers):
    return {"cols": list(df.columns),
            "rows": [[_cell(v) for v in row] for row in df.itertuples(index=False)],
            "hl": [sse_view.trig_cols(t) for t in triggers]}


def _num(v, nd):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) \
        else round(float(v), nd)


def collect() -> dict:
    sse = fetcher.fetch_sse_daily()
    if sse is None or sse.empty:
        raise SystemExit("上证行情是空的(market.db 没同步?), 不生成")
    sse = sse.sort_values("date").reset_index(drop=True)

    out = {
        "gen": sse_view.fmt_cst(time.time(), "%Y-%m-%d %H:%M"),
        "ranges": [[k, v] for k, v in sse_view.SSE_RANGES.items()],
        "rdef": sse_view.SSE_RANGE_DEFAULT,
        "qcolor": sse_view.QVIX_COLOR,
        "bg": sse_view.TRIG_BG,
        "sse": {"d": sse["date"].astype(str).tolist(),
                "c": [_num(v, 2) for v in sse["close"]],
                "p": [_num(v, 3) for v in sse["pct"]]},
    }

    hist = fetcher.load_qvix_self_history()
    if hist is not None and not hist.empty:
        hist = hist.sort_values("date").reset_index(drop=True)
        thr = sse_view.qvix_thresholds(hist)
        out["qv"] = {"d": hist["date"].astype(str).tolist(),
                     "v": [_num(v, 2) for v in hist["qvix"]],
                     "t": [_num(v, 3) for v in thr["thr"]],
                     # 实时路径收盘值(只作展示), 老库没有这列就全空
                     "r": [_num(v, 2) for v in hist.get("qvix_rt",
                                                        [None] * len(hist))],
                     "bk": [1 if b else 0
                            for b in sse_view.chain_broken_mask(hist)]}

    rv = {"intro": sse_view.md_inline(sse_view.REVIEW_INTRO)}
    bt_df, _p, bt_at, bt_id = fetcher.load_backtest_trades()
    if bt_df is None or bt_df.empty:
        rv["empty"] = sse_view.md_inline(
            "暂无回测明细——跑一次 `python3 backtest_qvix.py` "
            "会往策略库追加一条标准跑批, 页面随即显示。")
    else:
        rv["summary"] = sse_view.md_inline(
            sse_view.review_summary(bt_df, bt_id, bt_at))
        df, trig = sse_view.review_table(bt_df, fetcher.load_backtest_notes())
        rv["table"] = _table(df, trig)
    out["review"] = rv

    runs = []
    for r in fetcher.list_strategy_runs():
        if r["is_standard"]:
            continue
        x_df, x_params, x_at = fetcher.load_strategy_run(r["id"])
        item = {"label": sse_view.run_label(r)}
        if x_df is None or x_df.empty:
            item["caption"] = "这次跑批没有明细。"
        else:
            cap, view, trig = sse_view.run_detail(x_df, x_params, r["id"], x_at)
            item["caption"] = sse_view.md_inline(cap)
            item["table"] = _table(view, trig)
        runs.append(item)
    out["runs"] = runs
    return out


def ensure_echarts(out_dir: str):
    dest = os.path.join(out_dir, ECHARTS_FILE)
    if os.path.exists(dest):
        return
    r = requests.get(ECHARTS_URL, timeout=60)
    r.raise_for_status()
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    print(f"下载 {ECHARTS_FILE} ({len(r.content) / 1024:.0f}KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出目录(nginx 的 root)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ensure_echarts(a.out)

    data = json.dumps(collect(), ensure_ascii=False, separators=(",", ":"))
    # 嵌在 <script> 里, "</" 必须拆开, 否则备注里万一出现 </script> 会提前闭合
    data = data.replace("</", "<\\/")
    with open(TEMPLATE, encoding="utf-8") as f:
        page = (f.read().replace("__ECHARTS__", ECHARTS_FILE)
                .replace("__DATA__", data))
    # 先写临时文件再改名: nginx 任何时刻读到的都是完整的一份
    dest = os.path.join(a.out, "index.html")
    with open(dest + ".tmp", "w", encoding="utf-8") as f:
        f.write(page)
    os.replace(dest + ".tmp", dest)
    print(f"生成 {dest} ({len(page.encode()) / 1024:.0f}KB)")


if __name__ == "__main__":
    main()
