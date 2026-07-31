#!/usr/bin/env python3
"""QVIX 实时计算链路诊断脚本。

在 Streamlit Cloud 上跑：
  cd ~/.local/share/fund-analyzer  # 或项目目录
  python3 qvix_diag.py

逐个测试自算QVIX链路里的每个网络依赖,定位是哪一步被新浪封了IP
还是 optbbs 也一起挂了。每步都打印: 成功/失败 + 响应时间 + 关键信息。
"""

import time
import datetime as dt
from zoneinfo import ZoneInfo

_CST = ZoneInfo("Asia/Shanghai")

def _step(name, fn, timeout=15):
    """跑一步,打印耗时和结果。"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = fn(timeout)
        elapsed = time.time() - t0
        print(f"  ✅ 成功 ({elapsed:.1f}s)")
        if result:
            # 截断长输出
            s = str(result)
            print(f"  结果: {s[:200]}{'...' if len(s)>200 else ''}")
        return True, result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ❌ 失败 ({elapsed:.1f}s): {e}")
        return False, None


def main():
    now = dt.datetime.now(_CST)
    print(f"\nQVIX 链路诊断  ·  {now.strftime('%Y-%m-%d %H:%M:%S')} CST")
    print(f"运行环境: {'Streamlit Cloud' if __import__('os').environ.get('STREAMLIT_CLOUD') else '本地/其他'}")

    # ── Step 1: 基础网络连通性(能不能出墙) ──────────────────────────
    import requests

    def test_baidu(timeout):
        r = requests.get("https://www.baidu.com", timeout=timeout)
        return f"HTTP {r.status_code}, {len(r.text)} bytes"
    _step("Step 1: 基础网络连通性 (baidu.com)", test_baidu)

    # ── Step 2: 新浪行情接口 hq.sinajs.cn ─────────────────────────────
    # 这是自算QVIX的核心: 拉期权实时买卖盘报价
    def test_sina_hq(timeout):
        # 先拉当前活跃合约代码,再测试行情接口
        import akshare as ak
        codes_df = ak.option_sse_codes_sina(
            symbol="看涨期权", trade_date="2609",
            underlying="510050")
        if codes_df is None or codes_df.empty:
            codes_df = ak.option_sse_codes_sina(
                symbol="看涨期权", trade_date="2608",
                underlying="510050")
        if codes_df is None or codes_df.empty:
            return "拿不到合约代码,无法测试行情接口"
        first_code = str(codes_df["期权代码"].iloc[0])
        url = f"https://hq.sinajs.cn/list=CON_OP_{first_code}"
        headers = {
            "Referer": "https://stock.finance.sina.com.cn/",
            "User-Agent": "Mozilla/5.0",
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        r.encoding = "gbk"
        text = r.text
        import re
        m = re.search(r'CON_OP_\d+="([^"]*)"', text)
        if m and m.group(1):
            parts = m.group(1).split(",")
            if len(parts) >= 8:
                return (f"HTTP {r.status_code}, 合约{first_code}, "
                        f"行权价={parts[7]}, 买价={parts[1]}, 卖价={parts[3]}")
        # 空引号 = 合约存在但当前无报价(盘前/已到期),连通性本身没问题
        if 'hq_str_CON_OP_' in text:
            return (f"HTTP {r.status_code}, 合约{first_code}返回空报价 "
                    f"(盘前/非交易时段正常, 连通性OK)")
        return f"HTTP {r.status_code}, 响应异常: {text[:100]}"

    ok_sina, _ = _step("Step 2: 新浪行情接口 (hq.sinajs.cn) — 自算QVIX核心依赖", test_sina_hq)

    # ── Step 3: akshare 期权合约代码列表 ──────────────────────────────
    # compute_qvix → _fetch_chain → option_sse_codes_sina
    def test_opt_codes(timeout):
        import akshare as ak
        df = ak.option_sse_codes_sina(symbol="看涨期权", trade_date="2608",
                                      underlying="510050")
        return f"拿到 {len(df)} 个认购合约代码"
    _step("Step 3: akshare option_sse_codes_sina (期权代码列表)", test_opt_codes, timeout=20)

    # ── Step 4: akshare 到期日探测 ─────────────────────────────────────
    # compute_qvix → _expiry_candidates → option_sse_expire_day_sina
    def test_expire_day(timeout):
        import akshare as ak
        result = ak.option_sse_expire_day_sina(trade_date="2608", symbol="50ETF")
        return f"到期日={result}"
    _step("Step 4: akshare option_sse_expire_day_sina (到期日探测)", test_expire_day, timeout=20)

    # ── Step 5: 完整自算QVIX ───────────────────────────────────────────
    def test_compute_qvix(timeout):
        import fetcher
        fetcher.init_db()
        import qvix_calc
        r = qvix_calc.compute_qvix()
        if r is not None:
            return f"QVIX={r[0]}, 时间={r[1]}"
        return "compute_qvix 返回 None (链路内部某步失败,见上面Step 2-4)"
    ok_self, _ = _step("Step 5: 完整自算QVIX (compute_qvix)", test_compute_qvix, timeout=45)

    # ── Step 6: optbbs 兜底 ───────────────────────────────────────────
    def test_optbbs(timeout):
        import akshare as ak
        df = ak.index_option_50etf_min_qvix()
        if df is not None and not df.empty:
            df["qvix"] = __import__("pandas").to_numeric(df["qvix"], errors="coerce")
            df = df.dropna(subset=["qvix"])
            if not df.empty:
                last = df.iloc[-1]
                return f"optbbs QVIX={last['qvix']:.2f}, 时间={last['time']}"
        return "optbbs 返回空数据"
    ok_optbbs, _ = _step("Step 6: optbbs 兜底 (index_option_50etf_min_qvix)", test_optbbs, timeout=25)

    # ── 汇总 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  诊断汇总")
    print(f"{'='*60}")
    if not ok_sina:
        print("  🔴 新浪 hq.sinajs.cn 连不上 → 自算QVIX必然失败")
        print("     原因: 新浪对境外IP限制/封禁, 或Streamlit Cloud出口IP被拉黑")
        print("     对策: 自算在云端不可用, 需依赖optbbs兜底")
    else:
        print("  🟢 新浪 hq.sinajs.cn 可连通 → 自算QVIX网络层没问题")
        if not ok_self:
            print("     但自算仍失败 → 可能是akshare接口/数据解析问题,需看具体日志")

    if not ok_optbbs:
        print("  🔴 optbbs 也连不上 → 兜底也失败, 页面显示「暂不可用」")
        print("     原因: optbbs(1.optbbs.com)同样可能限制境外IP")
    else:
        print("  🟢 optbbs 可用 → 即使自算失败也有兜底")

    if not ok_sina and not ok_optbbs:
        print("\n  ⚠️  新浪和optbbs都连不上: 页面的实时QVIX完全不可用。")
        print("     建议: 在GitHub Actions(境内/中转IP)里跑一个定时任务,")
        print("     把盘中QVIX写入DB Release, 云端读缓存而非实时拉取。")

    print()


if __name__ == "__main__":
    main()
