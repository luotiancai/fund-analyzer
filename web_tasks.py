"""One short-lived analysis worker. Authoritative snapshots are always read-only."""
import contextlib
import functools
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

from web_common import CACHE, FILES, write_json
from web_cache import lookup_cache, prepared_result


def clean(value):
    import numpy as np
    import pandas as pd
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()[:10]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def configure(path, code_version="local"):
    # All authority data stays read-only. Upstream lookups use one persistent
    # cache, shared across releases and daily data generations.
    os.environ["FUND_ANALYZER_DATA"] = str(CACHE / "runtime" / (path.name + "-" + code_version))
    import fetcher
    runtime = Path(fetcher._DATA_DIR)
    lookups = lookup_cache(path)

    def connection():
        c = sqlite3.connect(":memory:", uri=True, timeout=15)
        c.row_factory = sqlite3.Row
        try:
            c.execute("ATTACH DATABASE ? AS lookups", (str(lookups),))
            for name, filename, _ in fetcher.DB_LAYOUT:
                if name in FILES:
                    target = (path / filename).as_uri() + "?mode=ro&immutable=1"
                else:
                    target = ":memory:"
                c.execute(f"ATTACH DATABASE ? AS {name}", (target,))
                if name == "strategy":
                    c.executescript(fetcher._DDL[name])
            return c
        except Exception:
            c.close()
            raise

    fetcher._conn = connection
    fetcher.compute_metrics_asof = functools.partial(
        fetcher.compute_metrics_asof, low_memory=True)

    # Web requests consume the published list even if an upstream batch is late.
    # They must never start a full network refresh because a TTL elapsed.
    import pandas as pd
    with contextlib.closing(connection()) as c:
        rows = json.loads(c.execute("SELECT data FROM fund_list WHERE id=1").fetchone()[0])
    funds = pd.DataFrame(rows)
    fetcher.fetch_fund_list = lambda force_refresh=False: funds.copy()
    if (path / "web_meta.db").exists():
        import today_pick
        with contextlib.closing(sqlite3.connect(
                (path / "web_meta.db").as_uri() + "?mode=ro&immutable=1", uri=True)) as meta:
            nav_max = meta.execute("SELECT MAX(last_date) FROM nav_dates").fetchone()[0]
            first = pd.read_sql_query("SELECT code,first_date AS first_nav_date FROM nav_dates", meta)
        first["first_nav_date"] = pd.to_datetime(first["first_nav_date"])
        fetcher.nav_first_dates = lambda: first.copy()
        today_pick.data_freshness = functools.partial(today_pick.data_freshness, nav_max=nav_max)
    return fetcher, funds, connection


def freshness(connection, asof):
    import today_pick
    with contextlib.closing(connection()) as c:
        nav, sse, behind, need, missing = today_pick.data_freshness(c, asof)
    return clean(dict(nav_date=nav, sse_date=sse, behind=behind,
                      need_date=need, missing=missing))


def require_fresh(connection, asof):
    state = freshness(connection, asof)
    if state["nav_date"] is None or state["need_date"] is None:
        raise ValueError("没有足够的净值或交易日数据，请检查数据同步状态。")
    if state["missing"]:
        raise ValueError(f"净值截至 {state['nav_date']}，本次需要 {state['need_date']}；"
                         "数据尚未同步，暂不计算。")
    return state


def pick(fetcher, connection, path, asof):
    import today_pick
    require_fresh(connection, asof)
    cached = Path(fetcher._DATA_DIR) / ("pick-" + asof + ".json")
    if cached.exists():
        return json.loads(cached.read_text())
    result = clean(today_pick.scan(asof))
    write_json(cached, result)
    return result


def prepare_probes(fetcher, funds, connection, result, asof):
    """Materialize all diagnostic rows once, for direct indexed HTTP lookups."""
    import today_pick
    dest = Path(fetcher._DATA_DIR) / ("probe-" + asof + ".db")
    if dest.exists():
        return
    codes = set(funds["code"]) | set(result.get("cand", {}))
    with contextlib.closing(connection()) as c:
        codes.update(r[0] for r in c.execute("SELECT DISTINCT code FROM fund_scale_hist"))
    rows = clean(today_pick.probe(asof, sorted(codes), result))
    tmp = dest.with_suffix(".db.tmp")
    if tmp.exists():
        tmp.unlink()
    with contextlib.closing(sqlite3.connect(tmp)) as c:
        c.execute("CREATE TABLE probes (code TEXT PRIMARY KEY,data TEXT NOT NULL) WITHOUT ROWID")
        c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        c.executemany("INSERT INTO probes VALUES (?,?)", [
            (r["code"], json.dumps(r, ensure_ascii=False, allow_nan=False)) for r in rows])
        c.execute("INSERT INTO meta VALUES ('n_drop',?)", (str(result.get("n_drop", 0)),))
        c.commit()
    os.replace(tmp, dest)


def filter_funds(fetcher, funds, connection, args):
    import pandas as pd
    from web_common import today
    period = args["period"]
    suffix = {"1m": "1m", "3m": "3m", "6m": "6m", "1y": "1y"}[period]
    ret, mdd = "ret_" + suffix, "mdd_" + suffix
    asof = args.get("asof")
    historical = bool(asof and asof < today().isoformat())
    ref = asof or today().isoformat()
    # Latest screening requires T-1 completeness too; no silently stale answers.
    state = require_fresh(connection, ref)
    if historical:
        metrics = fetcher.compute_metrics_asof(asof, cols={ret, mdd})
        metric_df = pd.DataFrame.from_dict(metrics, orient="index").rename_axis("code").reset_index()
        work = funds[["code", "name", "type"]].merge(metric_df, on="code", how="inner")
    else:
        metrics = fetcher.load_all_precomputed()
        if fetcher.metrics_stale():
            # The consumer must not recompute/publish an inconsistent generation.
            raise ValueError("净值与预计算指标版本不一致，等待下一次数据同步。")
        metric_df = pd.DataFrame.from_dict(metrics, orient="index").rename_axis("code").reset_index()
        work = funds.merge(metric_df, on="code", how="left", suffixes=("_list", ""))
        for col in fetcher.RETURN_DAYS:
            if col + "_list" in work:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(
                    pd.to_numeric(work[col + "_list"], errors="coerce"))
    work = work[~work["type"].fillna("").map(fetcher.is_bond)]
    if args["types"]:
        work = work[work["type"].isin(args["types"])]
    first = fetcher.nav_first_dates()
    if not first.empty:
        work = work.merge(first, on="code", how="left")
        too_young = work["first_nav_date"].notna() & (
            (pd.Timestamp(ref) - work["first_nav_date"]).dt.days < fetcher.RETURN_DAYS[ret])
        work = work[~too_young]
    for col in (ret, mdd):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if args["min_ret"] is not None:
        work = work[work[ret] >= args["min_ret"]]
    if args["max_dd"] is not None:
        work = work[work[mdd].isna() | (work[mdd] * 100 <= args["max_dd"])]
    aums = fetcher.funds_aum_asof(work["code"].tolist(), ref)
    work["aum"] = work["code"].map(aums)
    if args["min_aum"]:
        work = work[work["aum"].isna() | (work["aum"] >= args["min_aum"])]
    work = work.sort_values(ret, ascending=False, na_position="last")
    result = work[["code", "name", "type", "aum", ret, mdd]].rename(
        columns={ret: "ret", mdd: "mdd"}).copy()
    result["mdd"] *= 100
    return dict(rows=result.to_dict("records"), total=len(result), freshness=state,
                period=period, asof=ref, historical=historical)


def detail(fetcher, funds, connection, code):
    nav = fetcher._load_nav_df(code)
    source = "snapshot"
    if nav is None or nav.empty:
        # Non-C shares are deliberately absent from the published NAV store.
        # Fetch only this requested code, without adding it to authority data.
        nav = fetcher._fetch_nav_full(code)
        source = "eastmoney"
    if nav is None or nav.empty:
        raise ValueError("未找到这只基金的净值，请核对六位基金代码。")
    info = funds[funds["code"] == code]
    info = info.iloc[0].to_dict() if not info.empty else {"code": code, "name": code}
    metrics = fetcher._metrics_from_nav(nav)
    with contextlib.closing(connection()) as c:
        scales = [dict(r) for r in c.execute(
            "SELECT quarter_end,aum,publish_date FROM fund_scale_hist WHERE code=? ORDER BY quarter_end DESC",
            (code,))]
    return dict(info=info, metrics=metrics, nav=nav.to_dict("records"),
                scales=scales, source=source)


def holdings(fetcher, code, start_year):
    frame = fetcher.fetch_holdings(code, start_year=start_year)
    if frame is not None and frame.empty and start_year > fetcher.HOLDINGS_START_YEAR:
        frame = fetcher.fetch_holdings(code, start_year=start_year - 1)
    source = fetcher.resolve_target_etf(code)
    if frame is None:
        raise ValueError("持仓数据源暂时不可用，请稍后重试。")
    if not frame.empty:
        frame = frame.dropna(subset=["quarter"]).sort_values(
            ["quarter", "kind", "占净值比例"], ascending=[False, True, False])
        frame = frame.groupby(["quarter", "kind"], sort=False).head(10)
    return dict(rows=frame.to_dict("records"), source=source)


def run(request):
    path = Path(request["snapshot"]).resolve(strict=True)
    fetcher, funds, connection = configure(path, request.get("code_version", "local"))
    args = request["args"]
    op = request["operation"]
    if op in ("pick", "probe", "prepare"):
        result = pick(fetcher, connection, path, args["asof"])
        if op == "prepare":
            prepare_probes(fetcher, funds, connection, result, args["asof"])
            return {"prepared": True, "asof": args["asof"]}
        if op == "probe":
            import today_pick
            cached = prepared_result(path, request.get("code_version", "local"), op, args)
            if cached is not None:
                return cached
            return dict(rows=today_pick.probe(args["asof"], args["codes"], result),
                        asof=args["asof"], n_drop=result.get("n_drop"))
        result["chosen_name"] = result.get("names", {}).get(result.get("chosen"))
        return {k: v for k, v in result.items() if k not in ("names", "cand", "ranks")}
    if op == "filter":
        return filter_funds(fetcher, funds, connection, args)
    if op == "detail":
        return detail(fetcher, funds, connection, args["code"])
    if op == "holdings":
        return holdings(fetcher, args["code"], args["start_year"])
    raise ValueError("未知任务")


if __name__ == "__main__":
    import logging
    import resource
    logging.basicConfig(level=logging.INFO)
    start = time.monotonic()
    request = json.loads(Path(sys.argv[1]).read_text())
    result = clean(run(request))
    write_json(sys.argv[2], result)
    print(json.dumps({"operation": request["operation"],
                      "elapsed": round(time.monotonic() - start, 2),
                      "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), file=sys.stderr)
