# 给 Claude Code 的项目规矩

这份文件跟着 git 走, 换一台电脑照样生效 —— 写在这里的是**跨机器都成立**的
约定, 只对某一台机器成立的东西不要往这里写。

## 一、算任何东西之前, 先把线上库拉下来

```bash
python3 sync_down.py --all      # market/rank/scale/nav, 条件下载, 远端没变零流量
```

**当天没拉过就先拉, 拉过了就不用再拉。** 不许走"本地库停在几天前 → 照算 →
在结论后面补一句免责声明"这条路: 那句声明救不了任何人, 而结论可能是反的。

2026-08-18 的实例: 本地 nav 停在 08-12, 算出 026834 近3月 **-37.87%**, 被 30%
跌幅上限一刀切掉, 于是报给用户"这只今天不会买"; 拉完最新库(到 08-17)重算,
同一只是 **-29.96%**, 恰好过线, 而且**就是当天的选中**。窗口起点只挪了 3 个
交易日, 跌幅差 8 个百分点, 结论完全相反。

nav 库压缩后 61MB。用户常走手机流量, 但**流量不构成拿旧数据算的理由** ——
要省流量就别算, 别算个假的。

数据新鲜度随时可查(`today_pick.data_freshness`): 净值库比上证行情落后几个
交易日, 答案就过期几天。

## 二、谁写哪个库

| 库 | 权威写入方 | 本地怎么用 |
|---|---|---|
| `rank` / `nav` / `scale` | GitHub Actions 基金跑批 | **纯消费**, 随时可被整文件覆盖 |
| `market` / `qvix` | VPS 独立跑批 | **纯消费**, 随时可被整文件覆盖 |
| `strategy`(回测跑批/点评) | **本地**跑回测 | 本地自产, **只推不拉** |

`sync_down.py` 默认组**不含** strategy, 就是这个原因: 拉下来会盖掉
本地还没推的跑批。

网站查询和持仓缓存位于 `/var/lib/fund-web/cache`, 不发布到 Release。

⚠️ **多台电脑的坑**: 两台机器都能跑回测往 strategy 库追加跑批。在 B 机器上
跑回测之前, 先 `python3 sync_down.py strategy` 显式拉一次, 否则 B 推库时
`push_dbs.py` 的行数检查会拦下来(本地比云端少), 或者更糟——两边各攒各的,
谁后推谁把对方的跑批盖掉。推之前那道检查是最后一道防线, 别用 `--force` 绕。

### 为什么 qvix 不归云端跑批管(2026-08-28)

上交所的期权风险指标接口(`option_risk_indicator_sse`, QVIX 的唯一数据源)
**挡 GitHub runner 的地址段**。那天实测: 同一时刻从 VPS 出口拿得到 656 条
合约, 从 runner 上三次重试全失败(一次返回非 JSON 的错误页, 一次直接
connection reset)。不是接口没数据, 也不是笼统封境外 —— VPS 也在境外, 照样
通; 封的是被爬虫用烂的 runner 地址段。换个时间点重试解决不了, 原来那个
`qvix-retry.yml`(早上6点在 GitHub 上补跑)因此一并删了。

所以 QVIX 从 `market.db` 拆进独立的 `qvix.db`, 由 VPS 独占写入。**一张表
一个写入方** —— 跟 index_daily_cache 挤在一个库里的话, 跑批和 VPS 谁后推
谁把对方盖掉, 就是 2026-08-05 那类事故。跑批的 workflow 里那个库列表**没有
qvix.db, 别给它加回去**。

## 三、网站入口与验证

基金分析网站使用 `web_service.py` + `web/`, 部署见 `deploy/README.md`。
本地使用已同步的快照启动:

```bash
FUND_WEB_ROOT=/path/to/fund-web uvicorn web_service:app --host 127.0.0.1 --port 8010
```

查询服务与数据指标检查使用 `python3 -m unittest discover -s tests -v`。
上证页由 `build_site.py` + `sse_view.py` + `site/template.html` 生成,
VPS 的 `vps_site_build.sh` 仍在使用这条链路。
