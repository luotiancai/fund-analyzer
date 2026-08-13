#!/bin/bash
# 按标准策略看今天会买哪只, 结果直接打在屏幕上(纯本地, 不上传)。
#   macOS: Finder 里双击(双击 .command 会开终端执行它)
#   WSL:   ./today_pick.command
#   带日期复盘: ./today_pick.command 2026-03-23
cd "$(dirname "$0")" || exit 1

# venv 优先级同 qvix.command/run.sh:WSL 原生 ext4(~/.venvs)> 项目内 .venv
# > 系统 python3。依赖(pandas/numpy)一般装在 venv 里, 系统 python3 是干净的。
if [ -x "$HOME/.venvs/fund-analyzer/bin/python" ]; then
    PY="$HOME/.venvs/fund-analyzer/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

# 双击时没有参数 = 看今天; 命令行调用时把参数原样透传(日期/--top)。
"$PY" today_pick.py "$@"
rc=$?

# 双击运行时终端会立刻关掉, 留个按键把结果按住。非交互(WSL 里直接跑、管道、
# 脚本调用)下 stdin 不是终端, read 立即返回, 不会卡住。
echo "按任意键关闭…"
read -n 1 -s

# 用 today_pick.py 的退出码, 而不是 read 的 —— read 读到 EOF 会返回非零,
# 让每次非交互运行看起来都像失败了。
exit $rc
