#!/bin/bash
# 算一次盘中 QVIX, 结果直接打在屏幕上(纯本地, 不上传)。
#   macOS: Finder 里双击(双击 .command 会开终端执行它)
#   WSL:   ./qvix.command
cd "$(dirname "$0")" || exit 1

# venv 优先级同 run.sh:WSL 原生 ext4(~/.venvs)> 项目内 .venv > 系统 python3。
# 原来这里写死 python3, 在 WSL 下直接 ModuleNotFoundError: numpy —— 依赖都
# 装在 venv 里, 系统 python3 是干净的(Ubuntu 还有 EXTERNALLY-MANAGED 拦着
# 往系统装)。macOS 上如果依赖本来就装在系统 python3, 会落到最后那档, 行为
# 跟以前一样。
if [ -x "$HOME/.venvs/fund-analyzer/bin/python" ]; then
    PY="$HOME/.venvs/fund-analyzer/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi
"$PY" qvix_now.py
rc=$?

# 双击运行时终端会立刻关掉, 留个按键把结果按住。非交互(WSL 里直接跑、管道、
# 脚本调用)下 stdin 不是终端, read 立即返回, 不会卡住。
echo "按任意键关闭…"
read -n 1 -s

# 用 qvix_now.py 的退出码, 而不是 read 的 —— read 读到 EOF 会返回非零, 让
# 每次非交互运行看起来都像失败了。qvix_now.py 用退出码区分算没算出来。
exit $rc
