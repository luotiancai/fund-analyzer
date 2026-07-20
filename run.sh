#!/usr/bin/env bash
cd "$(dirname "$0")"

# venv 优先级:WSL 原生 ext4(~/.venvs)> 项目内 .venv > 系统 python3。
# 项目在 /mnt/c(9p 文件系统)上,venv 放这儿冷启动导入 pandas/numpy 等
# 要 ~20s+;挪到 ext4 后秒级。项目内 .venv 仅作旧机器回退。
if [ -x "$HOME/.venvs/fund-analyzer/bin/python" ]; then
    VENV="$HOME/.venvs/fund-analyzer"
elif [ -x ".venv/bin/python" ]; then
    VENV=".venv"
else
    VENV=""
fi
PY="${VENV:+$VENV/bin/python}"
PY="${PY:-python3}"

# requirements.txt 没变就跳过 pip(空跑也要 ~3s)
if [ -n "$VENV" ]; then
    REQ_STAMP="$VENV/.req-$(md5sum requirements.txt | cut -d' ' -f1)"
    if [ ! -f "$REQ_STAMP" ]; then
        "$PY" -m pip install -q -r requirements.txt \
            && rm -f "$VENV"/.req-* && touch "$REQ_STAMP"
    fi
fi

"$PY" -m streamlit run app.py --server.port 8501
