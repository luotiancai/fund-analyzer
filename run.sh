#!/usr/bin/env bash
cd "$(dirname "$0")"

# 优先使用项目自带的 .venv;不存在则回退到系统 python3
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

# requirements.txt 没变就跳过 pip(空跑也要 ~3s)
REQ_STAMP=".venv/.req-$(md5sum requirements.txt | cut -d' ' -f1)"
if [ ! -f "$REQ_STAMP" ]; then
    "$PY" -m pip install -q -r requirements.txt \
        && rm -f .venv/.req-* && touch "$REQ_STAMP"
fi

"$PY" -m streamlit run app.py --server.port 8501
