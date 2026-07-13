#!/usr/bin/env bash
cd "$(dirname "$0")"

# 优先使用项目自带的 .venv;不存在则回退到系统 python3
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

"$PY" -m pip install -q -r requirements.txt
"$PY" -m streamlit run app.py --server.port 8501
