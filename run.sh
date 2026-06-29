#!/usr/bin/env bash
cd "$(dirname "$0")"
pip3 install -q -r requirements.txt
python3 -m streamlit run app.py --server.port 8501
