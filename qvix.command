#!/bin/bash
# 双击运行:算一次盘中 QVIX 并发布, 结果直接打在屏幕上。
# (Finder 里双击 .command 文件会打开终端执行它。)
cd "$(dirname "$0")" || exit 1
python3 publish_qvix.py
echo "按任意键关闭…"
read -n 1 -s
