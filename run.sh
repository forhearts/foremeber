#!/bin/bash
# ForeMeber 运行脚本（uv 环境）
# 用法: bash run.sh [chat|webui|test]
PY=.venv/Scripts/python.exe
case "${1:-chat}" in
  chat)  $PY -X utf8 foremeber.py --character aila --scene "集市摊位" --interactive ;;
  webui) $PY -X utf8 webui.py ;;
  test)  $PY -m pytest tests/ -v ;;
  *) echo "用法: bash run.sh [chat|webui|test]" ;;
esac
