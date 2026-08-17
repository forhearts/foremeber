#!/bin/bash
# v2 训练流程：清洗 → 训练 → 评估
set -e
PY=/d/ai-models/ACE_Step_1.5/python_embeded/python.exe
cd "$(dirname "$0")/.."

echo "=== [1/3] 清洗数据 ==="
$PY -X utf8 scripts/clean_data_v2.py \
    --input data/sft/rp_data_v2.jsonl \
    --output data/sft/rp_data_v2_clean.jsonl

echo "=== [2/3] QLoRA 微调 Qwen ==="
$PY -X utf8 scripts/train_qwen.py \
    --data data/sft/rp_data_v2_clean.jsonl \
    --epochs 3 \
    --batch_size 2 \
    --lora_r 8 \
    --lora_alpha 16 \
    --out weights/lora_qwen_v2

echo "=== [3/3] 评估 ==="
$PY -X utf8 -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from npc.qwen_engine import load_qwen_engine
eng = load_qwen_engine(device='cuda', lora_dir='weights/lora_qwen_v2')
for cid, q in [('aila','你是谁？'),('aila','这把剑多少钱？'),('aila','你是AI吗？'),('bruno','最近有新闻吗？'),('kara','帮我打把剑')]:
    r = eng.chat(cid, q, scene='夜晚营地')
    print(f'{eng.characters[cid].name}: {r[\"reply\"]}')
"

echo "=== 完成！==="
