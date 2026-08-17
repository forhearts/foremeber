#!/bin/bash
# 一键训练脚本：生成数据 → LoRA 微调 → 评估
set -e
PY=/d/ai-models/ACE_Step_1.5/python_embeded/python.exe
cd "$(dirname "$0")/.."

echo "=== [1/3] 生成训练数据（如已有则跳过） ==="
if [ ! -f data/sft/rp_data.jsonl ]; then
    $PY -X utf8 scripts/generate_sft_data.py --count 300 --out data/sft/rp_data.jsonl --workers 4
else
    echo "已存在 data/sft/rp_data.jsonl ($(wc -l < data/sft/rp_data.jsonl) 条)"
fi

echo "=== [2/3] LoRA 微调 ==="
$PY -X utf8 scripts/train_sft.py \
    --data data/sft/rp_data.jsonl \
    --epochs 3 \
    --batch_size 4 \
    --max_len 512 \
    --lora \
    --lora_out weights/lora_rp \
    --lora_r 16 \
    --lora_alpha 32

echo "=== [3/3] 评估 ==="
$PY -X utf8 scripts/evaluate.py --lora weights/lora_rp --characters aila bruno

echo "=== 完成！启动 Demo ==="
echo "python scripts/demo_chat.py --lora weights/lora_rp"
