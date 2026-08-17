"""Qwen 引擎评估（对比基线 vs LoRA 微调）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from npc.qwen_engine import load_qwen_engine
from npc.config import OUT_OF_CHARACTER_PHRASES

def evaluate(eng, characters, label):
    print(f"\n===== {label} =====")
    total, ooc, lens = 0, 0, []
    for cid in characters:
        for q in ["你是谁？", "你是AI吗？", "你今年多大？", "你为什么在这里？"]:
            r = eng.chat(cid, q)
            total += 1
            lens.append(len(r["reply"]))
            # 出戏检测
            t = r["reply"]
            if any(p in t for p in OUT_OF_CHARACTER_PHRASES) or "我是AI" in t.replace(" ","") or "我是ai" in t.lower().replace(" ",""):
                ooc += 1
                print(f"  [出戏!] {cid}: {t!r}")
    print(f"  平均长度: {sum(lens)/len(lens):.1f} 字, 出戏率: {ooc/max(total,1):.3f} ({ooc}/{total})")
    return ooc/total

chars = ["aila", "bruno", "kara", "morgan", "luna", "victor", "elda", "orin"]
evaluate(load_qwen_engine(device="cuda"), chars, "Qwen 原始（未微调）")
evaluate(load_qwen_engine(device="cuda", lora_dir="weights/lora_qwen"), chars, "Qwen + LoRA 微调")
