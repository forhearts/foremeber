"""交互式 NPC 对话 Demo（命令行）

用法:
    python scripts/demo_chat.py --character aila --scene "夜晚营地"
    python scripts/demo_chat.py --character bruno --scene "酒馆" --interactive
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from npc.config import LLM_WEIGHT, GenerationConfig, SFT_WEIGHT
from npc.engine import DialogueEngine
from npc.model import MiniMindLM
from npc.tokenizer_utils import load_tokenizer


def load_engine(weight: str, device: str = "auto", lora_dir: str = None):
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] 权重: {weight} (device={device})")
    from npc.model import load_model
    model = load_model(weight, device=device, lora_dir=lora_dir)
    model.eval()
    tok = load_tokenizer()
    engine = DialogueEngine(model=model, tokenizer=tok)
    print(f"[ready] 可用角色: {list(engine.characters.keys())}")
    print(f"[ready] 世界书条目: {len(engine.lorebook.entries)}")
    return engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="aila")
    ap.add_argument("--model", default="qwen", choices=["minimind", "qwen"], help="基座模型")
    ap.add_argument("--scene", default="夜晚营地")
    ap.add_argument("--weight", default=str(LLM_WEIGHT))
    ap.add_argument("--lora", default=None, help="LoRA adapter 目录")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.model == "qwen":
        from npc.qwen_engine import load_qwen_engine
        engine = load_qwen_engine(device=args.device, lora_dir=args.lora)
    else:
        engine = load_engine(args.weight, args.device, args.lora)
    char = engine.characters.get(args.character)
    if char is None:
        print(f"角色 {args.character} 不存在！可用: {list(engine.characters.keys())}")
        sys.exit(1)
    print(f"\n=== 与 {char.name}（{char.identity}）对话 @ {args.scene} ===")
    print(f"{char.name}: {char.greetings[0] if char.greetings else ''}")

    if args.interactive:
        history = []
        while True:
            try:
                text = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text in ("quit", "exit", "退出"):
                break
            result = engine.chat(args.character, text, scene=args.scene, history=history)
            print(f"{char.name} > {result['reply']}")
            history.append(("玩家", text))
            history.append((char.name, result["reply"]))
            # 只保留最近几轮
            history = history[-6:]
    else:
        while True:
            try:
                text = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text in ("quit", "exit", "退出"):
                break
            result = engine.chat(args.character, text, scene=args.scene)
            print(f"{char.name} > {result['reply']}  [意图: {result['intent']}]")


if __name__ == "__main__":
    main()
