"""游戏接入 Demo：村庄场景多 NPC 对话

演示规划 1.2 数据流：意图识别 → 状态判断 → 记忆检索 → 模型生成 → 后处理。
这是一个文字冒险风格的迷你游戏壳，展示引擎如何被游戏逻辑驱动。

用法:
    python game/demo_game.py --weight weights/minimind-3o-pytorch/llm_768_moe.pth
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from npc.engine import DialogueEngine
from npc.model import MiniMindLM
from npc.tokenizer_utils import load_tokenizer


class VillageGame:
    """迷你村庄游戏：场景 + 角色 + 任务状态。"""

    SCENES = {
        "村口": "村口的老树下",
        "酒馆": "热闹的酒馆",
        "集市": "集市摊位",
        "铁匠铺": "铁匠铺门口",
    }

    def __init__(self, engine: DialogueEngine):
        self.engine = engine
        self.location = "村口"
        self.player_hp = 100
        self.gold = 50
        self.quest = "无"
        self.trust_map = {}
        self.affection_map = {}
        self.dialogue_history = {}

    def describe(self):
        print(f"\n=== {self.location} @ {self.SCENES[self.location]} ===")
        print(f"  生命 {self.player_hp} | 金币 {self.gold} | 任务: {self.quest}")

    def available_npcs(self):
        if self.location == "酒馆":
            return ["bruno", "morgan"]
        if self.location == "集市":
            return ["aila"]
        if self.location == "铁匠铺":
            return ["kara"]
        return ["orin", "elda"]

    def move(self, dest):
        if dest in self.SCENES:
            self.location = dest
            print(f"你来到了{dest}。")
        else:
            print(f"没有这个地方: {dest}，可选: {list(self.SCENES.keys())}")

    def talk(self, npc_id: str):
        char = self.engine.characters.get(npc_id)
        if char is None:
            print("没有这个 NPC")
            return
        # 初始化状态
        if npc_id not in self.affection_map:
            self.affection_map[npc_id] = 0
        print(f"\n[{char.name}（{char.identity}）] 好感度={self.affection_map[npc_id]}")
        greeting = char.greetings[0] if char.greetings else "..."
        print(f"{char.name}: {greeting}")

        history = self.dialogue_history.get(npc_id, [])
        while True:
            try:
                text = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text in ("quit", "exit", "离开对话"):
                break
            if not text:
                continue

            # 游戏逻辑控制状态（规划 1.1：剧情由游戏控制）
            state_updates = None
            if "帮忙" in text or "任务" in text:
                self.quest = "护送货物到集市"
                state_updates = {"任务": self.quest}
                print(f"[系统] 接受任务: {self.quest}")
            if "送" in text and ("你" in text or "给" in text):
                self.gold -= 10
                self.affection_map[npc_id] = min(100, self.affection_map[npc_id] + 15)
                state_updates = {"好感度": self.affection_map[npc_id]}
                print(f"[系统] 送礼，好感度 +15 → {self.affection_map[npc_id]}")
            if "滚" in text or "找死" in text:
                self.affection_map[npc_id] = max(0, self.affection_map[npc_id] - 10)
                state_updates = {"好感度": self.affection_map[npc_id]}
                print(f"[系统] 无礼，好感度 -10 → {self.affection_map[npc_id]}")

            result = self.engine.chat(
                npc_id, text,
                scene=self.SCENES[self.location],
                history=history,
                state_updates=state_updates,
            )
            print(f"{char.name} > {result['reply']}")
            history.append(("玩家", text))
            history.append((char.name, result["reply"]))
            history = history[-6:]
            self.dialogue_history[npc_id] = history

    def run(self):
        print("=" * 50)
        print("  边境村庄 — NPC 对话 Demo")
        print("  命令: move <地点> | talk <npc> | status | help | quit")
        print("  地点: 村口 酒馆 集市 铁匠铺")
        print("  可用 NPC: orin elda bruno morgan aila kara")
        print("=" * 50)
        while True:
            try:
                cmd = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd:
                continue
            parts = cmd.split(maxsplit=1)
            action = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            if action in ("quit", "exit"):
                break
            elif action == "help":
                print("move <地点> / talk <npc> / status / help / quit")
            elif action == "move":
                self.move(arg)
            elif action == "talk":
                self.talk(arg)
            elif action == "status":
                self.describe()
            elif action == "where":
                print(f"当前: {self.location}，NPC: {', '.join(self.available_npcs())}")
            else:
                print("未知命令，输入 help 查看帮助")


def main():
    import torch
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="weights/minimind-3o-pytorch/llm_768_moe.pth")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default="qwen", choices=["minimind", "qwen"])
    ap.add_argument("--lora", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "qwen":
        from npc.qwen_engine import load_qwen_engine
        engine = load_qwen_engine(device=device, lora_dir=args.lora)
    else:
        print(f"[load] {args.weight} (device={device})")
        from npc.model import MiniMindLM
        model = MiniMindLM.from_official_checkpoint(args.weight, device=device)
        engine = DialogueEngine(model=model, tokenizer=load_tokenizer())
    game = VillageGame(engine)
    game.run()


if __name__ == "__main__":
    main()
