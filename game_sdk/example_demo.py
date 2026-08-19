"""游戏嵌入示例：把 game_sdk 接进你的游戏

完整展示：状态驱动、记忆、送礼加好感、任务系统。
用法:
    python example_game.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from npc_api import NPCSystem


class MyGame:
    """一个迷你游戏壳，演示 SDK 怎么被游戏逻辑驱动。"""

    def __init__(self):
        self.npc = NPCSystem()  # 启动引擎 + 记忆
        self.hp = 100
        self.gold = 50

    def play(self):
        print("=== 边境村庄（SDK 示例）===")
        print("NPC: aila(艾拉) bruno(布鲁诺) kara(卡拉)")
        while True:
            try:
                cmd = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd in ("quit", "exit"):
                break
            parts = cmd.split(maxsplit=1)
            act = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if act == "talk":
                cid = arg or "aila"
                self.talk(cid)
            elif act == "give":
                self.give(arg or "aila")
            elif act == "quest":
                print("[游戏] 接受任务：护送货物到集市")
                self.npc.set_state("aila", 任务="护送货物")
            elif act == "event":
                print(self.npc.generate("aila", "事件", arg or "商队抵达", scene="集市"))
            elif act == "desc":
                print(self.npc.generate("aila", "描述", arg or "月光剑"))
            elif act == "sys":
                print(self.npc.generate("aila", "系统消息", arg or "任务完成"))
            else:
                print("命令: talk <npc> / give <npc> / quest / event <事> / desc <物> / sys <消息> / quit")

    def talk(self, cid):
        char = self.npc.chars.get(cid)
        if not char:
            print("无此 NPC")
            return
        print(f"[{char.name} 好感度 {self.npc.get_state(cid).get('好感度', 0)}]")
        while True:
            try:
                text = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text in ("quit", "exit", "back"):
                break
            reply = self.npc.chat(cid, text, scene="集市摊位")
            print(f"{char.name}> {reply}")

    def give(self, cid):
        """送礼：游戏逻辑更新好感度。"""
        st = self.npc.get_state(cid)
        st["好感度"] = min(100, st.get("好感度", 0) + 15)
        self.npc.set_state(cid, **st)
        print(f"[游戏] 送礼成功！{cid} 好感度 +15 → {st['好感度']}")
        reply = self.npc.chat(cid, "这个送给你，拿着吧。", scene="集市摊位")
        char = self.npc.chars[cid]
        print(f"{char.name}> {reply}")


if __name__ == "__main__":
    MyGame().play()
