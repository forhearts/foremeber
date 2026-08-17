"""Prompt Builder — 性格 + 记忆 + 用户对话 拼装（单文件核心）

把三部分拼装成发给文字模型的完整 prompt：
1. 性格（说话风格）：性格 + 说话特点 + 对话示范（persona）
2. 记忆：固定记忆（身份/背景/目标） + 动态记忆（价格/名字/事件）
3. 用户对话：当前玩家输入

依赖 npc/ 记忆系统（MemorySystem / memory_prompt）。
用法：
    from prompt_builder import build_prompt
    prompt = build_prompt("aila", "这剑多少钱？", scene="集市摊位")
    # prompt 可直接发给任意文字模型
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npc.character import load_all_characters
from npc.lorebook import load_lorebook
from npc.memory import MemorySystem
from npc.memory_prompt import build_memory_prompt

# 场景 → 活动描述（让 NPC 能回答"你在干什么"）
SCENE_ACTIVITIES = {
    "集市摊位": "正在集市摆摊，整理货物等顾客",
    "夜晚营地": "正在营地歇脚，收拾行囊",
    "热闹的酒馆": "正在酒馆里，看着来往的客人",
    "村口老树下": "正在村口老树下乘凉",
    "铁匠铺门口": "正在铁匠铺前，检查打好的铁器",
}


class PromptBuilder:
    """性格 + 记忆 + 用户对话 → 完整 prompt。"""

    def __init__(self, memory: MemorySystem | None = None):
        self.mem = memory if memory is not None else MemorySystem()
        self.chars = load_all_characters()
        self.lore = load_lorebook()

    # ---- 1. 性格（说话风格，persona 常驻）----
    def build_style(self, cid: str) -> str:
        """性格部分：性格 + 说话特点 + 示范。"""
        char = self.chars.get(cid)
        if not char:
            return ""
        persona_facts, examples = self.mem.get_persona(cid)
        parts = []
        for f in persona_facts:
            parts.append(f)
        if examples:
            parts.append("我说话的样子（示例）：\n" + "\n".join(examples[:3]))
        parts.append("规则：1.用我的口吻说话；2.回应简短自然；3.我从来不是AI或助手。")
        return "\n".join(parts)

    # ---- 2. 记忆 ----
    def build_memory(self, cid: str, player_input: str) -> str:
        """记忆部分：固定（身份/背景/目标）+ 动态（检索）。"""
        fixed = self.mem.fixed_memories(cid)      # 身份/背景/目标（第一人称）
        dynamic = self.mem.dynamic_memories(cid, player_input)  # 价格/名字/事件
        facts = (fixed + dynamic)[:4]
        return build_memory_prompt(cid, facts)

    # ---- 3. 用户对话 + 场景 ----
    def build_user(self, cid: str, player_input: str, scene: str = "") -> str:
        char = self.chars.get(cid)
        name = char.name if char else cid
        act = SCENE_ACTIVITIES.get(scene, scene)
        user = f"[场景] {act}\n"
        user += f"\n玩家对你说：\"{player_input}\"\n{name}直接回答（1~2句，说出口的话）："
        return user

    # ---- 拼装 ----
    def build_prompt(self, cid: str, player_input: str, scene: str = "") -> dict:
        """返回 {system, user, memory, style} 拼装结果。"""
        style = self.build_style(cid)
        memory = self.build_memory(cid, player_input)
        user = self.build_user(cid, player_input, scene)
        # 记忆注入到 user 部分（模型对末尾注意力最强）
        if memory:
            user = user.replace("\n玩家对你说", f"\n你记得：{memory}\n\n玩家对你说")
        return {"system": style, "user": user, "memory": memory, "style": style}


def build_prompt(cid: str, player_input: str, scene: str = "", memory=None) -> dict:
    """快捷函数：拼装性格+记忆+用户对话 prompt。"""
    return PromptBuilder(memory).build_prompt(cid, player_input, scene)


if __name__ == "__main__":
    import json
    # 演示
    result = build_prompt("aila", "这剑多少钱？", "集市摊位")
    print(json.dumps(result, ensure_ascii=False, indent=2))
