"""Prompt 构建器

对应规划第 6.4 节"Prompt 拼接模板（八股文）"：
[System] + [State] + [Lore] + [Memory] + [Recent] + [Input]
关键原则：
- 状态记忆 < 100 字、知识碎片 < 150 字、历史检索 < 150 字（预算纪律）
- 越重要信息越靠近 prompt 末尾
- 明确告知"以当前状态为准"
"""
from __future__ import annotations

from npc.character import Character, match_keywords
from npc.lorebook import Lorebook


def build_prompt(
    character: Character,
    player_input: str,
    state: dict | None = None,
    lorebook: Lorebook | None = None,
    memories: list[str] | None = None,
    history: list[tuple[str, str]] | None = None,
    scene: str = "",
    max_history_rounds: int = 3,
) -> str:
    """组装发给模型的完整 prompt（不含角色卡的 system 部分，由调用方拼接）。

    参数:
        character: 角色卡
        player_input: 玩家当前输入
        state: 结构化状态字典 {好感度, 任务, 地点...}
        lorebook: 世界书
        memories: 检索到的历史记忆（短句列表）
        history: 最近对话 [("玩家", "…"), ("艾拉", "…")]
        scene: 当前场景描述
        max_history_rounds: 保留最近几轮
    """
    state = state or {}
    memories = memories or []
    history = history or []

    blocks = []

    # [State] 结构化状态 —— 最稳定，永远最高优先级
    state_lines = []
    if scene:
        state_lines.append(f"场景：{scene}")
    if state.get("好感度") is not None:
        state_lines.append(f"好感度：{state['好感度']}")
    if state.get("任务"):
        state_lines.append(f"任务：{state['任务']}")
    if state.get("信任") is not None:
        state_lines.append(f"信任：{'是' if state['信任'] else '否'}")
    for k, v in state.items():
        if k not in ("好感度", "任务", "信任") and v:
            state_lines.append(f"{k}：{v}")
    if state_lines:
        blocks.append("[当前状态]\n" + "\n".join(state_lines))

    # [Lore] 世界书关键词触发
    query_text = player_input + scene + "".join(f"{p}{r}" for p, r in history)
    if lorebook is not None:
        lore_block = lorebook.to_prompt_block(query_text)
        if lore_block:
            blocks.append(lore_block)

    # [Memory] 检索到的历史记忆
    if memories:
        mem_text = "[相关记忆]\n" + "\n".join(f"- {m}" for m in memories[:3])
        blocks.append(mem_text)

    # [Recent] 短期工作记忆（最近 N 轮）
    if history:
        recent_lines = ["[最近对话]"]
        for speaker, text in history[-max_history_rounds:]:
            recent_lines.append(f"{speaker}：{text}")
        blocks.append("\n".join(recent_lines))

    # [Input] 当前玩家输入（放最末，模型对末尾注意力更强）
    blocks.append(f"[玩家]\n{player_input}")

    # 背景故事触发（角色卡 backstory_keywords）
    if character.background:
        hits = match_keywords(query_text, character.backstory_keywords)
        if hits:
            blocks.insert(1, f"[背景]\n{character.background[:150]}")

    return "\n\n".join(blocks)


def build_system_prompt(character: Character, output_limit: int = 60) -> str:
    """System prompt：只含说话风格（怎么说话），身份/背景/经历全由记忆系统提供。"""
    example = character.greetings[0] if character.greetings else "..."
    return (
        f"我的性格：{character.personality}。\n"
        f"我说话的特点：{character.speech_style}。\n"
        f"我说话的样子（示范）：\"{example}\"\n"
        f"规则：1. 我是游戏里的角色，用我的口吻说话；"
        f"2. 回应要简短自然，像现实对话；"
        f"3. 我不知道的事就按我的性格回应；"
        f"4. 我从来不是AI或助手。"
    )


SCENE_ACTIVITIES = {
    "集市摊位": "正在集市摆摊，整理货物等顾客",
    "夜晚营地": "正在营地歇脚，收拾行囊",
    "热闹的酒馆": "正在酒馆里，看着来往的客人",
    "村口老树下": "正在村口老树下乘凉",
    "铁匠铺门口": "正在铁匠铺前，检查打好的铁器",
}


def build_memory_pack(
    character: Character,
    player_input: str,
    intent: str = '',
    state: dict | None = None,
    memories: list[str] | None = None,
    lore_hits: list | None = None,
    scene: str = '',
) -> dict:
    """构建结构化记忆包（模型无关，任何底模都能消费）。"""
    state = state or {}
    memories = memories or []
    lore_hits = lore_hits or []
    parts = []
    state_str = chr(59).join(f'{k}：{v}' for k, v in state.items()
                             if v not in (None, '', False) and k != '_updated_at')
    if state_str:
        parts.append('[状态] ' + state_str)
    if scene:
        # 场景用活动描述（"集市摊位"→"正在集市摆摊"），让 NPC 能回答"你在干什么"
        act = SCENE_ACTIVITIES.get(scene, scene)
        parts.append('[场景] ' + act)
    if intent:
        parts.append('[意图] ' + intent)
    if lore_hits:
        parts.append('[设定] ' + chr(59).join(h.content if hasattr(h, 'content') else str(h) for h in lore_hits[:2]))
    return {
        'state': state,
        'scene': scene,
        'intent': intent,
        'memories': memories,
        'lore': [h.content if hasattr(h, 'content') else str(h) for h in lore_hits],
        'text': chr(10).join(parts),
    }
