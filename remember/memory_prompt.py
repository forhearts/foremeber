"""记忆提示词构建器：把记忆事实转成"角色视角"的自然提示词

核心设计（用户目标）：
- 记忆系统 = 组织记忆提示词（不是数据库 dump）
- 输出给模型的记忆提示词必须是"该角色记得的事"，用第一人称自然表述
- 常识（砍价不涨价/买卖逻辑）由模型自身能力处理，记忆系统不教

示例：
  数据库事实: aila把剑定价为五百金币
  → 记忆提示词: 这把剑我报过价，五百金币
"""
from __future__ import annotations

import re


def fact_to_memory_line(character_id: str, fact: str) -> str | None:
    """把一条数据库事实转成角色视角的记忆提示词。返回 None 表示不适合作为记忆。"""
    # 1. 价格事实: "aila把剑定价为五百金币" → "这把剑我定价五百金币"
    m = re.search(rf"^{character_id}把(.+?)定价为(.+)$", fact)
    if m:
        item, price = m.group(1), m.group(2)
        return f"这把{item}我定价{price}"

    # 2. 价格陈述: "aila说价格是五百金币" → "我说过价格是五百金币"
    m = re.search(rf"^{character_id}说价格是(.+)$", fact)
    if m:
        return f"我说过价格是{m.group(1)}"

    # 3. 玩家身份: "玩家自称林风" → "有个玩家说他叫林风"
    m = re.search(r"^玩家自称(.+)$", fact)
    if m:
        return f"有个玩家说他叫{m.group(1)}"

    # 4. 玩家来历: "玩家来自北方" → "有个玩家是从北方来的"
    m = re.search(r"^玩家来自(.+)$", fact)
    if m:
        return f"有个玩家是从{m.group(1)}来的"

    # 5. 玩家想要: "玩家想要一把好剑" → "有个玩家想要一把好剑"
    m = re.search(r"^玩家想要(.+)$", fact)
    if m:
        return f"有个玩家想要{m.group(1)}"

    # 6. 关系: "玩家帮助过aila" → "这个玩家帮过我"
    m = re.search(rf"^玩家(.+?)过{re.escape(character_id)}$", fact)
    if m:
        return f"这个玩家{m.group(1)}过我"

    # 7. 身份: "艾拉是流浪商人" → "我是流浪商人"（需要角色中文名，从事实推断）
    m = re.search(r"^([\u4e00-\u9fff]{1,4})是(.+)$", fact)
    if m:
        return f"我是{m.group(2)}"

    # 8. 人设类（性格/风格/目标/背景/禁忌）→ 转角色视角记忆
    m = re.search(rf"^[\u4e00-\u9fff]{{1,4}}的(性格|目标|背景|禁忌|说话风格)：(.+)$", fact)
    if m:
        kind, val = m.group(1), m.group(2)
        if kind == "性格":
            return f"我{val}"
        if kind == "目标":
            return f"我想{val}"
        if kind == "背景":
            return f"我{val}"
        if kind == "禁忌":
            return f"我禁忌{val}"
        return f"我{kind}：{val}"
    # 对玩家的态度
    m = re.search(rf"^[\u4e00-\u9fff]{{1,4}}对玩家的(?:初始)?态度：(.+)$", fact)
    if m:
        return f"我对陌生人的态度：{m.group(1)}"

    # 8. 兜底：原样但去角色名前缀
    for prefix in [f"{character_id}提到", f"{character_id}回应"]:
        if fact.startswith(prefix):
            return None  # 噪音不注入
    return fact


def build_memory_prompt(character_id: str, facts: list[str], max_lines: int = 3) -> str:
    """把多条事实转成角色视角记忆提示词（供 prompt 注入）。"""
    lines = []
    for fact in facts:
        line = fact_to_memory_line(character_id, fact)
        if line:
            lines.append(line)
        if len(lines) >= max_lines:
            break
    if not lines:
        return ""
    return "；".join(lines)
