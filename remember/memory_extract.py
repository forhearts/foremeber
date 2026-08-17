"""记忆提炼：从对话中提取 NPC 应记住的关键事实

核心设计（用户目标）：记忆不是"对话记录"，而是"NPC 知道的事实"。
这样任意底模拿到事实后都能自然使用（价格/名字/喜好/恩怨）。

提取策略（规则 + 关键词，不依赖 LLM）：
1. 价格/交易类：提到数字+单位 → "X 的价格是 N 金币"
2. 身份/名字类：我叫/我是 → "玩家叫 X，来自 Y"
3. 喜好/厌恶类：喜欢/讨厌/想要 → 记录偏好
4. 事件/恩怨类：偷/帮/送 → 记录关系变化
5. 通用：保留简短对话事实（NPC 视角）
"""
from __future__ import annotations

import re


def extract_facts(character_id: str, player_input: str, npc_reply: str) -> list[str]:
    """从一轮对话提炼 NPC 应记住的事实（多条）。"""
    facts = []
    combined = player_input + " " + npc_reply

    # 1. 价格/交易：只在玩家明确问价时存价格（"多少钱/怎么卖/什么价"），
    #    砍价/嫌贵/其他不存（那是讨论不是定价，避免错误价格入记忆）
    ASKING_PRICE = ["多少钱", "怎么卖", "什么价", "卖多少", "价格", "价", "卖吗"]
    is_asking = any(k in player_input for k in ASKING_PRICE)
    price_m = re.search(
        r"([0-9]+|[零一二三四五六七八九十百千万两半]+)\s*(枚|个|块|文)?\s*(金币|银币|铜币)", combined)
    if price_m and is_asking:
        num, qty, unit = price_m.group(1), price_m.group(2) or "", price_m.group(3)
        price_str = f"{num}{qty}{unit}" if qty else f"{num}{unit}"
        # 找物品（剑/匕首/货物...）
        item = None
        for kw in ["剑", "匕首", "刀", "盾", "铠甲", "货物", "干粮", "药水", "酒", "装备", "武器"]:
            if kw in combined:
                item = kw
                break
        if item:
            facts.append(f"{character_id}把{item}定价为{price_str}")
        else:
            facts.append(f"{character_id}说价格是{price_str}")

    # 2. 名字/身份
    name_m = re.search(r"我(?:叫|是|乃)([\u4e00-\u9fff]{1,4})[，,。]?", player_input)
    if name_m:
        facts.append(f"玩家自称{name_m.group(1)}")
    origin_m = re.search(r"(?:来自|从)([\u4e00-\u9fff]{1,6})(?:来|出发|赶到)", player_input)
    if origin_m:
        facts.append(f"玩家来自{origin_m.group(1)}")

    # 3. 喜好/厌恶
    like_m = re.search(r"(?:喜欢|想要|想买|需要)([\u4e00-\u9fff]{1,6})", player_input)
    if like_m:
        facts.append(f"玩家想要{like_m.group(1)}")

    # 4. 事件/恩怨
    if any(k in player_input for k in ["偷", "抢", "骗"]):
        facts.append(f"玩家对{character_id}有过不良行为")
    if any(k in player_input for k in ["帮", "救", "送"]) and "你" in player_input:
        facts.append(f"玩家帮助过{character_id}")

    # 5. 兜底：只保留明确的身份/背景事实（避免"提到X"噪音）
    if not facts:
        m = re.search(r"(?:我是|我是这里的|我是那)([\u4e00-\u9fff]{2,10})", npc_reply)
        if m:
            facts.append(f"{character_id}是{m.group(1)}")

    return facts


def memory_entry(character_id: str, player_input: str, npc_reply: str) -> str | None:
    """生成单条记忆（优先事实）。无事实返回 None（不存噪音）。"""
    facts = extract_facts(character_id, player_input, npc_reply)
    if facts:
        return "；".join(facts)
    return None
