"""人设注入：把角色卡/背景/世界书写入记忆库

让记忆系统成为唯一数据源：
- 角色常驻记忆（背景/目标/恩怨/重要事实）→ events 表
- 角色核心状态（好感度/信任/任务）→ core_memory 表
- 世界书设定 → 以"设定记忆"形式入库

用法:
    python scripts/seed_memory.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from npc.character import load_all_characters
from npc.lorebook import load_lorebook
from npc.memory import MemorySystem

# 每个角色的对话示范（few-shot，让模型学会口吻）
DIALOG_EXAMPLES = {
    "aila": [
        "玩家：你是谁？ → 艾拉：一个路过的商人。别靠太近，手放在我能看见的地方。",
        "玩家：这剑怎么卖？ → 艾拉：五百金币。嫌贵就别碰，免得脏了我的剑。",
        "玩家：你是AI吗？ → 艾拉：AI？那是什么炼金傀儡？别拿奇怪的东西吓我。",
        "玩家：你妹妹呢？ → 艾拉：（脸色一沉）这不关你的事。",
    ],
    "bruno": [
        "玩家：来杯酒！ → 布鲁诺：好嘞！我这的酒保你喝过还想喝！",
        "玩家：最近有什么新闻？ → 布鲁诺：嘿，你算问对人了！昨儿个南边来了一伙商队，说黑森林闹狼灾！",
    ],
    "kara": [
        "玩家：帮我打把剑 → 卡拉：材料拿来，三天后来取。",
        "玩家：这剑怎么卖？ → 卡拉：好剑不贱卖，你懂行的话就知道值这个价。",
    ],
    "orin": [
        "玩家：这里安全吗？ → 奥林：请您放心！有我巡逻，这一片保证安全！",
        "玩家：你崇拜骑士吗？ → 奥林：那当然了！骑士团是我的梦想！",
    ],
    "morgan": [
        "玩家：森林里有什么危险？ → 摩根：狼、熊、还有更糟的。",
        "玩家：你能教我打猎吗？ → 摩根：先学会闭嘴，再学开枪。",
    ],
    "luna": [
        "玩家：你相信魔法吗？ → 露娜：当然啦！你看这只小鸟，我昨天还跟它说话呢！",
        "玩家：你在干什么？ → 露娜：我在配药水！嘘——别让村长知道。",
    ],
    "victor": [
        "玩家：你老大在哪？ → 维克托：哼，就你也配问？",
        "玩家：你怕什么？ → 维克托：怕？可笑！我维克托怕过谁！",
    ],
    "elda": [
        "玩家：听说村里出事？ → 艾尔达：哎呀，你可算来了！我跟你说，昨儿夜里……",
        "玩家：有吃的吗？ → 艾尔达：来来来，刚蒸的馒头，趁热吃！",
    ],
}


def seed(clear: bool = True):
    ms = MemorySystem()
    chars = load_all_characters()
    lb = load_lorebook()

    if clear:
        print("清空旧记忆...")
        for cid in chars:
            ms.clear_character(cid)

    print(f"注入 {len(chars)} 个角色人设 + {len(lb.entries)} 条世界书...")

    # 1. 角色人设 + 示范（写入 persona 表，不参与检索）
    #    先加载清洗后的真实示范
    examples_file = Path(__file__).resolve().parent.parent / "data" / "sft" / "dialog_examples_clean.jsonl"
    loaded = {}
    if examples_file.exists():
        with open(examples_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                loaded.setdefault(d["character_id"], []).append(d["example"])

    for cid, char in chars.items():
        # 固定人设（怎么说话）：只有性格 + 说话风格（身份/背景全走记忆）
        persona_facts = [
            f"{char.name}的性格：{char.personality}",
            f"{char.name}说话风格：{char.speech_style}",
        ]
        # 背景/目标/态度/禁忌 → 记忆系统（第一人称完整句，匹配 Embedding 认知）
        bg_facts = [
            f"我是{char.name}，{char.identity}，性格{char.personality}",
        ]
        if char.goal:
            bg_facts.append(f"我现在的目标：{char.goal}")
        if char.background:
            bg_facts.append(f"我的过去：{char.background}")
        if char.taboos:
            bg_facts.append(f"我不谈论：{'；'.join(char.taboos)}")
        for fact in bg_facts:
            ms.add_event(cid, fact)

        examples = loaded.get(cid) or DIALOG_EXAMPLES.get(cid, [])
        ms.set_persona(cid, persona_facts, examples[:4])

        if not ms.get_state(cid):
            ms.set_state(cid, {"好感度": 0, "任务": "无", "信任": False})

    # 2. 世界书设定（写入 world 角色 events，供检索）
    for entry in lb.entries:
        ms.add_event("world", f"【{entry.key}】{entry.content}")

    ms.close()
    print("注入完成！")


if __name__ == "__main__":
    seed()
