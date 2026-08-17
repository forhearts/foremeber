"""生成世界书（Lorebook）条目文件"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "lorebook"
OUT.mkdir(exist_ok=True)

ENTRIES = [
    {"key": "wolf", "trigger": ["狼", "狼群", "红眼"], "content": "黑森林的狼最近变异了，眼睛是红色的，怕火。老猎人摩根说它们在月圆之夜最活跃。", "budget": 150, "priority": 3},
    {"key": "moon_sword", "trigger": ["月光剑", "宝剑", "神剑"], "content": "月光剑是传说中的武器，据说能斩断黑暗与邪恶。只在月圆之夜现身，握剑者需无惧黑暗。", "budget": 150, "priority": 4},
    {"key": "tavern", "trigger": ["酒馆", "客栈", "喝酒"], "content": "布鲁诺的酒馆是本地的消息集散地，商队、猎人、流浪者都在此歇脚。", "budget": 150, "priority": 2},
    {"key": "black_forest", "trigger": ["黑森林", "森林", "树林"], "content": "黑森林是危险区域，变异生物出没。猎人们只在白天结伴进入，夜晚绝无人敢留。", "budget": 150, "priority": 2},
    {"key": "king", "trigger": ["国王", "王都", "陛下"], "content": "远方的国王最近在征兵，边境气氛紧张。村民私下议论纷纷。", "budget": 150, "priority": 2},
    {"key": "war", "trigger": ["战争", "战乱", "边境战"], "content": "三年前的边境战争毁掉了许多村庄，很多人家破人亡，至今阴影未散。", "budget": 150, "priority": 3},
    {"key": "witch", "trigger": ["女巫", "魔法", "施法"], "content": "女巫被王国通缉，但村民私下觉得她们只是怪人，并未伤害过谁。", "budget": 150, "priority": 2},
    {"key": "mine", "trigger": ["矿洞", "废弃矿", "宝藏"], "content": "废弃矿洞据说有宝藏，也有人说闹鬼。没人敢下去太深。", "budget": 150, "priority": 2},
    {"key": "church", "trigger": ["教堂", "神父", "祈祷"], "content": "村里唯一的教堂，神父很神秘，总在深夜独自祈祷。", "budget": 150, "priority": 2},
    {"key": "caravan", "trigger": ["商队", "商人", "货物"], "content": "定期经过的商队带来远方消息和货物，也是艾拉这类流浪商人的同行。", "budget": 150, "priority": 2},
    {"key": "dragon", "trigger": ["龙", "巨龙", "北方山脉"], "content": "传说北方山脉栖息着一头巨龙，近百年来无人真正见过它。", "budget": 150, "priority": 2},
    {"key": "plague", "trigger": ["瘟疫", "疫病", "生病"], "content": "瘟疫曾在此地肆虐，村长家因此失去长子。村民至今谈之色变。", "budget": 150, "priority": 3},
]

for e in ENTRIES:
    with open(OUT / f"{e['key']}.json", "w", encoding="utf-8") as f:
        json.dump(e, f, ensure_ascii=False, indent=2)

print(f"[ok] 生成 {len(ENTRIES)} 个世界书条目 -> {OUT}")
