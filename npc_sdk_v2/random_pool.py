"""随机系统数据池：随机 NPC + 随机事件 的生成素材

系统提供"素材"，随机组合后交给 NPCSystem.generate 生成具体文本。
"""
import random

# ==================== 随机 NPC 素材 ====================
NPC_NAMES = ["阿泽", "翠花", "石头", "二狗", "小满", "老周", "铁牛", "阿莲",
             "旺财", "大壮", "小翠", "根生", "春花", "柱子", "香兰", "六子"]

NPC_IDENTITIES = [
    "农夫", "猎户", "渔夫", "木匠", "裁缝", "药农", "货郎", "车夫",
    "厨娘", "更夫", "渡夫", "樵夫", "磨坊主", "养蜂人", "陶匠", "织娘",
]

NPC_PERSONALITIES = [
    "老实憨厚", "精明圆滑", "沉默寡言", "热情开朗", "胆小谨慎",
    "爱占便宜", "倔强固执", "乐善好施", "爱吹牛", "心直口快",
]

NPC_SPEECH = [
    "简短、朴实", "带点土话", "说话慢吞吞", "嗓门大", "爱唠家常",
    "三句不离本行", "爱用比喻", "说话带笑", "粗声粗气", "爱反问",
]

NPC_GOALS = [
    "攒钱给家里盖房", "打听亲戚的下落", "想把货物卖个好价",
    "种好今年的地", "学一门手艺", "找个安稳的活计",
    "找回走丢的羊", "治好家里的病人", "把女儿嫁出去", "攒钱买头牛",
]

NPC_SCENES = ["集市", "村口", "夜晚营地", "酒馆", "铁匠铺"]

# ==================== 随机事件模板 ====================
# 每种事件：模板（{subject} 会被替换为角色）+ 关键词
EVENT_TEMPLATES = [
    # 遇袭类
    {
        "template": "一群野狼趁夜袭击了营地，{subject}在慌乱中受了伤。",
        "keywords": ["狼", "袭击", "夜"],
    },
    {
        "template": "商队在途中遭遇了山贼打劫，{subject}的货物被抢走大半。",
        "keywords": ["山贼", "商队", "抢劫"],
    },
    # 奇遇类
    {
        "template": "有人在河边发现了一柄生锈的古剑，{subject}闻讯赶来围观。",
        "keywords": ["古剑", "发现", "河边"],
    },
    {
        "template": "夜里营地外传来神秘的兽吼，{subject}说那声音从未听过。",
        "keywords": ["兽吼", "神秘", "夜"],
    },
    # 发现类
    {
        "template": "废弃矿洞深处传出挖掘声，{subject}怀疑有人偷偷进去。",
        "keywords": ["矿洞", "发现", "偷挖"],
    },
    {
        "template": "老树根下挖出了一个陶罐，{subject}说里面可能装着旧物。",
        "keywords": ["陶罐", "老树", "挖掘"],
    },
    # 冲突类
    {
        "template": "两伙人为了一桩旧账起了争执，{subject}夹在中间很为难。",
        "keywords": ["争执", "旧账", "冲突"],
    },
    {
        "template": "有人举报{subject}私藏违禁品，守卫们正在挨家搜查。",
        "keywords": ["搜查", "举报", "守卫"],
    },
    # 自然类
    {
        "template": "一场暴雨冲垮了村口的木桥，{subject}带头组织修补。",
        "keywords": ["暴雨", "木桥", "修补"],
    },
    {
        "template": "庄稼地里出现了大片虫害，{subject}愁得直叹气。",
        "keywords": ["虫害", "庄稼", "愁"],
    },
    # 消息类
    {
        "template": "远方传来消息，王都要开始征兵了，{subject}听后若有所思。",
        "keywords": ["征兵", "王都", "消息"],
    },
    {
        "template": "有人在酒馆吹嘘见过巨龙，{subject}半信半疑地摇头。",
        "keywords": ["巨龙", "酒馆", "传说"],
    },
]


# ==================== 随机生成器 ====================
def random_npc(seed=None):
    """生成一个随机 NPC 配置（数据组合，不调模型）。"""
    if seed is not None:
        random.seed(seed)
    return {
        "name": random.choice(NPC_NAMES),
        "identity": random.choice(NPC_IDENTITIES),
        "personality": random.choice(NPC_PERSONALITIES),
        "speech_style": random.choice(NPC_SPEECH),
        "goal": random.choice(NPC_GOALS),
        "scene": random.choice(NPC_SCENES),
        "greeting": f"（{random.choice(NPC_GOALS)}，{random.choice(NPC_IDENTITIES)}打量着你）",
    }


def random_event_template(seed=None):
    """随机选一个事件模板。"""
    if seed is not None:
        random.seed(seed)
    return random.choice(EVENT_TEMPLATES)
