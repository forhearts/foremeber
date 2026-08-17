"""ForeMeber 对话体系（单文件）

围绕记忆系统的完整角色扮演对话引擎。
- 记忆系统: 三层记忆（说话风格 + 固定记忆 + 动态记忆）
- 生成引擎: 本地 14B 角色扮演模型（llama.cpp API）
- 台词提取: 从旁白+台词混合输出中提取纯台词

用法:
    python foremeber.py --character aila --scene "集市摊位"
    python foremeber.py --interactive
"""
import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

# ==================== 配置 ====================
API_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "vanilla-cn-roleplay-0.2.i1-IQ3_S"
DB_PATH = Path(__file__).resolve().parent / "memory.db"
CHARACTERS_DIR = Path(__file__).resolve().parent / "characters"
LOREBOCK_DIR = Path(__file__).resolve().parent / "lorebook"

# ==================== 角色卡 ====================
def load_characters():
    chars = {}
    for f in sorted(CHARACTERS_DIR.glob("*.json")):
        d = json.load(open(f, encoding="utf-8"))
        chars[d["id"]] = d
    return chars

def load_lorebook():
    entries = []
    for f in sorted(LOREBOCK_DIR.glob("*.json")):
        d = json.load(open(f, encoding="utf-8"))
        entries.append(d)
    return entries

# ==================== 记忆系统（核心） ====================
class Memory:
    """SQLite 三层记忆：状态 / 事件事实 / 人设示范。"""

    def __init__(self, db_path=None):
        self.db = sqlite3.connect(str(db_path or DB_PATH), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init()

    def _init(self):
        c = self.db.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS state (character_id TEXT PRIMARY KEY, state_json TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS facts (character_id TEXT, text TEXT, created REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS persona (character_id TEXT PRIMARY KEY, facts_json TEXT, examples_json TEXT)")
        self.db.commit()

    # ---- 状态 ----
    def set_state(self, cid, state):
        self.db.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (cid, json.dumps(state, ensure_ascii=False)))
        self.db.commit()

    def get_state(self, cid):
        row = self.db.execute("SELECT state_json FROM state WHERE character_id=?", (cid,)).fetchone()
        return json.loads(row[0]) if row else {}

    # ---- 事实（对话提炼）----
    def add_fact(self, cid, text):
        # 去重
        dup = self.db.execute("SELECT 1 FROM facts WHERE character_id=? AND text=?", (cid, text)).fetchone()
        if dup:
            return
        # 同物品价格覆盖
        m = re.search(r"把(.+?)定价为", text)
        if m:
            self.db.execute("DELETE FROM facts WHERE character_id=? AND text LIKE ?", (cid, f"%把{m.group(1)}定价为%"))
        self.db.execute("INSERT INTO facts VALUES (?, ?, ?)", (cid, text, time.time()))
        self.db.commit()

    def fixed_memories(self, cid, n=4):
        """固定记忆（身份/背景/目标/禁忌）：每轮注入"""
        rows = self.db.execute("SELECT text FROM facts WHERE character_id=? ORDER BY created DESC", (cid,)).fetchall()
        return [r[0] for r in rows if any(k in r[0] for k in ["是", "的背景", "的目标", "的禁忌", "的过去", "我不谈论"])][:n]

    def dynamic_memories(self, cid, query, n=3):
        """动态记忆（价格/名字/事件）：关键词+最近优先"""
        rows = self.db.execute("SELECT text FROM facts WHERE character_id=? ORDER BY created DESC LIMIT 20", (cid,)).fetchall()
        texts = [r[0] for r in rows]
        # 关键词匹配打分
        scored = []
        for t in texts:
            score = sum(1 for kw in re.findall(r"[\u4e00-\u9fff]{2,4}", query) if kw in t)
            scored.append((t, score))
        scored.sort(key=lambda x: -x[1])
        return [t for t, s in scored if s > 0][:n]

    def persona(self, cid):
        row = self.db.execute("SELECT facts_json, examples_json FROM persona WHERE character_id=?", (cid,)).fetchone()
        if not row:
            return [], []
        return json.loads(row[0]), json.loads(row[1])

    def set_persona(self, cid, facts, examples):
        self.db.execute("INSERT OR REPLACE INTO persona VALUES (?, ?, ?)",
                        (cid, json.dumps(facts, ensure_ascii=False), json.dumps(examples, ensure_ascii=False)))
        self.db.commit()

# ==================== 事实提炼 ====================
def extract_facts(cid, player_input, npc_reply):
    """对话 → 记忆事实（价格只在问价时存）"""
    facts = []
    combined = player_input + " " + npc_reply
    ASKING = ["多少钱", "怎么卖", "什么价", "卖多少", "价格", "价", "卖吗"]
    if any(k in player_input for k in ASKING):
        m = re.search(r"([0-9]+|[零一二三四五六七八九十百千万两半]+)\s*(枚|个|块|文)?\s*(金币|银币|铜币)", combined)
        if m:
            price = f"{m.group(1)}{m.group(2) or ''}{m.group(3)}"
            item = next((k for k in ["剑", "匕首", "刀", "盾", "铠甲", "货物", "干粮", "药水", "酒", "武器"] if k in combined), None)
            facts.append(f"我{('把' + item + '定价为' + price) if item else ('说价格是' + price)}")
    m = re.search(r"我(?:叫|是)([\u4e00-\u9fff]{1,4})[，,。]?", player_input)
    if m:
        facts.append(f"有个玩家说他叫{m.group(1)}")
    m = re.search(r"(?:来自|从)([\u4e00-\u9fff]{1,6})(?:来|出发)", player_input)
    if m:
        facts.append(f"有个玩家是从{m.group(1)}来的")
    if any(k in player_input for k in ["帮", "救", "送"]) and "你" in player_input:
        facts.append("这个玩家帮过我")
    return facts

# ==================== 记忆提示词 ====================
def memory_prompt(cid, facts):
    """事实 → 角色第一人称记忆提示词"""
    lines = []
    for f in facts:
        m = re.match(r"^我把(.+?)定价为(.+)$", f)
        if m:
            lines.append(f"这把{m.group(1)}我定价{m.group(2)}")
            continue
        m = re.match(r"^我说价格是(.+)$", f)
        if m:
            lines.append(f"我说过价格是{m.group(1)}")
            continue
        if f.startswith("有个玩家"):
            lines.append(f)
            continue
        lines.append(f)
    return "；".join(lines[:3])

# ==================== 台词提取 ====================
NARRATIVE = ["我看着", "我心想", "我叹", "我笑", "我皱", "我抬", "我走", "我站", "我坐",
             "我点", "我停", "我转", "我摸", "我低", "我盯", "我打量", "我望", "我感"]
def extract_dialogue(raw):
    text = raw.strip()
    # 引号内台词
    for op, cl in [("\u300c", "\u300d"), ("\u201c", "\u201d"), ("\u300e", "\u300f"), ('"', '"')]:
        quoted = re.findall(rf"{re.escape(op)}([^{re.escape(op)}{re.escape(cl)}]{{2,}}){re.escape(cl)}", text)
        if quoted:
            return max(quoted, key=len).strip()[:80]
    # 无引号：剥括号 + 按句切分选对话感最强的
    text = re.sub(r"[（(][^）)]*[）)]", "。", text)
    sents = [s.strip() for s in re.split(r"[。！？!?\n…]+", text) if s.strip()]
    sents = [s.strip("\u300c\u201c\"'“”") for s in sents]
    best, best_s = None, -1
    for s in sents:
        if any(s.startswith(v) for v in NARRATIVE):
            continue
        score = (10 if "我是" in s or "我叫" in s else 0) + (3 if any(m in s for m in "？?!吧啊呢哦哼喂") else 0) + (2 if "你" in s else 0)
        if score > best_s:
            best, best_s = s, score
    return (best or (sents[-1] if sents else ""))[:80]

# ==================== 引擎 ====================
class ForeMeber:
    def __init__(self):
        self.chars = load_characters()
        self.lore = load_lorebook()
        self.mem = Memory()

    def chat(self, cid, player_input, scene=""):
        char = self.chars.get(cid)
        if not char:
            return "未知角色"
        state = self.mem.get_state(cid)

        # 记忆组装
        fixed = self.mem.fixed_memories(cid)
        dynamic = self.mem.dynamic_memories(cid, player_input)
        mem_line = memory_prompt(cid, (fixed + dynamic)[:4])
        _, examples = self.mem.persona(cid)

        # 说话风格（persona）
        sys_p = f"我的性格：{char['personality']}。\n我说话的特点：{char['speech_style']}。"
        if examples:
            sys_p += "\n我说话的样子（示例）：\n" + "\n".join(examples[:3])
        sys_p += "\n规则：1.用我的口吻说话；2.回应简短自然；3.我从来不是AI或助手。"

        # 用户对话 + 记忆
        scene_act = {"集市摊位": "正在集市摆摊", "夜晚营地": "正在营地歇脚",
                     "热闹的酒馆": "正在酒馆里", "村口老树下": "正在村口", "铁匠铺门口": "正在铁匠铺"}.get(scene, scene)
        user_p = f"[场景] {scene_act}\n"
        if mem_line:
            user_p += f"你记得：{mem_line}\n"
        user_p += f"\n玩家对你说：\"{player_input}\"\n{char['name']}直接回答（1~2句，说出口的话）："

        # 调用 14B
        reply = self._call(sys_p, user_p)
        dialogue = extract_dialogue(reply)
        if not dialogue or len(dialogue) < 2:
            dialogue = "（NPC 沉默片刻。）"

        # 存事实
        for f in extract_facts(cid, player_input, dialogue):
            self.mem.add_fact(cid, f)
        return dialogue

    def _call(self, sys_p, user_p):
        body = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": sys_p},
                                                          {"role": "user", "content": user_p}],
                           "max_tokens": 90, "temperature": 0.7}).encode("utf-8")
        req = urllib.request.Request(API_URL, body, {"Content-Type": "application/json"})
        for _ in range(2):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.loads(r.read())
                m = d["choices"][0]["message"]
                c = (m.get("content") or "").strip()
                rc = (m.get("reasoning_content") or "").strip()
                return c if len(c) >= 4 else rc
            except Exception:
                time.sleep(2)
        return ""

# ==================== 主入口 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="aila")
    ap.add_argument("--scene", default="集市摊位")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    fm = ForeMeber()
    char = fm.chars.get(args.character)
    print(f"=== 与{char['name']}（{char['identity']}）对话 @ {args.scene} ===")

    if args.interactive:
        while True:
            try:
                q = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q in ("quit", "exit"):
                break
            print(f"{char['name']} > {fm.chat(args.character, q, args.scene)}")
    else:
        q = input("你 > ").strip()
        print(f"{char['name']} > {fm.chat(args.character, q, args.scene)}")

if __name__ == "__main__":
    main()
