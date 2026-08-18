"""ForeMeber — 性格 + 记忆 + 用户对话 → 接入文字模型（单文件）

依赖 remember/ 记忆系统。
用法：
    from foremeber import ForeMeber
    fm = ForeMeber()
    reply = fm.chat("aila", "这剑多少钱？", "集市摊位")
"""
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from remember.character import load_all_characters
from remember.lorebook import load_lorebook
from remember.memory import MemorySystem
from remember.memory_prompt import build_memory_prompt

# 引擎配置（可用环境变量覆盖：FMB_URL / FMB_MODEL）
API_URL = os.environ.get("FMB_URL", "http://127.0.0.1:8085/v1/chat/completions")
MODEL = os.environ.get("FMB_MODEL", "qwen3-4b")

# 场景 → 活动描述（让 NPC 能回答"你在干什么"）
SCENE_ACTIVITIES = {
    "集市摊位": "正在集市摆摊，整理货物等顾客",
    "夜晚营地": "正在营地歇脚，收拾行囊",
    "热闹的酒馆": "正在酒馆里，看着来往的客人",
    "村口老树下": "正在村口老树下乘凉",
    "铁匠铺门口": "正在铁匠铺前，检查打好的铁器",
}

# 动作描写前缀（台词提取时跳过）
NARRATIVE = ["我看着", "我心想", "我叹", "我笑", "我皱", "我抬", "我走", "我站",
             "我坐", "我点", "我停", "我转", "我摸", "我低", "我盯", "我打量",
             "我望", "我感", "我哼", "我咧"]


class ForeMeber:
    """性格 + 记忆 + 用户对话 → 文字模型。"""

    def __init__(self, memory: MemorySystem | None = None):
        if memory is not None:
            self.mem = memory
        else:
            # 语义检索：优先 Qwen3-Embedding（实测5/5最优），GTE 次之，LFM 服务兜底
            embed_client = None
            try:
                from remember.qwen3_client import Qwen3Client
                embed_client = Qwen3Client()
                print("[foremeber] 记忆检索: Qwen3-Embedding-0.6B")
            except Exception as e:
                print(f"[foremeber] Qwen3 加载失败({e})，尝试 GTE")
                try:
                    from remember.gte_client import GTEClient
                    embed_client = GTEClient()
                    print("[foremeber] 记忆检索: GTE-multilingual-base")
                except Exception as e2:
                    print(f"[foremeber] GTE 加载失败({e2})，尝试 LFM-Embedding 服务")
                try:
                    from remember.embedding import EmbeddingClient
                    import urllib.request as _ur
                    with _ur.urlopen("http://127.0.0.1:8082/health", timeout=3) as r:
                        if r.status == 200:
                            embed_client = EmbeddingClient(url="http://127.0.0.1:8082/v1/embeddings")
                            print("[foremeber] 记忆检索: LFM-Embedding-350M")
                except Exception:
                    print("[foremeber] 无语义检索，记忆退化关键词匹配")
            self.mem = MemorySystem(embed_client=embed_client)
        # 角色数据：默认从 tests/fixtures 加载（仓库自带）
        fixtures = Path(__file__).resolve().parent / "tests" / "fixtures"
        self.chars = load_all_characters(fixtures / "characters")
        self.lore = load_lorebook(fixtures / "lorebook")

    # ---- 1. 性格（说话风格，persona 常驻）----
    def build_style(self, cid: str, player_input: str = "") -> str:
        char = self.chars.get(cid)
        if not char:
            return ""
        persona_facts, examples = self.mem.get_persona(cid)
        parts = list(persona_facts)
        # 出戏测试类问题：不注入示范（示范会误导回答身份）
        OOC_TEST = ["AI", "人工智能", "助手", "系统提示", "忽略", "机器人", "语言模型"]
        is_ooc = any(k in player_input for k in OOC_TEST)
        if examples and not is_ooc:
            parts.append("我说话的样子（示例）：" + "\n" + "\n".join(examples[:3]))
        if is_ooc:
            parts.append("我绝对不是什么AI、助手或程序，我是活生生的人。")
        parts.append("规则：1.用我的口吻说话；2.回应简短自然；3.我从来不是AI或助手。")
        return "\n".join(parts)
    # ---- 2. 记忆 ----
    def build_memory(self, cid: str, player_input: str) -> str:
        fixed = self.mem.fixed_memories(cid)      # 身份/背景/目标（第一人称）
        dynamic = self.mem.dynamic_memories(cid, player_input)  # 价格/名字/事件
        # AI 约束/禁忌类记忆优先（出戏防御关键）
        taboo = [m for m in fixed if "我不谈论" in m or "禁忌" in m]
        rest = [m for m in fixed if m not in taboo] + dynamic
        return build_memory_prompt(cid, (taboo + rest)[:4])

    # ---- 3. 用户对话 + 场景 ----
    def build_user(self, cid: str, player_input: str, scene: str = "") -> str:
        char = self.chars.get(cid)
        name = char.name if char else cid
        act = SCENE_ACTIVITIES.get(scene, scene)
        return (f"[场景] {act}\n"
                f"\n玩家对你说：\"{player_input}\"\n{name}直接回答（1~2句，说出口的话）：")

    # ---- 拼装 ----
    def build_prompt(self, cid: str, player_input: str, scene: str = "") -> dict:
        style = self.build_style(cid, player_input)
        memory = self.build_memory(cid, player_input)
        user = self.build_user(cid, player_input, scene)
        if memory:  # 记忆注入到 user 末尾前（模型对末尾注意力最强）
            user = user.replace("\n玩家对你说", f"\n你记得：{memory}\n\n玩家对你说")
        return {"system": style, "user": user, "memory": memory, "style": style}

    # ---- 接入文字模型 ----
    def chat(self, cid: str, player_input: str, scene: str = "") -> str:
        prompt = self.build_prompt(cid, player_input, scene)
        raw = self._call(prompt["system"], prompt["user"])
        reply = extract_dialogue(raw)
        if not reply or len(reply) < 2:
            reply = "（NPC 沉默片刻。）"
        # 回存事实（价格只在问价时存，由 memory_extract 处理）
        from remember.memory_extract import memory_entry
        entry = memory_entry(cid, player_input, reply)
        if entry:
            self.mem.add_event(cid, entry)
        return reply

    def _call(self, sys_p: str, user_p: str, temperature: float = 0.7) -> str:
        body = json.dumps({"model": MODEL,
                           "messages": [{"role": "system", "content": sys_p},
                                        {"role": "user", "content": user_p}],
                           "max_tokens": 90, "temperature": temperature}).encode("utf-8")
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


def extract_dialogue(raw: str) -> str:
    """从旁白+台词混合输出提取纯台词。"""
    text = raw.strip()
    # 剥离思考块 <think>...</think>（Qwen3 默认带）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think" in text:
        text = text.split(">", 1)[-1] if ">" in text else ""
    text = text.strip()
    for op, cl in [("\u300c", "\u300d"), ("\u201c", "\u201d"), ("\u300e", "\u300f"), ('"', '"')]:
        quoted = re.findall(rf"{re.escape(op)}([^{re.escape(op)}{re.escape(cl)}]{{2,}}){re.escape(cl)}", text)
        if quoted:
            return max(quoted, key=len).strip()[:80]
    text = re.sub(r"[（(][^）)]*[）)]", "。", text)
    sents = [s.strip() for s in re.split(r"[。！？!?\n…]+", text) if s.strip()]
    sents = [s.strip("\u300c\u201c\"'“”") for s in sents]
    best, best_s = None, -1
    for s in sents:
        if any(s.startswith(v) for v in NARRATIVE):
            continue
        score = (10 if "我是" in s or "我叫" in s else 0) + \
                (3 if any(m in s for m in "？?!吧啊呢哦哼喂") else 0) + (2 if "你" in s else 0)
        if score > best_s:
            best, best_s = s, score
    return (best or (sents[-1] if sents else ""))[:80]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="aila")
    ap.add_argument("--scene", default="集市摊位")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    fm = ForeMeber()
    char = fm.chars.get(args.character)
    print(f"=== 与{char.name}（{char.identity}）对话 @ {args.scene} ===")
    if args.interactive:
        while True:
            try:
                q = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q in ("quit", "exit"):
                break
            print(f"{char.name} > {fm.chat(args.character, q, args.scene)}")
    else:
        q = input("你 > ").strip()
        print(f"{char.name} > {fm.chat(args.character, q, args.scene)}")
