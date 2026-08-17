"""Engine — 接入文字模型（本地 14B 角色扮演 API）

把 PromptBuilder 拼装的 prompt 发给文字模型，提取台词，回存记忆。
用法：
    from engine_vanilla import VanillaEngine
    eng = VanillaEngine()
    reply = eng.chat("aila", "这剑多少钱？", "集市摊位")
"""
import json
import re
import time
import urllib.request

from prompt_builder import PromptBuilder

API_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "vanilla-cn-roleplay-0.2.i1-IQ3_S"

# 动作描写前缀（台词提取时跳过）
NARRATIVE = ["我看着", "我心想", "我叹", "我笑", "我皱", "我抬", "我走", "我站",
             "我坐", "我点", "我停", "我转", "我摸", "我低", "我盯", "我打量",
             "我望", "我感", "我哼", "我咧"]


def extract_dialogue(raw: str) -> str:
    """从旁白+台词混合输出提取纯台词。"""
    text = raw.strip()
    # 引号内台词优先
    for op, cl in [("\u300c", "\u300d"), ("\u201c", "\u201d"), ("\u300e", "\u300f"), ('"', '"')]:
        quoted = re.findall(rf"{re.escape(op)}([^{re.escape(op)}{re.escape(cl)}]{{2,}}){re.escape(cl)}", text)
        if quoted:
            return max(quoted, key=len).strip()[:80]
    # 无引号：剥括号 + 按句选对话感最强的
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


class VanillaEngine:
    """接入本地 14B 角色扮演模型。"""

    def __init__(self, builder: PromptBuilder | None = None):
        self.builder = builder if builder is not None else PromptBuilder()

    def chat(self, cid: str, player_input: str, scene: str = "") -> str:
        """完整一轮：拼装 prompt → 调模型 → 提取台词 → 回存记忆。"""
        prompt = self.builder.build_prompt(cid, player_input, scene)
        raw = self._call(prompt["system"], prompt["user"])
        reply = extract_dialogue(raw)
        if not reply or len(reply) < 2:
            reply = "（NPC 沉默片刻。）"

        # 回存事实（价格只在问价时存等，由 memory_extract 处理）
        from npc.memory_extract import memory_entry
        entry = memory_entry(cid, player_input, reply)
        if entry:
            self.builder.mem.add_event(cid, entry)
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="aila")
    ap.add_argument("--scene", default="集市摊位")
    args = ap.parse_args()
    eng = VanillaEngine()
    print(eng.chat(args.character, "你是谁？", args.scene))
