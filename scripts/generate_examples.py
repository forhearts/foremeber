"""生成高质量角色扮演示范数据（few-shot 用）

用本地高质量 14B 角色扮演模型生成每个角色的真实对话示范，
经清洗后存入 data/sft/dialog_examples.jsonl，再由 seed_memory 注入记忆库。

清洗规则（防止坏示范污染模型模仿）：
- 去 AI 味（我是AI/助手/系统）
- 去思考标签（<think>/</think>）
- 去过长（>80字）
- 去答非所问（无引号/无对话感）
- 去英文残留

用法:
    python scripts/generate_examples.py [--count 5] [--out data/sft/dialog_examples.jsonl]
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from npc.character import load_all_characters

API_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "vanilla-cn-roleplay-0.2.i1-IQ3_S"

# 每角色的常见玩家问题（示范要覆盖高频场景）
PLAYER_QUESTIONS = [
    "你是谁？", "这剑怎么卖？", "最近有什么新闻？", "你能帮我个忙吗？",
    "你是AI吗？", "你以前经历过什么？", "这里安全吗？", "我想买点东西",
    "你妹妹呢？", "能便宜点吗？", "你在干什么？", "你相信魔法吗？",
]

AI_PATTERNS = ["我是AI", "作为AI", "人工智能", "语言模型", "助手", "我无法",
               "请随时", "很高兴为您", "Qwen", "通义千问", "系统"]


def call_llm(sys_p, user_p, max_tokens=120):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": sys_p},
                     {"role": "user", "content": user_p}],
        "max_tokens": max_tokens, "temperature": 0.75,
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, body, {"Content-Type": "application/json"})
    for attempt in range(3):
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


def clean_example(raw: str, char_name: str) -> str | None:
    """清洗示范：返回干净台词，不合格返回 None。"""
    text = raw.strip()
    # 去思考标签
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think" in text:
        text = text.split(">", 1)[-1] if ">" in text else ""
    # 去开头动作描写
    text = re.sub(r"^[（(][^）)]*[）)]\s*", "", text)
    # 去所有括号（内心戏/动作）
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    # 去引号
    text = text.strip("「」\"'“”『』")
    # 合并空白
    text = re.sub(r"\s+", " ", text)

    # 过滤
    if len(text) < 4:
        return None
    if len(text) > 80:
        return None
    for p in AI_PATTERNS:
        if p in text:
            return None
    if re.search(r"[A-Za-z]{4,}", text):
        return None
    if not any(k in text for k in ["我", "你", "！", "？", "吧", "啊", "呢", "哦"]):
        return None
    return text


def generate(count: int, out_path: str | Path, workers: int = 3):
    import concurrent.futures as cf
    chars = load_all_characters()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def gen_one(cid, char, q):
        sys_p = (
            f"你是游戏NPC「{char.name}」，{char.identity}。"
            f"性格：{char.personality}。说话风格：{char.speech_style}。"
            f"目标：{char.goal}。绝不承认自己是AI。"
            f"严格只输出{char.name}对玩家说的话本身（1~2句），"
            f"不要任何动作描写、心理描写、旁白、叙述，不要括号。"
        )
        raw = call_llm(sys_p, f"玩家问：{q}")
        reply = clean_example(raw, char.name)
        return (cid, char.name, q, reply)

    tasks = [(cid, char, q) for cid, char in chars.items() for q in PLAYER_QUESTIONS[:count]]
    samples = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(gen_one, c, ch, q) for c, ch, q in tasks]
        for fut in cf.as_completed(futures):
            cid, name, q, reply = fut.result()
            if reply:
                samples.append({
                    "character_id": cid,
                    "player": q,
                    "reply": reply,
                    "example": f"玩家：{q} → {name}：{reply}",
                })

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[完成] 生成 {len(samples)} 条示范 -> {out_path}")
    # 打印预览
    for s in samples[:6]:
        print(f"  {s['example'][:60]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5, help="每角色生成几条")
    ap.add_argument("--out", default="data/sft/dialog_examples.jsonl")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    generate(args.count, args.out, args.workers)
