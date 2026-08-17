"""示范数据清洗器：把 14B 生成的对话清洗成纯台词

目标：从"动作描写+台词"混合输出中提取纯台词（few-shot 示范用）。
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AI_PATTERNS = ["我是AI", "作为AI", "人工智能", "语言模型", "助手", "我无法",
               "请随时", "很高兴为您", "Qwen", "通义千问", "系统", "程序"]

# 动作描写前缀（我+动词开头）
ACTION_PREFIXES = [
    "我停下", "我抬起头", "我抬头", "我冷笑", "我笑", "我瞥", "我看了", "我看着",
    "我望", "我低", "我点", "我皱", "我叹", "我耸", "我转", "我摸", "我擦",
    "我走", "我站", "我坐", "我靠", "我哼", "我嗯", "我咧", "我露", "我盯",
    "我打量", "我沉默", "我咳", "我伸", "我拿", "我放", "我拿起", "我放下",
    "我温和", "我大胆", "我歪", "我挑", "我扬", "我抿", "我轻轻", "我缓缓",
    "我向前", "我后退", "我侧", "我闪", "我摇", "我闭", "我睁", "我拍",
    "抬眼", "大胆地", "用一种", "用一", "面带", "嘴角", "目光", "眼神",
]


def clean_reply(text: str) -> str:
    """清洗成纯台词。策略：优先取引号内内容，否则剥动作描写。"""
    t = text.strip()
    # 去思考标签
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    if "<think" in t:
        t = t.split(">", 1)[-1] if ">" in t else ""

    # 1. 优先：提取引号内内容（台词通常在引号里）
    import re as _re
    quoted = _re.findall(r"[\u300c\u201c\u300e]([^\u300d\u201d\u300f]{2,})[\u300d\u201d\u300f]", t)
    if not quoted:
        quoted = _re.findall(r'"([^"]{2,})"', t)
    if quoted:
        # 取最长的像台词的
        best = max(quoted, key=len).strip()
        if len(best) >= 3:
            t = best
            return t.strip()

    # 2. 无闭合引号：去开头动作描写（我+动词，切到第一个逗号/冒号/句号）
    for p in ACTION_PREFIXES:
        if t.startswith(p):
            # 切到第一个逗号/冒号/句号/引号后的内容
            m = re.search(r"[,，:：。！？!?\u201c\u300c\"']", t)
            if m:
                t = t[m.start()+1:]
            else:
                t = ""
            break
    # 3. 去所有括号内容
    t = re.sub(r"[（(][^）)]*[）)]", "", t)
    # 去引号
    t = t.strip("「」\"'“”『』")
    # 去结尾叙述残留
    t = re.sub(r"(说完|说罢|说着|随后|然后)[^。！？!?]*$", "", t)
    t = re.sub(r"(语气|目光|眼神|神情|表情)[^。！？!?]*$", "", t)
    t = re.sub(r"(一边|接着|继续)[^。！？!?]*$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.lstrip("「」\"'“”『』：: ")
    return t


def is_bad(text: str) -> tuple[bool, str]:
    t = text.strip()
    if len(t) < 4:
        return True, "过短"
    if len(t) > 60:
        return True, "过长"
    for p in AI_PATTERNS:
        if p in t:
            return True, f"AI味:{p}"
    if "<think" in t or "</think" in t:
        return True, "思考标签"
    if re.search(r"[A-Za-z]{4,}", t):
        return True, "英文残留"
    # 日文假名残留
    if re.search(r"[\u3040-\u30ff]", t):
        return True, "日文残留"
    # 仍含动作描写（清洗不干净）
    for p in ACTION_PREFIXES:
        if t.startswith(p):
            return True, "动作残留"
    if not any(k in t for k in ["我", "你", "！", "？", "吧", "啊", "呢", "哦", "哼", "嘿", "别", "喂"]):
        return True, "无对话感"
    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sft/dialog_examples.jsonl")
    ap.add_argument("--output", default="data/sft/dialog_examples_clean.jsonl")
    args = ap.parse_args()

    inp, outp = Path(args.input), Path(args.output)
    total = kept = removed = 0
    reasons = {}
    with open(inp, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    with open(outp, "w", encoding="utf-8") as f:
        for line in lines:
            total += 1
            try:
                d = json.loads(line)
                cid = d.get("character_id", "?")
                q = d.get("player", "")
                reply = d.get("reply", "")
                cleaned = clean_reply(reply)
                bad, reason = is_bad(cleaned)
                if bad:
                    removed += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                name = cleaned.split("：")[0] if "：" in cleaned[:6] else cid
                d["reply"] = cleaned
                d["example"] = f"玩家：{q} → {name}：{cleaned}"
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                kept += 1
            except Exception:
                removed += 1
                reasons["解析失败"] = reasons.get("解析失败", 0) + 1

    print(f"[clean] {total} -> {kept} (移除 {removed})")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")


if __name__ == "__main__":
    main()
