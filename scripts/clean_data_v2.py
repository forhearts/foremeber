"""SFT 数据清洗器 v2（针对 14B 角色扮演模型生成的数据）

过滤低质量台词：AI味 / 任务报告 / 纯动作 / 模板残留 / 英文残留 / 重复字符
用法:
    python scripts/clean_data_v2.py [--input data/sft/rp_data_v2.jsonl] [--output data/sft/rp_data_v2_clean.jsonl]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AI_PATTERNS = [
    "我是AI", "作为AI", "人工智能", "语言模型", "助手", "我无法",
    "请随时", "很高兴为您", "如果你有任何问题", "Qwen", "通义千问",
    "系统", "程序", "虚拟", "模型",
]

META_PATTERNS = [
    "任务目标", "目标达成", "步骤", "草稿", "选项", "润色", "检查",
    "符合要求", "字数", "内心活动", "内心独白", "叙述", "旁白",
]

ACTION_ONLY_PREFIXES = [
    "我眯起眼睛", "我看着", "我笑了笑", "我温柔地", "我抬起", "我停下",
    "我站在", "我靠在", "我眨了眨", "我点了点头", "我转身", "我打量",
]

TEMPLATE_LEFTOVERS = ["<think", "<reply", "【", "】", "<|im", "</think>"]


def is_bad(text: str) -> tuple[bool, str]:
    """返回 (是否低质量, 原因)。"""
    t = text.strip()
    if len(t) < 6:
        return True, "过短"
    if len(t) > 80:
        return True, "过长"
    for p in AI_PATTERNS:
        if p in t:
            return True, f"AI味:{p}"
    for p in META_PATTERNS:
        if p in t:
            return True, f"元语言:{p}"
    for p in TEMPLATE_LEFTOVERS:
        if p in t:
            return True, f"模板残留:{p}"
    # 英文残留（4+ 连续字母）
    if re.search(r"[A-Za-z]{4,}", t):
        return True, "英文残留"
    # 纯动作开头（无引号台词）
    for p in ACTION_ONLY_PREFIXES:
        if t.startswith(p) and not re.search(r"[\"「『“]", t):
            return True, "纯动作"
    # 重复字符（5+ 相同）
    if re.search(r"(.)\1{4,}", t):
        return True, "重复字符"
    return False, ""


def clean_text(text: str) -> str:
    """轻度清洗：去首尾引号/空格，合并空白。"""
    t = text.strip().strip("「」\"'“”《》")
    t = re.sub(r"\s+", " ", t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sft/rp_data_v2.jsonl")
    ap.add_argument("--output", default="data/sft/rp_data_v2_clean.jsonl")
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
                msgs = d["messages"]
                asst = msgs[2]["content"]
                bad, reason = is_bad(asst)
                if bad:
                    removed += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                msgs[2]["content"] = clean_text(asst)
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
