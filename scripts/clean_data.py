"""SFT 数据清洗：过滤 AI 味/低质量样本

规划 3.6 人工筛选/清洗环节的自动化版。
过滤规则：
1. 出戏/AI 味（"我是AI"、"请随时"、"谢谢您的"等）
2. 过长（>60字）
3. 过短（<4字）
4. 含英文残留
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# AI 味 / 出戏特征
AI_PATTERNS = [
    "我是AI", "作为AI", "人工智能", "语言模型", "助手", "请随时",
    "随时提醒", "谢谢您的", "感谢您的", "祝您", "很高兴为您",
    "if you", "please", "help", "assistant", "I am", "Let's",
    "希望您", "有什么可以帮您", "请输入", "系统提示",
]

# 元语言残留
META_PATTERNS = [
    "草稿", "选项", "步骤", "润色", "检查", "符合要求",
    "<think", "<reply", "字数", "改写",
]


def is_bad(text: str) -> tuple[bool, str]:
    t = text.strip()
    if len(t) < 4:
        return True, "过短"
    if len(t) > 60:
        return True, "过长"
    for p in AI_PATTERNS:
        if p.lower() in t.lower():
            return True, f"AI味:{p}"
    for p in META_PATTERNS:
        if p in t:
            return True, f"元语言:{p}"
    # 英文单词残留（>2 个连续英文字母）
    if re.search(r"[A-Za-z]{3,}", t):
        return True, "英文残留"
    return False, ""


def clean_data(in_path: str | Path, out_path: str | Path):
    in_path, out_path = Path(in_path), Path(out_path)
    total = kept = removed = 0
    reasons = {}
    with open(in_path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            total += 1
            try:
                d = json.loads(line)
                asst = d["messages"][2]["content"]
                bad, reason = is_bad(asst)
                if bad:
                    removed += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                f.write(line)
                kept += 1
            except Exception:
                removed += 1
                reasons["解析失败"] = reasons.get("解析失败", 0) + 1
    print(f"[clean] {total} -> {kept} (移除 {removed})")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")


if __name__ == "__main__":
    clean_data(
        sys.argv[1] if len(sys.argv) > 1 else "data/sft/rp_data.jsonl",
        sys.argv[2] if len(sys.argv) > 2 else "data/sft/rp_data_clean.jsonl",
    )
