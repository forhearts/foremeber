"""后处理模块

对应规划第 7 章：
- 7.1 出戏关键词过滤（命中 → 重试 → 兜底）
- 7.2 长度控制（只保留第一句 / 截断）
"""
from __future__ import annotations

import re

from npc.config import OUT_OF_CHARACTER_PHRASES


class PostProcessor:
    """模型输出后处理：出戏检测 + 长度控制 + 兜底。"""

    def __init__(
        self,
        phrases: list[str] | None = None,
        max_chars: int = 60,
        fallback: str = "（NPC 若有所思地看了看你，没有回答。）",
    ):
        self.phrases = phrases or OUT_OF_CHARACTER_PHRASES
        self.max_chars = max_chars
        self.fallback = fallback

    def check_out_of_character(self, text: str) -> bool:
        """检测是否包含出戏关键词（含空格/AI变体）。"""
        t = re.sub(r"\s+", "", text)  # 去空格防 "A I" 绕过
        for p in self.phrases:
            if p and p in text:
                return True
        # 变体：A I / a i / A1 等
        if re.search(r"我是\s*[Aa]\s*[Ii]", text):
            return True
        if re.search(r"[Aa]\s*[Ii]", t) and "AI" in t.upper():
            return True
        return False

    def enforce_length(self, text: str, max_chars: int | None = None) -> str:
        """长度控制：超长则只保留第一句，再截断。"""
        limit = max_chars or self.max_chars
        if len(text) <= limit:
            return text
        # 按句号/问号/感叹号/省略号切第一句
        m = re.split(r"[。！？!?…]+", text)
        first = m[0]
        if first and len(first) < limit:
            return first
        if first:
            return first[:limit]
        return text[:limit]

    def clean(self, text: str) -> str:
        """基础清洗：去空白、去首尾引号、剥离动作旁白。"""
        t = text.strip().strip("「」\"'“”")
        # 剥离动作旁白（括号内容），保留台词
        t = re.sub(r"（.*?）", "", t)
        t = re.sub(r"\(.*?\)", "", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    @staticmethod
    def truncate_repetition(text: str, min_rep: int = 4) -> str:
        """截断重复片段：检测连续重复（如“会会会…”或“有趣有趣…”）并截掉。"""
        # 单字符连续重复
        m = re.search(r"(.)\1{%d,}" % (min_rep - 1), text)
        if m:
            return text[: m.start()]
        # 多字符短语重复（2~6字）
        for span in range(2, 7):
            m = re.search(rf"(.{{{span}}})\1{{2,}}", text)
            if m and len(set(m.group(1))) > 1:
                return text[: m.start()]
        return text

    def process(self, raw: str, retry_limit: int = 1) -> str:
        """完整后处理管线：清洗 → 重复截断 → 出戏检测 → 长度控制 → 兜底。"""
        cleaned = self.clean(raw)
        # 重复截断（未微调模型常见）
        cleaned = self.truncate_repetition(cleaned)

        # 出戏检测：命中则直接触发兜底（生成侧已做重试，此处不再循环）
        if self.check_out_of_character(cleaned):
            return self.fallback

        # 长度控制
        return self.enforce_length(cleaned)
