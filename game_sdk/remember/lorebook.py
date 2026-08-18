"""世界书（Lorebook / World Info）模块

对应规划第 6.2 节：基于关键词触发，不全量塞入设定。
玩家输入或当前场景包含"国王""魔法剑"等词，自动提取条目注入 prompt。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pathlib import Path

LOREBOCK_DIR = Path(__file__).resolve().parent.parent / "lorebook"


@dataclass
class LoreEntry:
    key: str = ""
    trigger: list[str] = field(default_factory=list)
    content: str = ""
    budget: int = 150
    priority: int = 1

    def to_dict(self) -> dict:
        return {"key": self.key, "trigger": self.trigger,
                "content": self.content, "budget": self.budget,
                "priority": self.priority}

    @classmethod
    def from_dict(cls, d: dict) -> "LoreEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Lorebook:
    def __init__(self, entries: list[LoreEntry] | None = None):
        self.entries = entries or []

    def query(self, text: str, max_entries: int = 2, max_budget: int = 300) -> list[LoreEntry]:
        """按关键词触发检索条目，按 priority 排序，受预算限制。"""
        hits = []
        used = 0
        for e in sorted(self.entries, key=lambda x: -x.priority):
            if any(t and t in text for t in e.trigger):
                if used + len(e.content) > max_budget:
                    continue
                hits.append(e)
                used += len(e.content)
                if len(hits) >= max_entries:
                    break
        return hits

    def to_prompt_block(self, text: str, max_entries: int = 2, max_budget: int = 300) -> str:
        """返回注入 prompt 的世界书文本块（无命中返回空串）。"""
        hits = self.query(text, max_entries, max_budget)
        if not hits:
            return ""
        lines = ["[世界设定]", *(f"- {e.content}" for e in hits)]
        return "\n".join(lines)


def load_lorebook(dir_path: str | Path = None) -> Lorebook:
    """加载目录下所有世界书条目。"""
    d = Path(dir_path) if dir_path else LOREBOCK_DIR
    entries = []
    if d.exists():
        for f in sorted(d.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    entries.append(LoreEntry.from_dict(json.load(fp)))
            except Exception as e:
                print(f"[warn] 加载世界书失败 {f.name}: {e}")
    return Lorebook(entries)
