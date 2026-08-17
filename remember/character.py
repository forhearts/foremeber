"""角色卡（Character Card）模块

对应规划第 6.2 节"知识碎片/设定集记忆"与第 13 节"角色卡设计要适合小模型"。
角色卡必须精炼，关键信息拆成短条件，避免长段落。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pathlib import Path

CHARACTERS_DIR = Path(__file__).resolve().parent.parent / "characters"


@dataclass
class Character:
    """精炼角色卡：短字段、关键词触发、低记忆预算。"""
    name: str = ""
    id: str = ""
    identity: str = ""
    personality: str = ""
    speech_style: str = ""
    taboos: list[str] = field(default_factory=list)
    goal: str = ""
    attitude_to_player: str = ""
    greetings: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    background: str = ""
    backstory_keywords: list[str] = field(default_factory=list)
    memory_budget: int = 300

    # ---- 序列化为注入 prompt 的精炼文本 ----
    def to_prompt_block(self) -> str:
        lines = [
            f"姓名：{self.name}",
            f"身份：{self.identity}",
            f"性格：{self.personality}",
            f"说话风格：{self.speech_style}",
        ]
        if self.goal:
            lines.append(f"当前目标：{self.goal}")
        if self.attitude_to_player:
            lines.append(f"对玩家态度：{self.attitude_to_player}")
        if self.taboos:
            lines.append(f"禁忌：{'；'.join(self.taboos)}")
        return "\n".join(lines)

    def background_block(self) -> str:
        """背景故事（仅在触发关键词时注入）。"""
        return self.background if self.background else ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "id": self.id, "identity": self.identity,
            "personality": self.personality, "speech_style": self.speech_style,
            "taboos": self.taboos, "goal": self.goal,
            "attitude_to_player": self.attitude_to_player,
            "greetings": self.greetings, "keywords": self.keywords,
            "background": self.background, "backstory_keywords": self.backstory_keywords,
            "memory_budget": self.memory_budget,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Character":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_character(path: str | Path) -> Character:
    """从 JSON 文件加载角色卡。"""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Character.from_dict(data)


def load_all_characters(dir_path: str | Path = None) -> dict[str, Character]:
    """加载所有角色卡：目录存在则读目录，否则返回空（角色数据由调用方提供）。"""
    result = {}
    d = Path(dir_path) if dir_path else CHARACTERS_DIR
    if d.exists():
        for f in sorted(d.glob("*.json")):
            try:
                c = load_character(f)
                result[c.id] = c
            except Exception as e:
                print(f"[warn] 加载角色卡失败 {f.name}: {e}")
    return result


def match_keywords(text: str, keywords: list[str]) -> list[str]:
    """判断文本是否命中角色关键词（用于背景故事触发）。"""
    return [k for k in keywords if k and k in text]
