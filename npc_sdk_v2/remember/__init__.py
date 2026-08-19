"""游戏 NPC 记忆系统（精简版）"""
from remember.character import Character, load_character, load_all_characters
from remember.memory import MemorySystem
from remember.memory_extract import extract_facts, memory_entry
from remember.memory_prompt import fact_to_memory_line, build_memory_prompt

__all__ = [
    "Character", "load_character", "load_all_characters",
    "MemorySystem",
    "extract_facts", "memory_entry",
    "fact_to_memory_line", "build_memory_prompt",
]
