"""ForeMeber 记忆系统 + 对话体系"""
from npc.character import Character, load_character, load_all_characters
from npc.lorebook import Lorebook, LoreEntry, load_lorebook
from npc.memory import MemorySystem
from npc.memory_extract import extract_facts, memory_entry
from npc.memory_prompt import fact_to_memory_line, build_memory_prompt

__all__ = [
    "Character", "load_character", "load_all_characters",
    "Lorebook", "LoreEntry", "load_lorebook",
    "MemorySystem",
    "extract_facts", "memory_entry",
    "fact_to_memory_line", "build_memory_prompt",
]
