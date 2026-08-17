"""记忆系统核心包（独立，不依赖对话引擎/模型）"""
from npc.character import Character, load_character, load_all_characters
from npc.lorebook import Lorebook, LoreEntry, load_lorebook
from npc.memory import MemorySystem
from npc.memory_extract import extract_facts, memory_entry
from npc.memory_prompt import fact_to_memory_line, build_memory_prompt
from npc.embedding import EmbeddingClient
from npc.colbert_memory import ColBERTRetriever
from npc.encoder_router import EncoderRouter

__all__ = [
    "Character", "load_character", "load_all_characters",
    "Lorebook", "LoreEntry", "load_lorebook",
    "MemorySystem",
    "extract_facts", "memory_entry",
    "fact_to_memory_line", "build_memory_prompt",
    "EmbeddingClient", "ColBERTRetriever", "EncoderRouter",
]
