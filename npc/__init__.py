"""NPC 角色扮演对话系统核心包"""
from npc.config import (
    ModelConfig, GenerationConfig, PROJECT_ROOT,
    WEIGHTS_DIR, MODEL_DIR, CHARACTERS_DIR, LOREBOCK_DIR,
    OUT_OF_CHARACTER_PHRASES,
)
from npc.character import Character, load_character, load_all_characters
from npc.model import load_model
from npc.lorebook import Lorebook, load_lorebook
from npc.prompt import build_prompt

__all__ = [
    "ModelConfig", "GenerationConfig",
    "PROJECT_ROOT", "WEIGHTS_DIR", "MODEL_DIR", "CHARACTERS_DIR", "LOREBOCK_DIR",
    "OUT_OF_CHARACTER_PHRASES",
    "Character", "load_character", "load_all_characters",
    "Lorebook", "load_lorebook",
    "build_prompt",
]
