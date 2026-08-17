"""NPC 角色扮演对话系统 - 统一配置"""
from dataclasses import dataclass, field
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 路径
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_DIR = WEIGHTS_DIR / "minimind-3o-pytorch"
LLM_WEIGHT = MODEL_DIR / "llm_768_moe.pth"          # 语言基座 (MoE)
OMNI_WEIGHT = MODEL_DIR / "sft_omni_768_moe.pth"    # 完整 Omni
SFT_WEIGHT = PROJECT_ROOT / "weights" / "sft_rp.pth"  # 微调后输出
LORA_DIR = PROJECT_ROOT / "weights" / "lora_rp"       # LoRA 输出
QWEN_DIR = WEIGHTS_DIR / "qwen" / "Qwen2.5-0.5B-Instruct"  # Qwen 基座（推荐）
LORA_QWEN = PROJECT_ROOT / "weights" / "lora_qwen"       # Qwen LoRA
CHARACTERS_DIR = PROJECT_ROOT / "characters"
LOREBOCK_DIR = PROJECT_ROOT / "lorebook"
DATA_SFT_DIR = PROJECT_ROOT / "data" / "sft"
DATA_SEEDS_DIR = PROJECT_ROOT / "data" / "seeds"

# 模型配置（与 minimind-3o-moe 官方一致）
@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 2432          # ceil(768*pi/64)*64
    vocab_size: int = 6400
    max_position_embeddings: int = 32768
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 1
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    flash_attn: bool = True
    # Omni 专属
    num_talker_hidden_layers: int = 4
    talker_hidden_size: int = 768
    audio_vocab_size: int = 2112
    bridge_layer: int = field(default=None)

    def __post_init__(self):
        if self.bridge_layer is None:
            self.bridge_layer = self.num_hidden_layers // 2 - 1

# 推理参数（规划 4.5）
@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_new_tokens: int = 48   # 短回复（角色台词 1-2 句）
    repetition_penalty: float = 1.2
    eos_token_id: int = 2

# 后处理（规划 7.1）
OUT_OF_CHARACTER_PHRASES = [
    "我是AI", "作为语言模型", "作为助手", "抱歉，我不能", "我不能回答",
    "忽略之前的设定", "系统提示", "prompt", "我是一个AI", "语言模型",
    "我是一个人工智能", "AI助手", "作为一个人工智能",
]
