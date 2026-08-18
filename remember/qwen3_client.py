"""Qwen3-Embedding 检索客户端：用 Qwen3-Embedding-0.6B 做语义检索

对标 EmbeddingClient 接口，便于切换。
Qwen3-Embedding 支持指令格式，检索效果最佳（实测 5/5）。
"""
from __future__ import annotations

import math

MODEL_DIR = "D:/ai-models/qwen3-embedding-0.6b"

# 指令前缀（检索任务）
QUERY_INSTRUCTION = "Instruct: 检索相关记忆\nQuery: "


class Qwen3Client:
    """基于 Qwen3-Embedding-0.6B 的语义检索客户端。"""

    def __init__(self, model_dir: str = MODEL_DIR, device: str = "cpu"):
        # 默认 CPU：避免与 14B 生成模型抢显存（8GB 不够同时跑）
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()
        self._dim = None
        print(f"[qwen3-emb] 已加载 {model_dir} (device={device})")

    def embed(self, text: str, is_query: bool = False):
        """返回文本向量。查询用指令前缀。失败返回 None。"""
        if not text or not text.strip():
            return None
        import torch
        import torch.nn.functional as F
        t = QUERY_INSTRUCTION + text if is_query else text
        inputs = self.tokenizer(t, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        v = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        v = F.normalize(v, p=2, dim=-1)
        self._dim = v.shape[-1]
        return v[0].cpu().tolist()

    def cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0
