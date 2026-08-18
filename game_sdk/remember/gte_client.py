"""GTE 检索客户端：用 gte-multilingual-base 做语义检索

对标 embedding.py 的 EmbeddingClient 接口，便于切换对比。
gte-multilingual-base: 多语言文本嵌入，768 维，sentence-transformers 格式。
"""
from __future__ import annotations

import math
import re

MODEL_DIR = "D:/ai-models/gte-multilingual-base"


class GTEClient:
    """基于 gte-multilingual-base 的语义检索客户端。"""

    def __init__(self, model_dir: str = MODEL_DIR, device: str = "auto"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()
        self._dim = None
        print(f"[gte] 已加载 {model_dir} (device={device})")

    def embed(self, text: str):
        """返回文本向量（768 维，mean pooling + L2 norm）。失败返回 None。"""
        if not text or not text.strip():
            return None
        import torch
        import torch.nn.functional as F
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        # mean pooling（忽略 pad）
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        vec = F.normalize(vec, p=2, dim=-1)
        self._dim = vec.shape[-1]
        return vec[0].cpu().tolist()

    def cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    g = GTEClient()
    v1 = g.embed("黑森林的狼最近变异了，红眼怕火")
    v2 = g.embed("狼群在森林里出没")
    v3 = g.embed("今天天气很好适合散步")
    print("维度:", len(v1))
    print("狼相关:", round(g.cosine(v1, v2), 4))
    print("狼vs天气:", round(g.cosine(v1, v3), 4))
