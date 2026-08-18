"""Embedding 客户端：连接本地 llama-server embedding 服务（LFM2.5-Embedding-350M）

用法：
    embed_client = EmbeddingClient()
    vec = embed_client.embed("黑森林的狼最近变异了")
"""
from __future__ import annotations

import json
import urllib.request

# llama-server embedding 端点
EMBEDDING_URL = "http://127.0.0.1:8081/v1/embeddings"  # 若单独起服务可改端口
MODEL = "LFM2.5-Embedding-350M-Q4_K_M"  # 按实际下载文件名调整


class EmbeddingClient:
    def __init__(self, url: str = EMBEDDING_URL, model: str = MODEL):
        self.url = url
        self.model = model
        self._dim = None

    def embed(self, text: str) -> list[float] | None:
        """返回文本向量（1024 维），失败返回 None。"""
        if not text or not text.strip():
            return None
        body = json.dumps({
            "input": text,
            "model": self.model,
        }).encode("utf-8")
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
            vec = d["data"][0]["embedding"]
            self._dim = len(vec)
            return vec
        except Exception as e:
            print(f"[embedding] 失败: {e}")
            return None

    def cosine(self, a: list[float], b: list[float]) -> float:
        """余弦相似度。"""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


def check_embedding_service() -> bool:
    """检查 embedding 服务是否可用。"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False
