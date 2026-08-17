"""ColBERT 记忆管理器：用 LFM2.5-ColBERT-350M 做语义检索（token 级 MaxSim）

比稠密 embedding（Embedding-350M）准确率更高，适合做记忆管理。
参考官方 colbert-rerank.py：query token 向量 × 文档 token 向量 MaxSim。

用法：
    from npc.colbert_memory import ColBERTRetriever
    retriever = ColBERTRetriever()
    retriever.add("黑森林的狼变异了", "wolf_event_1")
    results = retriever.search("狼有什么危险？", top_k=3)
"""
from __future__ import annotations

import json
import threading

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = "D:/ai-models/LFM2.5-ColBERT-350M-transformers"


class ColBERTRetriever:
    """基于 LFM2.5-ColBERT-350M 的记忆检索器（token 级 MaxSim）。"""

    def __init__(self, model_dir: str = MODEL_DIR, device: str = "auto"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 加载 config（前缀/长度/skiplist）
        config_path = f"{model_dir}/config_sentence_transformers.json"
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        self.query_prefix = cfg.get("query_prefix", "")
        self.doc_prefix = cfg.get("document_prefix", "")
        self.query_len = cfg.get("query_length", 32)
        self.doc_len = cfg.get("document_length", 180)
        skiplist = cfg.get("skiplist_words", [])
        self.skiplist = set(
            t for w in skiplist
            for t in self.tokenizer.encode(w, add_special_tokens=False))

        # 加载模型（bidirectional encoder）
        import importlib.util, sys as _sys
        # 官方自定义 modeling 文件
        from transformers import AutoModel
        try:
            self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        except Exception:
            # 自定义 modeling 需要从本地加载
            spec = importlib.util.spec_from_file_location(
                "lfm2_bidir", f"{model_dir}/modeling_lfm2_bidirectional.py")
            mod = importlib.util.module_from_spec(spec)
            _sys.modules["lfm2_bidir"] = mod
            spec.loader.exec_module(mod)
            self.model = mod.LFM2BidirectionalModel.from_pretrained(
                model_dir, trust_remote_code=True)
        self.model.to(self.device).eval()
        print(f"[colbert] 已加载 {model_dir} (device={device})")

        self._lock = threading.Lock()
        # 内存索引：text -> token 向量（文档级缓存）
        self._index: dict[str, np.ndarray] = {}

    def _encode(self, text: str, is_query: bool) -> torch.Tensor:
        """编码为 token 向量 (n_tokens, dim)。官方格式：加 Q/D 前缀 + skiplist mask。"""
        prefix = self.query_prefix if is_query else self.doc_prefix
        toks = self.tokenizer.encode(prefix + text)
        max_len = self.query_len if is_query else self.doc_len
        if is_query:
            pad = self.tokenizer.pad_token_id or 0
            toks = (toks + [pad] * max_len)[:max_len]
        else:
            toks = toks[:max_len]
        # 文档：mask skiplist token（标点），query 不 mask
        mask = None if is_query else [t not in self.skiplist for t in toks]
        input_ids = torch.tensor([toks], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            emb = outputs.last_hidden_state[0] if hasattr(outputs, "last_hidden_state") else outputs[0][0]
        # 应用 skiplist mask（文档 token 置 0）
        if mask is not None:
            mask_t = torch.tensor(mask, dtype=emb.dtype, device=emb.device).unsqueeze(-1)
            emb = emb * mask_t
        # L2 normalize 每个 token 向量（官方 colbert-rerank.py）
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.float()

    def add(self, text: str):
        """索引一条记忆（文档级 token 向量）。"""
        with self._lock:
            emb = self._encode(text, is_query=False).cpu().numpy()
            self._index[text] = emb

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """MaxSim 检索：返回 [(text, score)] 降序。"""
        with self._lock:
            if not self._index:
                return []
            q = self._encode(query, is_query=True).cpu().numpy()  # (q_tokens, dim)
            results = []
            for text, demb in self._index.items():
                # MaxSim: sum over query tokens of max over doc tokens
                sims = q @ demb.T  # (q_tokens, d_tokens)
                score = float(sims.max(axis=1).sum())
                results.append((text, score))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def search_events(self, events: list[str], query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """对给定事件列表做 MaxSim 检索（不维护索引，适合小批量）。"""
        if not events:
            return []
        q = self._encode(query, is_query=True).cpu().numpy()
        results = []
        with self._lock:
            for text in events:
                demb = self._encode(text, is_query=False).cpu().numpy()
                sims = q @ demb.T
                score = float(sims.max(axis=1).sum())
                results.append((text, score))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def clear(self):
        with self._lock:
            self._index.clear()


if __name__ == "__main__":
    r = ColBERTRetriever()
    r.add("黑森林的狼最近变异了，眼睛是红色的，怕火")
    r.add("布鲁诺的酒馆是本地的消息集散地")
    r.add("国王最近在征兵，边境气氛紧张")
    for q in ["狼有什么危险？", "酒馆在哪里？", "边境怎么了？"]:
        print(f"\nQ: {q}")
        for t, s in r.search(q, 2):
            print(f"  {s:.2f}: {t[:30]}")
