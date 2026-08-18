"""多模型检索对比测试：GTE / bge-base-zh / bge-m3 / LFM-Embedding"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DOCS = [
    "我是艾拉，流浪商人，性格警惕爱钱",
    "我的过去：妹妹被掳走，一直在暗中寻找",
    "我把剑定价为五百金币",
    "有个玩家说他叫林风，从北方来",
    "玩家偷过我的钱袋，我记恨他",
]
QUERIES = [
    ("这剑多少钱？", "定价"),
    ("你妹妹在哪？", "妹妹"),
    ("你还记得我名字吗？", "林风"),
    ("你是谁？", "流浪商人"),
    ("你以前经历过什么？", "过去"),
]


class HFEmbedder:
    def __init__(self, model_dir, query_prefix="", max_len=512):
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).eval()
        self.query_prefix = query_prefix
        self.max_len = max_len
        self.name = model_dir.split("/")[-1]
        print(f"[{self.name}] 加载完成")

    def embed(self, text, is_query=False):
        if is_query and self.query_prefix:
            text = self.query_prefix + text
        inputs = self.tok(text, return_tensors="pt", max_length=self.max_len, truncation=True)
        with torch.no_grad():
            out = self.model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return F.normalize(vec, p=2, dim=-1)[0]

    def cosine(self, a, b):
        return float((a * b).sum())


def evaluate(emb, name):
    print(f"\n=== {name} ===")
    ok = 0
    for q, kw in QUERIES:
        qv = emb.embed(q, is_query=True)
        sims = [(d, emb.cosine(qv, emb.embed(d))) for d in DOCS]
        top = max(sims, key=lambda x: x[1])
        hit = kw in top[0]
        ok += hit
        print(f"  {q} -> {'OK' if hit else 'X '} ({top[1]:.3f}) {top[0][:18]}")
    print(f"  准确率: {ok}/{len(QUERIES)}")
    return ok


def main():
    models = {
        "GTE-multilingual-base": ("D:/ai-models/gte-multilingual-base", ""),
        "bge-base-zh-v1.5": ("D:/ai-models/bge-base-zh-v1.5",
                             "为这个句子生成表示以用于检索相关文章："),
        "bge-m3": ("D:/ai-models/bge-m3", ""),
    }
    for name, (path, prefix) in models.items():
        try:
            emb = HFEmbedder(path, prefix)
            evaluate(emb, name)
        except Exception as e:
            print(f"\n=== {name} === 加载失败: {str(e)[:60]}")


if __name__ == "__main__":
    main()
