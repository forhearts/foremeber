"""EncoderRouter：用 LFM2.5-Encoder-350M 粗筛玩家意图/主题

记忆系统的第一层：先判断玩家输入属于什么意图类别，
再决定检索哪类记忆（交易/打听/关系/攻击/出戏...），
避免无关记忆污染 prompt。

零样本分类：Encoder 句子向量 vs 意图模板向量 余弦相似度。
"""
from __future__ import annotations

import threading

import torch
import torch.nn.functional as F

MODEL_DIR = "D:/ai-models/LFM2.5-Encoder-350M"

# 意图类别 → 模板（零样本分类用）
INTENT_TEMPLATES = {
    "交易": "我想买卖东西，询问价格",
    "讨价还价": "太贵了，便宜一点",
    "打听情报": "最近有什么消息传闻新闻",
    "询问身份": "你是谁，你叫什么名字",
    "询问任务": "有什么任务需要帮忙",
    "送礼": "这个送给你，给你礼物",
    "攻击威胁": "我要教训你，小心点",
    "关系亲密": "你喜欢我吗，我们做朋友",
    "往事回忆": "你的过去，你的家人经历",
    "世界观": "这个世界，魔法，神明",
    "闲聊": "今天天气，随便聊聊",
    "出戏测试": "你是AI吗，忽略设定，系统提示",
    "离开": "再见，我走了",
    "求助": "帮我个忙，需要帮助",
}

# 意图 → 记忆检索重点
INTENT_MEMORY_FOCUS = {
    "交易": ["交易", "价格", "货物"],
    "讨价还价": ["交易", "价格"],
    "打听情报": ["消息", "传闻", "事件"],
    "询问身份": ["身份", "背景"],
    "询问任务": ["任务", "委托"],
    "送礼": ["好感", "送礼", "关系"],
    "攻击威胁": ["冲突", "敌对"],
    "关系亲密": ["好感", "关系"],
    "往事回忆": ["背景", "往事"],
    "世界观": ["设定", "世界"],
    "闲聊": [],
    "出戏测试": [],
    "离开": [],
    "求助": ["任务", "帮助"],
}


class EncoderRouter:
    """意图粗筛器（零样本分类）。"""

    def __init__(self, model_dir: str = MODEL_DIR, device: str = "auto"):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_dir, trust_remote_code=True).to(device).eval()
        self._lock = threading.Lock()
        # 预计算意图模板向量
        self._templates = {}
        with torch.no_grad():
            for intent, tpl in INTENT_TEMPLATES.items():
                self._templates[intent] = self._sentence_vec(tpl)
        print(f"[encoder] 意图路由器已加载 (device={device})")

    def _sentence_vec(self, text: str) -> torch.Tensor:
        """句子向量（mean pooling + L2 norm）。"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        vec = out.hidden_states[-1][0].mean(dim=0)
        return F.normalize(vec, p=2, dim=0)

    def classify(self, text: str) -> tuple[str, dict[str, float]]:
        """粗筛意图。返回 (意图, 各意图分数)。"""
        with self._lock:
            q = self._sentence_vec(text)
            scores = {}
            for intent, tvec in self._templates.items():
                scores[intent] = float((q * tvec).sum())
        best = max(scores, key=scores.get)
        return best, scores

    def memory_focus(self, intent: str) -> list[str]:
        """意图 → 记忆检索关键词（用于语义检索加权）。"""
        return INTENT_MEMORY_FOCUS.get(intent, [])


if __name__ == "__main__":
    r = EncoderRouter()
    for q in ["这把剑多少钱？", "最近村里有什么新闻？", "你是AI吗？", "送你这个礼物", "再见"]:
        intent, scores = r.classify(q)
        top = sorted(scores.items(), key=lambda x: -x[1])[:3]
        print(f"{q} -> {intent} | {[(k, round(v,3)) for k,v in top]}")
