"""记忆系统（参考 Mem0 / Letta / 社区 SQLite 记忆方案）

融合社区成熟设计（见规划 6.7）：
- Mem0: add() 写 → search() 检索 → 应用决定注入（本实现用引擎事件直写，不用 LLM 提取）
- Letta/MemGPT: Memory Blocks（核心记忆块，常驻）+ Archival Memory（外部存档，按需检索）
- sqlite-memory-mcp / ClawMemory: SQLite + FTS5/BM25 混合检索 + 时间衰减
- AgentDB: 短期/中期/长期分层

设计：
- 核心记忆块（Core Memory）: 常驻 prompt 的结构化状态（好感度/任务/信任）——对应第一层
- 事件存档（Archival Memory）: FTS5 全文索引的全部事件——对应第三层
- 混合检索: FTS5 BM25 + 时间衰减 + 去重
- 冲突解决: 状态永远优先于记忆（规划 6.6）
- 线程安全: check_same_thread=False + RLock，支持 Gradio 多线程
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MemorySystem:
    """SQLite 记忆系统：核心记忆块 + 事件存档（语义+关键词混合检索）。

    检索策略（规划 6.7 社区方案）：
    - 语义检索：LFM2.5-Embedding-350M 向量相似度（小模型，质量优先）
    - 关键词检索：FTS5 BM25（兜底，零模型开销）
    - 两者 RRF 融合 + 时间衰减
    低端设备：无 embedding 服务时自动退化 FTS5。
    线程安全：所有操作通过 RLock 串行化。
    """

    def __init__(self, db_path: str | Path | None = None, embed_client=None, colbert=None):
        self.db_path = Path(db_path) if db_path else PROJECT_ROOT / "data" / "memory.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: 允许跨线程访问（配合 RLock 保证串行）
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        # 语义检索：优先 ColBERT（MaxSim），其次 EmbeddingClient（余弦），都无则 FTS5
        self._colbert = colbert
        self._embed = embed_client
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            c = self.conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS core_memory (
                    character_id TEXT PRIMARY KEY,
                    block_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            c.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS events USING fts5(
                    character_id UNINDEXED, event_text, original_text, created_at UNINDEXED
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS event_meta (
                    fingerprint TEXT PRIMARY KEY,
                    character_id TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            # 人设/示范层（不参与检索，常驻供 prompt）
            c.execute("""
                CREATE TABLE IF NOT EXISTS persona (
                    character_id TEXT PRIMARY KEY,
                    facts_json TEXT NOT NULL,
                    examples_json TEXT NOT NULL
                )
            """)
            self.conn.commit()

    # ================= 人设/示范层（Persona，不参与检索） =================

    def set_persona(self, character_id: str, facts: list[str], examples: list[str]):
        """存人设事实 + 对话示范（few-shot），常驻供 prompt。"""
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO persona (character_id, facts_json, examples_json) VALUES (?, ?, ?)",
                (character_id, json.dumps(facts, ensure_ascii=False),
                 json.dumps(examples, ensure_ascii=False)),
            )
            self.conn.commit()

    def get_persona(self, character_id: str) -> tuple[list[str], list[str]]:
        """返回 (人设事实列表, 示范列表)。"""
        with self._lock:
            row = self.conn.execute(
                "SELECT facts_json, examples_json FROM persona WHERE character_id = ?",
                (character_id,),
            ).fetchone()
        if row is None:
            return [], []
        return json.loads(row["facts_json"]), json.loads(row["examples_json"])

    # ================= 核心记忆块（Core Memory，常驻） =================
    def set_state(self, character_id: str, state: dict):
        """游戏引擎事件直写核心记忆块（好感度/任务/物品/关系）。"""
        state["_updated_at"] = time.time()
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO core_memory (character_id, block_json, updated_at) VALUES (?, ?, ?)",
                (character_id, json.dumps(state, ensure_ascii=False), time.time()),
            )
            self.conn.commit()

    def get_state(self, character_id: str) -> dict:
        with self._lock:
            row = self.conn.execute(
                "SELECT block_json FROM core_memory WHERE character_id = ?", (character_id,)
            ).fetchone()
        if row is None:
            return {}
        d = json.loads(row["block_json"])
        d.pop("_updated_at", None)
        return d

    def get_state_snapshot(self, character_id: str, max_chars: int = 200) -> str:
        """返回状态块文本（预算纪律：<100字量级，超出截断）。"""
        state = self.get_state(character_id)
        if not state:
            return ""
        parts = []
        for k, v in state.items():
            if v is None or v == "":
                continue
            if isinstance(v, bool):
                v = "是" if v else "否"
            parts.append(f"{k}：{v}")
        text = "；".join(parts)
        return text[:max_chars]

    # ================= 事件存档（Archival Memory，写） =================

    def add_event(self, character_id: str, event_text: str, fingerprint: str | None = None):
        """写入一条事件到存档。fingerprint 用于去重（可选）。

        中文 bigram 预处理：把连续 CJK 切为空格分隔的 2-gram，
        使 FTS5 能对中文子串做 BM25 匹配（unicode61 不切中文）。
        """
        text = event_text.strip()
        if not text:
            return
        fp = fingerprint or self._fingerprint(text)
        indexed = self._cjk_bigram(text)
        now = time.time()
        with self._lock:
            # 去重：同一指纹不重复写入
            dup = self.conn.execute(
                "SELECT 1 FROM event_meta WHERE fingerprint = ?", (fp,)
            ).fetchone()
            if dup:
                return
            self.conn.execute(
                "INSERT INTO events (character_id, event_text, original_text, created_at) VALUES (?, ?, ?, ?)",
                (character_id, indexed, text, now),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO event_meta (fingerprint, character_id, created_at) VALUES (?, ?, ?)",
                (fp, character_id, now),
            )
            self.conn.commit()

    @staticmethod
    def _cjk_bigram(text: str) -> str:
        """把连续中文切为 [单字 + 2-gram] 混合 token（保证单字可检索）。"""
        out = []
        seg = ""

        def flush():
            nonlocal seg
            if seg:
                chars = list(seg)
                out.extend(chars)  # 单字
                if len(chars) >= 2:
                    out.extend(seg[i:i+2] for i in range(len(chars)-1))  # 2-gram
            seg = ""

        buf = ""
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                if buf:
                    out.append(buf)
                    buf = ""
                seg += ch
            elif ch.isalnum():
                flush()
                buf += ch
            else:
                flush()
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(ch)
        flush()
        if buf:
            out.append(buf)
        return " ".join(out)

    @staticmethod
    def _restore_cjk(indexed: str) -> str:
        """还原 bigram 化文本为原文：去掉 2-gram 之间插入的空格。"""
        return re.sub(
            r"((?:[\u4e00-\u9fff] )+[\u4e00-\u9fff])",
            lambda m: m.group(1).replace(" ", ""), indexed,
        )

    @staticmethod
    def _fingerprint(text: str) -> str:
        """稳定指纹：规范化后 md5（跨进程一致，保证去重生效）。"""
        import hashlib
        norm = re.sub(r"\s+", "", text)
        return hashlib.md5(norm.encode("utf-8")).hexdigest()[:16]

    def add_dialogue_turn(
        self, character_id: str, player_text: str, npc_text: str,
        state_snapshot: dict | None = None,
    ):
        """记录一轮对话，提炼为 NPC 记住的事实（非对话记录）。

        冲突解决：状态永远优先于记忆（规划 6.6）。
        """
        from npc.memory_extract import memory_entry
        entry = memory_entry(character_id, player_text, npc_text)
        if entry:
            # 价格事实覆盖：同物品已有定价则删旧（只留最新价）
            import re as _re
            price_new = _re.search(r"把(.+?)定价为", entry)
            if price_new:
                item = price_new.group(1)
                with self._lock:
                    self.conn.execute(
                        "DELETE FROM events WHERE character_id=? AND original_text LIKE ?",
                        (character_id, f"%把{item}定价为%"))
                    self.conn.commit()
            self.add_event(character_id, entry)

    # ================= 混合检索（BM25 + 时间衰减） =================

    def _fts_search(self, character_id: str, query: str, limit: int = 10) -> list[tuple[str, float, float]]:
        """FTS5 BM25 检索。返回 [(text, score, created_at)]，score 越低越相关（BM25）。"""
        terms = self._tokenize(query)
        if not terms:
            return []
        # 查询词同样 bigram 化（与存储侧 _cjk_bigram 一致），保证能命中索引
        query_bigrams = [t for t in self._cjk_bigram(query).split() if len(t) >= 2]
        if not query_bigrams:
            query_bigrams = terms
        # 单字中文查询：直接用单字（FTS5 能匹配包含该字的 token 前缀）
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", query)
        query_bigrams += [c for c in cjk_chars if c not in query_bigrams]
        # 用 OR 组合（命中越多 BM25 分越高）
        search = " OR ".join(f'"{t}"' for t in query_bigrams[:8])
        with self._lock:
            rows = self.conn.execute(
                "SELECT original_text, created_at, bm25(events) AS score FROM events "
                "WHERE character_id = ? AND events MATCH ? ORDER BY score LIMIT ?",
                (character_id, search, limit),
            ).fetchall()
        return [(r["original_text"] or r["event_text"], r["score"], r["created_at"]) for r in rows]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """分词：英文按词 + 中文 bigram（FTS5 默认分词器不支持中文）。"""
        text = text.strip()
        if not text:
            return []
        tokens = []
        for w in re.findall(r"[A-Za-z0-9_]{2,}", text):
            tokens.append(w)
        cjk = re.findall(r"[\u4e00-\u9fff]+", text)
        for seg in cjk:
            if len(seg) == 1:
                tokens.append(seg)
                continue
            for i in range(len(seg) - 1):
                tokens.append(seg[i:i+2])
        return tokens

    def _time_decay(self, created_at: float, half_life_days: float = 30.0) -> float:
        """指数时间衰减权重（0~1，越新越高）。"""
        age_days = (time.time() - created_at) / 86400.0
        return 2.0 ** (-age_days / half_life_days)

    def search(
        self,
        character_id: str,
        query: str,
        top_k: int = 3,
        half_life_days: float = 30.0,
        recency_boost: float = 1.5,
    ) -> list[dict]:
        """混合检索：语义（embedding）+ 关键词（FTS5）RRF 融合 + 时间衰减。

        参考 Mem0 / sqlite-memory-mcp 的混合检索设计。
        - 有 embedding 服务：语义相似度为主
        - 无 embedding 服务：退化 FTS5 BM25
        返回 [{"text":..., "score":..., "created_at":...}]
        """
        candidates = {}  # text -> {score, created}

        # 1. 语义检索：优先 ColBERT MaxSim，其次 embedding 余弦
        recent = self.recent_events_raw(character_id, n=50)
        if self._colbert is not None and recent:
            texts = [t for t, _ in recent]
            hits = self._colbert.search_events(texts, query, top_k=top_k * 3)
            created_map = {t: c for t, c in recent}
            if hits:
                max_s = max(h[1] for h in hits) or 1.0
                for text, score in hits:
                    norm = score / max_s
                    decay = self._time_decay(created_map.get(text, time.time()), half_life_days)
                    s = norm * (0.5 + 0.5 * decay) * recency_boost
                    if text not in candidates or s > candidates[text]["score"]:
                        candidates[text] = {"score": s, "created": created_map.get(text, time.time())}
        elif self._embed is not None and recent:
            qvec = self._embed.embed(query)
            if qvec:
                for text, created in recent:
                    evec = self._embed.embed(text)
                    if evec is None:
                        continue
                    sim = self._embed.cosine(qvec, evec)
                    if sim > 0.3:  # 阈值过滤无关
                        decay = self._time_decay(created, half_life_days)
                        score = sim * (0.5 + 0.5 * decay) * recency_boost
                        if text not in candidates or score > candidates[text]["score"]:
                            candidates[text] = {"score": score, "created": created}

        # 2. 交易/关系上下文关联：交易类意图（嫌贵/打折/砍价）强制关联最近交易记忆
        #    这是游戏场景强先验：玩家砍价必然关于刚才谈的价格
        trade_hint = any(k in query for k in ["贵", "便宜", "打折", "砍价", "优惠", "降价", "太贵", "少点"])
        if trade_hint:
            for text, created in recent[:5]:  # 最近 5 条
                if any(k in text for k in ["定价", "价格", "金币", "银币", "铜币", "卖", "买"]):
                    decay = self._time_decay(created, half_life_days)
                    # 强权重：交易上下文
                    score = 1.2 * (0.6 + 0.4 * decay)
                    if text not in candidates or score > candidates[text]["score"]:
                        candidates[text] = {"score": score, "created": created}
        # 关系上下文：提到名字/记忆时关联身份类记忆
        elif any(k in query for k in ["名字", "记住", "记得", "叫什么", "从哪里"]):
            for text, created in recent[:5]:
                if any(k in text for k in ["玩家", "自称", "来自", "名字"]):
                    decay = self._time_decay(created, half_life_days)
                    score = 1.2 * (0.6 + 0.4 * decay)
                    if text not in candidates or score > candidates[text]["score"]:
                        candidates[text] = {"score": score, "created": created}

        # 2. 关键词检索（FTS5，兜底 + 补充）
        hits = self._fts_search(character_id, query, limit=top_k * 4)
        for text, bm25, created in hits:
            rel = 1.0 / (1.0 + max(bm25, 0))
            decay = self._time_decay(created, half_life_days)
            score = rel * (0.5 + 0.5 * decay) * recency_boost
            if text not in candidates or score > candidates[text]["score"]:
                candidates[text] = {"score": score, "created": created}

        if not candidates:
            return []
        scored = [{"text": t, "score": v["score"], "created_at": v["created"]}
                  for t, v in candidates.items()]
        scored.sort(key=lambda x: -x["score"])
        # 去重（按文本前缀相似度）
        unique = []
        seen = set()
        for item in scored:
            key = item["text"][:20]
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique[:top_k]

    def build_memory_context(
        self,
        character_id: str,
        player_input: str,
        top_k: int = 3,
        max_chars_each: int = 100,
    ) -> list[str]:
        """构造 [Memory] 块短句列表（预算纪律：每条百字内）。"""
        items = self.search(character_id, player_input, top_k=top_k)
        out = []
        for it in items:
            text = it["text"]
            out.append(text if len(text) <= max_chars_each else text[:max_chars_each] + "…")
        return out

    def recent_events(self, character_id: str, n: int = 10) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT original_text FROM events WHERE character_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (character_id, n),
            ).fetchall()
        return [r["original_text"] or "" for r in reversed(rows)]

    def recent_events_raw(self, character_id: str, n: int = 50) -> list[tuple[str, float]]:
        """返回 (文本, created_at) 对（供语义检索）。"""
        with self._lock:
            rows = self.conn.execute(
                "SELECT original_text, created_at FROM events WHERE character_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (character_id, n),
            ).fetchall()
        return [(r["original_text"] or "", r["created_at"]) for r in rows]

    def clear_character(self, character_id: str):
        with self._lock:
            self.conn.execute("DELETE FROM core_memory WHERE character_id = ?", (character_id,))
            self.conn.execute("DELETE FROM events WHERE character_id = ?", (character_id,))
            self.conn.execute("DELETE FROM event_meta WHERE character_id = ?", (character_id,))
            self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()


    # ================= 记忆分层获取（PromptBuilder 使用） =================

    def fixed_memories(self, character_id: str, n: int = 4) -> list[str]:
        """固定记忆：身份/背景/目标/禁忌（NPC 总是知道，每轮注入）。"""
        evts = self.recent_events(character_id, 20)
        return [e for e in evts
                if any(k in e for k in ["是", "的背景", "的目标", "的禁忌", "的过去", "我不谈论"])][:n]

    def dynamic_memories(self, character_id: str, query: str, n: int = 3) -> list[str]:
        """动态记忆：价格/名字/事件（语义检索）。"""
        return self.build_memory_context(character_id, query, top_k=n)
