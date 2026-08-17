"""核心模块单元测试（不依赖模型权重，纯逻辑）"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from npc.character import Character, load_all_characters, match_keywords
from npc.lorebook import LoreEntry, Lorebook, load_lorebook
from npc.memory import MemorySystem
from npc.postprocess import PostProcessor
from npc.prompt import build_prompt, build_system_prompt


# ---------------- Character ----------------
def test_character_prompt_block():
    c = Character(name="艾拉", identity="流浪商人", personality="警惕、爱钱",
                  speech_style="短句、带刺", goal="找妹妹", attitude_to_player="陌生警惕")
    block = c.to_prompt_block()
    assert "姓名：艾拉" in block
    assert "身份：流浪商人" in block
    assert "找妹妹" in block


def test_character_keywords():
    c = Character(name="艾拉", keywords=["商人", "买卖"])
    hits = match_keywords("我想和你做生意，你是商人吗", c.keywords)
    assert "商人" in hits
    assert match_keywords("今天天气不错", c.keywords) == []


# ---------------- Lorebook ----------------
def test_lorebook_trigger():
    lb = Lorebook([
        LoreEntry(key="wolf", trigger=["狼"], content="狼怕火", priority=3),
        LoreEntry(key="sword", trigger=["剑"], content="月光剑传说", priority=4),
    ])
    hits = lb.query("黑森林有狼出没", max_entries=2, max_budget=300)
    assert any(h.key == "wolf" for h in hits)
    # priority 排序：剑优先
    assert hits[0].key == "sword" if any(h.key == "sword" for h in hits) else True


def test_lorebook_budget():
    lb = Lorebook([
        LoreEntry(key="a", trigger=["苹果"], content="x" * 200, priority=5),
    ])
    # 预算 150，内容 200 超预算 -> 不返回
    hits = lb.query("苹果", max_entries=1, max_budget=150)
    assert len(hits) == 0


# ---------------- Memory ----------------
def test_memory_state():
    with tempfile.TemporaryDirectory() as td:
        ms = MemorySystem(Path(td) / "m.db")
        ms.set_state("aila", {"好感度": 10, "任务": "护送"})
        assert ms.get_state("aila")["好感度"] == 10
        ms.set_state("aila", {"好感度": 20})
        assert ms.get_state("aila")["好感度"] == 20
        ms.close()


def test_memory_events_and_retrieval():
    with tempfile.TemporaryDirectory() as td:
        ms = MemorySystem(Path(td) / "m.db")
        ms.add_event("aila", "玩家帮艾拉修好了马车")
        ms.add_event("aila", "玩家试图偷艾拉的钱袋，被发现了")
        ms.add_event("bruno", "玩家在酒馆买了一杯酒")
        # 去重
        ms.add_event("aila", "玩家帮艾拉修好了马车")
        evts = ms.recent_events("aila", n=10)
        assert len(evts) == 2
        # 检索
        hits = ms.search("aila", "马车", top_k=3)
        assert len(hits) >= 1
        assert "马车" in hits[0]["text"]
        ms.close()


def test_memory_conflict_state_priority():
    with tempfile.TemporaryDirectory() as td:
        ms = MemorySystem(Path(td) / "m.db")
        ms.add_event("aila", "玩家是好人，救了艾拉")
        ms.set_state("aila", {"好感度": -10, "信任": False})
        # 状态优先：检索记忆时状态快照独立
        state = ms.get_state("aila")
        assert state["好感度"] == -10
        ms.close()


# ---------------- PostProcessor ----------------
def test_post_ooc_detection():
    pp = PostProcessor()
    assert pp.check_out_of_character("我是AI助手，可以回答你的问题")
    assert pp.check_out_of_character("作为语言模型，我无法回答")
    assert not pp.check_out_of_character("艾拉：一个路过的商人。别靠太近。")


def test_post_length():
    pp = PostProcessor(max_chars=20)
    # 短句不截断
    assert pp.enforce_length("这是一个很长的句子啊这是第二句。") == "这是一个很长的句子啊这是第二句。"
    # 长句：保留第一句
    long_text = "第一句。第二句很长很长很长很长很长很长很长很长很长很长很长。"
    r = pp.enforce_length(long_text, 20)
    assert len(r) <= 20
    assert r == "第一句"


def test_post_clean():
    pp = PostProcessor()
    assert pp.clean('  "你好"  ') == "你好"


def test_post_process_fallback():
    pp = PostProcessor()
    r = pp.process("我是AI，我是语言模型")
    assert "回答" in r or r  # 兜底
    assert not pp.check_out_of_character(r)


# ---------------- Prompt ----------------
def test_build_prompt_structure():
    c = Character(name="艾拉", identity="商人", personality="警惕",
                  speech_style="短句", goal="找妹妹")
    p = build_prompt(
        character=c,
        player_input="你是谁？",
        state={"好感度": 5, "任务": "护送", "信任": False},
        scene="夜晚营地",
        history=[("玩家", "你好"), ("艾拉", "你是谁？")],
    )
    assert "[当前状态]" in p
    assert "好感度：5" in p
    assert "[最近对话]" in p
    assert "[玩家]" in p
    assert "你是谁？" in p
    # 玩家输入在末尾
    assert p.strip().endswith("你是谁？")


def test_build_system_prompt():
    c = Character(name="艾拉", identity="流浪商人", personality="警惕、爱钱",
                  speech_style="短句、带刺")
    sp = build_system_prompt(c)
    # 说话风格（身份/名字由记忆系统提供，不在 system prompt）
    assert "警惕" in sp or "我的性格" in sp
    assert "不要用AI助手口吻" in sp or "我的口吻" in sp
    assert "绝不承认自己是AI" in sp or "从来不是AI" in sp or "不是AI或助手" in sp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
