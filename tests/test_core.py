"""记忆系统单元测试"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from remember.character import Character, load_all_characters, match_keywords
from remember.lorebook import LoreEntry, Lorebook, load_lorebook
from remember.memory import MemorySystem
from remember.memory_extract import extract_facts, memory_entry
from remember.memory_prompt import fact_to_memory_line, build_memory_prompt
from remember.prompt import build_prompt, build_memory_pack


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


def test_lorebook_budget():
    lb = Lorebook([
        LoreEntry(key="a", trigger=["苹果"], content="x" * 200, priority=5),
    ])
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


def test_memory_events_and_dedup():
    with tempfile.TemporaryDirectory() as td:
        ms = MemorySystem(Path(td) / "m.db")
        ms.add_event("aila", "我是艾拉，流浪商人")
        ms.add_event("aila", "我是艾拉，流浪商人")  # 去重
        ms.add_event("aila", "我是艾拉，流浪商人")  # 去重
        ms.add_event("aila", "这把剑我定价五百金币")
        evts = ms.recent_events("aila", 10)
        assert len(evts) == 2  # 去重生效
        ms.close()


def test_memory_persona():
    with tempfile.TemporaryDirectory() as td:
        ms = MemorySystem(Path(td) / "m.db")
        ms.set_persona("aila", ["我的性格：警惕爱钱"], ["玩家：你是谁？ → 艾拉：别靠太近。"])
        facts, examples = ms.get_persona("aila")
        assert facts == ["我的性格：警惕爱钱"]
        assert len(examples) == 1
        # clear_character 不清 persona
        ms.clear_character("aila")
        facts2, examples2 = ms.get_persona("aila")
        assert facts2 == ["我的性格：警惕爱钱"]
        ms.close()


def test_memory_conflict_state_priority():
    with tempfile.TemporaryDirectory() as td:
        ms = MemorySystem(Path(td) / "m.db")
        ms.add_event("aila", "玩家是好人，救了艾拉")
        ms.set_state("aila", {"好感度": -10, "信任": False})
        state = ms.get_state("aila")
        assert state["好感度"] == -10
        ms.close()


# ---------------- MemoryExtract ----------------
def test_extract_price():
    facts = extract_facts("aila", "这把剑多少钱？", "五百金币")
    assert any("定价" in f or "价格" in f for f in facts)


def test_extract_price_only_when_asking():
    # 砍价不产生价格事实（讨论不是定价）
    facts = extract_facts("aila", "太贵了，便宜点吧", "五百金币不讲价")
    assert not any("定价" in f for f in facts)


def test_extract_name():
    facts = extract_facts("aila", "我叫林风，从北方来", "哦")
    assert any("林风" in f for f in facts)
    assert any("北方" in f for f in facts)


def test_memory_entry_none_for_noise():
    entry = memory_entry("aila", "今天天气不错", "嗯")
    assert entry is None  # 无事实不存噪音


# ---------------- MemoryPrompt ----------------
def test_fact_to_memory_line():
    assert fact_to_memory_line("aila", "aila把剑定价为五百金币") == "这把剑我定价五百金币"
    assert fact_to_memory_line("aila", "艾拉是流浪商人") == "我是流浪商人"
    assert fact_to_memory_line("aila", "玩家自称林风") == "有个玩家说他叫林风"
    assert fact_to_memory_line("aila", "aila提到卖东西的") is None  # 噪音跳过


def test_build_memory_prompt():
    facts = ["aila把剑定价为五百金币", "艾拉是流浪商人", "玩家自称林风"]
    p = build_memory_prompt("aila", facts)
    assert "这把剑我定价五百金币" in p
    assert "我是流浪商人" in p


# ---------------- Prompt ----------------
def test_build_memory_pack():
    c = Character(name="艾拉", identity="流浪商人", personality="警惕、爱钱")
    pack = build_memory_pack(
        character=c, player_input="你是谁？", intent="询问身份",
        state={"好感度": 5}, scene="集市摊位",
        memories=["这把剑我定价五百金币"],
    )
    assert pack["intent"] == "询问身份"
    assert pack["scene"] == "集市摊位"
    assert "我定价" in pack["memories"][0] or "定价" in pack["memories"][0]
    assert "场景" in pack["text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
