"""ForeMeber 测试：记忆系统 + 拼装 + 对话（使用 tests/fixtures 角色数据）"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from remember.character import Character, load_all_characters
from remember.lorebook import Lorebook, load_lorebook
from remember.memory import MemorySystem
from remember.memory_extract import extract_facts, memory_entry
from remember.memory_prompt import fact_to_memory_line, build_memory_prompt

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def characters():
    return load_all_characters(FIXTURES / "characters")


@pytest.fixture
def lorebook():
    return load_lorebook(FIXTURES / "lorebook")


@pytest.fixture
def memory():
    with tempfile.TemporaryDirectory() as td:
        ms = MemorySystem(Path(td) / "test.db")
        yield ms
        ms.close()


# ---------------- 角色数据 ----------------
def test_characters_loaded(characters):
    assert len(characters) == 8
    assert "aila" in characters
    assert characters["aila"].name == "艾拉"


def test_lorebook_loaded(lorebook):
    assert len(lorebook.entries) == 12


# ---------------- 记忆系统 ----------------
def test_memory_state(memory):
    memory.set_state("aila", {"好感度": 10, "任务": "护送"})
    assert memory.get_state("aila")["好感度"] == 10


def test_memory_dedup(memory):
    memory.add_event("aila", "我是艾拉，流浪商人")
    memory.add_event("aila", "我是艾拉，流浪商人")
    memory.add_event("aila", "我把剑定价为五百金币")
    evts = memory.recent_events("aila", 10)
    assert len(evts) == 2  # 去重


def test_memory_price_overwrite(memory):
    memory.add_event("aila", "我把剑定价为五百金币")
    memory.add_event("aila", "我把剑定价为六百金币")
    evts = memory.recent_events("aila", 10)
    assert len(evts) == 1  # 同物品价格覆盖
    assert "六百" in evts[0]


def test_memory_persona(memory):
    memory.set_persona("aila", ["我的性格：警惕爱钱"], ["玩家：你是谁？ → 艾拉：别靠太近。"])
    facts, examples = memory.get_persona("aila")
    assert facts == ["我的性格：警惕爱钱"]
    assert len(examples) == 1
    # clear 不清 persona
    memory.clear_character("aila")
    facts2, _ = memory.get_persona("aila")
    assert facts2 == ["我的性格：警惕爱钱"]


def test_memory_fixed_and_dynamic(memory):
    memory.add_event("aila", "我是艾拉，流浪商人")
    memory.add_event("aila", "我的过去：妹妹被掳走")
    memory.add_event("aila", "这把剑我定价五百金币")
    fixed = memory.fixed_memories("aila")
    assert any("我是" in f for f in fixed)
    assert any("过去" in f for f in fixed)
    dynamic = memory.dynamic_memories("aila", "剑多少钱")
    assert any("剑" in f for f in dynamic)


# ---------------- 事实提炼 ----------------
def test_extract_price_only_when_asking():
    facts = extract_facts("aila", "这把剑多少钱？", "五百金币")
    assert any("定价" in f for f in facts)
    # 砍价不产生价格事实
    facts2 = extract_facts("aila", "太贵了，便宜点吧", "五百金币不讲价")
    assert not any("定价" in f for f in facts2)


def test_extract_name():
    facts = extract_facts("aila", "我叫林风，从北方来", "哦")
    assert any("林风" in f for f in facts)


def test_memory_entry_none_for_noise():
    assert memory_entry("aila", "今天天气不错", "嗯") is None


# ---------------- 记忆提示词 ----------------
def test_memory_prompt_conversion():
    assert fact_to_memory_line("aila", "aila把剑定价为五百金币") == "这把剑我定价五百金币"
    assert fact_to_memory_line("aila", "艾拉是流浪商人") == "我是流浪商人"
    assert fact_to_memory_line("aila", "aila提到卖东西的") is None


def test_build_memory_prompt():
    p = build_memory_prompt("aila", ["aila把剑定价为五百金币", "艾拉是流浪商人"])
    assert "这把剑我定价五百金币" in p
    assert "我是流浪商人" in p


# ---------------- 拼装（foremeber） ----------------
def test_prompt_build(characters, memory):
    from foremeber import ForeMeber
    fm = ForeMeber(memory=memory)
    fm.chars = characters
    memory.set_persona("aila", ["我的性格：警惕、爱钱"], ["玩家：你是谁？ → 艾拉：别靠太近。"])
    p = fm.build_prompt("aila", "这剑多少钱？", "集市摊位")
    assert "我的性格" in p["system"]
    assert "玩家对你说" in p["user"]
    assert "场景" in p["user"]


def test_dialogue_extraction():
    from foremeber import extract_dialogue
    assert extract_dialogue('（我抬起头）"五百金币。"') == "五百金币。"
    assert "铁匠" in extract_dialogue("我是这里的铁匠。")
