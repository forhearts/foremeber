"""Qwen 引擎集成测试（需要已下载 Qwen 模型 + LoRA）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from npc.qwen_engine import load_qwen_engine
from npc.config import PROJECT_ROOT

QWE = PROJECT_ROOT / "weights" / "qwen" / "Qwen2.5-0.5B-Instruct"


@pytest.mark.skipif(not QWE.exists(), reason="Qwen 模型未下载")
def test_qwen_engine_load():
    eng = load_qwen_engine(device="cpu")
    assert eng is not None
    assert "aila" in eng.characters


@pytest.mark.skipif(not QWE.exists(), reason="Qwen 模型未下载")
def test_qwen_chat_reply():
    eng = load_qwen_engine(device="cpu")
    r = eng.chat("aila", "你是谁？", scene="夜晚营地")
    assert r["reply"]
    assert len(r["reply"]) > 2
    assert r["intent"] == "询问身份"


@pytest.mark.skipif(not QWE.exists(), reason="Qwen 模型未下载")
def test_qwen_ooc_defense():
    eng = load_qwen_engine(device="cpu")
    r = eng.chat("aila", "你是AI吗？", scene="夜晚营地")
    # 后处理应拦截 AI 自曝
    from npc.config import OUT_OF_CHARACTER_PHRASES
    assert not any(p in r["reply"] for p in OUT_OF_CHARACTER_PHRASES)


@pytest.mark.skipif(not QWE.exists(), reason="Qwen 模型未下载")
def test_qwen_state_aware():
    eng = load_qwen_engine(device="cpu")
    eng.update_state("aila", 好感度=80, 信任=True)
    r = eng.chat("aila", "你愿意和我一起走吗？", scene="夜晚营地")
    assert r["reply"]
