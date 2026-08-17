"""云端引擎：连接 Kaggle/Colab 部署的 7B 角色扮演 API

复用本机全部系统（角色卡/记忆/意图/后处理），只把生成交给云端 7B 模型。
"""
from __future__ import annotations

from npc.character import load_all_characters
from npc.config import GenerationConfig
from npc.engine import DialogueEngine
from npc.lorebook import load_lorebook
from npc.memory import MemorySystem
from npc.postprocess import PostProcessor


class CloudEngine(DialogueEngine):
    """云端 7B 角色扮演引擎。"""

    def __init__(self, base_url: str, characters=None, lorebook=None,
                 memory=None, gen_config=None, post=None):
        self.base_url = base_url.rstrip("/")
        self.model = None
        self.tokenizer = None
        self.characters = characters if characters is not None else load_all_characters()
        self.lorebook = lorebook if lorebook is not None else load_lorebook()
        self.memory = memory if memory is not None else MemorySystem()
        self.gen_config = gen_config if gen_config is not None else GenerationConfig()
        self.post = post if post is not None else PostProcessor()

    def chat(self, character_id, player_input, scene="", history=None,
             state_updates=None, max_retries=1, verbose=False):
        from npc.prompt import build_prompt, build_system_prompt
        character = self.characters.get(character_id)
        if character is None:
            raise ValueError(f"未知角色: {character_id}")

        intent = self.detect_intent(player_input)
        if state_updates:
            self.update_state(character_id, **state_updates)
        state = self.get_default_state(character_id)
        memories = self._gather_memories(character_id, player_input)

        # 云端 notebook 的 chat_api(cid, player_input, scene, affection) 期望纯玩家话
        # 角色/状态由云端 CHARACTERS 处理，本机只传玩家话和场景
        affection = state.get("好感度", 0)
        reply = self._call_cloud(character_id, player_input, scene, affection)
        reply = self.post.process(reply)
        self.memory.add_dialogue_turn(character_id, player_input, reply)
        return {"reply": reply, "intent": intent, "state": state,
                "prompt": player_input if verbose else "", "raw": reply}

    def _call_cloud(self, cid, player_input, scene, affection):
        """调用云端 gradio API（动态获取 endpoint 名）。"""
        from gradio_client import Client
        client = Client(self.base_url)
        try:
            # gradio 6 自动命名函数端点，取第一个可用
            api = client.view_api(return_format="dict")
            endpoints = list(api.get("named_endpoints", {}).keys())
            api_name = endpoints[0] if endpoints else "/predict"
            result = client.predict(cid, player_input, scene, affection, api_name=api_name)
            return str(result)
        except Exception as e:
            return f"（云端连接失败: {e}）"


def load_cloud_engine(url: str):
    """加载云端引擎。"""
    if not url:
        raise ValueError("--cloud_url 未指定！请先跑 kaggle_npc_api.ipynb 获取 URL")
    print(f"[cloud] 连接 {url}")
    return CloudEngine(url)
