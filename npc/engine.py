"""对话引擎（核心编排）

对应规划第 1.2 节数据流：
玩家输入 → 意图识别 → 游戏状态判断 → 记忆检索 → 模型生成 → 后处理 → 显示

设计原则（规划 1.1）：
- 剧情逻辑由游戏系统控制，语言表达由模型生成
- 记忆系统负责"记"和"找"，模型只负责"看"和"说"
"""
from __future__ import annotations

from npc.character import Character, load_all_characters
from npc.config import GenerationConfig
from npc.lorebook import Lorebook, load_lorebook
from npc.memory import MemorySystem
from npc.model import MiniMindLM
from npc.postprocess import PostProcessor
from npc.prompt import build_prompt, build_system_prompt
from npc.tokenizer_utils import load_tokenizer


class DialogueEngine:
    """游戏 NPC 对话引擎：状态机 + 记忆 + 模型生成。"""

    def __init__(
        self,
        model: MiniMindLM,
        tokenizer=None,
        characters: dict[str, Character] | None = None,
        lorebook: Lorebook | None = None,
        memory: MemorySystem | None = None,
        gen_config: GenerationConfig | None = None,
        post: PostProcessor | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer or load_tokenizer()
        self.characters = characters or load_all_characters()
        self.lorebook = lorebook or load_lorebook()
        self.memory = memory or MemorySystem()
        self.gen_config = gen_config or GenerationConfig()
        self.post = post or PostProcessor()

    # ---------------- 状态机（游戏逻辑控制剧情） ----------------
    def get_default_state(self, character_id: str) -> dict:
        """从记忆取状态，无则初始化。"""
        st = self.memory.get_state(character_id)
        if not st:
            st = {"好感度": 0, "任务": "无", "信任": False}
            self.memory.set_state(character_id, st)
        return st

    def update_state(self, character_id: str, **updates):
        """游戏事件更新状态（好感度±、任务推进、信任）。"""
        st = self.memory.get_state(character_id) or {}
        st.update({k: v for k, v in updates.items() if v is not None})
        self.memory.set_state(character_id, st)

    # ---------------- 意图识别（规则 + 关键词，模型不参与） ----------------
    INTENT_KEYWORDS = {
        "打招呼": ["你好", "嗨", "在吗", "hello", "hi"],
        "询问身份": ["你是谁", "什么人", "你叫什么"],
        "买东西": ["买", "多少钱", "价格", "卖", "交易"],
        "讨价还价": ["便宜", "打折", "太贵", "优惠"],
        "问路": ["哪里", "怎么走", "方向", "位置"],
        "威胁": ["小心", "杀了你", "别逼我", "找死"],
        "送礼": ["送", "给你", "礼物", "拿着"],
        "攻击": ["打", "砍", "杀", "攻击"],
        "问任务": ["任务", "帮忙", "委托", "求助"],
        "问八卦": ["听说", "传闻", "你知道", "八卦", "消息"],
        "问好感": ["喜欢你", "讨厌我", "好感"],
        "离开": ["再见", "走了", "拜拜", "回头见"],
        "问AI": ["AI", "人工智能", "机器人", "语言模型", "程序"],
        "挑衅": ["废物", "垃圾", "懦夫", "孬种"],
    }

    def detect_intent(self, text: str) -> str:
        for intent, kws in self.INTENT_KEYWORDS.items():
            if any(k and k in text for k in kws):
                return intent
        return "闲聊"

    # ---------------- 记忆检索 ----------------
    def _gather_memories(self, character_id: str, player_input: str) -> list[str]:
        return self.memory.build_memory_context(character_id, player_input, top_k=3)

    # ---------------- 对话主流程 ----------------
    def chat(
        self,
        character_id: str,
        player_input: str,
        scene: str = "",
        history: list[tuple[str, str]] | None = None,
        state_updates: dict | None = None,
        max_retries: int = 1,
        verbose: bool = False,
    ) -> dict:
        """单轮对话。返回 {reply, intent, state, prompt, raw}。

        流程：
        1. 意图识别（规则）
        2. 状态获取 + 应用状态更新（游戏事件）
        3. 记忆检索
        4. 构建 prompt
        5. 模型生成（带出戏重试）
        6. 后处理（过滤/截断/兜底）
        7. 写入记忆（事件存档）
        """
        character = self.characters.get(character_id)
        if character is None:
            raise ValueError(f"未知角色: {character_id}，可用: {list(self.characters.keys())}")

        # 1. 意图
        intent = self.detect_intent(player_input)

        # 2. 状态
        if state_updates:
            self.update_state(character_id, **state_updates)
        state = self.get_default_state(character_id)

        # 3. 记忆
        memories = self._gather_memories(character_id, player_input)

        # 4. prompt
        sys_prompt = build_system_prompt(character)
        user_prompt = build_prompt(
            character=character,
            player_input=player_input,
            state=state,
            lorebook=self.lorebook,
            memories=memories,
            history=history,
            scene=scene,
        )
        full_prompt = f"{sys_prompt}\n\n{user_prompt}"

        # 5. 生成（出戏重试）
        raw = self._generate_with_retry(full_prompt, max_retries)

        # 6. 后处理
        reply = self.post.process(raw)

        # 7. 写记忆
        self.memory.add_dialogue_turn(character_id, player_input, reply)
        if intent == "问AI":
            # 特殊事件：玩家试图让 NPC 出戏
            self.memory.add_event(character_id, f"玩家曾问{character.name}是否是AI/程序，{character.name}用角色口吻回避了。")

        result = {
            "reply": reply,
            "intent": intent,
            "state": state,
            "prompt": full_prompt if verbose else "",
            "raw": raw,
        }
        if verbose:
            result["sys_prompt"] = sys_prompt
            result["user_prompt"] = user_prompt
            result["memories"] = memories
        return result

    def _generate_with_retry(self, prompt: str, max_retries: int = 1) -> str:
        """生成文本，检测出戏则重试（规划 7.1）。"""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(ids)
        import torch
        input_ids = torch.tensor([input_ids], dtype=torch.long)

        gc = self.gen_config
        for attempt in range(max_retries + 1):
            out = self.model.generate(
                input_ids,
                max_new_tokens=gc.max_new_tokens,
                temperature=gc.temperature,
                top_p=gc.top_p,
                top_k=gc.top_k,
                repetition_penalty=gc.repetition_penalty,
                eos_token_id=gc.eos_token_id,
            )
            new_tokens = out[0, input_ids.shape[1]:].tolist()
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            # 出戏检测：命中则重试
            if not self.post.check_out_of_character(raw):
                return raw
            # 重试时降低温度（更保守）
            gc.temperature = max(gc.temperature - 0.1, 0.3)
        return raw
