"""MiniMind-3 引擎：官方 chat template + LoRA 微调版

对应规划：用中文训练充分的小模型做角色扮演。
关键点：
- 官方 chat_template（open_thinking=False 加空 think 块，让模型直接回答）
- LoRA 微调后的角色扮演能力
- think 块剥离（防止 reasoning 泄漏）
"""
from __future__ import annotations

import re

import torch

from npc.config import PROJECT_ROOT
from npc.engine import DialogueEngine


def strip_think(raw: str) -> str:
    """剥离 <think>...</think> 块（含未闭合）。"""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    if "<think" in text:
        text = text.split(">", 1)[-1] if ">" in text else ""
    return text.strip()


class MiniMind3Engine(DialogueEngine):
    """MiniMind-3 对话引擎：官方 chat template + LoRA。"""

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

        sys_prompt = build_system_prompt(character)
        user_prompt = build_prompt(
            character=character, player_input=player_input, state=state,
            lorebook=self.lorebook, memories=memories, history=history, scene=scene,
        )

        # 官方 chat template
        messages = [{"role": "system", "content": sys_prompt}]
        if history:
            for speaker, text in history[-6:]:
                role = "user" if speaker == "玩家" else "assistant"
                messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt")
        dev = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(dev)

        gc = self.gen_config
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=gc.max_new_tokens,
                temperature=gc.temperature,
                top_p=gc.top_p,
                top_k=gc.top_k,
                repetition_penalty=gc.repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, input_ids.shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw = strip_think(raw)

        reply = self.post.process(raw)
        self.memory.add_dialogue_turn(character_id, player_input, reply)
        return {"reply": reply, "intent": intent, "state": state,
                "prompt": text if verbose else "", "raw": raw}


def load_minimind3_engine(device="auto", lora_dir=None, weight=None):
    """加载 MiniMind-3 + LoRA。"""
    from transformers import AutoTokenizer
    from npc.model import MiniMindLM

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = weight or str(PROJECT_ROOT / "weights" / "minimind3" / "pytorch" / "full_sft_768_moe.pth")

    print(f"[minimind3] 加载基座 {weight} (device={device})")
    model = MiniMindLM.from_official_checkpoint(weight, device=device)

    if lora_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        print(f"[minimind3] LoRA 已加载: {lora_dir}")

    model.eval()
    tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "weights" / "minimind3"),
                                        trust_remote_code=True)
    return MiniMind3Engine(model=model, tokenizer=tok)
