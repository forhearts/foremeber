"""Qwen 基座适配：用 Qwen2.5-0.5B-Instruct 替代 MiniMind 作为对话生成器。

保留系统的全部能力（记忆/角色/后处理/意图/状态），只替换模型。
对应规划 10 混合方案：关键 NPC 用稍大模型（0.5B 中文强）。
"""
from __future__ import annotations

import torch
from pathlib import Path

from npc.config import PROJECT_ROOT
from npc.engine import DialogueEngine


def load_qwen_engine(
    model_dir: str | Path | None = None,
    device: str = "auto",
    lora_dir: str | None = None,
):
    """加载 Qwen2.5-0.5B-Instruct + DialogueEngine。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "weights" / "qwen" / "Qwen2.5-0.5B-Instruct"
    if not model_dir.exists():
        raise FileNotFoundError(f"Qwen 模型不存在: {model_dir}\n请运行 python scripts/download_qwen.py")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[qwen] 加载 {model_dir} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    if lora_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        print(f"[qwen] LoRA 已加载: {lora_dir}")

    model.eval()
    engine = QwenChatEngine(model=model, tokenizer=tokenizer)
    return engine


class QwenChatEngine(DialogueEngine):
    """Qwen 对话引擎：重写 chat 用 chat template + transformers generate。"""

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

        # Qwen chat template
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if history:
            messages = [messages[0]] + [
                {"role": "user" if i % 2 == 0 else "assistant", "content": t}
                for i, (_, t) in enumerate(history[-6:])
            ] + [messages[1]]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gc = self.gen_config
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=gc.max_new_tokens,
                temperature=gc.temperature,
                top_p=gc.top_p,
                top_k=gc.top_k,
                repetition_penalty=gc.repetition_penalty,
                do_sample=True,
                pad_token_id=pad_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        reply = self.post.process(raw)
        self.memory.add_dialogue_turn(character_id, player_input, reply)
        return {"reply": reply, "intent": intent, "state": state,
                "prompt": text if verbose else "", "raw": raw}
