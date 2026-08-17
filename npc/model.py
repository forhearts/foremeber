"""MiniMind 推理引擎（自包含，纯 PyTorch 实现）

对应规划 5.3 节：官方代码为 PyTorch 原生，结构简单，适合直接移植。
本模块实现与 minimind-3o-moe 官方一致的：
- GQA 注意力（8 头 / KV 4 头）
- RoPE 旋转位置编码
- MoE FFN（4 专家 top-1）
- RMSNorm
支持从官方 .pth 权重加载，纯文本推理（不加载视觉/语音旁路）。

轻量自包含实现，不依赖 transformers 模型类，便于后续移植 ONNX/llama.cpp。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- 基础组件（与官方 model_minimind.py 一致） ----------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self._norm(x.float())).type_as(x)


def precompute_freqs_cis(dim: int, end: int = 32768, rope_base: float = 1e6):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    # 与官方一致：cat 两次使频率维度 = head_dim（rotate_half 需要）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, hidden_size, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        if past_kv is not None:
            pk, pv = past_kv
            xk = torch.cat([pk, xk], dim=1)
            xv = torch.cat([pv, xv], dim=1)
        past_kv_new = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk.transpose(1, 2), self.n_rep)
        xv = repeat_kv(xv.transpose(1, 2), self.n_rep)

        if attention_mask is not None:
            # additive mask: (bsz, 1, 1, seq) 0=attend, -inf=ignore
            attn_mask = (attention_mask[:, None, None, :] == 0).to(xq.dtype) * -1e9
            out = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, is_causal=(past_kv is None))
        else:
            out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=(past_kv is None))
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out), past_kv_new


def repeat_kv(x, n_rep):
    """x: (bsz, n_kv_heads, seq, head_dim) -> (bsz, n_kv_heads*n_rep, seq, head_dim)"""
    if n_rep == 1:
        return x
    bsz, n_kv, slen, d = x.shape
    return x[:, :, None, :, :].expand(bsz, n_kv, n_rep, slen, d).reshape(bsz, n_kv * n_rep, slen, d)


class FFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """4 专家 top-1 MoE（与官方一致）。"""

    def __init__(self, hidden_size, intermediate_size, num_experts=4):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            FFN(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=1, dim=-1)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)              # (n, 1)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero(as_tuple=True)[0]  # 选中 token
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
        return y.view(bsz, seq_len, dim)


class Block(nn.Module):
    def __init__(self, hidden_size, n_heads, n_kv_heads, head_dim,
                 intermediate_size, use_moe, num_experts):
        super().__init__()
        self.self_attn = Attention(hidden_size, n_heads, n_kv_heads, head_dim)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = MoEFFN(hidden_size, intermediate_size, num_experts) if use_moe \
            else FFN(hidden_size, intermediate_size)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, attention_mask=None):
        h, past_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv, use_cache, attention_mask)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, past_kv


class MiniMindLM(nn.Module):
    """MiniMind 语言模型（Thinker）。纯文本，支持 KV cache。"""

    def __init__(
        self,
        hidden_size=768,
        num_layers=8,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=6400,
        intermediate_size=2432,
        max_pos=32768,
        rope_theta=1e6,
        use_moe=True,
        num_experts=4,
        eps=1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads
        self.vocab_size = vocab_size
        self.use_moe = use_moe

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Block(hidden_size, n_heads, n_kv_heads, self.head_dim,
                  intermediate_size, use_moe, num_experts)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # tie embeddings
        self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_freqs_cis(self.head_dim, max_pos, rope_theta)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, input_ids, past_key_values=None, use_cache=False, attention_mask=None):
        """input_ids: (bsz, seq)。返回 (logits, past_key_values)。"""
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        cos = self.freqs_cos[start_pos:start_pos + seq_len]
        sin = self.freqs_sin[start_pos:start_pos + seq_len]
        new_past = []
        for i, layer in enumerate(self.layers):
            x, pkv = layer(x, cos, sin, past_key_values[i], use_cache, attention_mask)
            new_past.append(pkv)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_past

    # ---------------- 权重加载 ----------------
    @classmethod
    def from_official_checkpoint(cls, ckpt_path: str | Path, device="cpu") -> "MiniMindLM":
        """从官方 minimind-3o pth 加载权重。

        兼容:
        - llm_768.pth / llm_768_moe.pth (纯语言基座, 键名 model.*)
        - sft_omni_768*.pth (完整 Omni, 键名含 talker.*, 只取 thinker/model.*)
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # 兼容 dict 包装
        if isinstance(sd, dict) and "model" in sd and hasattr(sd["model"], "keys"):
            state = sd["model"]
        elif isinstance(sd, dict):
            # 扁平键: model.*
            state = {k: v for k, v in sd.items() if k.startswith("model.")}
            if not state:
                raise ValueError(f"无法识别的权重格式: {list(sd.keys())[:10]}")
        else:
            raise ValueError(f"无法识别的权重类型: {type(sd)}")

        # 从权重推断配置
        embed_w = state.get("model.embed_tokens.weight")
        if embed_w is None:
            embed_w = state.get("embed_tokens.weight")
        vocab_size, hidden_size = embed_w.shape
        n_layers = max(int(k.split(".")[2]) for k in state if ".layers." in k) + 1

        q_w = next(v for k, v in state.items() if k.endswith("self_attn.q_proj.weight"))
        n_heads = q_w.shape[0] // (hidden_size // 8)
        kv_w = next(v for k, v in state.items() if k.endswith("self_attn.k_proj.weight"))
        n_kv_heads = kv_w.shape[0] // (hidden_size // 8)

        # MoE 检测
        use_moe = any("mlp.experts." in k for k in state)
        num_experts = 4
        if use_moe:
            import re
            exps = set()
            for k in state:
                m = re.search(r"mlp\.experts\.(\d+)", k)
                if m:
                    exps.add(int(m.group(1)))
            num_experts = len(exps) if exps else 4

        # intermediate
        if use_moe:
            gw = next(v for k, v in state.items()
                      if "mlp.experts.0.gate_proj.weight" in k)
        else:
            gw = next(v for k, v in state.items()
                      if "mlp.gate_proj.weight" in k)
        intermediate_size = gw.shape[0]

        model = cls(
            hidden_size=hidden_size,
            num_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            use_moe=use_moe,
            num_experts=num_experts,
        )

        # 键名映射：官方前缀 model.layers.X.self_attn.* 与本地一致
        mapping = {}
        prefix = "model."
        for k, v in state.items():
            local_k = k[len(prefix):] if k.startswith(prefix) else k
            mapping[local_k] = v
        # lm_head tied -> 不单独加载
        missing, unexpected = model.load_state_dict(mapping, strict=False)
        if missing:
            # 允许缺失 lm_head.weight (tied)
            real_missing = [m for m in missing if m != "lm_head.weight"]
            if real_missing:
                print(f"[warn] 缺失权重: {real_missing[:10]}")
        return model.to(device)

    # ---------------- 生成 ----------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 96,
        temperature: float = 0.85,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """自回归生成。input_ids: (1, seq)。返回完整序列 (1, seq+new)。"""
        self.eval()
        input_ids = input_ids.to(self.embed_tokens.weight.device)
        generated = input_ids
        past_kv = None
        for _ in range(max_new_tokens):
            if past_kv is None:
                logits, past_kv = self.forward(generated, use_cache=True)
            else:
                logits, past_kv = self.forward(generated[:, -1:], past_kv, use_cache=True)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for i in range(logits.shape[0]):
                    seen = torch.unique(generated[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k)[0][..., -1, None]
                logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                # 标准 top-p: 累积概率超过 top_p 的尾部置 -inf
                remove_mask = cumsum - sorted_probs > top_p
                remove_mask[..., 0] = False
                remove_mask = remove_mask.scatter(-1, sorted_indices, remove_mask)
                logits = torch.where(remove_mask,
                                     torch.full_like(logits, -float("inf")), logits)
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any():
                # 兜底：全部被屏蔽时退化为均匀
                probs = torch.ones_like(probs) / probs.size(-1)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == eos_token_id:
                break
        return generated

def load_model(weight: str, device: str = "cpu", lora_dir: str | None = None):
    """加载模型（可选加载 peft LoRA adapter）。"""
    model = MiniMindLM.from_official_checkpoint(weight, device=device)
    if lora_dir and Path(lora_dir).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        print(f"[lora] 已加载: {lora_dir}")
    return model
