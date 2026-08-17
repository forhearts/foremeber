"""SFT 微调脚本（LoRA）

对应规划第 4 章：让 MiniMind 学会"角色扮演格式 + 短回复 + 不出戏"。
用 peft 对自实现 MiniMindLM 做 LoRA 微调，消费级显卡可跑。

用法:
    python scripts/train_sft.py --data data/sft/template_rp.jsonl --epochs 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from npc.config import LLM_WEIGHT, SFT_WEIGHT, LORA_DIR
from npc.model import MiniMindLM
from npc.tokenizer_utils import load_tokenizer


class ChatDataset(Dataset):
    """从 jsonl（chat template）构造指令数据。"""

    def __init__(self, path: str | Path, tokenizer, max_len: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
        print(f"[data] 加载 {len(self.samples)} 条")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        msgs = self.samples[idx]["messages"]
        # 简单拼接：system + user -> assistant
        sys_c = next(m["content"] for m in msgs if m["role"] == "system")
        user_c = next(m["content"] for m in msgs if m["role"] == "user")
        asst_c = next(m["content"] for m in msgs if m["role"] == "assistant")

        prompt = f"{sys_c}\n{user_c}\n{asst_c}"
        full_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        # 截断到 max_len
        full_ids = full_ids[: self.max_len]
        input_len = len(self.tokenizer.encode(f"{sys_c}\n{user_c}\n", add_special_tokens=False))
        labels = [-100] * input_len + full_ids[input_len:]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_id: int = 0):
    """pad 到 batch 内最大长度。"""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        mask[i, :n] = True
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    tok = load_tokenizer()
    model = MiniMindLM.from_official_checkpoint(args.weight, device=device)
    model.train()
    print(f"[train] 模型参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # LoRA 包装（peft）—— 只训 attention，冻结 MoE experts（防过拟合+省显存）
    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    ds = ChatDataset(args.data, tok, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # 只训练 lora 参数（若用 lora）
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(input_ids)
            # cross entropy with labels
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item()
            global_step += 1
            if (step + 1) % args.log_every == 0:
                print(f"  epoch {epoch+1}/{args.epochs} step {step+1}/{len(loader)} "
                      f"loss={loss.item():.4f}")
        print(f"  [epoch {epoch+1}] avg_loss={total_loss/max(len(loader),1):.4f}")

    # 保存
    if args.lora:
        save_dir = Path(args.lora_out)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        tok.save_pretrained(str(save_dir))
        print(f"[save] LoRA 已保存 -> {save_dir}")
    else:
        torch.save({"model": model.state_dict()}, args.out)
        print(f"[save] 全量模型已保存 -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft/template_rp.jsonl")
    ap.add_argument("--weight", default=str(LLM_WEIGHT))
    ap.add_argument("--out", default=str(SFT_WEIGHT))
    ap.add_argument("--lora", action="store_true", help="使用 LoRA 微调")
    ap.add_argument("--lora_out", default=str(LORA_DIR))
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_modules", nargs="*", default=["q_proj", "k_proj", "v_proj", "o_proj"],
                    help="LoRA 目标模块（默认只训 attention）")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()
    train(args)
