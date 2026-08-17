"""MiniMind-3 LoRA 微调脚本（用官方 chat template + 空 think 块）

基座: minimind-3 full_sft（中文能力强）
数据: rp_data_clean.jsonl（角色扮演 chat template）
用法:
    python scripts/train_minimind3.py --data data/sft/rp_data_clean.jsonl --epochs 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader

from npc.model import MiniMindLM
from npc.config import PROJECT_ROOT


class ChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=768):
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len
        print(f"[data] {len(self.items)} 条")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        msgs = self.items[idx]["messages"]
        # 训练输入：完整对话（system+user+assistant）
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, open_thinking=False)
        ids = self.tok(text, max_length=self.max_len, truncation=True)["input_ids"]
        # labels: 只对 assistant 部分计算 loss（从 <|im_start|>assistant 标记开始）
        marker = "<|im_start|>assistant\n"
        marker_ids = self.tok(marker, add_special_tokens=False)["input_ids"]
        # 找最后一个 assistant 标记位置
        start = -1
        for i in range(len(ids) - len(marker_ids) + 1):
            if ids[i:i+len(marker_ids)] == marker_ids:
                start = i
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "label_start": start}


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids = b["input_ids"]
        input_ids[i, :n] = ids
        if b["label_start"] >= 0:
            labels[i, b["label_start"]:n] = ids[b["label_start"]:]
    return {"input_ids": input_ids, "labels": labels}


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}, 基座: {args.weight}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "weights" / "minimind3"),
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = MiniMindLM.from_official_checkpoint(args.weight, device=device)
    model.train()
    print(f"[train] 基座参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = ChatDataset(args.data, tok, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        total = 0.0
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # batch=2 长度接近，无需 attention_mask（避免 SDPA mask 冲突）
            logits, _ = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            total += loss.item()
            if (step + 1) % args.log_every == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(loader)} loss={loss.item():.4f}")
        print(f"  [epoch {epoch+1}] avg_loss={total/max(len(loader),1):.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[save] -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft/rp_data_clean.jsonl")
    ap.add_argument("--weight", default=str(PROJECT_ROOT / "weights" / "minimind3" / "pytorch" / "full_sft_768_moe.pth"))
    ap.add_argument("--out", default="weights/lora_minimind3")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()
    train(args)
