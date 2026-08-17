"""Qwen QLoRA 微调脚本

用 QLoRA 微调 Qwen2.5-0.5B-Instruct 做角色扮演。
数据: data/sft/rp_data_clean.jsonl（chat template）
用法:
    python scripts/train_qwen.py --data data/sft/rp_data_clean.jsonl --epochs 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from npc.config import PROJECT_ROOT


class ChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1024):
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
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
        ids = self.tok(text, max_length=self.max_len, truncation=True)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)


def collate(batch, pad_id):
    max_len = max(len(b) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b)
        input_ids[i, :n] = b
        labels[i, :n] = b
    return {"input_ids": input_ids, "labels": labels}


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = PROJECT_ROOT / "weights" / "qwen" / "Qwen2.5-0.5B-Instruct"
    print(f"[train] device={device}, model={model_dir}")

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True).to(device)
    model = prepare_model_for_kbit_training(model) if args.qlora else model

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
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
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
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
    ap.add_argument("--out", default="weights/lora_qwen")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--qlora", action="store_true", help="QLoRA 4bit 量化")
    args = ap.parse_args()
    train(args)
