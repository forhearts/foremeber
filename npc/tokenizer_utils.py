"""MiniMind tokenizer 加载（使用官方 tokenizer.json，纯文本模式）"""
from pathlib import Path
from transformers import PreTrainedTokenizerFast

from npc.config import MODEL_DIR


def load_tokenizer(path: Path = None):
    """加载官方 tokenizer（fast）。缺失时给出明确报错。"""
    path = path or (MODEL_DIR / "tokenizer.json")
    if not path.exists():
        raise FileNotFoundError(
            f"tokenizer.json 未找到: {path}\n"
            "请先运行: python scripts/download_weights.py"
        )
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=32768,
    )
    return tok


if __name__ == "__main__":
    t = load_tokenizer()
    print("vocab:", t.vocab_size)
    ids = t.encode("你好，你是谁？")
    print("ids:", ids)
    print("decode:", t.decode(ids))
