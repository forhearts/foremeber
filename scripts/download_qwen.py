"""下载 Qwen2.5-0.5B-Instruct（ModelScope）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from modelscope import snapshot_download
    target = Path(__file__).resolve().parent.parent / "weights" / "qwen" / "Qwen2.5-0.5B-Instruct"
    target.mkdir(parents=True, exist_ok=True)
    print(f"下载 Qwen/Qwen2.5-0.5B-Instruct -> {target}")
    p = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir=str(target))
    print(f"完成: {p}")


if __name__ == "__main__":
    main()
