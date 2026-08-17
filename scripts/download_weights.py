"""下载 MiniMind 权重（ModelScope，中文网络可用）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("需要 modelscope: pip install modelscope")
        raise

    target = Path(__file__).resolve().parent.parent / "weights" / "minimind-3o-pytorch"
    target.mkdir(parents=True, exist_ok=True)
    print(f"下载 gongjy/minimind-3o-pytorch -> {target}")
    path = snapshot_download("gongjy/minimind-3o-pytorch", local_dir=str(target))
    print(f"完成: {path}")
    for f in sorted(target.glob("*.pth")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f}MB")


if __name__ == "__main__":
    main()
