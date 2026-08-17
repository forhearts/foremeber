"""云端引擎：连接 Kaggle/Colab 部署的 7B 角色扮演 API

用法:
    python scripts/cloud_engine.py --url https://xxx.trycloudflare.com
    # 然后 WebUI 用 --engine cloud 接入
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_gradio_endpoint(base_url: str):
    """从 gradio 页面提取 /call 端点（gradio 4+）。"""
    # 尝试直接调 gradio api
    return base_url.rstrip("/")


def call_cloud(base_url: str, cid: str, player_input: str, scene: str, affection: int = 0) -> str:
    """调用云端 gradio API。"""
    from gradio_client import Client
    client = Client(base_url)
    result = client.predict(
        cid, player_input, scene, affection, api_name="/predict")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Kaggle cloudflared 公网 URL")
    ap.add_argument("--test", action="store_true", help="测试连接")
    args = ap.parse_args()

    if args.test:
        try:
            r = call_cloud(args.url, "aila", "你是谁？", "集市摊位", 0)
            print(f"✅ 云端连接成功! 艾拉: {r}")
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            sys.exit(1)
    else:
        print("用法: python scripts/cloud_engine.py --url <URL> --test")


if __name__ == "__main__":
    main()
