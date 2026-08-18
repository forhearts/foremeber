"""引擎启动/切换管理

支持：
- qwen3-4b   (Qwen3-4B-Q4_K_M, 推荐, 4.5GB 显存)
- qwen3-1.7b (Qwen3-1.7B-Q8_0, 甜点位, 3GB 显存)
- vanilla    (14B 角色扮演, 原引擎, 4GB 显存)

用法:
    python engine_server.py start qwen3-4b     # 启动 Qwen3-4B (端口8085)
    python engine_server.py stop                # 停止所有引擎服务
    python engine_server.py status              # 查看状态
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LLAMA = r"D:\ai-models\llama\llama-server.exe"
MODELS = {
    "qwen3-4b": {
        "path": r"D:\ai-models\Qwen3\Qwen3-4B-Q4_K_M.gguf",
        "port": 8085, "c": 8192,
        "args": ["--reasoning", "off", "--reasoning-format", "none"],
    },
    "qwen3-1.7b": {
        "path": r"D:\ai-models\Qwen3\Qwen3-1.7B-Q8_0.gguf",
        "port": 8085, "c": 8192,
        "args": ["--reasoning", "off", "--reasoning-format", "none"],
    },
    "vanilla": {
        "path": r"D:\ai-models\vanilla-cn-roleplay-0.2\vanilla-cn-roleplay-0.2.i1-IQ3_S.gguf",
        "port": 8081, "c": 16384,
        "args": [],
    },
}


def find_port_pids(port):
    """找到占用端口的进程 PID。"""
    pids = set()
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            pids.add(line.split()[-1])
    return pids


def start(name):
    cfg = MODELS.get(name)
    if not cfg:
        print(f"未知引擎: {name}, 可选: {list(MODELS.keys())}")
        return
    if not Path(cfg["path"]).exists():
        print(f"模型不存在: {cfg['path']}")
        return
    # 停掉同端口旧服务
    for pid in find_port_pids(cfg["port"]):
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        print(f"已停止旧进程 {pid}")
    time.sleep(1)
    cmd = [LLAMA, "--model", cfg["path"], "--host", "127.0.0.1",
           f"--port", str(cfg["port"]), "-ngl", "99", "-c", str(cfg["c"]),
           "--jinja", "--temp", "0.7"] + cfg["args"]
    log = Path(f"D:/ai-models/llama/engine-{name}.log")
    log.parent.mkdir(exist_ok=True)
    with open(log, "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"启动 {name} -> 端口 {cfg['port']} (日志: {log})")
    # 等待健康
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{cfg['port']}/health", timeout=2):
                print(f"[OK] {name} 就绪 (http://127.0.0.1:{cfg['port']})")
                return
        except Exception:
            time.sleep(2)
    print("[WARN] 启动超时，检查日志")


def stop():
    import subprocess
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)
    print("已停止所有 llama-server 服务")


def status():
    for name, cfg in MODELS.items():
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{cfg['port']}/health", timeout=2):
                print(f"[OK] {name} 运行中 (端口 {cfg['port']})")
        except Exception:
            print(f"[X] {name} 未运行 (端口 {cfg['port']})")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "start":
        start(sys.argv[2] if len(sys.argv) > 2 else "qwen3-4b")
    elif action == "stop":
        stop()
    else:
        status()
