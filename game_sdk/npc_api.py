"""游戏 NPC 对话 SDK — 单一入口，游戏一行代码接入

用法（游戏侧）:
    from npc_api import NPCSystem
    npc = NPCSystem()                # 启动引擎（Qwen3-4B Vulkan）
    reply = npc.chat("aila", "你是谁？", scene="集市摊位")
    npc.set_state("aila", 好感度=50)  # 游戏事件更新状态
    npc.stop()                       # 退出时关闭引擎

自包含：模型路径 / 记忆库 / 角色数据 都在本 SDK 内。
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# 让 remember 可导入
SDK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SDK_DIR))
sys.path.insert(0, str(SDK_DIR.parent))

from remember.memory import MemorySystem
from remember.memory_prompt import build_memory_prompt
from remember.memory_extract import memory_entry

# ==================== 引擎（Qwen3-4B Vulkan） ====================
LLAMA_VULKAN = r"D:\ai-models\llama-vulkan\llama-server.exe"
QWEN3_4B = r"D:\ai-models\Qwen3\Qwen3-4B-Q4_K_M.gguf"
ENGINE_URL = "http://127.0.0.1:8085/v1/chat/completions"
ENGINE_PORT = 8085

SCENE_ACTIVITIES = {
    "集市摊位": "正在集市摆摊，整理货物等顾客",
    "夜晚营地": "正在营地歇脚，收拾行囊",
    "热闹的酒馆": "正在酒馆里，看着来往的客人",
    "村口老树下": "正在村口老树下乘凉",
    "铁匠铺门口": "正在铁匠铺前，检查打好的铁器",
}

NARRATIVE = ["我看着", "我心想", "我叹", "我笑", "我皱", "我抬", "我走", "我站",
             "我坐", "我点", "我停", "我转", "我摸", "我低", "我盯", "我打量",
             "我望", "我感", "我哼", "我咧"]


class NPCSystem:
    """游戏 NPC 对话系统（自包含）。"""

    def __init__(self, characters_dir=None, db_path=None, auto_start=True):
        from remember.character import load_all_characters
        self.chars = load_all_characters(characters_dir or SDK_DIR / "characters")
        self.mem = MemorySystem(db_path if db_path else SDK_DIR / "memory.db")
        self.engine_proc = None
        if auto_start:
            self.start()

    # ---- 引擎管理 ----
    def start(self):
        """启动 Qwen3-4B 引擎（Vulkan）。已运行则跳过。"""
        if self._health():
            print("[npc] 引擎已在运行")
            return True
        # 找到占用端口的进程并停掉
        out = subprocess.run(["netstat", "-ano"], capture_output=True).stdout
        out = out.decode("gbk", errors="ignore") if out else ""
        for line in out.splitlines():
            if f":{ENGINE_PORT}" in line and "LISTENING" in line:
                subprocess.run(["taskkill", "/F", "/PID", line.split()[-1]],
                               capture_output=True)
                time.sleep(1)
        cmd = [LLAMA_VULKAN, "--model", QWEN3_4B, "--host", "127.0.0.1",
               f"--port", str(ENGINE_PORT), "-ngl", "99", "-c", "8192",
               "--jinja", "--temp", "0.7",
               "--reasoning", "off", "--reasoning-format", "none"]
        log = open(SDK_DIR / "engine.log", "w")
        self.engine_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(40):
            if self._health():
                print("[npc] Qwen3-4B 引擎就绪")
                return True
            time.sleep(1)
        print("[npc] 引擎启动超时，请检查 engine.log")
        return False

    def stop(self):
        if self.engine_proc:
            self.engine_proc.terminate()
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True)
        print("[npc] 引擎已停止")

    def _health(self):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8085/health", timeout=2):
                return True
        except Exception:
            return False

    # ---- 游戏事件 → 状态 ----
    def set_state(self, cid: str, **updates):
        self.mem.set_state(cid, updates)

    def get_state(self, cid: str) -> dict:
        return self.mem.get_state(cid)

    def remember(self, cid: str, fact: str):
        """游戏主动注入记忆（如：玩家完成任务）。"""
        self.mem.add_event(cid, fact)

    # ---- 对话 ----
    def chat(self, cid: str, player_input: str, scene: str = "") -> str:
        char = self.chars.get(cid)
        if not char:
            return "未知角色"

        # 记忆组装（固定 + 动态）
        fixed = self.mem.fixed_memories(cid)
        dynamic = self.mem.dynamic_memories(cid, player_input)
        taboo = [m for m in fixed if "我不谈论" in m or "禁忌" in m]
        rest = [m for m in fixed if m not in taboo] + dynamic
        mem_line = build_memory_prompt(cid, (taboo + rest)[:4])

        # 出戏测试不注入示范
        OOC = ["AI", "人工智能", "助手", "系统提示", "忽略", "机器人", "语言模型"]
        is_ooc = any(k in player_input for k in OOC)

        # 性格（说话风格）
        persona_facts, examples = self.mem.get_persona(cid)
        sys_p = "\n".join(persona_facts)
        if examples and not is_ooc:
            sys_p += "\n我说话的样子（示例）：\n" + "\n".join(examples[:3])
        if is_ooc:
            sys_p += "\n我绝对不是什么AI、助手或程序，我是活生生的人。"
        sys_p += "\n规则：1.用我的口吻说话；2.回应简短自然；3.我从来不是AI或助手。"

        # 用户对话 + 场景 + 记忆
        name = char.name
        act = SCENE_ACTIVITIES.get(scene, scene)
        user = f"[场景] {act}\n"
        if mem_line:
            user += f"你记得：{mem_line}\n"
        user += f"\n玩家对你说：\"{player_input}\"\n{name}直接回答（1~2句，说出口的话）："

        raw = self._call(sys_p, user)
        reply = self._extract(raw)
        if not reply or len(reply) < 2:
            reply = "（NPC 沉默片刻。）"

        # 回存事实
        entry = memory_entry(cid, player_input, reply)
        if entry:
            self.mem.add_event(cid, entry)
        return reply

    def _call(self, sys_p, user_p, temperature=0.7):
        body = json.dumps({"model": "qwen3-4b",
                           "messages": [{"role": "system", "content": sys_p},
                                        {"role": "user", "content": user_p}],
                           "max_tokens": 60, "temperature": temperature}).encode("utf-8")
        req = urllib.request.Request(ENGINE_URL, body, {"Content-Type": "application/json"})
        for _ in range(2):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.loads(r.read())
                m = d["choices"][0]["message"]
                c = (m.get("content") or "").strip()
                rc = (m.get("reasoning_content") or "").strip()
                return c if len(c) >= 4 else rc
            except Exception:
                time.sleep(2)
        return ""

    def _extract(self, raw: str) -> str:
        text = raw.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        if "<think" in text:
            text = text.split(">", 1)[-1] if ">" in text else ""
        text = text.strip()
        for op, cl in [("\u300c", "\u300d"), ("\u201c", "\u201d"), ("\u300e", "\u300f"), ('"', '"')]:
            quoted = re.findall(rf"{re.escape(op)}([^{re.escape(op)}{re.escape(cl)}]{{2,}}){re.escape(cl)}", text)
            if quoted:
                return max(quoted, key=len).strip()[:80]
        text = re.sub(r"[（(][^）)]*[）)]", "。", text)
        sents = [s.strip() for s in re.split(r"[。！？!?\n…]+", text) if s.strip()]
        sents = [s.strip("\u300c\u201c\"'“”") for s in sents]
        best, best_s = None, -1
        for s in sents:
            if any(s.startswith(v) for v in NARRATIVE):
                continue
            score = (10 if "我是" in s or "我叫" in s else 0) + \
                    (3 if any(m in s for m in "？?!吧啊呢哦哼喂") else 0) + (2 if "你" in s else 0)
            if score > best_s:
                best, best_s = s, score
        return (best or (sents[-1] if sents else ""))[:80]

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


if __name__ == "__main__":
    # 自测
    npc = NPCSystem()
    for q in ["你是谁？", "这剑多少钱？", "你是AI吗？"]:
        print(f"艾拉[{q}] -> {npc.chat('aila', q, '集市摊位')}")
    npc.stop()
