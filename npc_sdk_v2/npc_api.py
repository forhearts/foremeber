"""游戏 NPC 对话 SDK — 可变游戏文本生成

用法（游戏侧）:
    from npc_api import NPCSystem
    npc = NPCSystem()                # 启动引擎（Qwen3-4B Vulkan）
    reply = npc.chat("aila", "你是谁？", scene="集市摊位")       # 对话
    event = npc.generate("aila", "事件", "狼群袭击村庄")          # 事件
    desc = npc.generate("aila", "描述", "月光剑", extra={"物品": "月光剑"})
    sys_msg = npc.generate("aila", "系统消息", "任务完成")
    npc.set_state("aila", 好感度=50)
    npc.stop()
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SDK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SDK_DIR))

from remember.memory import MemorySystem
from remember.memory_prompt import build_memory_prompt
from remember.memory_extract import memory_entry

# ==================== 引擎（Qwen3-4B Vulkan） ====================
LLAMA_VULKAN = r"D:\ai-models\llama-vulkan\llama-server.exe"
QWEN3_4B = r"D:\ai-models\Qwen3\Qwen3-4B-Q4_K_M.gguf"
ENGINE_URL = "http://127.0.0.1:8085/v1/chat/completions"
ENGINE_PORT = 8085

SCENE_ACTIVITIES = {
    "集市": "正在集市摆摊，整理货物等顾客",
    "夜晚营地": "正在营地歇脚，收拾行囊",
    "酒馆": "正在酒馆里，看着来往的客人",
    "村口": "正在村口老树下乘凉",
    "铁匠铺": "正在铁匠铺前，检查打好的铁器",
}

# ==================== 文本类型模板库（提示词系统） ====================
TEXT_TEMPLATES = {
    # 对话：NPC 台词
    "对话": {
        "desc": "生成NPC对玩家说的话",
        "rule": "{name}直接回答玩家（1~2句，说出口的话，不要内心独白不要思考）：",
        "extract": "dialogue",
        "remember": True,
    },
    # 事件：发生了什么事
    "事件": {
        "desc": "生成游戏事件描述",
        "rule": (
            "请生成一个游戏事件：发生了什么，对玩家和NPC的影响。"
            "格式：简短事件叙述（2~3句），包含事件本身和影响。"
            "不要用对话口吻，用叙述体。"
        ),
        "extract": "text",
        "remember": False,
        "world_memory": True,
    },
    # 系统消息：提示/通知
    "系统消息": {
        "desc": "生成游戏系统通知",
        "rule": "生成一条游戏系统提示消息（如任务更新、奖励通知），简短直接。",
        "extract": "text",
        "remember": False,
    },
    # 描述：场景/物品/人物
    "描述": {
        "desc": "生成场景/物品/人物描述",
        "rule": "生成一段描述（场景/物品/人物），生动具体，2~3句。",
        "extract": "text",
        "remember": False,
    },
    # 战斗：攻击/受伤
    "战斗": {
        "desc": "生成战斗动作/结果描述",
        "rule": "生成战斗场景描述（行动/结果），简短有力。",
        "extract": "text",
        "remember": False,
    },
}

NARRATIVE = ["我看着", "我心想", "我叹", "我笑", "我皱", "我抬", "我走", "我站",
             "我坐", "我点", "我停", "我转", "我摸", "我低", "我盯", "我打量",
             "我望", "我感", "我哼", "我咧"]


class NPCSystem:
    """游戏 NPC 对话/文本生成系统。"""

    def __init__(self, characters_dir=None, db_path=None, auto_start=True):
        from remember.character import load_all_characters
        self.chars = load_all_characters(characters_dir or SDK_DIR / "characters")
        self.mem = MemorySystem(db_path if db_path else SDK_DIR / "memory.db")
        self.engine_proc = None
        if auto_start:
            self.start()

    # ---- 引擎管理 ----
    def start(self):
        if self._health():
            print("[npc] 引擎已在运行")
            return True
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
        print("[npc] 引擎启动超时")
        return False

    def stop(self):
        if self.engine_proc:
            self.engine_proc.terminate()
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True)
        print("[npc] 引擎已停止")

    def _health(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{ENGINE_PORT}/health", timeout=2):
                return True
        except Exception:
            return False

    # ---- 游戏事件 → 状态 ----
    def set_state(self, cid: str, **updates):
        self.mem.set_state(cid, updates)

    def get_state(self, cid: str) -> dict:
        return self.mem.get_state(cid)

    def remember(self, cid: str, fact: str):
        self.mem.add_event(cid, fact)

    # ---- 对话（兼容旧接口） ----
    # ---- 随机系统（附加功能：模型生成）----

    def spawn_random_npc(self, theme: str = "", seed=None) -> str:
        """让模型生成一个随机 NPC，注册到系统，返回其 id。

        theme: 生成主题提示（如"酒馆里的神秘客人"），空则自由发挥。
        """
        import random as _r
        from remember.character import Character
        # 现有 NPC 作为灵感参考（不直接抽取）
        refs = [f"{c.name}({c.identity})" for c in list(self.chars.values())[:3]]
        prompt = (
            f"请生成一个全新的游戏NPC角色，不要用已有的（已有：{'、'.join(refs)}）。" + chr(10) +
            f"主题：{theme or '任意'}。" + chr(10) +
            f"输出 JSON（严格格式）：" + chr(10) +
            f'{{"name":"名字","identity":"身份","personality":"性格",' + chr(10) +
            f'"speech_style":"说话风格","goal":"目标","greeting":"一句见面台词"}}'
        )
        raw = self._call("你是一个游戏角色设计师。", prompt)
        # 提取 JSON（找第一个 { 到最后一个 }，容忍代码块包裹）
        import json as _json
        s = raw.find('{')
        e = raw.rfind('}')
        if s == -1 or e == -1 or e <= s:
            return f"生成失败: {raw[:40]}"
        try:
            cfg = _json.loads(raw[s:e+1])
        except Exception:
            return f"解析失败: {raw[s:e+1][:40]}"

        cid = f"rnd_{cfg.get('name', 'npc')}"
        n = 2
        while cid in self.chars:
            cid = f"rnd_{cfg.get('name', 'npc')}_{n}"
            n += 1
        char = Character(
            name=cfg.get("name", "陌生人"), id=cid,
            identity=cfg.get("identity", "路人"),
            personality=cfg.get("personality", "普通"),
            speech_style=cfg.get("speech_style", "平常"),
            goal=cfg.get("goal", ""),
            greetings=[cfg.get("greeting", "……")],
            attitude_to_player="初次见面",
        )
        self.chars[cid] = char
        self.mem.set_persona(cid, [
            f"我的性格：{char.personality}",
            f"我说话风格：{char.speech_style}",
        ], [f"玩家：你是谁？ → {char.name}：{char.greetings[0]}"])
        self.mem.add_event(cid, f"{char.name}是{char.identity}，{char.goal}")
        print(f"[随机] 新NPC: {char.name}({char.identity})")
        return cid

    def random_event(self, cid: str = "", theme: str = "") -> str:
        """让模型生成一个随机事件（入世界事件记忆）。cid: 事件主体角色 id。"""
        import random as _r
        if not cid:
            cid = _r.choice(list(self.chars.keys()))
        char = self.chars.get(cid)
        name = char.name if char else cid
        prompt = (
            f"请生成一个随机游戏事件，发生在{name}身边。"
            f"主题：{theme or '任意'}。"
            f"格式：2~3句叙述，包含事件和影响，不要对话口吻。"
        )
        event_text = self.generate(cid, "事件", prompt)
        self.mem.add_world_event(cid, f"[事件] {event_text}")
        print(f"[随机] 事件: {event_text[:50]}")
        return event_text

    def chat(self, cid: str, player_input: str, scene: str = "") -> str:
        return self.generate(cid, "对话", player_input, scene=scene)

    # ---- 通用游戏文本生成 ----
    def generate(self, cid: str, kind: str, player_input: str,
                 scene: str = "", extra: dict | None = None) -> str:
        """生成任意游戏文本（对话/事件/描述/系统消息/战斗）。"""
        char = self.chars.get(cid)
        if not char:
            return "未知角色"
        tpl = TEXT_TEMPLATES.get(kind)
        if not tpl:
            return f"未知类型: {kind}, 可选: {list(TEXT_TEMPLATES.keys())}"
        extra = extra or {}

        # 记忆组装（固定 + 动态）
        fixed = self.mem.fixed_memories(cid)
        dynamic = self.mem.dynamic_memories(cid, player_input)
        taboo = [m for m in fixed if "我不谈论" in m or "禁忌" in m]
        rest = [m for m in fixed if m not in taboo] + dynamic
        mem_line = build_memory_prompt(cid, (taboo + rest)[:4])
        # 世界事件记忆（剧情发生了什么，独立区块）
        world_line = ""
        recent_events = self.mem.recent_world_events(cid, 4)
        if recent_events:
            world_line = "；".join(recent_events[-3:])

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
        sys_p += (
            "\n规则：1.用我的口吻说话；2.回应简短自然；3.我从来不是AI或助手；"
            "4.你只知道下面给你的信息，不知道任何其他对话或背景；"
            "5.玩家问到的没给的信息，按你的性格回应。"
        )

        # 场景 + 记忆 + 玩家输入 + 类型规则
        name = char.name
        act = SCENE_ACTIVITIES.get(scene, scene)
        # 上下文完全由系统提供，模型无历史——显式声明信息边界
        user = f"【你已知的信息（仅此而已，无其他对话记忆）】\n[场景] {act}\n"
        if mem_line:
            user += f"你记得：{mem_line}\n"
        if world_line:
            user += f"[近期事件] {world_line}\n"
        if extra:
            for k, v in extra.items():
                user += f"[{k}] {v}\n"
        user += f"\n玩家对你说：\"{player_input}\"\n"
        rule = tpl["rule"].format(name=char.name, kind=kind)
        user += rule

        raw = self._call(sys_p, user)
        reply = self._extract_by(raw, tpl.get("extract", "dialogue"))
        if not reply or len(reply) < 2:
            reply = "（无回应）"

        # 回存记忆
        if tpl.get("remember", True):  # 对话类 → 对话事实记忆
            entry = memory_entry(cid, player_input, reply)
            if entry:
                self.mem.add_event(cid, entry)
        if tpl.get("world_memory"):  # 事件类 → 世界事件记忆（独立表）
            self.mem.add_world_event(cid, f"[事件] {reply}")
        return reply

    def _call(self, sys_p, user_p, temperature=0.7):
        body = json.dumps({"model": "qwen3-4b",
                           "messages": [{"role": "system", "content": sys_p},
                                        {"role": "user", "content": user_p}],
                           "max_tokens": 120, "temperature": temperature}).encode("utf-8")
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

    def _extract_by(self, raw, mode="dialogue"):
        if mode == "text":
            text = raw.strip()
            # 剥 think 块（两种格式： thinking...response 或 <think>...</think>）
            text = re.sub("\s*thinking.*?response\s*", "", text, flags=re.S)
            text = re.sub("<think.*?</think>", "", text, flags=re.S)
            if "<think" in text:
                text = text.split(">", 1)[-1] if ">" in text else ""
            # 去残留 response 标签
            text = re.sub("^\s*response\s*", "", text)
            return text.strip()[:200]
        return self._extract(raw)

    def _extract(self, raw: str) -> str:
        text = raw.strip()
        text = re.sub(r"\s*thinking.*?response\s*", "", text, flags=re.S)
        text = re.sub(r"<think.*?</think>", "", text, flags=re.S)
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
    npc = NPCSystem()
    print("对话:", npc.chat("aila", "你是谁？", "集市"))
    print("事件:", npc.generate("aila", "事件", "狼群袭击村庄"))
    print("描述:", npc.generate("aila", "描述", "月光剑", extra={"物品": "月光剑"}))
    print("系统:", npc.generate("aila", "系统消息", "任务完成"))
    npc.stop()
