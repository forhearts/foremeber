"""Vanilla 14B 角色扮演引擎：本地 llama.cpp API 后端

用高质量 14B 模型（vanilla-cn-roleplay-0.2）做运行时推理。
复用全部系统：角色卡/记忆/意图/后处理，只替换生成后端。
含台词提取：从模型输出的旁白+台词混合文本中提取纯台词。
"""
from __future__ import annotations

import json
import re
import time
import urllib.request

from npc.character import load_all_characters
from npc.config import GenerationConfig
from npc.engine import DialogueEngine
from npc.lorebook import load_lorebook
from npc.memory import MemorySystem
from npc.postprocess import PostProcessor

API_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "vanilla-cn-roleplay-0.2.i1-IQ3_S"

# 引号对（中英文）
QUOTE_PAIRS = [
    ("\u300c", "\u300d"),  # 「」
    ("\u201c", "\u201d"),  # “”
    ("\u300e", "\u300f"),  # 『』
    ('"', '"'),              # 英文双引号 ""
]

NARRATIVE_VERBS = ["我看着", "我心想", "我叹", "我笑", "我皱", "我抬", "我走",
                   "我站", "我坐", "我点", "我停", "我转", "我摸", "我低", "我盯",
                   "我打量", "我望着", "我盯着", "我望向", "我看", "我望", "我感", "我觉"]
DIALOG_MARKERS = ["你", "吧", "啊", "呢", "吗", "呀", "喂", "哼", "嘿", "别",
                  "！", "？", "!", "?", "哦", "嘛"]


def extract_dialogue(raw: str) -> str:
    """从旁白+台词混合输出中提取纯台词。"""
    text = raw.strip()
    # 0. 先剥离孤立括号残留
    text = re.sub(r"^[）)]+\s*", "", text)
    # 1. 引号内容（优先，模型台词常用引号）
    best_quoted = None
    for open_q, close_q in QUOTE_PAIRS:
        quoted = re.findall(
            rf"{re.escape(open_q)}([^{re.escape(open_q)}{re.escape(close_q)}]{{1,}}){re.escape(close_q)}",
            text)
        if quoted:
            for q in quoted:
                q = q.strip()
                if not q:
                    continue
                if any(m in q for m in DIALOG_MARKERS):
                    return q[:80]
                if best_quoted is None or len(q) > len(best_quoted):
                    best_quoted = q
    if best_quoted:
        return best_quoted[:80]

    # 2. 无引号：剥掉所有括号内容（动作/心理），用句号替代（保持切分）
    text = re.sub(r"[（(][^）)]*[）)]", "。", text)
    sentences = re.split(r"[。！？!?\n…]|\.\.\.+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s.strip("「」\"'“”") for s in sentences]
    sentences = [s for s in sentences if s]
    if not sentences:
        return ""
    identity_markers = ["我是", "我叫", "这里是", "我乃", "本人", "老身", "在下", "咱"]
    best = None
    best_score = -1
    for s in sentences:
        if any(s.startswith(v) for v in NARRATIVE_VERBS):
            continue
        score = 0
        if any(m in s for m in identity_markers):
            score += 10
        if any(m in s for m in ["？", "?", "吧", "啊", "哦", "呢", "呀", "哼", "喂"]):
            score += 3
        if "你" in s:
            score += 2
        if len(s) <= 30:
            score += 1
        if score > best_score:
            best_score = score
            best = s
    return (best or sentences[-1])[:80]


class VanillaEngine(DialogueEngine):
    """14B 角色扮演引擎（API 后端，无本地模型加载）。"""

    def __init__(self, characters=None, lorebook=None, memory=None,
                 gen_config=None, post=None):
        self.model = None
        self.tokenizer = None
        self.router = None  # Encoder-350M 意图粗筛（惰性加载）
        self.characters = characters if characters is not None else load_all_characters()
        self.lorebook = lorebook if lorebook is not None else load_lorebook()
        self.memory = memory if memory is not None else MemorySystem()
        self.gen_config = gen_config if gen_config is not None else GenerationConfig()
        self.post = post if post is not None else PostProcessor()

    def chat(self, character_id, player_input, scene="", history=None,
             state_updates=None, max_retries=1, verbose=False):
        from npc.prompt import build_system_prompt, build_memory_pack
        from npc.encoder_router import EncoderRouter
        character = self.characters.get(character_id)
        if character is None:
            raise ValueError(f"未知角色: {character_id}")

        # 意图粗筛：规则优先（精准），Encoder-350M 兜底（规则未命中时）
        intent = self.detect_intent(player_input)
        if intent == "闲聊":
            if self.router is None:
                from npc.encoder_router import EncoderRouter
                self.router = EncoderRouter()
            intent, _ = self.router.classify(player_input)
        # 闲聊/世界观/离开 等不检索记忆（避免噪音）
        NO_MEMORY_INTENTS = {"闲聊", "离开", "出戏测试"}

        if state_updates:
            self.update_state(character_id, **state_updates)
        state = self.get_default_state(character_id)

        # ===== 记忆系统（通用层）：按意图聚焦检索 =====
        # ===== 记忆组装 =====
        # 1. 固定记忆（身份/背景/目标/禁忌）：NPC 总是知道自己是谁，每轮注入
        fixed = [m for m in self.memory.recent_events(character_id, 20)
                 if any(k in m for k in ["是", "的背景", "的目标", "的禁忌", "对玩家的"])][:4]
        # 2. 动态记忆（价格/名字/事件）：语义检索
        dynamic = []
        if intent not in NO_MEMORY_INTENTS:
            dynamic = self._gather_memories(character_id, player_input)
            BAD_MEM = ["沉默", "没有接话", "（"]
            dynamic = [m for m in dynamic
                       if len(m) >= 6 and not any(b in m for b in BAD_MEM)][:3]
        memories = fixed + dynamic
        # 世界书关键词触发
        lore_hits = self.lorebook.query(player_input + scene, max_budget=200)

        # 结构化记忆包（模型无关）
        pack = build_memory_pack(
            character=character, player_input=player_input, intent=intent,
            state=state, memories=memories, lore_hits=lore_hits, scene=scene,
        )

        sys_prompt = build_system_prompt(character)
        # 示范（persona 表，说话风格参考）
        _, demo_lines = self.memory.get_persona(character_id)
        if demo_lines:
            demos = "\n".join(demo_lines[:3])
            sys_prompt += f"\n我说话的样子（示例）：\n{demos}"
        info_hint = ""
        if intent in ("打听情报", "询问任务", "世界观"):
            info_hint = f"（{character.name}知道相关情况，直接告诉玩家）"

        # 记忆优先 prompt：状态/场景在前，关键事实强调放最末尾（模型注意力最强处）
        context_parts = []
        if pack["state"]:
            st = "；".join(f"{k}：{v}" for k, v in pack["state"].items()
                            if v not in (None, "", False) and k != "_updated_at")
            if st:
                context_parts.append(f"[状态] {st}")
        if pack["scene"]:
            from npc.prompt import SCENE_ACTIVITIES
            act = SCENE_ACTIVITIES.get(pack["scene"], pack["scene"])
            context_parts.append(f"[场景] {act}")
        if pack["lore"]:
            context_parts.append("[设定] " + "；".join(pack["lore"][:2]))
        ctx = "\n".join(context_parts)
        # 记忆提示词：把事实转成角色视角（"这把剑我定价五百金币"）
        from npc.memory_prompt import build_memory_prompt
        mem_line = build_memory_prompt(character_id, pack["memories"], max_lines=2)
        gen_prompt = (
            f"你是{character.name}。\n{ctx}\n\n"
            f"玩家对你说：\"{player_input}\"{info_hint}\n"
        )
        if mem_line:
            gen_prompt += f"你记得：{mem_line}\n"
        gen_prompt += f"{character.name}直接回答玩家（第一人称，1~2句，直接说出口的话，不要内心独白不要思考）："

        messages = [{"role": "system", "content": sys_prompt}]
        messages.append({"role": "user", "content": gen_prompt})

        raw = self._call_api(messages)
        # 空/过短回复重试一次（14B 3bit 偶发）
        if len(raw.strip()) < 4:
            raw = self._call_api(messages, temperature=0.85)
        dialogue = extract_dialogue(raw)
        reply = self.post.process(dialogue)
        # 意图-回复一致性校验：交易类问题必须含价格/交易词，否则重试（答非所问修复）
        for attempt in range(2):
            if reply and self._is_relevant(intent, reply):
                break
            # 重试：换 prompt 强调直接回答
            retry_hint = self._retry_prompt(character, player_input, intent, pack)
            retry_msgs = [{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": retry_hint}]
            raw = self._call_api(retry_msgs, temperature=0.9)
            dialogue = extract_dialogue(raw)
            reply = self.post.process(dialogue)
        # 最后防线：交易类仍答非所问 → 用记忆里的价格事实兜底
        if not (reply and self._is_relevant(intent, reply)):
            fb = self._price_fallback(character, player_input, intent, memories)
            if fb:
                reply = fb
        # 兜底：提取后仍空则用角色化兜底（不写入记忆，避免坏记忆污染）
        if not reply or len(reply) < 2:
            return {"reply": "（NPC 沉默片刻，没有接话。）", "intent": intent,
                    "state": state, "prompt": "" if not verbose else gen_prompt, "raw": raw}
        self.memory.add_dialogue_turn(character_id, player_input, reply)
        return {"reply": reply, "intent": intent, "state": state,
                "prompt": "" if not verbose else gen_prompt, "raw": raw}

    @staticmethod
    def _is_relevant(intent: str, reply: str) -> bool:
        """意图-回复一致性：交易类必须提价格（数字+币）；身份类必须提身份。"""
        if not reply:
            return False
        if intent in ("交易", "买东西", "讨价还价"):
            # 必须含价格特征：数字/中文数字 + 币种，或明确"X钱""XX金"
            import re
            return bool(re.search(r"[0-9零一二三四五六七八九十百千万两]+\s*(枚|个|块|文)?\s*(金币|银币|铜币|金|银)", reply))
        if intent in ("询问身份", "问AI"):
            return any(k in reply for k in ["我", "艾", "商", "是", "不", "别"])
        return True

    @staticmethod
    def _retry_prompt(character, player_input, intent, pack) -> str:
        """重试 prompt：更直接地要求回答。"""
        if intent in ("交易", "买东西", "讨价还价"):
            return (
                f"你是{character.name}。玩家问\"{player_input}\"。\n"
                f"直接告诉玩家价格（如：五百金币），1句话，不要其他话。"
            )
        return (
            f"你是{character.name}。玩家问\"{player_input}\"。\n"
            f"直接回答玩家的问题，1句话。"
        )

    def _price_fallback(self, character, player_input, intent, memories) -> str | None:
        """确定性价格兜底：从记忆提取价格事实，生成模板回复。"""
        if intent not in ("交易", "买东西", "讨价还价"):
            return None
        for m in memories:
            import re
            pm = re.search(r"定价为([0-9零一二三四五六七八九十百千万两]+(?:枚|个|块|文)?(?:金币|银币|铜币))", m)
            if pm:
                return f"{character.name}：{pm.group(1)}，不讲价。"
        return None

    def _call_api(self, messages, temperature=0.75, max_tokens=90):
        body = json.dumps({
            "model": MODEL, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(API_URL, body, {"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.loads(r.read())
                m = d["choices"][0]["message"]
                c = (m.get("content") or "").strip()
                rc = (m.get("reasoning_content") or "").strip()
                raw = c if len(c) >= 4 else rc
                # 元语言残留检测（模型把指令当回复），命中则重试
                if any(k in raw for k in ["在回答中", "作为AI", "我需要", "首先，", "最后，", "总结"]):
                    if attempt < 2:
                        time.sleep(1)
                        continue
                return raw
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    return f"（{e}）"
        return ""


def load_vanilla_engine():
    """加载 14B 引擎（检查 API 可用性 + 接入语义记忆）。"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=5) as r:
            if r.status != 200:
                raise ConnectionError("vanilla API 不可用")
    except Exception:
        raise ConnectionError(
            "vanilla-cn-roleplay 服务未启动！请先运行: D:\\ai-models\\start-vanilla-roleplay.cmd")

    # 语义记忆：只用 Embedding-350M（轻量，350M 显存小）；ColBERT 关闭（省资源）
    colbert = None
    embed_client = None
    try:
        from npc.embedding import EmbeddingClient
        with urllib.request.urlopen("http://127.0.0.1:8082/health", timeout=3) as r:
            if r.status == 200:
                embed_client = EmbeddingClient(url="http://127.0.0.1:8082/v1/embeddings")
                print("[vanilla] 记忆检索: Embedding-350M (余弦, 轻量)")
    except Exception:
        print("[vanilla] Embedding-350M 不可用，记忆退化为关键词检索")

    print(f"[vanilla] 连接 {API_URL}")
    from npc.memory import MemorySystem
    eng = VanillaEngine()
    eng.memory = MemorySystem(embed_client=embed_client, colbert=colbert)
    return eng
