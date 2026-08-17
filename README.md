# ForeMeber — 游戏 NPC 记忆系统

为游戏角色扮演设计的**独立记忆系统**：说话风格 + 记忆 + 用户对话 三部分架构。

> 目标：任意玩家输入 → 记忆系统组织记忆提示词 → 任意文字模型（换 lora/底模）都能输出
> 常识（砍价不涨价、买卖逻辑）由模型自身能力处理，记忆系统只负责"记住该记住的"。

## 架构

```
玩家输入
  → 意图粗筛（规则 + Encoder-350M）
  → 记忆检索（Embedding-350M 语义 + 固定记忆每轮注入）
  → 记忆提示词（事实 → 角色第一人称视角）
  → 输出给任意模型
```

### 三层记忆
| 层 | 内容 | 方式 |
|---|---|---|
| **说话风格**（固定） | 性格 + 说话特点 + 对话示范 | system prompt 常驻 |
| **固定记忆** | 身份/背景/目标/禁忌（第一人称） | 每轮注入 |
| **动态记忆** | 价格/名字/事件/恩怨 | 语义检索 |

### 记忆提示词（核心）
数据库事实 → 角色视角自然表述：
```
aila把剑定价为五百金币  → 这把剑我定价五百金币
艾拉是流浪商人          → 我是流浪商人
玩家自称林风            → 有个玩家说他叫林风
```

## 模块

| 文件 | 职责 |
|---|---|
| `npc/memory.py` | SQLite 记忆系统（状态/事件/人设分层，去重） |
| `npc/memory_extract.py` | 对话 → 事实提炼（价格只在问价时存，砍价不产生） |
| `npc/memory_prompt.py` | 事实 → 角色视角记忆提示词 |
| `npc/embedding.py` | Embedding-350M 语义检索客户端 |
| `npc/colbert_memory.py` | ColBERT-350M MaxSim 检索（备选） |
| `npc/encoder_router.py` | 意图粗筛（规则兜底） |
| `npc/character.py` | 角色卡 |
| `npc/lorebook.py` | 世界书（关键词触发） |
| `npc/prompt.py` | 记忆包组装（build_memory_pack） |

## 脚本

| 脚本 | 用途 |
|---|---|
| `scripts/seed_memory.py` | 人设/示范注入记忆库 |
| `scripts/generate_examples.py` | 用高质量 14B 生成对话示范 |
| `scripts/clean_examples.py` | 示范清洗（去动作/AI味/噪音） |

## 使用

```bash
# 1. 注入人设到记忆库
python scripts/seed_memory.py

# 2. 生成高质量示范（可选）
python scripts/generate_examples.py --count 5
python scripts/clean_examples.py

# 3. 在任意引擎中使用
from npc import MemorySystem
ms = MemorySystem()
facts, examples = ms.get_persona("aila")   # 说话风格 + 示范
memories = ms.build_memory_context("aila", "这剑多少钱", top_k=3)  # 动态记忆
```

## 测试

```bash
python -m pytest tests/ -v
```
