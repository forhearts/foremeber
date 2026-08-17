# ForeMeber — 以记忆为核心的游戏 NPC 对话体系

一个**围绕记忆系统构建的完整角色扮演对话体系**。记忆系统是重中之重，但整体是能跑的对话系统。

> 定位：任意玩家输入 → 记忆系统组织记忆提示词 → 任意文字模型（换 lora/底模）输出
> 架构：**说话风格 + 记忆 + 用户对话** 三部分

## 架构

```
玩家输入
  → 意图粗筛（规则 + Encoder-350M）
  → 记忆检索（Embedding-350M 语义 + 固定记忆每轮注入）
  → 记忆提示词（事实 → 角色第一人称视角）
  → 生成引擎（vanilla14B / Qwen / MiniMind3 / 云端7B 可切换）
  → 台词提取 + 后处理 → 输出
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

### 关键设计
- **常识不靠记忆教**：砍价不涨价等由模型自身能力，记忆只负责"记住该记住的"
- **价格只在问价时存**：砍价是讨论不是定价，不产生错误事实
- **同物品价格覆盖**：只留最新真实价
- **坏记忆不注入**：沉默/兜底回复不进记忆

## 模块

| 文件 | 职责 |
|---|---|
| `npc/memory.py` | SQLite 分层记忆（状态/事件/人设）+ 去重 + 价格覆盖 |
| `npc/memory_extract.py` | 对话 → 事实提炼 |
| `npc/memory_prompt.py` | 事实 → 角色视角记忆提示词 |
| `npc/embedding.py` / `colbert_memory.py` | 语义检索（Embedding/ColBERT） |
| `npc/encoder_router.py` | 意图粗筛 |
| `npc/vanilla_engine.py` | 本地 14B 生成引擎（记忆注入 + 台词提取） |
| `npc/qwen_engine.py` / `minimind3_engine.py` / `cloud_engine.py` | 其他底模引擎 |
| `npc/character.py` / `lorebook.py` | 角色卡 / 世界书 |
| `npc/prompt.py` | 记忆包组装 |
| `webui.py` | 对话 WebUI |
| `game/demo_game.py` | 村庄场景 demo |

## 脚本

| 脚本 | 用途 |
|---|---|
| `scripts/seed_memory.py` | 人设/示范注入记忆库 |
| `scripts/generate_examples.py` | 高质量 14B 生成对话示范 |
| `scripts/clean_examples.py` | 示范清洗 |
| `scripts/demo_chat.py` | 命令行对话 |

## 使用

```bash
# 1. 注入人设到记忆库
python scripts/seed_memory.py

# 2. 启动本地 14B 服务（vanilla-cn-roleplay）后对话
python webui.py --engine vanilla

# 3. 命令行对话
python scripts/demo_chat.py --character aila --scene "集市摊位"
```

## 测试

```bash
python -m pytest tests/ -v
```
