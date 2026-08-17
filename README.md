# ForeMeber — 以记忆为核心的游戏 NPC 对话体系

围绕记忆系统构建的角色扮演对话体系。

> 架构：**记忆系统 + [性格+记忆+用户对话拼装] + 接入文字模型**

## 组成

| 部分 | 文件 | 职责 |
|---|---|---|
| **记忆系统** | `npc/` | 完整记忆系统库（核心） |
| **拼装** | `prompt_builder.py` | 性格+记忆+用户对话 → 完整 prompt（单文件核心） |
| **接入模型** | `engine_vanilla.py` | 把 prompt 发给文字模型，提取台词，回存记忆 |

## 快速开始

```bash
# 需先启动本地 14B 角色扮演服务（llama.cpp, port 8081）

# 1. 注入性格（persona）到记忆
python -c "
from npc.memory import MemorySystem
ms = MemorySystem()
ms.set_persona('aila', ['我的性格：警惕、爱钱、嘴硬心软', '我说话风格：短句、带刺'],
    ['玩家：你是谁？ → 艾拉：一个路过的商人。别靠太近。'])
"

# 2. 对话
python engine_vanilla.py --character aila --scene "集市摊位"
```

## 拼装（prompt_builder.py）

```
system（性格）: 性格 + 说话特点 + 对话示范（persona 常驻）
user（记忆+对话）:
  [场景] 正在集市摆摊...
  你记得：我是艾拉，流浪商人，性格警惕爱钱；我想赚钱找妹妹；我的过去...
  玩家对你说："这剑多少钱？"
  艾拉直接回答...
```

## 记忆系统（npc/）

| 文件 | 职责 |
|---|---|
| `memory.py` | SQLite 三层记忆（状态/事件/人设）+ 去重 + 价格覆盖 + 分层获取 |
| `memory_extract.py` | 对话 → 事实提炼（价格只在问价时存） |
| `memory_prompt.py` | 事实 → 角色第一人称记忆提示词 |
| `embedding.py` / `colbert_memory.py` | 语义检索 |
| `encoder_router.py` | 意图粗筛 |
| `character.py` / `lorebook.py` | 角色卡 / 世界书 |

### 关键设计
- 常识不靠记忆教（砍价不涨价由模型能力）
- 价格只在问价时存，同物品覆盖
- 坏回复不进记忆

## 测试

```bash
python -m pytest tests/ -v   # 记忆系统单元测试
```
