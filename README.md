# ForeMeber — 以记忆为核心的游戏 NPC 对话体系

围绕记忆系统构建的**完整角色扮演对话体系**。记忆系统是重中之重。

> 架构：**说话风格 + 记忆 + 用户对话**
> 定位：任意玩家输入 → 记忆系统组织记忆提示词 → 任意文字模型输出

## 快速开始

对话体系是**单文件** `foremeber.py`（内置记忆 + 引擎 + 台词提取）：

```bash
# 需先启动本地 14B 角色扮演服务（llama.cpp, port 8081）
python foremeber.py --character aila --scene "集市摊位"     # 单轮
python foremeber.py --character aila --interactive          # 多轮对话
```

## 架构

```
玩家输入
  → 记忆组装（固定记忆每轮注入 + 动态记忆关键词检索）
  → 记忆提示词（事实 → 角色第一人称"你记得：..."）
  → 生成引擎（14B 角色扮演模型）
  → 台词提取 + 后处理 → 输出
```

### 三层记忆
| 层 | 内容 | 方式 |
|---|---|---|
| **说话风格**（固定） | 性格 + 说话特点 + 对话示范 | system prompt 常驻 |
| **固定记忆** | 身份/背景/目标/禁忌（第一人称） | 每轮注入 |
| **动态记忆** | 价格/名字/事件/恩怨 | 关键词检索 |

### 关键设计
- **常识不靠记忆教**：砍价不涨价等由模型自身能力
- **价格只在问价时存**：砍价是讨论不是定价，不产生错误事实
- **同物品价格覆盖**：只留最新真实价
- **坏回复不进记忆**：沉默/兜底不存

## 记忆系统库（npc/）

可独立使用的完整记忆系统（不依赖对话体系）：

| 文件 | 职责 |
|---|---|
| `npc/memory.py` | SQLite 分层记忆（状态/事件/人设）+ 去重 + 价格覆盖 |
| `npc/memory_extract.py` | 对话 → 事实提炼 |
| `npc/memory_prompt.py` | 事实 → 角色视角记忆提示词 |
| `npc/embedding.py` | Embedding-350M 语义检索 |
| `npc/colbert_memory.py` | ColBERT-350M MaxSim 检索 |
| `npc/encoder_router.py` | 意图粗筛 |
| `npc/character.py` / `lorebook.py` | 角色卡 / 世界书 |

## 脚本

| 脚本 | 用途 |
|---|---|
| `scripts/seed_memory.py` | 人设/示范注入记忆库 |
| `scripts/generate_examples.py` | 高质量 14B 生成对话示范 |
| `scripts/clean_examples.py` | 示范清洗 |

## 测试

```bash
python -m pytest tests/ -v   # 15 个记忆系统单元测试
```
