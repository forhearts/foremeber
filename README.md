# ForeMeber — 以记忆为核心的游戏 NPC 对话体系

**remember/**（记忆系统）+ **foremeber.py**（性格+记忆+用户对话 → 接入文字模型）

## 组成

| 部分 | 文件 | 职责 |
|---|---|---|
| **记忆系统** | `remember/` | 完整记忆系统库（核心） |
| **For 部分** | `foremeber.py` | 性格+记忆+用户对话拼装 + 接入文字模型（单文件） |

## 快速开始

```bash
# 需先启动本地 14B 角色扮演服务（llama.cpp, port 8081）
python foremeber.py --character aila --scene "集市摊位"    # 单轮
python foremeber.py --character aila --interactive         # 多轮
```

## 拼装结构（foremeber.py）

```
system（性格）: 我的性格+说话风格+对话示范（persona 常驻）
user（记忆+对话）:
  [场景] 正在集市摆摊...
  你记得：我是艾拉，流浪商人，性格警惕爱钱；我想赚钱找妹妹；我的过去...
  玩家对你说："这剑多少钱？"
  艾拉直接回答...
```

## 记忆系统（remember/）

| 文件 | 职责 |
|---|---|
| `memory.py` | SQLite 三层记忆 + 去重 + 价格覆盖 + 分层获取 |
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
