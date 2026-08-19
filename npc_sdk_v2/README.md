# npc_sdk_v2 — 游戏 NPC SDK（精简自包含版）

游戏嵌入用完整对话/文本生成系统。**上下文完全由系统管理，模型无状态**。

## 接入

```python
# 把 npc_sdk_v2/ 拷到游戏项目
from npc_api import NPCSystem

npc = NPCSystem()   # 自动启动 Qwen3-4B 引擎（Vulkan）

# 对话（无状态，记忆系统管理上下文）
npc.chat("aila", "这把剑多少钱？", scene="集市")

# 可变游戏文本
npc.generate("aila", "事件", "狼群袭击村庄")     # 事件 → 世界事件记忆
npc.generate("aila", "描述", "月光剑")           # 描述
npc.generate("aila", "系统消息", "任务完成")     # 系统通知
npc.generate("morgan", "战斗", "与黑狼交战")     # 战斗

# 游戏事件 → 状态
npc.set_state("aila", 好感度=50)
npc.remember("aila", "玩家帮艾拉找到丢失的货物")  # 主动注入记忆

npc.stop()  # 退出时关引擎
```

## 无状态上下文（核心设计）

```
每次模型调用只收到：
  system: [性格/说话风格] + 规则（含"你只知道下面给你的信息"）
  user:   【你已知的信息（仅此而已，无其他对话记忆）】
          [场景] ...
          你记得：{对话事实记忆}       ← 系统从记忆库查
          [近期事件] {世界事件记忆}     ← 系统从事件库查
          玩家对你说：{当前输入}        ← 唯一"新"信息
```

模型**不接收任何历史对话**，全部上下文由记忆系统注入。

## 记忆分层

| 记忆 | 表 | 注入区块 | 例子 |
|---|---|---|---|
| 对话事实 | events | 你记得 | "这把剑我定价五百金币" |
| **世界事件** | world_events | [近期事件] | "狼群袭击村庄，谷仓被烧" |
| 人设 | persona | system | 性格/说话风格/示范 |

- 事件与对话**独立存储**，互不污染
- 事件类生成自动写入世界事件，对话时自动引用

## 依赖

- `D:\ai-models\llama-vulkan\llama-server.exe`（Vulkan 版 llama.cpp）
- `D:\ai-models\Qwen3\Qwen3-4B-Q4_K_M.gguf`（Qwen3-4B）
- Python 3.10+（纯标准库，无第三方依赖）

> 路径不同改 `npc_api.py` 顶部 `LLAMA_VULKAN` / `QWEN3_4B`。

## 目录

```
npc_sdk_v2/
├── npc_api.py          ← 主入口（NPCSystem）
├── characters/         ← 角色卡（8 NPC，可改）
├── lorebook/           ← 世界书（12 条设定）
├── remember/           ← 记忆系统（5 个必需模块，纯标准库）
└── memory.db           ← 记忆库（运行时生成）
```

## 随机系统（附加功能）

模型生成全新 NPC 和事件（不是从池子抽取）：

```python
# 生成随机 NPC（返回 id，可对话）
cid = npc.spawn_random_npc(theme="酒馆里的神秘客人")
npc.chat(cid, "你是谁？")   # 和随机 NPC 对话

# 生成随机事件（入世界事件记忆）
npc.random_event()                    # 随机主体
npc.random_event("aila", "发现宝藏")   # 指定角色 + 主题
```

- `spawn_random_npc(theme)`：模型生成角色 JSON（名字/身份/性格/风格/目标），注册后即可对话
- `random_event(cid, theme)`：模型生成事件叙述，写入世界事件记忆，后续对话会引用

## 文本类型

| 类型 | 生成内容 | 记忆 |
|---|---|---|
| 对话 | NPC 台词 | 对话事实 |
| 事件 | 剧情叙述 | 世界事件 |
| 描述 | 场景/物品/人物 | - |
| 系统消息 | 任务/奖励通知 | - |
| 战斗 | 行动/结果 | - |
