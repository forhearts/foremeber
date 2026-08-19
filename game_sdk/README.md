# game_sdk — 游戏 NPC 对话 SDK（自包含）

给游戏嵌入用的完整对话系统。**一行代码接入**，自动管理记忆、引擎、角色。

## 接入方式

```python
# 把 game_sdk/ 整个文件夹复制到你的游戏项目
from npc_api import NPCSystem

npc = NPCSystem()   # 自动启动 Qwen3-4B 引擎（Vulkan）

# 对话
reply = npc.chat("aila", "这把剑多少钱？", scene="集市摊位")

# 可变游戏文本（事件/描述/系统消息/战斗）
event = npc.generate("aila", "事件", "狼群袭击村庄", scene="夜晚营地")
desc = npc.generate("aila", "描述", "月光剑", extra={"物品": "月光剑"})
msg = npc.generate("aila", "系统消息", "任务完成")
battle = npc.generate("morgan", "战斗", "与黑狼交战", extra={"敌人": "变异狼"})

# 游戏事件 → 更新状态（好感度/任务）
npc.set_state("aila", 好感度=50, 任务="护送货物")

# 游戏主动注入记忆（如任务完成）
npc.remember("aila", "玩家帮艾拉找到了丢失的货物")

# 查询状态
state = npc.get_state("aila")

# 退出时停止引擎
npc.stop()
```

## 依赖

- `D:\ai-models\llama-vulkan\llama-server.exe`（Vulkan 版，含 ggml-vulkan.dll）
- `D:\ai-models\Qwen3\Qwen3-4B-Q4_K_M.gguf`（Qwen3-4B 模型）
- Python + transformers（仅记忆检索用，Qwen3-Embedding）

> 如果模型路径不同，改 `npc_api.py` 顶部的 `LLAMA_VULKAN` / `QWEN3_4B`。

## 目录

```
game_sdk/
├── npc_api.py      ← 主入口（NPCSystem 类）
├── characters/     ← 角色卡（8 个 NPC，可改）
├── lorebook/       ← 世界书（12 条设定）
├── memory.db       ← 记忆库（运行时生成）
├── engine.log      ← 引擎日志
└── remember/       ← 记忆系统库（如需要可引入）
```

## 示例（游戏对话循环）

```python
from npc_api import NPCSystem

npc = NPCSystem()
npc.set_state("aila", 好感度=10)

while True:
    player = input("你> ")
    if player in ("quit", "exit"):
        break
    # 游戏逻辑：送礼加好感
    if "送" in player:
        npc.set_state("aila", 好感度=npc.get_state("aila").get("好感度", 0) + 10)
    reply = npc.chat("aila", player, scene="集市摊位")
    print(f"艾拉> {reply}")

npc.stop()
```
