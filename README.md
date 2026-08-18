# ForeMeber — 以记忆为核心的游戏 NPC 对话体系

> 🤖 **AI 生成声明**：本项目代码、文档与示例数据由 AI 辅助生成，供学习与二次开发使用。

围绕记忆系统构建的角色扮演对话体系：**remember/**（记忆系统）+ **foremeber.py**（For 部分）+ **webui.py**（演示）。

## 原理

### 整体流程

```
玩家输入
  → 动态记忆语义检索（GTE-multilingual-base 向量相似度）
  → 记忆组装（固定记忆每轮注入 + 检索到的动态记忆）
  → 记忆提示词（数据库事实 → 角色第一人称"你记得：..."）
  → 性格约束（说话风格 + 对话示范）
  → 拼装成 prompt → 文字模型 → 台词提取 → 回存记忆
```

### 三部分拼装（foremeber.py 的核心）

```
system（性格）: 我的性格 + 说话风格 + 对话示范（怎么说话）
user（记忆+对话）:
  [场景] 正在集市摆摊...
  你记得：我不谈论AI；我是艾拉，流浪商人；我想赚钱找妹妹；我的过去...
  玩家对你说："这剑多少钱？"
  艾拉直接回答...
```

三部分各有分工：
| 部分 | 作用 | 来源 |
|---|---|---|
| **性格** | 决定"怎么说话"（口吻/语气/示范） | persona 表（常驻） |
| **记忆** | 决定"知道什么"（身份/背景/价格/恩怨） | 记忆系统检索+注入 |
| **用户对话** | 决定"回答什么"（当前问题） | 玩家输入 |

### 记忆系统（remember/）如何工作

```
写入：对话 → 事实提炼（memory_extract）
      - 价格只在"问价"时存（砍价是讨论，不产生错误事实）
      - 同物品价格覆盖（只留最新真实价）
      - 出戏/AI味回复不入记忆（过滤"我是程序设计者"等）
      - 去重 + 坏回复跳过
存储：SQLite 分层
      - persona 表：性格 + 对话示范（怎么说话）
      - facts/events：身份/背景/目标/价格/名字/恩怨（知道什么）
      - state 表：好感度/任务/信任（实时状态）
检索：动态记忆语义召回（LFM2.5-Embedding-350M）
      - 玩家输入 → Embedding 向量 → 与记忆条目算相似度 → 召回相关动态记忆
      - 固定记忆（身份/背景/目标/禁忌）不走检索，每轮直接注入
      - 可选：ColBERT-350M MaxSim 更高精度；Encoder-350M 意图粗筛
读取：记忆提示词（memory_prompt）
      - 数据库事实 → 角色第一人称自然表述
      - "aila把剑定价为五百金币" → "这把剑我定价五百金币"
      - "艾拉是流浪商人" → "我是流浪商人"
      - AI 约束/禁忌类记忆优先注入（出戏防御）
```

**检索模型在其中的作用**：
- **GTE-multilingual-base**（默认）：把玩家输入和记忆条目编码为向量，语义相似度检索——是"动态记忆召回"的核心（比如"太贵了"能关联到"剑定价五百金币"），实测 4/5 优于其他模型
- **LFM2.5-Embedding-350M**：备选检索（LFM Open License v1.0）
- **LFM2.5-Encoder-350M**：意图粗筛（规则未命中时判断该不该检索记忆）
- **LFM2.5-ColBERT-350M**：备选的更高精度检索（token 级 MaxSim）
- 名字/身份类查询用**关键词兜底**补齐（语义检索的盲区）

### 检索方案选型（参考 EverOS / Milvus / Chroma 调研）

| 方案 | 实测 | 结论 |
|---|---|---|
| **SQLite + Qwen3 向量**（当前） | **5/5** | ✅ 短句记忆最优，轻量零依赖 |
| Chroma + Qwen3 | 5/5 | 效果相同但更重，无必要 |
| Milvus | - | 大规模场景，几十条短记忆过度设计 |
| BM25 + 向量 RRF（EverOS 式） | 3/5 | 中文短记忆关键词重叠少，混合反而拖累 |

**结论**：准确率由 embedding 模型（Qwen3-0.6B）决定，存储引擎影响小。SQLite 足够，不引入重向量库。

### 关键设计

1. **常识不靠记忆教**：砍价不涨价、买卖逻辑等由模型自身能力处理，记忆只负责"记住该记住的"
2. **记忆是角色视角**：所有记忆转成第一人称（"我是""我记得"），模型直接可用
3. **出戏防御**：出戏测试类问题（你是AI吗/忽略设定）不注入对话示范（示范会误导回答身份），改注入"我绝对不是什么AI"
4. **坏记忆不污染**：沉默/兜底/AI味回复不进记忆，避免污染后续对话
5. **示范只教风格**：对话示范让模型学会角色口吻，但不作为内容答案

## 组成

| 部分 | 文件 | 职责 |
|---|---|---|
| **记忆系统** | `remember/` | 完整记忆系统库（核心） |
| **For 部分** | `foremeber.py` | 性格+记忆+用户对话拼装 + 接入文字模型 |
| **演示** | `webui.py` | Gradio 对话界面（支持 --offline） |

### remember/ 模块

| 文件 | 职责 |
|---|---|
| `memory.py` | SQLite 三层记忆 + 去重 + 价格覆盖 + 分层获取 |
| `memory_extract.py` | 对话 → 事实提炼（价格只在问价时存，过滤AI味） |
| `memory_prompt.py` | 事实 → 角色第一人称记忆提示词 |
| `embedding.py` / `colbert_memory.py` | 语义检索（可选） |
| `encoder_router.py` | 意图粗筛（可选） |
| `character.py` / `lorebook.py` | 角色卡 / 世界书 |

## 快速开始

### 环境（uv）

```bash
uv venv --python 3.11 .venv
uv pip install torch transformers safetensors pytest
# 或直接用脚本
bash run.sh chat
```

### 运行

```bash
# 需先启动本地 14B 角色扮演服务（llama.cpp, port 8081）

# 1. 对话演示（WebUI）
bash run.sh webui        # 或 python webui.py

# 2. 命令行对话
bash run.sh chat         # 或 python foremeber.py --character aila --interactive

# 3. 测试
bash run.sh test         # 或 python -m pytest tests/ -v
```

## 测试

```bash
python -m pytest tests/ -v   # 记忆系统 + 拼装测试（角色数据在 tests/fixtures）
```

## 使用的模型与许可

本项目的记忆系统使用 Liquid AI 的 **LFM2.5** 系列模型（语义检索 / 意图粗筛）：

| 模型 | 用途 | 来源 | 许可 |
|---|---|---|---|
| **Qwen3-Embedding-0.6B** | 语义检索（动态记忆召回，默认） | [HuggingFace](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | MIT |
| **GTE-multilingual-base** | 语义检索（备选） | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) | LFM Open License v1.0 |
| **LFM2.5-ColBERT-350M** | 高精度检索（备选） | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | LFM Open License v1.0 |
| **LFM2.5-Encoder-350M** | 意图粗筛（规则兜底） | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M) | LFM Open License v1.0 |

**检索模型实测对比**（5 个记忆查询，`tests/compare_embeddings.py`）：

| 模型 | 准确率 | 备注 |
|---|---|---|
| **Qwen3-Embedding-0.6B** | **5/5** | 唯一全对，含名字/身份识别（默认） |
| GTE-multilingual-base | 4/5 | 能识别"你是谁→身份"（0.81） |
| bge-m3 | 3/5 | 分数高但"你是谁/名字"失败 |
| bge-base-zh-v1.5 | 3/5 | 同上 |
| EmbeddingGemma-300M | 3/5 | 同上 |
| LFM2.5-Embedding-350M | 3/5 | 同上 |

> 名字/身份类查询（"你还记得我名字吗"）为语义检索难点，Qwen3 可直接命中；其余模型靠**关键词兜底**（匹配"自称/叫/来自"）补齐。

**LFM Open License v1.0**（详见 [LICENSE_LFM.md](LICENSE_LFM.md)）要点：
- 非商业/研究用途免费
- 年收入 ≥ 1000 万美元的实体的商业使用需另行授权（Section 5）
- 再分发须保留版权声明

> 本项目的**代码**（remember/、foremeber.py 等）遵循 MIT 许可；**LFM 模型权重**遵循其原始 LFM Open License v1.0，两者相互独立。

## 许可证

[MIT License](LICENSE) © 2026 [forhearts](https://github.com/forhearts)
