# ForeMeber — 以记忆为核心的游戏 NPC 对话体系

> 🤖 **AI 生成声明**：本项目代码、文档与示例数据由 AI 辅助生成，供学习与二次开发使用。

围绕记忆系统构建的角色扮演对话体系：**remember/**（记忆系统）+ **foremeber.py**（For 部分）+ **webui.py**（演示）。

## 原理

### 整体流程

```
玩家输入
  → 动态记忆语义检索（Qwen3-Embedding-0.6B 向量相似度）
  → 记忆组装（固定记忆每轮注入 + 检索到的动态记忆）
  → 记忆提示词（数据库事实 → 角色第一人称"你记得：..."）
  → 性格约束（说话风格 + 对话示范）
  → 拼装成 prompt → 文字模型 → 台词提取 → 回存记忆
```

### 三部分拼装

```
system（性格）: 我的性格 + 说话风格 + 对话示范（怎么说话）
user（记忆+对话）:
  [场景] 正在集市摆摊...
  你记得：我不谈论AI；我是艾拉，流浪商人；我想赚钱找妹妹；我的过去...
  玩家对你说："这剑多少钱？"
  艾拉直接回答...
```

| 部分 | 作用 | 来源 |
|---|---|---|
| **性格** | 决定"怎么说话" | persona 表（常驻） |
| **记忆** | 决定"知道什么"（身份/背景/价格/恩怨） | 记忆系统检索+注入 |
| **用户对话** | 决定"回答什么"（当前问题） | 玩家输入 |

### 记忆系统

```
写入：对话 → 事实提炼（memory_extract）
      - 价格只在"问价"时存（砍价是讨论，不产生错误事实）
      - 同物品价格覆盖（只留最新真实价）
      - 出戏/AI味回复不入记忆
      - 去重 + 坏回复跳过
存储：SQLite 分层（persona 表 / facts / state 表）
检索：Qwen3-Embedding-0.6B 语义召回动态记忆（固定记忆每轮直接注入）
读取：记忆提示词（memory_prompt）——事实转第一人称
      - "aila把剑定价为五百金币" → "这把剑我定价五百金币"
      - AI 约束/禁忌类记忆优先注入（出戏防御）
```

### 关键设计

1. **常识不靠记忆教**：砍价不涨价等由模型自身能力处理
2. **记忆是角色视角**：所有记忆转第一人称，模型直接可用
3. **出戏防御**：出戏测试类问题不注入对话示范，改注入"我绝对不是什么AI"
4. **坏记忆不污染**：沉默/兜底/AI味回复不进记忆
5. **示范只教风格**：对话示范让模型学会口吻，不作内容答案

## 组成

| 部分 | 文件 | 职责 |
|---|---|---|
| **记忆系统** | `remember/` | 记忆系统库（核心） |
| **For 部分** | `foremeber.py` | 性格+记忆+用户对话拼装 + 接入文字模型 |
| **引擎管理** | `engine_server.py` | 一键启动/切换对话引擎（Qwen3-4B/1.7B/vanilla） |
| **演示** | `webui.py` | Gradio 对话界面（支持 --offline） |

| `remember/` 文件 | 职责 |
|---|---|
| `memory.py` | SQLite 三层记忆 + 去重 + 价格覆盖 |
| `memory_extract.py` | 对话 → 事实提炼 |
| `memory_prompt.py` | 事实 → 角色第一人称记忆提示词 |
| `qwen3_client.py` | Qwen3-Embedding 检索（默认，CPU 模式） |
| `gte_client.py` / `embedding.py` / `colbert_memory.py` / `encoder_router.py` | 备选检索组件 |

## 快速开始

```bash
# 环境（uv）
uv venv --python 3.11 .venv
uv pip install torch transformers safetensors pytest

# 运行（默认 Qwen3-4B 引擎）
python engine_server.py start qwen3-4b   # 启动对话引擎（或 qwen3-1.7b / vanilla）
bash run.sh chat    # 命令行对话
bash run.sh webui   # WebUI 演示
bash run.sh test    # 测试
```

## 检索模型对比

`tests/compare_embeddings.py` 实测（5 个记忆查询）：

| 模型 | 准确率 |
|---|---|
| **Qwen3-Embedding-0.6B**（默认） | **5/5** |
| GTE-multilingual-base | 4/5 |
| bge-m3 | 3/5 |
| bge-base-zh-v1.5 | 3/5 |
| EmbeddingGemma-300M | 3/5 |

## 使用的模型与许可

| 模型 | 用途 | 来源 | 许可 |
|---|---|---|---|
| **Qwen3-Embedding-0.6B** | 语义检索（默认） | [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Apache-2.0 |
| **GTE-multilingual-base** | 语义检索（备选） | [HuggingFace](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | MIT |
| **LFM2.5-Embedding/ColBERT/Encoder-350M** | 检索组件（备选） | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M) | LFM Open License v1.0 |

**LFM Open License v1.0**（[LICENSE_LFM.md](LICENSE_LFM.md)）要点：
- 非商业/研究免费；年收入 ≥ 1000 万美元的实体商业使用需授权（Section 5）
- 再分发须保留版权声明

> 项目**代码**遵循 MIT；**LFM 模型权重**遵循 LFM Open License v1.0，相互独立。

## 许可证

[MIT License](LICENSE) © 2026 [forhearts](https://github.com/forhearts)
