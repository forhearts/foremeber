# Smart Character — 游戏 NPC 角色扮演对话系统

游戏内 NPC 角色扮演对话引擎：端侧可部署、低延迟、强角色一致性。

**基座模型**：Qwen2.5-0.5B-Instruct（0.5B，中文强，端侧可行）+ QLoRA 微调 —— **实测效果最佳**（角色一致 + 中文流畅 + 出戏防御）

**备选基座**：
- **MiniMind-3-MoE**（198M，中文预训练 + SFT）—— 中文流畅但需更多角色数据微调才能入戏
- **MiniMind-3o-MoE**（198M，Omni）—— 英文强但中文 tokenizer 弱，不推荐中文游戏

> MiniMind 出处：[GitHub: jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache-2.0）
> Qwen 出处：Qwen2.5-0.5B-Instruct（Apache-2.0，ModelScope 下载）

## 核心特性

- 🎭 **强角色一致性**：角色卡 + 关键词触发世界书 + 出戏防御
- 🧠 **三层记忆系统**：结构化状态（SQLite）+ 世界书（关键词触发）+ 历史事件（FTS5 混合检索 + 时间衰减）
- 🚀 **端侧友好**：0.5B 模型，INT4 量化后 ~300MB
- 🎮 **游戏系统控制剧情**：意图识别 / 状态判断由游戏代码完成，模型只负责"说"
- ✂️ **后处理管线**：出戏检测、重复截断、长度控制、兜底

## 项目结构

```
smart-character/
├── npc/                    # 核心引擎
│   ├── engine.py           # 对话编排（意图→状态→记忆→生成→后处理）
│   ├── qwen_engine.py      # Qwen 基座引擎（chat template）
│   ├── model.py            # MiniMind 自包含推理引擎（备选）
│   ├── memory.py           # SQLite 三层记忆（FTS5 中文 bigram 检索）
│   ├── character.py        # 角色卡
│   ├── lorebook.py         # 世界书（关键词触发）
│   ├── prompt.py           # Prompt 构建（八股文模板）
│   ├── postprocess.py      # 后处理（出戏/重复/长度/兜底）
│   └── tokenizer_utils.py  # MiniMind tokenizer
├── characters/             # 角色卡 JSON（aila/bruno/kara/...）
├── lorebook/               # 世界书 JSON（狼/月光剑/酒馆/...）
├── data/
│   ├── memory.db           # SQLite 记忆库
│   └── sft/                # SFT 训练数据
├── game/
│   └── demo_game.py        # 村庄场景文字冒险 Demo
├── scripts/
│   ├── download_qwen.py    # 下载 Qwen 权重（ModelScope）
│   ├── download_weights.py # 下载 MiniMind 权重（ModelScope）
│   ├── demo_chat.py        # 命令行对话 Demo
│   ├── generate_sft_data.py# 数据生成（本机 LLM 蒸馏）
│   ├── clean_data.py       # 数据清洗
│   ├── train_qwen.py       # Qwen QLoRA 微调
│   ├── train_sft.py        # MiniMind LoRA 微调
│   ├── evaluate.py         # MiniMind 评估
│   └── eval_qwen.py        # Qwen 评估对比
├── weights/                # 模型权重（下载后）
└── tests/                  # 单元测试
```

## 快速开始

### 1. 环境

```bash
# Python 3.10+，需要 torch / transformers / peft / numpy
pip install torch transformers peft numpy modelscope
```

### 2. 下载 Qwen 基座（~1GB）

```bash
python scripts/download_qwen.py
```

### 3. 对话（推荐本地 14B 引擎）

```bash
# 先启动本地 14B 角色扮演服务（Qwen3-14B 中文角色扮演）
# 运行: D:i-models\start-vanilla-roleplay.cmd

# 🌐 WebUI（默认 vanilla 14B 引擎，效果最佳）
python webui.py

# 命令行对话
python scripts/demo_chat.py --character aila --model qwen  # 0.5B（弱）
python game/demo_game.py --model qwen
```

**引擎选择**（`webui.py --engine X`）：
| 引擎 | 模型 | 质量 | 说明 |
|---|---|---|---|
| **vanilla**（默认） | 本地 14B 角色扮演 | ⭐⭐⭐ | 最佳，需先启动 vanilla 服务 |
| qwen | 本地 0.5B+LoRA | ⭐ | 端侧可用但答非所问 |
| minimind3 | 本地 198M+LoRA | ⭐⭐ | 中文流畅但角色弱 |
| cloud | Kaggle 7B | ⭐⭐⭐ | 需部署 kaggle_npc_api.ipynb |

**WebUI 功能**：
- 8 个 NPC 角色切换（艾拉/布鲁诺/卡拉/奥林/摩根/露娜/维克托/艾尔达）
- 场景选择 + 好感度/信任/任务状态控制（游戏逻辑驱动剧情）
- 多轮对话历史 + 出戏防御测试
- 🧠 记忆系统查看（核心状态 + 最近事件）
- 📖 世界书关键词触发测试

### 4. 微调（可选，提升角色一致性，需 GPU）

```bash
# 生成训练数据（需本机 llama.cpp 服务器 http://127.0.0.1:8080，或用模板数据）
python scripts/generate_sft_data.py --count 300 --out data/sft/rp_data.jsonl --workers 4
python scripts/clean_data.py data/sft/rp_data.jsonl data/sft/rp_data_clean.jsonl

# QLoRA 微调 Qwen
python scripts/train_qwen.py --data data/sft/rp_data_clean.jsonl --epochs 3 --out weights/lora_qwen

# 加载微调模型对话
python scripts/demo_chat.py --model qwen --lora weights/lora_qwen
```

### 5. 评估

```bash
# Qwen 原始 vs LoRA 微调对比
python scripts/eval_qwen.py
```

## 架构设计

```
玩家输入
  ↓
意图识别 / 关键词 / 规则（游戏代码）
  ↓
游戏状态判断（好感度/任务/信任 → NPC 能说什么）
  ↓
记忆系统检索 + 压缩上下文（状态 < 100字, 世界书 < 150字, 历史 < 150字）
  ↓
Qwen 微调模型生成角色化短回复
  ↓
后处理（出戏过滤 / 重复截断 / 长度控制 / 兜底）
  ↓
显示给玩家
```

**核心原则**：剧情逻辑由游戏系统控制，语言表达由 LLM 生成；
记忆系统负责"记"和"找"，模型只负责"看"和"说"。

## 记忆系统（模型无关通用层）

**目标**：任意玩家输入 → 结构化记忆包 → 任意底模（换 lora 即可输出）。

```
玩家输入
  → 规则/Encoder-350M 意图粗筛（哪些记忆需要检索）
  → Embedding-350M 语义检索召回相关事件
  → build_memory_pack() 输出结构化记忆包（状态/场景/意图/记忆/设定）
  → 任意底模消费（vanilla14B / Qwen0.5B / MiniMind3 / 云端7B）
```

**三个 LFM2.5 模型分工（实测）**：
| 模型 | 职责 | 效果 |
|---|---|---|
| Encoder-350M | 意图粗筛（规则未命中时兜底） | 中文短句区分度弱，作辅助 |
| Embedding-350M | 语义检索（主） | 5/6 准确，快 |
| ColBERT-350M | 备选（长记忆/跨语言） | 4/6，慢，留扩展 |

- **核心记忆块**（Core Memory）：常驻结构化状态（好感度/任务/信任）
- **事件存档**（Archival Memory）：FTS5 全文索引 + Embedding 语义
- **混合检索**：语义 + BM25 + 时间衰减 + 去重
- **坏记忆过滤**：兜底/沉默回复不写入、不注入
- **冲突解决**：状态永远优先于记忆

## 测试

```bash
python -m pytest tests/ -v
```

## License

Apache-2.0（模型与代码）

## ☁️ 云端部署（7B 高质量角色扮演）

本地 0.5B/14B 效果有限时，可用 Kaggle 免费 GPU 跑 **Qwen2.5-7B**：

1. 打开 **https://www.kaggle.com** → New Notebook → Settings → Accelerator 选 **GPU T4 x2**
2. 导入 **`kaggle_npc_api.ipynb`**（File → Import Notebook），依次运行 6 个 cell
3. 最后 cell 打印公网 URL（trycloudflare.com），复制它
4. 本机接入：
   ```bash
   python scripts/cloud_engine.py --url https://xxx.trycloudflare.com --test
   python webui.py --engine cloud --cloud_url https://xxx.trycloudflare.com
   ```

详见 `CLOUD_DEPLOY.md`。
