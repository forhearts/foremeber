# ☁️ 云端部署 7B 角色扮演模型（Kaggle 免费）

## 为什么用云
- 本地 14B 是 IQ3_S 3bit 量化，约束遵守差，输出旁白多
- Kaggle 免费 **T4 16GB**（每周 30h），可跑 **Qwen2.5-7B** 全精度 fp16
- 7B 中文能力、指令遵循、角色扮演都远强于 0.5B

## 步骤（约 5 分钟）

### 1. 打开 Kaggle 并登录
浏览器访问 **https://www.kaggle.com** （你的浏览器已登录 Google，点"Sign in with Google"选账号）

### 2. 新建 Notebook
右上角 **New Notebook** → 选 **GPU T4 x2**（免费）
> 如果没 GPU 选项，先在 Settings 里关掉 Internet，再开 GPU（Kaggle 免费版 T4 需要 Internet 开启才能下模型）

### 3. 导入部署代码
菜单 **File → Import Notebook** → 选择本机文件：**`D:\forhearts\smart-character\kaggle_npc_api.ipynb`**

### 4. 依次运行所有 cell
- Cell 1: 装依赖（~1分钟）
- Cell 2: 下载加载 Qwen2.5-7B（~3分钟）
- Cell 3: 测试生成（看到"艾拉: ..."即成功）
- Cell 4: 启动 Gradio
- Cell 5: **打印公网 URL**（形如 `https://xxx.trycloudflare.com`）→ **复制它**

### 5. 本机接入
```bash
# 测试连接
python scripts/cloud_engine.py --url https://xxx.trycloudflare.com --test

# 启动 WebUI（用云端 7B）
python webui.py --engine cloud --cloud_url https://xxx.trycloudflare.com
```

## 备选：Colab
同 Kaggle，但免费 T4 额度更少。导入相同 notebook 即可。

## 备选：Modal
```bash
pip install modal
modal deploy scripts/modal_app.py
```
（需 `modal token new` 登录，免费额度每月 $30）

## 注意
- **保持 Kaggle notebook 运行**（别关页面/中断 cell 5），断开后 URL 失效
- cloudflared 免费隧道每次重启 URL 变化，需重新复制
- 每周免费 30h，够开发测试
