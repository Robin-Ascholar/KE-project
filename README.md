# Harry Potter LinearRAG

基于 [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) 的课程项目，用 Harry Potter 语料构建关系无关图检索增强生成（Graph RAG）系统。

当前仓库是团队开发的初始版本，已经包含 LinearRAG 核心代码、可复现的环境安装脚本、数据目录约定和本机 GPU 兼容修复。原始书籍、模型权重、API 密钥和运行产物不会提交到 Git。

## 当前状态

- Python 3.9 环境已验证
- PyTorch 2.1.2 + CUDA 12.1 已在 RTX 4060 上验证
- `en_core_web_trf` 3.6.1 已验证
- `all-mpnet-base-v2` 本地向量模型已验证
- Harry Potter 第一册的 PDF 清洗、章节切块、完整索引与检索测试已通过
- 待完成：扩展到其余六册、问题集制作与端到端问答评测

## 目录结构

```text
.
├── Book/                         # 本地原始书籍，不提交
├── dataset/harrypotter/          # 生成的数据，不提交；示例文件除外
├── docs/                         # 数据格式与上游说明
├── figure/                       # LinearRAG 上游示意图
├── model/                        # 本地向量模型，不提交
├── scripts/                      # 环境、模型和运行脚本
├── src/                          # LinearRAG 核心代码
├── .env.example                  # 环境变量示例
├── environment.yml              # Conda 基础环境
├── requirements.txt             # Python 依赖
└── run.py                        # 主入口
```

## 快速开始

### 1. 克隆并安装

Linux + NVIDIA GPU 推荐直接运行：

```bash
git clone https://github.com/Robin-Ascholar/KE-project.git
cd KE-project
bash scripts/setup_environment.sh
conda activate linearrag
```

脚本会创建 Python 3.9 环境、安装 CUDA 12.1 版 PyTorch、项目依赖和 spaCy 模型，并下载 `all-mpnet-base-v2` 到 `model/`。

如果环境已经存在，可以逐步安装：

```bash
conda env create -f environment.yml
conda activate linearrag
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m spacy download en_core_web_trf
python scripts/download_embedding_model.py
```

### 2. 配置 LLM

```bash
cp .env.example .env
```

`run.py` 读取环境变量，不会自动读取 `.env`。运行前请执行：

```bash
export OPENAI_API_KEY="your-api-key"
# 使用 OpenAI 兼容服务时再设置：
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

不要提交 `.env` 或真实密钥。

### 3. 准备本地书籍与数据

将有权使用的 PDF 放到 `Book/`。原始书籍不会上传到公开仓库，团队成员需要从获授权渠道各自取得。

LinearRAG 当前读取：

```text
dataset/harrypotter/chunks.json
dataset/harrypotter/questions.json
```

具体格式见 [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)。

第一册数据可通过脚本生成：

```bash
python scripts/prepare_harry_potter.py
```

该脚本会自动移除重复页眉和页码、识别章节，并生成适配 MPNet 384-token 上限的重叠文本块。生成语料保留在本地，不会提交到 Git。

不调用 LLM API 的建图与检索检查：

```bash
python scripts/index_and_test.py
```

### 4. 运行

```bash
python run.py \
  --spacy_model en_core_web_trf \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name harrypotter \
  --llm_model gpt-4o-mini \
  --max_workers 4 \
  --retrieval_top_k 8
```

首次运行会在 `import/harrypotter/` 建立索引，并把结果写入 `results/harrypotter/`。这两个目录均为本地产物。

## 上游项目与许可

本项目基于 DEEP-PolyU 的 LinearRAG，当前导入基线为提交 `bcc94e66c221f798801255efba09311d6fbcd8d6`。原始说明保存在 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)，修改与归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

由于上游代码采用 GPL-3.0，本仓库中的派生代码同样按 [GPL-3.0](LICENSE) 分发。
