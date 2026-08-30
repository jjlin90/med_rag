# 医疗知识问答系统 (Medical RAG System)

基于 RAG（检索增强生成）的医疗科普知识问答系统，支持**离线知识入库**与**在线智能问答**两条链路。
知识库来源于《默沙东诊疗手册（大众版）》公开科普内容，仅用于技术学习与演示。

## 核心特性

- **双通道问答架构**：①FAQ 快通道优先——先查 Redis 缓存，未命中走 MySQL + jieba BM25，得分经 **softmax 归一化**（阈值 0.85）后才允许直答，从机制上杜绝「头痛」误命中「声带息肉」式答非所问；②未命中自动降级 **RAG 深通道**（意图分类 → 策略选择 → 混合检索 → 重排 → 生成），意图分类全程只跑一次。
- **混合检索**：BGE-M3 一次前向同时产出**稠密向量**（语义）+**稀疏向量**（词项权重），在 Milvus 中按 sparse 0.7 / dense 1.0 加权融合（WeightedRanker），兼顾语义理解与关键词命中。
- **Small-to-Big 父子分块**：400 字符子块负责「精准检索」，2000 字符父块负责「给 LLM 完整上下文」；检索链路 Top-16 粗排召回 → Top-5 子块命中 → parent_id 回溯父块去重 → BGE-reranker 精排 Top-2。
- **chunk_type 过滤下推**：schema 顶层 `chunk_type` 字段 + 自适应探测（新库 `chunk_type=='child'`，旧库 `parent_id != ""`），父块不参与召回，20816 条旧数据无需重建集合，召回槽位浪费率 25% → 0。
- **分层降级策略**：原则是「降级路径必须更安全而非更粗糙」——L0 严格 Small-to-Big（orphan 父块打点剔除）→ L1 同粒度降级（放开过滤重查，命中父块现场切成 400 字子块，粒度不退化）→ L2 安全拒答（固定话术，**不调用 LLM**，避免参数知识编造）；降级水位经 `/health` 暴露，基础设施故障的降级结果不写缓存。
- **GPU 加速**：向量化/重排自动使用 CUDA（fp16），离线入库从纯 CPU 的十余小时降到约 30 分钟。
- **LLM 自动选策略**：医疗咨询由大模型自动判断检索策略（直接检索 / HyDE / 子查询 / 回溯抽象），无需用户手动选择。
- **多轮会话持久化**：MySQL `conversations` 表按 `session_id` 留存最近 5 轮，支持跨刷新续聊。
- **RAG 评估（Ragas）**：复用本地 BGE-M3 与 DashScope 跑 faithfulness / answer_relevancy / context_precision / context_recall 四项指标。

## 项目结构

```
med_rag/
├── .env.example              # 环境变量模板（LLM_API_KEY / LLM_BASE_URL / REDIS_PASSWORD）
├── pyproject.toml            # 依赖声明（uv 管理）
├── requirements.txt          # 与 pyproject 同步的依赖清单
├── uv.lock                   # uv 锁定文件
├── src/
│   ├── config/
│   │   └── settings.py       # 全局配置（路径/分块/检索/设备/模型）
│   ├── offline_pipeline/     # 离线入库流水线
│   │   ├── document_loader.py    # 多格式加载 + OCR
│   │   ├── data_cleaner.py       # 文本清洗、元数据抽取
│   │   ├── chunk_splitter.py     # 父子分层分块
│   │   ├── embedding_provider.py # BGE-M3 稠密+稀疏向量
│   │   └── milvus_store.py       # Milvus 建集合/索引/混合检索/入库（schema 顶层 chunk_type，自适应过滤）
│   ├── online_service/       # 在线问答服务
│   │   ├── rag_system.py         # RAG 核心编排（EduRAG 对齐六步：意图→策略→检索→重排→上下文→生成）
│   │   ├── main_api.py           # FastAPI 服务（RAGWebAPI 封装，双通道路由/缓存/FAQ/会话/评估/降级指标）
│   │   ├── cache_manager.py      # Redis 缓存（md5 稳定键）
│   │   ├── faq_search.py         # MySQL FAQ + jieba BM25（softmax 归一化，阈值 0.85）
│   │   ├── intent_classifier.py  # BERT 意图识别（已重训，general/medical）
│   │   ├── strategy_selector.py  # LLM 自动检索策略选择
│   │   ├── query_augmenter.py    # 四种 Query 增强
│   │   ├── retrieval.py          # Milvus 混合检索 + Small-to-Big + 分层降级（L0/L1/L2）
│   │   ├── reranker.py           # BGE-reranker-large 精排
│   │   ├── llm_generator.py      # LLM 生成回答（支持多轮历史）
│   │   ├── conversation_store.py # MySQL 会话历史（conversations 表）
│   │   └── rag_evaluator.py      # Ragas 评估（四项指标）
│   ├── utils/                # logger / common 工具
│   └── models/               # 本地预训练模型（git 已屏蔽权重）
├── scripts/
│   ├── run_offline_ingest.py # 离线入库（主入口，支持 --data-dir）
│   ├── run_api.py            # 启动在线 API（默认端口 8005）
│   ├── train_intent.py       # 微调 BERT 意图分类器
│   ├── build_intent_data.py  # 构造意图训练数据（医疗/通用各 ~2500 条）
│   ├── evaluate_rag.py       # Ragas 评估（--static 静态 / 实时管线）
│   ├── ingest_faq.py         # 导入 FAQ 到 MySQL
│   ├── clean_faq.py          # 清洗 FAQ 数据
│   ├── extract_msd.py        # MSD 原始数据抽取
│   ├── check_chunk_type_filter.py # Milvus 体检：chunk_type 过滤下推验证（--with-search 端到端）
│   ├── test_degrade_policy.py# 分层降级策略 mock 测试（L0/L1/L2 全场景）
│   ├── update_pptx_text.py   # 同步 presentation.pptx 中与代码脱节的表述（支持 --dry-run）
│   └── test_*.py / simple_*.py   # 测试与简化版工具
├── main.py                   # 命令行交互入口（EduRAG 式：选学科→输入问题→RAG 生成）
├── data/                     # 数据（git 已屏蔽）
│   ├── raw/                  # 原始 MSD 资源
│   ├── clean_md/             # 清洗后 Markdown（约 2570 篇）
│   ├── split_docs/           # 分块 JSON（约 20816 块）
│   └── test_query/           # 测试集 / 评估数据
└── docs/                     # architecture.md / data_source.md
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 向量数据库 | Milvus（IVF_FLAT + 稀疏向量，加权混合检索） |
| 向量模型 | BGE-M3（dense 1024 维 + sparse lexical weights） |
| 重排模型 | BGE-reranker-large（FlagReranker） |
| 意图分类 | BERT 中文（bert_query_classifier，已重训） |
| 策略选择 | LLM 自动（direct / hyde / subquery / backtracking） |
| 会话存储 | MySQL(PyMySQL) conversations 表 |
| 缓存/FAQ | Redis + MySQL(PyMySQL) + jieba BM25 |
| RAG 评估 | Ragas 0.2.x（BGE-M3 嵌入 + DashScope LLM） |
| LLM | OpenAI 兼容接口（如阿里云 DashScope） |
| 依赖管理 | uv + pyproject.toml |

## 快速开始

### 1. 环境准备

推荐使用 [uv](https://github.com/astral-sh/uv)（项目已用 uv 管理）：

```bash
# 创建虚拟环境并安装依赖（uv 会自动建 .venv 并按 uv.lock 安装）
uv sync
```

或使用 pip：

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
```

### 2. GPU（可选但强烈推荐）

有 NVIDIA 显卡时，安装 CUDA 版 PyTorch 可大幅加速向量化：

```bash
# RTX 40 系（驱动支持 CUDA 12.6）安装 cu126 版 torch
uv pip install "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126
```

> 用 `nvidia-smi` 右上角 `CUDA Version` 查看驱动支持的最高 CUDA 版本，
> 只要 torch 的 cuXXX ≤ 该值即可。无 GPU 时代码自动回退 CPU，无需改动。

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入 LLM_API_KEY 与 LLM_BASE_URL
# 若使用本机 Docker 里的 Redis（默认带 requirepass），还需填 REDIS_PASSWORD=1234
```

### 4. 准备模型与数据

- 模型放入 `src/models/`：`bge-m3`、`bert-base-chinese`、`bge-reranker-large`。
- 原始 MSD 数据放入 `data/raw/MSDZHConsumerMedicalTopics/`，
  用 `python scripts/extract_msd.py` 抽取、清洗到 `data/clean_md/`。

### 5. 启动中间件

- Milvus：`localhost:19530`
- Redis：`localhost:6379`（缓存，可选；本机通常用 Docker 容器 `milvus-redis`，启动带 `--requirepass 1234`，需在 `.env` 设 `REDIS_PASSWORD=1234` 才能连通，否则健康页 Redis 显示红、自动降级为无缓存模式）
- MySQL：`localhost:3306`（可选，FAQ/会话，未启动会自动降级）

### 6. 运行

```bash
# 离线入库（一次性，GPU 上约 30 分钟）
python scripts/run_offline_ingest.py
# 也可指定目录：python scripts/run_offline_ingest.py --data-dir ./data/clean_md

# 启动在线问答 API（默认端口 8005）
python scripts/run_api.py
# 或显式指定：python scripts/run_api.py --port 8005

# Windows / VS Code 终端若启动即崩溃（0xC0000005 原生库 segfault），改用净化环境启动器：
.\run_api_safe.ps1            # 收窄 PATH、规避多 Python 环境 DLL 冲突，并开启 faulthandler

# 命令行交互问答（EduRAG 式入口，无需前端）
python main.py

# 交互式问答测试
python scripts/test_query_pipeline.py
```

> ⚠️ 不要直接 `python src/online_service/main_api.py`——它用相对导入，
> 直接跑会报包导入错误。请用 `scripts/run_api.py` 或新建的 `python main.py`。

启动后访问 `http://localhost:8005/docs` 查看接口文档。

## API 示例

```python
import requests
resp = requests.post("http://localhost:8005/query", json={
    "question": "什么是高血压？",
    # strategy 可省略，省略时由 LLM 自动判断（direct/hyde/subquery/backtracking）
    "strategy": "direct"
})
print(resp.json()["answer"])
```

主要接口：

| 接口 | 说明 |
|------|------|
| `POST /query` | 单轮问答（含 FAQ/缓存/意图/策略/检索/生成全流程） |
| `POST /chat` | 多轮对话（携带 `session_id` 续聊） |
| `GET /health` | 健康检查（含各组件状态与降级水位指标） |
| `GET /stats` | 系统统计 |
| `GET /available_strategies` | 可用检索策略列表 |
| `GET /intent_example` | 意图分类示例 |
| `GET /conversation/{session_id}` | 获取某会话历史 |
| `DELETE /conversation/{session_id}` | 清空某会话历史 |
| `POST /evaluate` | RAG 评估（Ragas 四项指标） |

## 关键配置（src/config/settings.py）

```python
# 父子分块
PARENT_CHUNK_SIZE = 2000   # 父块（上下文）
CHILD_CHUNK_SIZE  = 400    # 子块（检索）
CHUNK_OVERLAP     = 60

# 混合检索加权融合（WeightedRanker）
SPARSE_WEIGHT = 0.7        # 稀疏（词权）权重，侧重关键词精确命中
DENSE_WEIGHT  = 1.0        # 稠密（语义）权重，侧重语义相似

# 检索与重排（Small-to-Big 链路）
TOP_K_RETRIEVE = 16        # 混合检索粗排召回量
TOP_K_CHILDREN = 5         # 子块召回数（精细定位）
TOP_K_RERANK   = 2         # CrossEncoder 精排最终输出（Top-2 父块）

# FAQ 快通道
FAQ_NORMALIZED_THRESHOLD = 0.85   # BM25 softmax 归一化后的命中阈值

# 分层降级策略
ENABLE_CHILD_FILTER_FALLBACK = True   # L1：同粒度降级（父块现场切成子块）
ALLOW_LLM_WHEN_NO_CONTEXT    = False  # L2：检索为空时安全拒答，不让 LLM 硬答
DEGRADE_ALERT_LEVEL          = 1      # 降级水位告警阈值

# 设备：自动检测，有 CUDA 用 cuda，否则 cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 性能与调优

- **向量生成批大小**：`embedding_provider.py` 与 `run_offline_ingest.py` 中 `batch_size`（默认 64，8G 显存实测安全）。
  持续跑时若显存顶格导致降速，代码已每批自动 `torch.cuda.empty_cache()`。
- **max_length**：默认 2048（块最长约 1300 token），勿盲目调大以免浪费显存。
- **Milvus 索引**：`MILVUS_NLIST` / `MILVUS_NPROBE` 可按数据量调整。

## 故障排查

| 问题 | 排查 |
|------|------|
| 模型加载失败 | 确认 `src/models/` 下模型完整；FlagEmbedding 需 ≥1.3（API 已适配） |
| CUDA 不可用 | `python -c "import torch;print(torch.cuda.is_available())"`，装 cuXXX 版 torch |
| Milvus 连接失败 | 确认服务在 19530 端口运行 |
| Redis 健康页显示红 | 多为密码未配：本机 Docker Redis 带 `requirepass`，需在 `.env` 设 `REDIS_PASSWORD=1234`；未配则自动降级为无缓存 |
| Windows 启动即崩溃(0xC0000005) | 原生 DLL 冲突：改用 `.\run_api_safe.ps1` 以最小化 PATH 启动；或确保 `.venv\Scripts` 含 VC++ 运行时 DLL（重建 venv 后需重新复制） |
| Redis/MySQL 报错 | 可选组件，未启动会自动降级，不影响主 RAG 链路 |
| API 启动报相对导入错 | 用 `python scripts/run_api.py`，勿直接跑 main_api.py |

## 免责声明

本项目仅用于**健康科普学习与技术演示**，不构成任何医疗诊断、治疗或用药建议。
如有身体不适，请前往正规医疗机构就诊。

## 许可证

MIT License
