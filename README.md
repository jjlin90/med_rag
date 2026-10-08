# 医疗知识问答系统 (Medical RAG System)

基于 RAG（检索增强生成）的医疗科普知识问答系统，支持**离线知识入库**与**在线智能问答**两条链路。
语料按本地《默沙东诊疗手册（大众版）》目录组织，项目定位为医疗科普技术学习与演示。来源与数据使用说明见 [数据文档](docs/data_source.md)。

## 核心特性

- **双通道问答架构**：①FAQ 快通道优先——先查 Redis 缓存，未命中走 MySQL + jieba BM25，得分经 **softmax 归一化**（阈值 0.85）后才允许直答，该阈值是候选相对分布，不是正确率，已拒绝原始 BM25 零分候选；仍需独立负样本校准；②FAQ 未命中后转入 **RAG 深通道**（意图分类 → 策略选择 → 混合检索 → 重排 → 生成），仅深通道执行一次意图分类，快通道命中不执行分类。
- **混合检索**：BGE-M3 一次前向同时产出**稠密向量**（语义）+**稀疏向量**（词项权重），在 Milvus 中按 sparse 0.7 / dense 1.0 加权融合（WeightedRanker），兼顾语义理解与关键词命中。
- **Small-to-Big 父子分块**：400 字符子块负责精准检索，2000 字符父块负责生成上下文；Top-16 粗排后取 Top-5 子块，按 `parent_id` 聚合，每个父块用最高分命中子块参与 BGE-reranker 精排，最终最多返回2条上下文；父正文缺失可回退子正文，L1使用子片段。
- **子块过滤下推**：新集合使用 `chunk_type == 'child' and parent_id != ''`，旧集合按非空父 ID 过滤。2026-10-08 已备份修正本地 3936 条旧父块标签，父子引用与冗余正文核对一致；在线集合使用 `scripts/check_chunk_type_filter.py --with-search` 单独验收。
- **分层降级策略**：原则是「降级路径必须更安全而非更粗糙」——L0 严格 Small-to-Big（orphan 父块打点剔除）→ L1 同粒度降级（放开过滤重查，命中父块现场切成 400 字子块，粒度不退化）→ L2 默认固定拒答（不调用最终答案生成器；此前策略选择仍可能调用LLM）；降级水位经 `/health` 暴露，返回level>=2的结果不写缓存；Milvus/向量化异常向上传播，任一子查询故障升级 L2；重排故障也标记 L2。
- **设备选择**：向量化/重排按 `torch.cuda.is_available()` 选择 CUDA 或 CPU；CUDA 时启用 fp16。仓库没有可复核的固定耗时基准。
- **LLM 自动选策略**：医疗咨询由大模型自动判断检索策略（直接检索 / HyDE / 子查询 / 回溯抽象），无需用户手动选择。
- **会话存储**：MySQL `conversations` 表按 `session_id` 保存历史；Streamlit 会读取历史恢复界面。服务端生成不会自动读取已保存历史，`/chat` 依赖调用方提交 messages。
- **RAG 评估（Ragas）**：本地 BGE-M3 + OpenAI 兼容裁判接口评估四项指标。当前 210/210 完整报告：Faithfulness 0.8163、Answer Relevancy 0.5007、Context Precision 0.8405、Context Recall 0.7619，等权综合 0.7298；主裁判为 GLM-4.6V，缺失单元由 DeepSeek 补评，报告保留逐项裁判来源。

## 文档导航

| 阅读目的 | 文档 |
|---|---|
| 环境准备与运行 | [快速上手](GETTING_STARTED.md) |
| 当前调用链路与组件职责 | [架构说明](docs/architecture.md) |
| 数据来源、格式支持与存量数据边界 | [数据来源与处理边界](docs/data_source.md) |
| 系统原理与面试复习 | [学习与面试全解](docs/med_rag_学习与面试全解.md) |
| 已修复问题、实现原因与验收依据 | [工程修订与面试详解](docs/20260927_工程修订与面试详解.md) |
| 历史 210 题报告与失败样本分析 | [Ragas 评估与 badcase 分析](docs/Ragas评估与badcase分析.md)（历史评测，不代表本次代码修订后重测） |
| 父子块过滤、降级与迁移边界 | [Small-to-Big 缺陷修复报告](docs/diagrams/Small-to-Big缺陷修复报告.md)（区分当前实现与历史运行记录） |

旧版 [面试全解](docs/med_rag_面试全解.md) 和 [学习指南](docs/rag_learning_guide.md) 保留为合并文档的导航入口，不再分别维护完整正文。本地演示和审查资料存于已忽略的 `artifacts/`，不随源码或 wheel 发布；仓库清理范围见 [仓库卫生与历史清理](docs/repository-hygiene.md)。

## 项目结构

以下列出主要源码、测试、运行入口与发布配置。

```
med_rag/
├── .env.example              # 环境变量模板（LLM_API_KEY / LLM_BASE_URL / REDIS_PASSWORD）
├── .gitattributes            # Windows 批处理脚本的 CRLF 换行规则
├── MANIFEST.in               # 源码包（sdist）的文件包含与排除规则
├── pyproject.toml            # 依赖声明（uv 管理）
├── requirements.txt          # pip直接依赖清单，与pyproject同步
├── uv.lock                   # uv 锁定文件
├── src/
│   ├── config/
│   │   └── settings.py       # 全局配置（路径/分块/检索/设备/模型）
│   ├── offline_pipeline/     # 离线入库流水线
│   │   ├── document_loader.py    # 多格式加载；OCR 接口当前为占位
│   │   ├── data_cleaner.py       # 文本清洗、元数据抽取
│   │   ├── chunk_splitter.py     # 父子分层分块
│   │   ├── embedding_provider.py # BGE-M3 稠密+稀疏向量
│   │   └── milvus_store.py       # Milvus 建集合/索引/混合检索/入库（schema 顶层 chunk_type，自适应过滤）
│   ├── online_service/       # 在线问答服务
│   │   ├── rag_system.py         # RAG 核心编排（EduRAG 对齐六步：意图→策略→检索→重排→上下文→生成）
│   │   ├── main_api.py           # FastAPI 服务（RAGWebAPI 封装，双通道路由/缓存/FAQ/会话/评估/降级指标）
│   │   ├── cache_manager.py      # Redis 缓存（v2；query 使用 SHA-256，FAQ 使用 MD5）
│   │   ├── faq_search.py         # MySQL FAQ + jieba BM25（softmax 归一化，阈值 0.85）
│   │   ├── intent_classifier.py  # BERT 意图识别（general/medical）
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
│   ├── update_pptx_text.py   # 更新本地 PPT（不随仓库分发；用 --pptx 指定文件，支持 --dry-run）
│   └── test_*.py / simple_*.py   # 测试与简化版工具
├── tests/                    # unittest 回归测试
│   ├── test_entrypoints.py   # 启动入口与包级导出兼容性
│   ├── test_regressions.py   # 数据处理、问答与评测边界回归
│   └── test_streamlit_ui.py  # Streamlit 交互与展示回归
├── main.py                   # 命令行交互入口（直接输入问题→RAG 生成，不含学科选择）
├── run_api_safe.ps1          # Windows API 启动器，收窄 PATH 并启用故障追踪
├── data/                     # 数据（git 已屏蔽）
│   ├── raw/                  # 原始 MSD 资源
│   ├── clean_md/             # 清洗后 Markdown（约 2570 篇）
│   ├── split_docs/docs.json  # 20816 条记录（按 parent_id：16880 子块 + 3936 父块）
│   └── test_query/           # 测试集 / 评估数据
├── web_demo/                 # Streamlit 问答与评测界面
├── frontend/                 # Vite 项目演示页面
└── docs/                     # 架构、数据来源、学习与面试、评测分析和仓库核查文档
```

[MANIFEST.in](MANIFEST.in) 管理源码分发包（sdist）的 `include`、`recursive-include`、`prune` 和 `global-exclude`：纳入使用文档、环境变量模板、Windows API 启动器、Streamlit 文件及前端源码与配置；排除本地数据、模型目录、审查产物、虚拟环境、前端依赖、构建输出、缓存与日志。wheel 的 Python 模块与包由 `pyproject.toml` 中的 setuptools 配置声明。[.gitattributes](.gitattributes) 固定 `.bat` 文件使用 CRLF，换行规则的核查记录见 [仓库卫生说明](docs/repository-hygiene.md)。

## 技术栈

| 模块 | 技术 |
|------|------|
| 向量数据库 | Milvus（IVF_FLAT + 稀疏向量，加权混合检索） |
| 向量模型 | BGE-M3（dense 1024 维 + sparse lexical weights） |
| 重排模型 | BGE-reranker-large（FlagReranker） |
| 意图分类 | BERT 中文（bert_query_classifier；本地分类器目录包含模型权重与训练参数文件，二者均不由 Git 跟踪；训练参数文件不等于训练日志） |
| 策略选择 | LLM 自动（direct / hyde / subquery / backtracking） |
| 会话存储 | MySQL(PyMySQL) conversations 表 |
| 缓存/FAQ | Redis + MySQL(PyMySQL) + jieba BM25 |
| RAG 评估 | Ragas 0.2.x（本地 BGE-M3 嵌入 + OpenAI 兼容裁判） |
| LLM | OpenAI 兼容接口，模型由 `.env` 的 `LLM_MODEL_NAME` 指定 |
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

### 2. GPU（可选）

有 NVIDIA 显卡时，安装 CUDA 版 PyTorch 可大幅加速向量化：

```bash
# 示例：安装与本机驱动、Python 和项目锁文件兼容的 CUDA 版 PyTorch
# 具体命令请按 PyTorch 官方安装矩阵选择，不要照抄固定版本。
```

> 用 `nvidia-smi` 和 `python -c "import torch; print(torch.cuda.is_available())"` 核验。无 GPU 时代码回退 CPU。

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入 LLM_API_KEY 与 LLM_BASE_URL
# REDIS_PASSWORD 必须与实际 Redis 配置一致；.env.example 的本地示例值为 1234
```

### 4. 准备模型与数据

- 在线模型放入 `src/models/`：`bge-m3`、`bert_query_classifier`、`bge-reranker-large`。训练分类器还需要 `bert-base-chinese`；仅离线向量化需要 `bge-m3`。
- 原始 MSD 数据放入 `data/raw/MSDZHConsumerMedicalTopics/`，
  运行 `python scripts/extract_msd.py`，默认输出到 `data/clean_md/`；可用 `--input-dir`、`--output-dir` 覆盖路径。

### 5. 启动中间件

- Milvus：`localhost:19530`
- Redis：`localhost:6379`（缓存，可选；密码由 `REDIS_PASSWORD` 指定。`.env.example` 的本地示例值为 1234，必须与实际 Redis 配置一致）
- MySQL：`localhost:3306`（FAQ/会话，可选；不可用时相关能力受限）

### 6. 运行

```bash
# 离线入库（耗时取决于设备、数据与依赖状态）
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

> 推荐使用 `python scripts/run_api.py`；`python -m src.online_service.main_api`
> 和 `python src/online_service/main_api.py` 共用相同启动参数，默认端口均为 8005。
> `python main.py` 是命令行问答，不启动 HTTP 服务。

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
| `GET /` | API 名称与版本 |
| `POST /chat` | 多轮对话（上下文来自请求中的 `messages`；`session_id` 用于保存） |
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

- **向量生成批大小**：离线入口传入 64，`generate_embeddings` 内部也固定为 64；不同设备是否可用需实测。
  持续跑时若显存顶格导致降速，代码已每批自动 `torch.cuda.empty_cache()`。
- **max_length**：代码固定为 2048；是否适合新数据需根据 tokenizer 后长度分布验证。
- **Milvus 索引**：`MILVUS_NLIST` / `MILVUS_NPROBE` 可按数据量调整。

## 故障排查

| 问题 | 排查 |
|------|------|
| 模型加载失败 | 确认 `src/models/` 下模型完整；项目锁定 `flagembedding==1.3.5` |
| CUDA 不可用 | `python -c "import torch;print(torch.cuda.is_available())"`，装 cuXXX 版 torch |
| Milvus 连接失败 | 确认服务在 19530 端口运行 |
| Redis 健康页显示红 | 检查地址和密码；`REDIS_PASSWORD` 必须与实际服务一致，连接失败时降级为无缓存 |
| Windows 启动即崩溃(0xC0000005) | 原生 DLL 冲突：改用 `.\run_api_safe.ps1` 以最小化 PATH 启动；或确保 `.venv\Scripts` 含 VC++ 运行时 DLL（重建 venv 后需重新复制） |
| Redis/MySQL 报错 | 可选组件，未启动会自动降级，不影响主 RAG 链路 |
| API 启动入口 | 推荐 `python scripts/run_api.py`；模块与文件入口共用同一组参数 |

## 免责声明

本项目仅用于**健康科普学习与技术演示**，不构成任何医疗诊断、治疗或用药建议。
如有身体不适，请前往正规医疗机构就诊。

项目使用和分发许可由仓库所有者确认。

## 当前修订与验证入口（2026-10-08）

已修复 DOCX 加载、数值清洗、缓存语义与会话保存、历史消息 timestamp、故障传播及评测统计。本轮补充意图故障、分块失败、完整率和舍入回归，详见 [检查记录](docs/review_20261008.md)。当前准确流程见 [架构说明](docs/architecture.md)，完整讲解与面试追问保留在 [学习与面试全解](docs/med_rag_学习与面试全解.md)，具体修复与验收依据见 [工程修订与面试详解](docs/20260927_工程修订与面试详解.md)。

query 缓存按问题、source_filter、strategy、history 生成 v2 键；只去首尾空白。带历史、来源限制或显式策略的请求跳过 FAQ。`use_cache=false` 同时关闭 query 和 FAQ 缓存。缓存命中也保存本轮会话。API 仍是技术演示，未实现身份鉴权和会话访问授权，不应直接暴露为公网多用户服务。

```bash
# 主环境；Streamlit 为可选演示依赖
uv sync --extra demo
python -m streamlit run web_demo/app.py

# 无外部模型调用的边界回归；Streamlit 未安装时其交互测试会跳过
python -m unittest discover -s tests -v
python scripts/test_quality_optimizations.py
python scripts/test_degrade_policy.py
python scripts/audit_static.py
```

本轮验收记录覆盖回归、静态检查、本地数据、安装包和页面渲染；210题分数保留历史评测标识，当前质量复评使用固定题集另行执行。
