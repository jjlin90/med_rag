# Medical RAG System — 快速开始指南

本指南帮你最快跑通医疗知识问答系统。更完整的架构与配置说明见 `README.md` 与 `docs/`。

## 前置条件

- Python 3.10（项目用 uv 管理，会自动使用 `.venv`）
- 可选：NVIDIA GPU（显著加速向量化，RTX 40 系推荐）
- 可选：Milvus(19530) / Redis(6379) / MySQL(3306)（后两者未启动会自动降级；Redis 若用本机 Docker 版需设 `REDIS_PASSWORD=1234`）

## 五步跑通

### 1. 安装依赖

```bash
uv sync                 # 推荐：按 uv.lock 一键建 .venv 并安装
# 或：pip install -r requirements.txt
```

### 2.（有 GPU 才做）安装 CUDA 版 PyTorch

```bash
# 先 nvidia-smi 看驱动支持的 CUDA 版本（右上角），选 ≤ 它的 cuXXX
uv pip install "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入 LLM_API_KEY 与 LLM_BASE_URL
# Redis 若用本机 Docker 版（milvus-redis，带 requirepass），需补 REDIS_PASSWORD=1234
```

### 4. 准备模型与数据

- 模型放 `src/models/`：`bge-m3`、`bert-base-chinese`、`bge-reranker-large`。
- 原始 MSD 数据放 `data/raw/MSDZHConsumerMedicalTopics/`，
  运行 `python scripts/extract_msd.py` 生成 `data/clean_md/`。

### 5. 运行

```bash
# 离线入库（一次性；GPU 约 30 分钟，CPU 十余小时）
python scripts/run_offline_ingest.py
# 也可指定目录：python scripts/run_offline_ingest.py --data-dir ./data/clean_md

# 启动在线 API（默认端口 8005，勿直接 python src/online_service/main_api.py）
python scripts/run_api.py
# 或显式指定端口：python scripts/run_api.py --port 8005

# Windows / VS Code 终端启动即崩溃（0xC0000005）时，改用净化环境启动器：
.\run_api_safe.ps1

# 命令行交互问答（EduRAG 式入口，无需前端）
python main.py

# 交互式问答测试
python scripts/test_query_pipeline.py

# Milvus 体检：验证子块过滤是否下推（秒级静态体检；加 --with-search 走端到端）
python scripts/check_chunk_type_filter.py
python scripts/check_chunk_type_filter.py --with-search --query "一型糖尿病和二型糖尿病有什么区别"

# 分层降级策略回归测试（9 项 mock 场景，无需 Milvus）
python scripts/test_degrade_policy.py
```

启动后访问 `http://localhost:8005/docs` 看接口文档。

## 没有模型/数据库？先用简化版

```bash
python scripts/simple_offline_ingest.py   # 不需要向量模型
python scripts/simple_query_test.py       # 不需要 LLM API
```

## 常见问题

| 问题 | 处理 |
|------|------|
| FlagEmbedding 报错 | 需 ≥1.3（新版 API 为直接实例化，代码已适配） |
| CUDA 不可用 | `python -c "import torch;print(torch.cuda.is_available())"`；装 cuXXX 版 torch |
| 显存不足(OOM) | 把 `embedding_provider.py` 和 `run_offline_ingest.py` 的 `batch_size` 从 64 调小到 32 |
| Milvus 连不上 | 确认服务在 19530；离线入库必须先启动 Milvus |
| API 启动报相对导入错 | 用 `python scripts/run_api.py`，勿直接跑 `src/online_service/main_api.py` |
| Redis 健康页显示红 | 本机 Docker Redis 带 requirepass，需在 `.env` 设 `REDIS_PASSWORD=1234`；未配则降级为无缓存 |
| Windows 启动崩溃(0xC0000005) | 原生 DLL 冲突，改用 `.\run_api_safe.ps1` 最小化 PATH 启动 |
| Redis/MySQL 报错 | 可选组件，未启动自动降级，不影响主链路 |

## 训练与评估

```bash
# 构造意图分类训练数据（医疗/通用各 ~2500 条，写入 data/intent_train/）
python scripts/build_intent_data.py

# 微调 BERT 意图分类器（RTX 4060 约 3-4 分钟，保存最佳模型到 src/models/bert_query_classifier）
python scripts/train_intent.py

# RAG 评估（Ragas 四项指标）
python scripts/evaluate_rag.py --static data/test_query/rag_evaluate_data.json   # 静态，无需 Milvus/MySQL
python scripts/evaluate_rag.py                                                    # 实时管线评估（需服务在线）
```

## 查看日志

```bash
# 离线入库日志
tail -f logs/offline_ingest.log
```

---

详细架构见 `docs/architecture.md`，数据合规见 `docs/data_source.md`。
