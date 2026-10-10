# Medical RAG System — 快速开始指南

本指南帮你最快跑通医疗知识问答系统。更完整的架构与配置说明见 `README.md` 与 `docs/`。

## 前置条件

- Python 3.10（项目用 uv 管理，会自动使用 `.venv`）
- 运行平台为 Windows/Linux，uv 的解析范围限定为这两类平台；macOS 用户需在 Windows/Linux CUDA 环境运行项目
- NVIDIA GPU 及兼容 CUDA 12.6 的驱动；正式模型计算与意图训练统一使用 CUDA，可用 batch 按显存实测
- 真实 RAG/API 启动需要 Milvus(19530)；Redis(6379) 与 MySQL(3306) 连接失败时相关缓存、FAQ、会话能力会降级

## 五步跑通

### 1. 安装依赖

```bash
uv sync --locked          # 推荐：按 uv.lock 安装；Windows/Linux 采用官方 cu126 索引
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux
```

使用 pip 时先创建并激活 Python 3.10 虚拟环境，再执行：

```bash
python -m pip install "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

pip 不读取 uv 的索引配置，需先单独安装 CUDA wheel。`requirements.txt` 的 `torch==2.13.0` 接受已安装的 `2.13.0+cu126`；其余依赖使用普通索引。uv 安装路径无需额外执行这两步。

### 2. 校验 GPU 环境

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print(torch.cuda.get_device_name(0)); print(torch.ones(4, device='cuda').sum().item())"
python scripts/audit_static.py
```

预期运行版本 `2.13.0+cu126`、CUDA 构建 `12.6`，GPU 张量求和输出 `4.0`。`Config` 会在启动时检查 CUDA 可用性，环境未就绪时明确报错。BGE-M3、BGE-reranker-large 和 BERT 意图模型共享 CUDA 设备配置；向量化、重排及意图训练启用 fp16。

旧环境若混入 CPU wheel，可用已激活环境执行 `python -m pip install --force-reinstall --no-deps "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126`，然后重新执行上述校验。`uv lock --check` 检查声明与锁文件关系，实际安装构建由静态核查和 GPU 张量计算确认。

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入 LLM_API_KEY 与 LLM_BASE_URL
# REDIS_PASSWORD 必须与实际 Redis 服务一致；.env.example 的本地示例值为 1234
```

质量试验可设置 `RETRIEVAL_TITLE_ANCHOR=true` 和 `LLM_GROUNDING_REVIEW=true`：前者读取本地 `data/split_docs/docs.json` 中的真实标题，后者对医学初稿增加一次结构化事实复核。标题补召回失败时告警并保留主检索已成功的资料；主检索故障仍按L2处理。复核失败时返回原文摘录L2/llm_unavailable，不发布初稿。两项默认关闭，可按实测质量与用时选择。固定20题对照及中文AR评测口径见[质量试验记录](docs/quality_pilot_20261010.md)。

### 4. 准备模型与数据

- 模型放 `src/models/`：`bge-m3`、`bert_query_classifier`、`bge-reranker-large`。训练分类器另需 `bert-base-chinese`；离线向量化只需要 `bge-m3`。
- 原始 MSD 数据放 `data/raw/MSDZHConsumerMedicalTopics/`，
  运行 `python scripts/extract_msd.py` 生成 `data/clean_md/`。

### 5. 运行

```bash
# 离线入库（一次性；耗时需按当前设备实测）
python scripts/run_offline_ingest.py
# 也可指定目录：python scripts/run_offline_ingest.py --data-dir ./data/clean_md

# 启动在线 API（默认端口 8005，各入口共用启动参数）
python scripts/run_api.py
# 或显式指定端口：python scripts/run_api.py --port 8005

# Windows / VS Code 终端启动即崩溃（0xC0000005）时，改用净化环境启动器：
.\run_api_safe.ps1

# 命令行交互问答（EduRAG 式入口，无需前端）
python main.py

# 交互式问答测试
python scripts/test_query_pipeline.py

# Milvus 体检：验证子块过滤是否下推（连接集合并检查类型；加 --with-search 验证实际检索）
python scripts/check_chunk_type_filter.py
python scripts/check_chunk_type_filter.py --with-search --query "一型糖尿病和二型糖尿病有什么区别"

# 分层降级策略 mock 测试：构造 Config 时需可用 CUDA GPU，无需 Milvus 或模型权重
python scripts/test_degrade_policy.py
```

启动后访问 `http://localhost:8005/docs` 看接口文档。

## 先验证环境和处理流程

```bash
python -m unittest discover -s tests -v   # 不构造正式 Config 的边界回归，使用 mock
python scripts/audit_static.py            # 源码、链接与安装依赖核查
python scripts/simple_offline_ingest.py    # 简化处理示例
python scripts/simple_query_test.py        # 省略 LLM 生成，仍需本地模型与 Milvus
```

## 常见问题

| 问题 | 处理 |
|------|------|
| FlagEmbedding 报错 | 项目锁定 `flagembedding==1.3.5`，先确认安装版本与本地模型完整 |
| CUDA 不可用 | 执行第二步，核对 NVIDIA 驱动、`torch.__version__`、`torch.version.cuda` 和 GPU 张量；按上文重装 cu126 wheel |
| 显存不足(OOM) | 把 `embedding_provider.py` 和 `run_offline_ingest.py` 的 `batch_size` 从 64 调小到 32 |
| Milvus 连不上 | 确认服务在 19530；离线入库必须先启动 Milvus |
| API 启动入口 | 推荐 `python scripts/run_api.py`；`python -m src.online_service.main_api` 与直接运行该文件也可使用 |
| Redis 健康页显示红 | 检查服务地址和密码；`REDIS_PASSWORD` 必须与实际 Redis 配置一致，失败时降级为无缓存 |
| Windows 启动崩溃(0xC0000005) | 原生 DLL 冲突，改用 `.\run_api_safe.ps1` 最小化 PATH 启动 |
| Redis/MySQL 报错 | 可选组件，未启动自动降级，不影响主链路 |
| MySQL 8 认证提示缺少 cryptography | 依赖清单使用 `pymysql[rsa]==1.1.1`，按安装步骤同步依赖，支持 `caching_sha2_password` / `sha256_password` 认证 |

## 训练与评估

```bash
# 构造意图分类训练数据（医疗/通用各 ~2500 条，写入 data/intent_train/）
python scripts/build_intent_data.py

# 微调 BERT 意图分类器（保存模型到 src/models/bert_query_classifier；耗时需实测）
python scripts/train_intent.py

# RAG 评估（Ragas 四项指标）
python scripts/evaluate_rag.py --static data/test_query/rag_evaluate_data.json   # 需 CUDA/BGE-M3/裁判接口，无需 Milvus/MySQL
python scripts/evaluate_rag.py                                                    # 实时管线评估（需服务在线）
```

## 查看日志

```bash
# 离线入库日志
tail -f logs/offline_ingest.log
```

---

详细架构见 [架构说明](docs/architecture.md)，数据来源与处理边界见 [数据说明](docs/data_source.md)，更多专题见 [README 文档导航](README.md#文档导航)。


## 2026-09-27 补充检查

- LLM_API_KEY、LLM_BASE_URL、LLM_MODEL_NAME 必须属于同一可用供应商；模板已补全模型名和中间件环境变量。
- Streamlit：`uv sync --extra demo` 后执行 `python -m streamlit run web_demo/app.py`。
- 轻量回归：`python -m unittest discover -s tests -v`；静态扫描：`python scripts/audit_static.py`。
- 原始数据抽取可显式传 `--input-dir`、`--output-dir`；不会在 import 时创建输出目录。
- `.doc`、`.ppt` 旧二进制格式需先转换，OCR 仍未实现。DOCX 基本文本读取已通过真实文件回归；复杂表格仍需单独验收。
