# med_rag 项目详解与 RAG 学习指南

> 本文目标：帮你**边读项目边学 RAG**。前半部分科普 RAG 的核心概念，后半部分对照 `med_rag` 代码逐模块拆解"它到底是怎么做的、为什么这样做"。
> 适合人群：刚接触 RAG、想通过一个真实项目理解端到端流程的开发者。
> 免责声明：本项目知识库来自《默沙东诊疗手册（大众版）》公开科普内容，**仅用于技术学习与演示，不构成任何医疗建议**。

---

## 目录

1. [先搞懂：什么是 RAG](#1-先搞懂什么是-rag)
2. [med_rag 项目全景](#2-med_rag-项目全景)
3. [离线链路：如何把"书"变成"可检索的知识库"](#3-离线链路知识库构建)
4. [在线链路：一个问题是怎么变成答案的（六步主流程）](#4-在线链路问答生成六步主流程)
5. [四种检索策略详解](#5-四种检索策略详解)
6. [核心知识点深挖（RAG 学习重点）](#6-核心知识点深挖rag-学习重点)
7. [RAG 质量评估：Ragas 四指标](#7-rag-质量评估ragas-四指标)
8. [上手实践：把它跑起来 + 学习练习](#8-上手实践把它跑起来--学习练习)
9. [关键配置参数速查](#9-关键配置参数速查)
10. [术语表](#10-术语表)

---

## 1. 先搞懂：什么是 RAG

### 1.1 RAG 是什么

**RAG（Retrieval-Augmented Generation，检索增强生成）** = 在让大模型（LLM）生成答案之前，先**从外部知识库里检索出相关材料**，再把"材料 + 问题"一起交给 LLM 生成回答。

一句话记忆：**先查资料，再作答**。

### 1.2 为什么需要 RAG（而不是只靠 LLM）

LLM 本身有三大硬伤，RAG 恰好能补：

| 问题 | 说明 | RAG 的解法 |
|------|------|------------|
| **幻觉（编造）** | 模型不知道时就"一本正经地编" | 强制要求"基于检索到的上下文回答"，无依据就说不知道 |
| **知识过期** | 模型训练数据有截止时间 | 知识库可随时更新，模型无需重新训练 |
| **领域知识缺失** | 通用模型不懂你公司的/专业的私有资料 | 把私有知识入库，检索后喂给模型 |

> 在医疗这种**容错率极低、且强依赖权威资料**的场景，RAG 几乎是必选项——你不能让模型"凭记忆开药"。

### 1.3 RAG 的两个阶段

RAG 工程上必然拆成两个阶段，对应本项目的两条链路：

```
【离线阶段 / Indexing（建库）】
   原始文档 → 清洗 → 切片(分块) → 向量化 → 存入向量库
   目标：把"人能读的文章"变成"机器能秒级检索的向量索引"
   本项目代码：src/offline_pipeline/*  +  python main.py --data-processing

【在线阶段 / Generation（问答）】
   用户提问 → 检索相关块 → 重排序 → 组装上下文 → LLM 生成
   目标：用最快的路径，把"最相关的资料"塞进模型，产出靠谱答案
   本项目代码：src/online_service/rag_system.py 的 RAGSystem.generate()
```

### 1.4 一个最小 RAG 的伪代码

读懂下面 10 行，就懂了 RAG 的骨架：

```python
# 离线（一次性）
chunks = split(document)              # 1. 把文档切成小段
vectors = embed(chunks)               # 2. 每段变成向量
vector_db.insert(chunks, vectors)     # 3. 入库

# 在线（每次提问）
query_vec = embed(question)           # 4. 把问题也变成向量
hits = vector_db.search(query_vec, k=4)  # 5. 找最相似的 4 段
context = "\n".join(hits)             # 6. 拼成上下文
answer = llm(f"根据资料回答：{context}\n问题：{question}")  # 7. 生成
```

**med_rag 做的事情，就是把上面 7 步里的每一步都做"厚"了**：更聪明的分块、更好的向量、混合检索、重排序、意图分流、自动选策略、多轮会话……下面逐一展开。

---

## 2. med_rag 项目全景

### 2.1 项目在做什么

一个基于 RAG 的**医疗科普问答系统**：

- 知识库来源：《默沙东诊疗手册（大众版）》公开科普内容（清洗后约 **2570 篇** Markdown，切成约 **20816 个块**）。
- 能力：用户用自然语言问健康问题 → 系统检索权威资料 → 生成严谨、可溯源的科普回答。
- 两条链路：**离线入库**（建知识库）与**在线问答**（用知识库回答）。

### 2.2 技术栈总览

| 模块 | 技术选型 | 在本项目中的作用 |
|------|----------|------------------|
| 向量数据库 | **Milvus** | 存储稠密+稀疏向量，支持混合检索（端口 19530） |
| 向量模型 | **BGE-M3** | 一次前向同时产出稠密向量(1024维) + 稀疏词权重 |
| 重排模型 | **BGE-reranker-large** | 交叉编码精排，过滤噪声 |
| 意图分类 | **BERT（bert-base-chinese 微调）** | 判断问题属于"通用知识"还是"医疗咨询" |
| 策略选择 | **LLM 自动判断** | 在 direct/hyde/subquery/backtracking 中选检索策略 |
| 会话存储 | **MySQL（PyMySQL）** | 按 session_id 存最近 5 轮对话（端口 3306） |
| 缓存 / FAQ | **Redis + MySQL + jieba BM25** | 高频问题直答，Redis 未启动自动降级（端口 6379） |
| RAG 评估 | **Ragas 0.2.x** | 跑 faithfulness / answer_relevancy / context_precision / context_recall |
| 生成模型 | **OpenAI 兼容接口**（模型由环境变量配置） | 最终答案生成 |
| 依赖管理 | **uv** + pyproject.toml | 虚拟环境与依赖锁定 |

### 2.3 目录结构导览（带"它是干嘛的"）

```
med_rag/
├── src/
│   ├── config/settings.py        # 全局配置：路径、分块大小、检索权重、模型名、设备
│   ├── offline_pipeline/         # 离线入库流水线（建库）
│   │   ├── document_loader.py    #   多格式加载（HTML/MD）
│   │   ├── data_cleaner.py       #   文本清洗、去噪声
│   │   ├── chunk_splitter.py     #   父子分层分块（核心优化点）
│   │   ├── embedding_provider.py #   BGE-M3 稠密+稀疏向量
│   │   └── milvus_store.py       #   Milvus 建集合/索引/混合检索/入库
│   ├── online_service/           # 在线问答服务（问答）
│   │   ├── rag_system.py         #   ★RAG 核心编排（六步主流程）
│   │   ├── main_api.py           #   FastAPI 服务（缓存/FAQ/会话/评估 外壳）
│   │   ├── cache_manager.py      #   Redis 缓存
│   │   ├── faq_search.py         #   MySQL FAQ + jieba BM25（一级/二级缓存）
│   │   ├── intent_classifier.py  #   BERT 意图分类（general / medical）
│   │   ├── strategy_selector.py  #   LLM 自动选检索策略
│   │   ├── query_augmenter.py    #   四种 Query 增强（direct/hyde/subquery/backtracking）
│   │   ├── retrieval.py          #   Milvus 混合检索 + 多查询合并
│   │   ├── reranker.py           #   BGE-reranker 精排
│   │   ├── llm_generator.py      #   LLM 调用 + 医疗 Prompt 组装
│   │   ├── conversation_store.py #   MySQL 会话历史
│   │   └── rag_evaluator.py      #   Ragas 评估
│   └── utils/                    # logger / common 工具
├── scripts/                      # 可执行脚本入口
│   ├── run_offline_ingest.py     #   离线入库主入口（支持 --data-dir）
│   ├── run_api.py                #   启动在线 API（端口 8005）
│   ├── train_intent.py           #   微调 BERT 意图分类器
│   ├── build_intent_data.py      #   构造意图训练数据
│   ├── evaluate_rag.py           #   Ragas 评估脚本
│   └── ...
├── main.py                       # 命令行交互入口（EduRAG 式：选模式→问答）
├── web_demo/                     # Streamlit 前端（端口 8501）
└── data/                         # 数据（git 已屏蔽）
    ├── clean_md/                 #   清洗后 Markdown（约 2570 篇）
    ├── split_docs/docs.json      #   分块结果（20816 块；现存文件需按 parent_id 判父子）
    └── test_query/               #   测试/评估数据
```

> 代码阅读顺序建议：**先读 `src/online_service/rag_system.py`**（在线主流程一目了然），再回头读 `offline_pipeline` 理解"库是怎么来的"。

---

## 3. 离线链路（知识库构建）

目标：把 2570 篇 Markdown 医疗文章，变成 Milvus 里能被秒级检索的"向量块"。

### 3.1 数据清洗（document_loader + data_cleaner）

- 剔除网页噪声：导航栏、页脚、广告、版权、作者信息、自测题、外链等。
- 结构化重构：统一标题层级、规范段落、去除无效短句。
- 产出：干净、结构统一的 Markdown，存 `data/clean_md/`。

> 学习点：**RAG 质量上限由数据质量决定**。再好的检索也救不回一堆噪声文档。清洗是性价比最高的一步。

### 3.2 父子分层分块（chunk_splitter.py）— 核心优化点

**问题**：分块（Chunking）是 RAG 最关键的超参数之一。
- 块太大 → 检索"糙"（一段里只有一句话相关，却把整段塞给模型，引入噪声）。
- 块太小 → 检索"准"但上下文断裂（模型拿到一句，却不知道它在讲什么病）。

**本项目的做法：父子双层分块（Parent-Child Chunking）**

```
一篇文章
  └─ 切成 父块（~2000 字，保留完整上下文，chunk_type='parent'）
        └─ 每个父块再切成 子块（~400 字，粒度细，chunk_type='child'）
```

- **子块（400 字）**：用于向量检索，命中精准。
- **父块（2000 字）**：检索命中子块后，通过 `parent_id` 回溯到父块，把**父块的完整正文**当作上下文喂给 LLM。

> 一句话：**用子块做"精准检索"，用父块做"完整上下文"**，兼顾精度与完整性。
> 代码里 `chunk_splitter.py` 用 `RecursiveCharacterTextSplitter`（langchain），按 `\n\n`→`\n`→`。！？；：` 等中文标点优先级切分，重叠 60 字符防割裂。

### 3.3 双模态向量化（embedding_provider.py）— BGE-M3

每个块（实际用子块）送入 **BGE-M3**，一次前向同时产出两种向量：

- **稠密向量（dense, 1024 维）**：编码深层语义。擅长"口语化、模糊、同义改写"的召回（如"血压高怎么办" ↔ "高血压的处理"）。
- **稀疏向量（sparse, lexical weights）**：BGE-M3 同时输出的词项权重 `{token_id: 权重}`，类似 BM25 但由模型学习得到。擅长**专业关键词精确命中**（如"胰岛素""糖化血红蛋白"）。

> 学习点：这就是**混合检索（Hybrid Search）**的基础——一条查询同时走"语义"和"关键词"两路，互补长短。

关键工程细节（来自代码注释，值得记）：
- BGE-M3 约 2.3GB 权重；GPU 上开 `fp16` 降到约 1.1GB，能塞进 8G 显存并提速；CPU 不支持 fp16，保持 fp32。
- 批大小 `batch_size=64`、长文本 `max_length=2048`，每批后 `torch.cuda.empty_cache()` 防显存碎片（8G 显存实测易顶格）。

### 3.4 入库与混合索引（milvus_store.py）

集合 schema 设计（Milvus 一张表）：

| 字段 | 类型 | 作用 |
|------|------|------|
| `id` | VARCHAR(主键) | 块唯一 ID |
| `text` | VARCHAR | 块正文（子块用于检索、父块用于上下文） |
| `dense_vector` | FLOAT_VECTOR(1024) | 稠密向量，IVF_FLAT 索引 |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | 稀疏向量，倒排索引 |
| `parent_id` | VARCHAR | 所属父块 ID（子块回溯用；父块统一为空串 `""`） |
| `parent_content` | VARCHAR | 冗余存父块正文（避免二次查询） |
| `chunk_type` | VARCHAR | **`"child"` / `"parent"`**——检索过滤下推的关键字段 |
| `source` | VARCHAR | 来源/科室（可按来源过滤） |
| `metadata` | JSON | 标题、关键词等附加信息 |

- dense 索引：`IVF_FLAT`，`nlist=256`，`metric_type=IP`（内积）。
- sparse 索引：`SPARSE_INVERTED_INDEX`，`drop_ratio_build=0.2`。
- 混合检索：`MilvusClient.hybrid_search` 对两路分别 ANN 检索，再用 `WeightedRanker` 按 `SPARSE_WEIGHT=0.7 / DENSE_WEIGHT=1.0` 加权融合、去重。
- **检索过滤下推**：新 schema 使用 `chunk_type == "child"`；旧 schema 无该列时使用 `parent_id != ""`。注意现存 `docs.json` 是修复前产物，顶层 `chunk_type` 全为 `child`，只能按 `parent_id` 判断逻辑父子；若用新 schema 重新入库，应先重生成分块文件。当前审查时 Milvus 未运行，线上 collection 需用诊断脚本验证。

---

## 4. 在线链路（问答生成，六步主流程）

核心编排在 `src/online_service/rag_system.py` 的 `RAGSystem.generate()`。
注意：**纯 RAG 的六步是从"意图分类"开始的**；而"FAQ/缓存优先"是 `main_api.py` 在外层套的加速壳，命中就直接返回，不进 RAG。

### 4.0 外层：双通道架构（main_api.py）

本项目**不再用意图预判决定走不走 FAQ**，而是改成「快通道优先、深通道兜底」的双通道：

```
用户提问
  │
  ├─【通道① FAQ 快通道】所有问题都先过这一关（不再做意图预判）
  │      → Redis 缓存命中 → 直接返回
  │      → 未命中 → MySQL + jieba BM25 检索
  │            → BM25 原始分（~7 量级）经 softmax 归一化到 (0,1]
  │            → 归一化分 ≥ 0.85 → 命中直答（写回 Redis）
  │
  └─【通道② RAG 深通道】快通道未命中时自动降级进来
            → ① 意图分类(BERT)  ② 策略选择(LLM)
            → ③ Small-to-Big 检索  ④ 重排  ⑤ 构建上下文  ⑥ LLM 生成
```

> 学习点 1：**缓存是 RAG 系统的"性价比外挂"**。高频重复问题没必要每次都跑全套检索+LLM，命中即答，省时省钱。
>
> 学习点 2（重要）：**为什么必须做 softmax 归一化？** BM25 的原始分数量纲不稳定——
> 「高血压吃什么药」可能得 7.2 分，「头痛」和「声带息肉」可能得 6.8 分，
> 想靠一个固定阈值（曾经用 0.5）区分"命中/未命中"根本不可能，于是出现「头痛」误答「声带息肉」。
> 归一化把分数压到 (0,1] 且按候选集内相对强弱分布，阈值 0.85 才真正有意义（相当于"明显比其它候选强"）。
> 这就是**先修量纲、再定阈值**的工程思路——比加一个"关键词重叠守卫"的补丁优雅得多。
>
> 学习点 3：**双通道取代了意图预判**。原来的做法是"医疗问题跳过 FAQ"，缺点是必须先跑一次 BERT
> 才能决定路由；现在所有问题统一先过快通道（成本极低），未命中才降级深通道，**意图分类全程只跑一次**。

### 4.1 第一步：意图分类（intent_classifier.py，BERT）

轻量 **BERT 中文分类器**判断问题类型：

- `general`（通用知识，label=0）：如"什么是人工智能""怎么学英语" → **不查医学库，直接让 LLM 回答**。
- `medical`（医疗咨询，label=1）：如"高血压吃什么""宝宝发烧怎么办" → 进入 RAG 检索链路。

代码逻辑（`rag_system.py`）：

```python
intent_result = self.intent_classifier.predict(query)
intent = intent_result.get("intent", "medical")
if intent == "general":                      # 通用知识：跳过检索
    answer = self.llm_generator.generate_with_context(query, context="")
    return {... "strategy": "direct", "sources": []}
# 否则走 medical 的检索增强流程
```

> 为什么用 BERT 而不是 LLM 做意图分类？见 [6.4](#64-为什么意图分类用-bert-而不是-llm)。

### 4.2 第二步：策略选择（strategy_selector.py，LLM 自动选）

医疗问题不再"一刀切"直接用原问题检索，而是由 **LLM 自动判断用哪种检索策略**（direct/hyde/subquery/backtracking，详见第 5 节）。

```python
strategy = self.strategy_selector.select_strategy(query)   # LLM 返回 direct/hyde/subquery/backtracking
```

LLM 返回后经 `_normalize_strategy` 清洗（兼容中文策略名、JSON 包裹等情况），兜底为 `direct`。

> 学习点：这是 RAG 从"能用"到"好用"的关键——**不同问题适合不同检索方式**。复杂/抽象问题用原问题直接搜往往搜不准，需要"改写查询"再搜。

### 4.3 第三步：检索与合并（retrieval.py + query_augmenter.py）

按策略执行检索：

```python
retrieval_results = self._retrieve_and_merge(query, source_filter, strategy)
```

- `direct` → 直接拿原问题检索。
- `hyde` → 先让 LLM 生成一段"假设答案"，用假设答案去检索（见 5.2）。
- `subquery` → 把复杂问题拆成多个子问题，分别检索后合并去重（`search_multi_queries`）。
- `backtracking` → 把具体问题抽象成更基础的问题再检索。

底层统一走 `Retrieval.search_child_to_parent()`（**Small-to-Big**，见 6.1）：

```python
# retrieval.py —— 检索到父块回填的完整链路
children = hybrid_search(query, filter='chunk_type == "child"', top_k=TOP_K_RETRIEVE)  # Top-16 粗排
children = children[:TOP_K_CHILDREN]                                                    # Top-5 子块
parents  = dedup_by_parent_id(children)                                                 # parent_id 回溯去重
```

即：BGE-M3 把查询变成 dense+sparse 向量 → Milvus `hybrid_search` 加权融合（**且过滤下推只召回子块**）
→ **Top-16 粗排** → 取 **Top-5 子块** → 按 `parent_id` 回溯父块并去重 → 得到候选父块列表。

### 4.4 第四步：重排序（reranker.py，BGE-reranker）

粗排得到的候选父块里可能有噪声。系统保留每个父块的最佳命中子块为 `rerank_content`，用 **BGE-reranker-large（交叉编码 CrossEncoder）** 对该子块证据打分：

```python
reranked_results = self.reranker.rerank(query, retrieval_results, top_k=self.config.TOP_K_RERANK)
```

- 把 `(query, 候选文档)` 成对送入模型做**交互式编码**，相关性判断更准。
- 按子块证据分数排序，但返回对应的**完整父块**；取 Top-2（`TOP_K_RERANK=2`）喂给 LLM。
- 模型不可用时**优雅降级**：按原顺序返回前 2 条，不阻塞主流程。
- 注意：重排内部会调用 `documents.sort()` 原地排序，因此**传入的必须是普通 list**（`RetrievalResult.documents`），
  不能直接把 `RetrievalResult` 包装对象传进去。

> 为什么不直接用重排代替检索？见 [6.3](#63-双塔-bi-encoder-vs-交叉编码-cross-encoder)。

### 4.5 第五步：构建上下文（rag_system.py `_build_context`）

把精排后的文档拼成带编号的上下文：

```python
context = "【知识来源1】\n...父块正文...\n【知识来源2】\n..."
```

- 检索命中子块，但返回的是 `parent_content`（父块完整正文），保证 LLM 看到完整语境。
- **检索为空时返回固定安全拒答话术，并且不再调用 LLM**——详见 4.9 分层降级策略。
  （旧版是"让 LLM 走兜底回答"，这是医疗场景的风险行为，已废弃。）

### 4.6 第六步：LLM 生成（llm_generator.py）

用**医疗专属系统 Prompt**约束生成：

```
你是一个专业的医疗智能助手，基于提供的医学知识回答用户问题。
1. 基于提供的上下文内容回答，不要编造信息
2. 回答要准确、专业、易于理解
3. 如果信息不足，明确说明
4. 对于医疗问题，强调仅供参考，不能替代专业医疗建议
```

- 支持多轮历史（取最近 3 轮拼进提示）。
- 温度 `0.2`（低随机，保证严谨）。
- 最终返回 `{answer, intent, strategy, sources, confidence}`。

### 4.7 多轮会话（conversation_store.py）

- 对话历史按 `session_id` 存 MySQL `conversations` 表，保留最近 5 轮。
- 未命中/MySQL 不可用时降级为内存态，不影响单轮问答。

### 4.8 在线主流程一图流（双通道版）

```
用户提问
  │
  ├─【通道① FAQ 快通道】所有问题先过这一关，无意图预判
  │        ├─ Redis 缓存命中 → 直答
  │        ├─ MySQL + jieba BM25 → softmax 归一化 ≥ 0.85 → 命中直答（写回 Redis）
  │        └─ 未命中 ↓
  │
  └─【通道② RAG 深通道】(rag_system.py)
          ① 意图分类(BERT) ── general ──→ 直接 LLM（不检索）
                              │
                             medical
                              ↓
          ② 策略选择(LLM) ──→ direct / hyde / subquery / backtracking
                              ↓
          ③ Small-to-Big 检索：混合检索(过滤下推只召回子块) Top-16
                              → Top-5 子块 → parent_id 回溯父块去重
                              ↓
          ④ 重排序(BGE-reranker 对最佳命中子块打分，返回 Top-2 父块)
                              ↓
          ⑤ 构建上下文(父块正文 + 编号，水位为 L0/L1/L2)
                              ↓
          ⑥ LLM 生成(医疗 Prompt + 多轮历史)
                              ↓
          返回 {answer, intent, strategy, sources, degraded, degraded_reason}
```

### 4.9 分层降级策略（retrieval.py）—— 医疗场景的可靠性底线

检索不是每次都有结果。项目的原则是：**降级路径必须比主路径更安全，而不是更粗糙。**

| 层级 | 触发条件 | 行为 | 输出粒度 |
|------|----------|------|----------|
| **L0** | 正常 | 严格 Small-to-Big：`chunk_type` 过滤下推，只召回子块；命中却找不到父块的 orphan 记录 ERROR 并剔除 | 400 字子块 → 2000 字父块 |
| **L1** | 子块召回为空 | **同粒度降级**：放开子块过滤重查；若命中父块，用 `_split_parent_in_memory()` **现场切成 400 字子块**再返回（切分分隔符与离线建库共享同一常量，保证线上线下同构）；有候选上限防膨胀 | **仍是 400 字子块**（粒度不退化） |
| **L2** | L1 仍为空 / 基础设施故障 | **安全拒答**：返回固定话术，**不调用 LLM** | 无上下文 |

**三个必须理解的设计取舍：**

1. **为什么 L1 不直接返回 2000 字父块？** 父块稀释 LLM 注意力、token 翻倍，而且掩盖了"这块数据为什么没被切出子块"的真相。
   L1 的降级是**换检索入口（父块层→子块层）**，不是**降输出质量**。
2. **为什么 L2 不调 LLM？** 医疗场景下，检索不到依据时让 LLM 用参数知识硬答，是整条链路上**风险最高的行为**——
   它会编造，而且语气权威，用户分不清。宁可拒答。
3. **为什么区分"召回为空"和"基础设施故障"？** 二者都是 L2，但**故障导致的降级结果不写 Redis 缓存**
   （否则故障期间的空结果会被缓存污染后续请求），且这两种情况在 `/health` 的降级指标里分开计数。

降级水位通过 `RetrievalResult`（Sequence 代理，含 `.documents` / `.watermark` / `.reason` / `.failure`）在链路上传播，
最终体现在 `/query` 响应的 `degraded` / `degraded_reason` / `degraded_level` 三个字段上。
多路子查询合并时取**最差水位**（木桶效应，不允许"一路降级、一路正常"被平均成正常）。

> 想验证？跑 `python scripts/test_degrade_policy.py`（9 项 mock 场景全覆盖）。

---

## 5. 四种检索策略详解

为什么需要多种策略？因为"问法"千差万别，固定用原问题检索会漏检。本项目让 LLM 自动选，也可手动指定 `strategy` 参数。

| 策略 | 中文 | 做法 | 适合的问题 | 例子 |
|------|------|------|------------|------|
| `direct` | 直接检索 | 用原问题直接检索 | 意图明确、表述清晰的单点问题 | "高血压的正常范围是多少？" |
| `hyde` | 假设问题检索 | 先让 LLM 写一段"假设答案"，用这段答案去检索 | 抽象、需推理、原问题太短不好搜 | "糖尿病不控制会有什么后果？" |
| `subquery` | 子查询检索 | 把复杂问题拆成多个子问题，分别检索后合并去重 | 涉及多个实体/多方面的对比类问题 | "比较1型与2型糖尿病的病因与治疗" |
| `backtracking` | 回溯/抽象检索 | 把具体的个人化问题，抽象成更基础、更易检索的通用问题 | 带个人背景、表述冗长具体的问题 | "我家族史+口渴多尿，是不是糖尿病？"→"糖尿病的症状" |

**HyDE 原理小科普**：HyDE（Hypothetical Document Embeddings）认为——"假设答案"的语义分布比"问题"更接近真实文档。所以先编一段像样的答案，用它去搜，反而比用问题搜更准。这是 RAG 里非常经典且有效的查询改写技巧。

---

## 6. 核心知识点深挖（RAG 学习重点）

### 6.1 父子分块：RAG 最重要的"隐藏超参"

- **本质矛盾**：检索要"小"（精准），理解要"大"（完整）。
- **解法**：检索用子块，生成用父块。本项目子块 400 / 父块 2000，重叠 60。
- **Small-to-Big 检索链路**：Top-16 粗排 → Top-5 子块 → `parent_id` 回溯父块去重 → 最佳命中子块证据精排 → 返回 Top-2 完整父块。
- **数据构成**：现存 `docs.json` 共 20816 块；按 `parent_id` 统计为 **16880 逻辑子块 + 3936 逻辑父块**（父块占 18.9%）。
  早期版本里这些父块也参与召回，且过滤发生在 Python 侧——结果 Top-16 里平均 4 条是父块（**浪费 25% 槽位**），
  现在过滤已下推到 Milvus。
- **你实验时可以调**：把 `CHILD_CHUNK_SIZE` 调大/调小，观察召回质量和答案完整度的变化。

### 6.2 稠密 vs 稀疏 / 混合检索

- **稠密（语义）**：能理解"同义改写、口语化"，但可能漏掉关键专名。
- **稀疏（关键词）**：专名命中极准，但不懂语义。
- **混合检索**：两路都搜，加权融合。**BGE-M3 一个模型同时产出两种向量**，所以本项目混合检索几乎"零额外成本"。
- 权重默认 `SPARSE_WEIGHT=0.7 / DENSE_WEIGHT=1.0`——**稀疏权重被刻意调高**。
  原因：医疗语料里专名极多（"胰岛素""糖化血红蛋白""一型糖尿病"），用户提问往往直接带专名，
  关键词精确命中的收益大于语义泛化；同时 BGE-M3 的稀疏向量是模型产出的 lexical weights，
  质量远高于传统 BM25，值得给更高权重。纯口语化场景则相反，可下调。

### 6.3 双塔（Bi-Encoder） vs 交叉编码（Cross-Encoder）

这是 RAG 检索必须懂的一对概念：

| | 双塔 Bi-Encoder（如 BGE-M3 检索） | 交叉编码 Cross-Encoder（如 BGE-reranker） |
|---|---|---|
| 结构 | query 和 doc **各自独立编码**成向量 | query 和 doc **拼在一起**联合编码 |
| 速度 | 快（doc 向量可离线预计算，查询只算一次） | 慢（每对都要现场算） |
| 精度 | 中（两塔不交互，语义交互弱） | 高（联合建模，相关性判断准） |
| 本项目用途 | **粗排**：从逻辑子块中快速召回 16 条（Top-5 子块 → 回溯父块） | **精排**：对候选父块的最佳命中子块证据打分，返回 Top-2 父块 |

> 结论：**粗排用双塔（快），精排用交叉编码（准）**。两步结合，既快又准——这就是工业级 RAG 的标准做法。

### 6.4 为什么意图分类用 BERT 而不是 LLM

- **成本/延迟**：意图分类是每秒可能触发很多次的"前置闸门"，用 LLM 太贵太慢；BERT 是轻量分类模型，毫秒级、可本地 CPU 跑。
- **任务简单**：只分两类（通用/医疗），不需要 LLM 的强推理。
- **可控**：分类结果确定、可解释，不会出现 LLM 乱返回策略名的情况。
- LLM 则留给"真正需要推理"的环节：选检索策略、生成答案。

> 本项目踩过的坑：意图分类器一度"坏掉"（医疗问题全被判成 general，导致跳过检索、答非所问）。后来用 `build_intent_data.py` + `train_intent.py` 重训 BERT（验证集 100% 准确率）解决。详见 [8.4](#84-练习重训意图分类器理解微调)。

### 6.5 RAG 的常见痛点与本项目的解法

| 痛点 | 表现 | 本项目解法 |
|------|------|------------|
| 检索不准 | 召回无关段落 | 混合检索 + Small-to-Big 父子分块 + 重排序 |
| 上下文缺失 | 答案断章取义 | 子块检索、父块回填上下文 |
| 幻觉 | 编造医学知识 | 医疗 Prompt 约束 + **检索为空时安全拒答（不调 LLM）** |
| 噪声进 LLM | Token 浪费、答案被带偏 | 重排序只留 Top-2 父块 |
| 重复问题慢 | 每次都跑全套 | 双通道：FAQ 快通道（Redis + BM25 归一化）直答 |
| 通用闲聊也检索 | 浪费检索、答得奇怪 | BERT 意图分流，general 直接 LLM |
| 复杂问题搜不到 | 原问题表述复杂 | LLM 自动选 hyde/subquery/backtracking 改写查询 |
| FAQ 误答（答非所问） | 「头痛」答成「声带息肉」 | BM25 **softmax 归一化** + 阈值 0.85（先修量纲、再定阈值） |
| 父块抢召回槽位 | Top-16 里有 4 条是 2000 字父块，浪费 25% | `chunk_type` 过滤下推到 Milvus 侧 |
| 检索失败就幻觉 | 无依据时 LLM 硬答 | **L0/L1/L2 分层降级**（见 4.9） |

### 6.6 为什么评估也很重要

RAG 不是"跑通就能用"，需要量化指标看效果。本项目用 Ragas（见第 7 节）。**没有评估，你根本不知道改了分块/权重后是在变好还是变坏。**

---

## 7. RAG 质量评估：Ragas 四指标

代码：`src/online_service/rag_evaluator.py`，脚本：`scripts/evaluate_rag.py`。

当前留档的最新完整结果为：210/210；Faithfulness 0.8163、Answer Relevancy 0.5007、Context Precision 0.8405、Context Recall 0.7619、加权综合 0.7299。主裁判为 GLM-4.6V，少量 API 失败指标由 deepseek-v4-flash 仅补缺；因此它是完整运行报告，但不是纯单裁判报告。严格 A/B 应使用同一独立裁判全量配对复评。

Ragas 用 4 个 0~1 的指标评估 RAG 输出：

| 指标 | 含义 | 依赖 |
|------|------|------|
| **faithfulness（忠实度）** | 回答是否完全基于上下文、无幻觉 | 上下文 + 回答 |
| **answer_relevancy（答案相关性）** | 回答是否切题、完整、有用 | 问题 + 回答 |
| **context_precision（上下文精确度）** | 检索到的上下文是否聚焦、无噪声 | 问题 + 上下文 |
| **context_recall（上下文召回率）** | 上下文是否覆盖回答所需的关键信息 | 问题 + 上下文 + **标准答案(ground_truth)** |

- **LLM 接入**：Ragas 通过 OpenAI 兼容接口接入 judge（`ChatOpenAI` + `LangchainLLMWrapper`），模型名和地址由环境变量配置。
- **Embedding 接入**：复用本地 BGE-M3（保证与检索同分布、无需额外密钥）。
- **降级**：Ragas 依赖缺失或 BGE 不可用时，自动回退到内置 **LLM-as-judge**（faithfulness/context_precision/answer_relevancy 三项）。

> 学习点：`context_recall` 必须有"标准答案(ground_truth)"才能算，所以准备评估集时要带 ground_truth。本项目 `data/test_query/rag_evaluate_data.json` 即样例。

---

## 8. 上手实践：把它跑起来 + 学习练习

### 8.1 环境准备

```bash
# 用 uv（推荐，项目已用 uv 管理）
uv sync

# 配置 LLM 密钥
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY 与 LLM_BASE_URL（OpenAI 兼容接口）

# 模型放进 src/models/：bge-m3、bert-base-chinese、bge-reranker-large
# 数据：原始 MSD → python scripts/extract_msd.py → data/clean_md/
```

### 8.2 启动中间件

- Milvus `localhost:19530`（必须）
- Redis `localhost:6379`（可选，缓存；本机 Docker 版带 `requirepass`，需在 `.env` 设 `REDIS_PASSWORD=1234` 才能连通，否则降级为无缓存）
- MySQL `localhost:3306`（可选，FAQ/会话，未启动自动降级）

### 8.3 运行方式

```bash
# 离线入库（建知识库，GPU 约 30 分钟）
python scripts/run_offline_ingest.py --data-dir ./data/clean_md

# 命令行交互问答（无需前端，最适合学习）
python main.py

# 启动在线 API（端口 8005，前端 Streamlit 在 8501）
python scripts/run_api.py
# Windows / VS Code 启动即崩溃(0xC0000005) 时，改用净化环境启动器：.\run_api_safe.ps1
```

> ⚠️ 不要直接 `python src/online_service/main_api.py`（相对导入会报错），用 `scripts/run_api.py` 或 `python main.py`。

### 8.4 练习：重训意图分类器（理解"微调"）

本项目意图分类器是微调出来的，过程本身就是很好的学习素材：

```bash
# 1) 构造训练数据（医疗~2500 + 通用~2500，9:1 切分）
python scripts/build_intent_data.py
# 2) 微调 BERT（底座 bert-base-chinese，3 轮，batch 8，warmup 500）
python scripts/train_intent.py
# 产出保存到 src/models/bert_query_classifier/
```

要点：医疗类来自清洗后的 2570 篇词条 + 测试 QA；通用类用模板生成；最终约 4984 条，验证集准确率 100%。

### 8.5 学习练习清单（边做边学 RAG）

1. **跑通最小问答**：`python main.py`，问"什么是高血压"，观察返回的 `strategy` 与 `sources`（来源内容就是喂给 LLM 的上下文）。
2. **对比四种策略**：用 `/query` 接口分别传 `strategy=direct` 和 `strategy=hyde`，看召回的 `sources` 有何不同，理解"查询改写"的价值。
3. **调参看变化**：把 `TOP_K_RERANK` 从 2 改成 4，重问，观察上下文变长、答案是否更全（也可能更啰嗦）；
   把 `FAQ_NORMALIZED_THRESHOLD` 从 0.85 降到 0.6，观察 FAQ 是否开始误答（体会"量纲没修好时阈值就是玄学"）。
4. **跑评估**：`python scripts/evaluate_rag.py --static data/test_query/rag_evaluate_data.json`，看四项指标，思考哪项是短板。
5. **改分块**：把 `CHILD_CHUNK_SIZE` 改成 200 和 800 各跑一次入库+问答，体会"块大小"对检索的影响。
6. **读源码顺序**：`rag_system.py` → `retrieval.py` → `reranker.py` → `embedding_provider.py` → `milvus_store.py` → `chunk_splitter.py`。

---

## 9. 关键配置参数速查

全部在 `src/config/settings.py`：

```python
# 父子分块
PARENT_CHUNK_SIZE = 2000   # 父块（给 LLM 的上下文）
CHILD_CHUNK_SIZE  = 400    # 子块（用于检索）
CHUNK_OVERLAP     = 60     # 重叠字符，防割裂

# 混合检索加权融合（WeightedRanker）
SPARSE_WEIGHT = 0.7        # 稀疏（词权）权重——医疗专名多，刻意调高
DENSE_WEIGHT  = 1.0        # 稠密（语义）权重

# 检索与重排（Small-to-Big 链路）
TOP_K_RETRIEVE = 16        # 粗排召回数（只召回子块）
TOP_K_CHILDREN = 5         # 子块命中数（精细定位）
TOP_K_RERANK   = 2         # 精排最终数（Top-2 父块）

# FAQ 快通道
FAQ_NORMALIZED_THRESHOLD = 0.85   # BM25 softmax 归一化后的命中阈值

# 分层降级策略
ENABLE_CHILD_FILTER_FALLBACK = True   # L1：同粒度降级（父块现场切成子块）
ALLOW_LLM_WHEN_NO_CONTEXT    = False  # L2：检索为空时安全拒答，不让 LLM 硬答
DEGRADE_ALERT_LEVEL          = 1      # 降级水位告警阈值（0=不打点 1=L1及以上 2=仅L2）

# 模型
EMBED_MODEL_NAME = "BGE-M3"        # 1024 维
RERANK_MODEL_NAME = "BGE-reranker-large"
LLM_MODEL_NAME = "<兼容模型名>"     # 通过 .env 配置，例如 deepseek-v4-flash
LLM_TEMPERATURE = 0.2              # 低温度，严谨

# 端口约定
后端 8005 / 前端 8501 / Milvus 19530 / MySQL 3306 / Redis 6379
```

---

## 10. 术语表

| 术语 | 解释 |
|------|------|
| **RAG** | 检索增强生成：先检索资料再让 LLM 生成答案 |
| **Chunking（分块）** | 把长文档切成小段，便于检索与上下文控制 |
| **Parent-Child Chunking** | 父子分块：子块检索、父块补上下文 |
| **Small-to-Big** | 先检索小粒度子块，再回溯大粒度父块作为生成上下文的检索范式 |
| **Filter Pushdown（过滤下推）** | 把过滤条件下推到数据库侧执行，而非取回后再在 Python 里过滤 |
| **Graceful Degradation（降级）** | 主路径不可用时切备用路径；本项目要求"降级路径更安全而非更粗糙" |
| **Embedding（嵌入/向量化）** | 把文本变成定长数字向量，使语义相近的文本向量也相近 |
| **Dense / Sparse Vector** | 稠密向量（语义）/ 稀疏向量（关键词权重） |
| **Hybrid Search** | 混合检索：语义+关键词两路融合 |
| **Bi-Encoder（双塔）** | query/doc 独立编码，快，用于粗排 |
| **Cross-Encoder（交叉编码）** | query/doc 联合编码，准，用于精排 |
| **Rerank（重排）** | 对粗排结果重新打分、取最相关几条 |
| **HyDE** | 假设文档嵌入：先生成假设答案再用它检索 |
| **Intent Classification** | 意图分类：判断问题类型（通用/医疗）以决定走不走检索 |
| **Ground Truth** | 标准答案，评估 context_recall 时需要 |
| **Ragas** | 一个 RAG 质量评估框架，提供多项自动化指标 |
| **Milvus** | 开源向量数据库，支持十亿级向量检索 |
| **Faithfulness / Hallucination** | 忠实度 / 幻觉（编造不在上下文里的信息） |

---

> 读完这份指南，建议你立刻打开 `src/online_service/rag_system.py` 顺着 `generate()` 方法读一遍——代码就是上面文字的最精确注脚。
> 记住一句话：**RAG = 检索（找对资料）+ 生成（说对话）**，本项目把这两件事的每一步都做厚了，这正是它值得作为学习样本的原因。
