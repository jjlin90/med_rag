# med_rag 学习与面试全解（合并校订版）

> 校订日期：2026-09-15。合并原《rag_learning_guide》和《med_rag_面试全解》，按“实现 → 原因 → 面试追问 → 已知边界”组织。代码事实以当前工作区为准；历史评测和运行记录标明时间。本次按实际函数复核并修正文档和图示；重新核算本地数据与历史报告，运行离线回归。没有重建知识库或重新调用在线模型评测。工程缺陷按现状说明，不冒充已经修复。

## 阅读导航

- 初次学习：第 1～8 讲，沿数据与请求的流向阅读。
- 面试复习：各讲的“面试回答”，以及第 11 讲的完整介绍。
- 复现与排错：第 9～10 讲。
- 核实来源：第 12 讲代码索引。
- 附录：原文更正清单与后续优化优先级。

文中代码路径均相对于项目根目录 `D:\pythonProject\med_rag`。桌面副本用于阅读，仓库 `docs/med_rag_学习与面试全解.md` 为合并版主文件。

## 第 1 讲：入口、组件与职责

### 实际入口

| 入口 | 命令 / 文件 | 行为 |
|---|---|---|
| CLI 问答 | `python main.py` | 默认进入交互问答，直接调用 RAGSystem |
| CLI 入库 | `python main.py --data-processing --data-dir ./data/clean_md` | 调用离线处理函数 |
| 离线脚本 | `python scripts/run_offline_ingest.py` | 默认处理 Config.CLEAN_MD_DIR |
| API | `python scripts/run_api.py --port 8005` | 创建 RAGWebAPI |
| 问答界面 | `web_demo/app.py` | Streamlit，默认后端地址 localhost:8005 |
| 演示页面 | `frontend/` | Vite PPT 演示，不是问答客户端 |

`main.py` 没有 `--query` 或 `--process-data` 参数。其 `--session-id` 用于保存问答，CLI 不会因此自动恢复生成上下文。

`RAGSystem` 负责意图、策略、检索、重排、上下文与生成。`RAGWebAPI` 增加 Redis、FAQ、会话和评测接口。CLI 直接使用 RAGSystem，**没有经过 API 的缓存/FAQ 快通道**，因此 CLI 结果不能直接代表完整 API 链路。

组件初始化会加载模型、连接 Milvus，LLM 初始化还会探测连接。FAQ 命中可省去该次请求的生成调用，但不表示服务启动完全不需要模型或外部依赖。

面试回答：项目有离线建库与在线问答两个阶段，在线核心可复用；FastAPI 在核心外组装缓存、FAQ、会话等功能。说明评测入口时要区分核心 RAG 与完整 API。

## 第 2 讲：数据抽取、加载和清洗

### HTML 抽取与文档清洗是两个模块

`scripts/extract_msd.py` 使用 BeautifulSoup，确实按选择器移除 script/style、导航、页脚、侧栏、版权、作者等区块，并从 main/article 抽取正文。原文“项目没有导航/版权清洗”过于绝对；准确说法是 **DataCleaner 没有这些专用规则，HTML 抽取脚本有**。

抽取脚本仍有明显边界：

- ROOT_DIR 与 OUTPUT_MD_DIR 硬编码到旧 `D:\pythonProject\test_python\MSD-Manual-Portable-main\...` 目录；它不读取 Config 的 data/clean_md 路径。
- 章节标题规则含固定的“1 型糖尿病”列表，并非通用标题层次识别。
- 行过滤包含单字符 `v`，可能误删含该字符的有效行；末尾正则删除方括号内容，也不能等同于只删无用跳转。
- 使用 get_text 抽取文本，没有通用表格结构、图片语义或跨页表合并实现。
- 没有本次重新执行抽取，当前 2570 个文件不能仅靠脚本存在就证明完整生成溯源。

`DocumentLoader` 路由 PDF、DOCX/DOC、PPTX/PPT、TXT、MD。PDF 使用 PyMuPDF；DOC 与 DOCX 共用 python-docx，PPT 与 PPTX 共用 python-pptx，旧二进制格式无专用转换器。三个 OCR 方法仍为占位，不能承诺扫描件可用。

`DataCleaner` 做字符、页码样式、重复短行、纯符号行、连续重复行和空白处理，并提取元数据。清洗规则存在不等于数据完全无噪声；新语料需要抽样对照原文。

面试回答：先区分 HTML 抽取脚本和入库时的文本清洗器，再说明各自规则与边界。复杂表格、OCR、通用章节恢复仍需补。


### 本次补核：真正的加载与清洗边界

- DOCX 当前不只是“支持但需测试”：模块先导入 python-docx 的 Document，随后自定义同名 Document 类覆盖它。_load_docx 中 Document(file_path) 会因缺少 metadata 报错，被捕获后返回空列表。因此按当前代码不能承诺 Word 已可用。
- DataCleaner 的页码判断使用 re.search，不是整行匹配。例如含“120/80”的正文行会匹配页码样式并整行删除；纯数字/符号行也会删除。清洗有误删有效数值的风险，不能说已经完整保留单位和表格。
- “重复短行”这个名字不完全符合实现：长度小于2的行被删除，而重复次数判断没有最大行长限制；普通长行重复也可能被剔除。
- 抽取阶段删除版权/作者文本属于文本处理，不能推出授权已完成；公开原文和本地文件存在也不能证明可再分发。

## 第 3 讲：父子分块、字段与数据规模

### 参数与实际文件

| 项目 | 当前值 |
|---|---:|
| PARENT_CHUNK_SIZE | 2000 字符 |
| CHILD_CHUNK_SIZE | 400 字符 |
| CHUNK_OVERLAP | 60 字符 |
| CHUNK_SAVE_PATH | data/split_docs/docs.json |
| clean_md 文件数（2026-09-15 重算） | 2570 |
| 分块条数（2026-09-15 重算） | 20816 |
| parent_id 非空 / 为空 | 16880 / 3936 |
| 全部记录平均长度（字符） | 522.90 |
| 最短 / 最长（字符） | 3 / 2000 |

配置和文件都指向 `data/split_docs/docs.json`；旧文档反复写的 `data/processed/docs.json` 实际不存在。

分块使用 RecursiveCharacterTextSplitter，按字符计数，不是 token。2000/400 是切分规模配置，不意味着每条都恰好这么长，父块也不一定包含整篇文章或完整疾病章节。

关键字段：

| 字段 | 用途 |
|---|---|
| id | 块标识 |
| content | 块正文 |
| parent_id | 子块关联父块；逻辑父块为空 |
| parent_content | 子块内冗余的父块正文 |
| chunk_type | 显式父子类型 |
| metadata | 来源等信息 |

### 存量文件缺陷

当前 JSON 的顶层 chunk_type 全为 child，与 3936 个逻辑父块不一致。新分块代码显式赋类型；入库代码优先使用已有顶层 chunk_type，**不会自动纠正一个非空但错误的 child 值**。

使用新 Schema 前应重生成分块并验证父子关系，或实施经验证的数据迁移；仅重建 collection 不会自动修好旧 JSON。当前文档任务没有执行这些写库操作。

面试回答：子块用于较小粒度召回与重排，父块用于较大上下文；冗余 parent_content 减少父块二次读取，但增加存储与更新一致性成本。

## 第 4 讲：BGE-M3 与 Milvus

BGEEmbeddingProvider 调用 encode，同时启用 dense/sparse，关闭 ColBERT；dense 配置 1024 维，max_length=2048，内部 batch_size=64。batch_process 方法自身默认 32，而正式离线脚本传入 64，不能把所有调用的 batch 都说成同一个默认值。

代码只在 device=="cuda" 时启用 fp16，并在 CUDA 批次结束后调用 empty_cache；这不证明特定显卡内存一定足够，也不证明固定吞吐或提速比例。

Milvus 默认配置：

| 配置 | 值 |
|---|---|
| 地址 | localhost:19530，可由环境变量覆盖 |
| database / 连接 alias | MED / MED |
| collection | med_msd_consumer_chunk |
| dense 索引 | IVF_FLAT / IP，nlist=256 |
| 搜索参数 | nprobe=16 |
| sparse 索引 | SPARSE_INVERTED_INDEX / IP |
| WeightedRanker 权重 | sparse 0.7、dense 1.0 |

连接 alias 是客户端连接名，**不是 collection 发布别名**，不能据此声称灰度切换已实现。

索引为分簇候选搜索；不能把 IVF_FLAT 称为全库精确搜索。sparse 是 BGE-M3 lexical weights，FAQ 使用的是另一套 jieba/BM25。两路权重是代码配置，仓库没有证据证明它是最优权重或各自贡献了多少提升。

入库代码使用 collection.insert，未实现完整版本发布、旧块清理或幂等 upsert 工作流。重复入库前需要明确数据更新策略。

面试回答：说明双表示与两路检索各自作用，交代索引参数；增益需要在标注集上测 Recall@K/NDCG，不能由模型名或代码结构推出。

## 第 5 讲：Small-to-Big、重排与多查询

### 单次 L0 调用

```text
查询编码 → dense ANN / sparse 倒排 + 子块过滤 → 最多16个子块
        → 截前5个子块 → 按parent_id聚合
        → 最佳子块证据重排 → 最多2条文档送生成
```

新 Schema 用 chunk_type 过滤；旧 Schema 用 parent_id 非空过滤。缺 parent_id 的异常候选会打点并剔除。

回溯构造父块记录时：
- content 优先 parent_content；缺失时回退子块 content。因此“L0 始终返回完整父块”不严谨。
- rerank_content 使用该父块命中子块中最高融合分的正文。
- 多个子块映射到同一父块会去重，最终数量可能不足 2。
- Reranker 优先读取 rerank_content，否则读取 content；不可用或异常时保留输入顺序并截 top_k。
- rerank_score 是排序分数，没有经过正确率概率校准。

### 四种策略

| 策略 | 实际作用 |
|---|---|
| direct | 原问题直接检索 |
| hyde | 生成假设答案，以它检索 |
| subquery | 拆子问题，各自检索再合并 |
| backtracking | 生成较抽象问题后检索 |

外部指定 strategy 优先；否则 LLM 选择，选择失败回退 direct。`/query` 有 strategy 字段，`/chat` 当前没有该字段。

subquery 在列表推导中串行执行。**Top-16、Top-5 是每次子查询的限制**，合并后候选可超过 5，最后才统一用原用户问题重排并截 Top-2。合并按 id 去重、保留更高 score；有结果时取有产出分支中最好的降级级别，无结果时取最严重级别，任一路 error 仍保留。因此整体 L0 不代表所有子问题都成功，合并集合还可能同时包含 L0 父块与 L1 子块。

history 只用于最终答案 Prompt，意图分类、策略选择和检索没有历史改写。“它怎么治疗”之类追问，生成阶段虽看到历史，检索仍可能先因问题不完整而失败。

面试回答：准确报“每次检索最多16→前5→回溯、最终最多2”，主动说明多子查询、父正文缺失和重排故障的例外。

## 第 6 讲：FAQ、缓存与置信度

API 顺序：query cache → FAQ 内部 cache/BM25 → RAGSystem。CLI 不走该快通道。

FAQSearch 先检查 MySQL connection/cursor；不可用时直接转 RAG，甚至不会继续尝试 FAQ 专用缓存。FAQ BM25 是本项目 BM25Index 的手写实现，不是实际调用 rank_bm25.BM25Okapi。它对最多 5 个返回候选做 softmax，阈值 0.85，命中后读 MySQL 答案并写 FAQ cache。

softmax 表示候选间相对分布，不能解释成答案有 85% 正确率。例如仅返回一个候选时，即使 BM25 原始分为 0，softmax 也必为 1；需结合绝对分、负样本和人工标注校准。

| 字段或操作 | 精确含义 |
|---|---|
| query 缓存键 | query: + MD5(规范化问题) |
| FAQ 缓存键 | faq: + MD5(规范化问题) |
| 规范化 | 去首尾空白、转小写，再用正则剔除空白/标点等 |
| query TTL / FAQ TTL | 默认各3600秒 |
| use_cache=false | 禁用外层 query cache 读写；FAQ 内部缓存仍可能使用 |
| query cache 命中 | 直接返回，不写本轮 MySQL conversation |
| FAQ 命中 | 写 conversation；对外 confidence=0.95、sources=[] |
| used_cache | 只反映外层 query cache；FAQ 内部命中也返回 false |
| 普通 RAG confidence | 意图分类器分数，不是答案可信度 |
| L2 confidence | 0.0 |

所有返回 degrade_level>=2 的 RAG 结果都跳过 query cache，包括无召回、检索故障和 LLM 不可用；L1 可以缓存。

当前 query key 不含 session/history、source_filter、strategy、模型或 grounding 版本。规范化也会丢失小数点等符号，例如 1.5 与 15 在其余内容相同的情况下可能落到相同键，属于应补测试的语义碰撞风险。

source_filter 仅传入 RAG 检索；FAQ 不受该条件约束。检索表达式直接拼成 source == '输入值'，没有把来源字符串参数化或转义；它是来源精确匹配，不是学科分类或权限隔离。

## 第 7 讲：降级与生成

### 分支规则

| 状态 | 行为与边界 |
|---|---|
| L0 有子块 | 回溯后重排；父正文缺失可退回子正文 |
| 被 Retrieval.search 捕获并记录到 status.error 的错误 | 直接返回 L2，不先执行 L1；底层吞掉的异常不保证被正确识别 |
| L0 空且允许 L1 | 去掉子块过滤再查；仅处理前5个命中 |
| L1 命中父块 | 在内存按共享分隔符重切；子块候选总量最多16 |
| L1 成功 | 使用子块进行重排、生成 |
| L1 禁用或仍为空 | L2 |
| L2 且无可用文档 | 默认 ALLOW_LLM_WHEN_NO_CONTEXT=False 时固定拒答 |
| LLM 无答案且已有证据 | 最多2段原文，每段截500字符并可能附省略号，标记 llm_unavailable |

不能说“任何配置下检索空都不调用 LLM”：代码存在 ALLOW_LLM_WHEN_NO_CONTEXT 开关。默认关闭时才执行无上下文拦截。general 路由在该拦截之前，走空 context 生成。

L1 最多16个重排候选，单次 L0 回溯后最多5个，因此不能由“有上限”推出“L1 不比 L0 贵”。

默认生成温度0.2、最大输出1024，grounding 默认开。系统提示约束只依据相关知识，首句给结论，部分有依据时回答可支持部分，缺失主题具体说明；它是模型提示约束，不是形式化的事实验证器。

general 使用相同 generate_with_context，却传空 context；默认 grounding 下可能倾向拒答。该冲突由代码可见，发生频率需另测。

正常答案的 sources 与正文分开返回，没有强制内联编号对齐。无召回固定拒答也不等价于自动识别所有“证据不足”：只要召回了候选，代码仍可能进入生成。


### 离线备选分块并非可靠兜底

缺少langchain_text_splitters时，离线子块路径把Chunk传给_simple_split，后者读取page_content而Chunk只有content，可能导致整篇分块失败被跳过。简单分块还用i//chunk_size生成ID，而步长是chunk_size-overlap，可能产生重复ID；遇标点提前截断又固定步进，可能跳过部分字符。当前生产文件的ID无重复不证明这个备用路径安全。

向量化异常返回空数组后，batch_process用zip拼装结果，失败批次可能无结果但仍继续后续批次。因此分块条数、向量化条数、入库条数必须分别核验，不能仅凭完成日志判断全部数据入库。离线脚本在创建logs目录之前初始化FileHandler，干净环境还需先确保logs存在；--data-dir不会绕过对RAW_MSD_DIR的验证。

### 本次补核：降级不是全链路保证

1. 服务启动与单次请求是两个阶段。Milvus 初始化、BGE 模型加载、BERT 初始化失败可能直接导致服务无法启动；此时没有已经运行的 API 可以返回 L2。
2. MilvusStore.search 在 hybrid_search 的 MilvusException 分支直接返回空列表；BGEEmbeddingProvider.generate_embeddings 也会将异常转换为空数组。上层只有收到异常并填入 status.error 才能明确标记 retrieval_error。某些故障可能被当成空召回，或在后续阶段才暴露。
3. subquery 合并会保留任一分支的 error，但有产出时取最好的水位。例如 L0成功 + L2故障可能合成 level=0、error非空。generate 的拦截要求 level>=2，正常返回又不透传 error，API可能把部分故障结果按L0缓存。因此“有一路故障就必不缓存”不符合当前实现。
4. Reranker 缺模型或打分失败只回退原候选顺序，未自动把 QueryResponse 标为 degraded；意图 predict 返回 error 时，核心用 medical、confidence=0 兜底，也没有专用故障标记。
5. L1 没有重新编码现场切出的每一片，片段沿用父命中的融合分；候选按命中顺序及片段顺序截断，可能在重排前丢掉末尾证据。依赖缺失时 L1 还会改成固定滑窗，不能说任何环境都与离线切块完全一致。
6. L2“无模型调用”准确指默认拒答分支不调用最终答案生成器。到达该分支之前，自动策略选择、HyDE等可能已经调用LLM。不能说整次失败请求完全零LLM调用。

Grounding 是提示词约束，不是程序级事实校验。真正的程序分支是默认配置下的 L2 拦截；两者不能混称“硬约束”。

## 第 8 讲：多轮、API 和可观测性

MySQL 表名是 `conversations`，一条记录保存一问一答。MAX_HISTORY_TURNS=5，每次写入后删除更旧记录；这不是永久保存全部聊天。历史按 timestamp 排序，同秒记录的稳定顺序也值得补充 id 排序验证。

三种“历史”要区分：

1. 数据库：最多5轮问答记录，读取返回 question/answer。
2. 客户端：Streamlit 主动读取并恢复界面，将消息重新发送。
3. 模型：/chat 使用请求中除最后一条外的 messages，Prompt 只取最后3个 message 对象；grounding 下历史仅用于理解指代。

服务端不自动从 session_id 加载历史，CLI 也没有把存储历史传给 generate_answer。外层缓存命中还会跳过会话写入。

### API 契约

| 接口 | 要点 |
|---|---|
| POST /query | question 必需；source_filter/use_cache/strategy/session_id 可选 |
| POST /chat | messages 非空，最后一条 role 必须 user |
| GET/DELETE /conversation/{session_id} | 查询/删除会话记录 |
| GET /health | 组件健康与降级统计 |
| GET /stats | 组件统计 |
| GET /available_strategies、/intent_example | 策略和示例 |
| POST /evaluate | 对提交的答案/上下文评测 |

ChatHistoryItem 的 **role、content、timestamp 都必需**；/chat 未声明 strategy 字段。QueryResponse 包含 answer、sources、confidence、response_time、used_cache、intent、strategy、session_id、degraded、degrade_level、degrade_reason。

客户端恢复历史还有接口不匹配：load_history 只重建 role/content，没有 timestamp；chat_api 直接发送这些消息。因此从数据库恢复出非空历史后再发 /chat，会因恢复消息缺必需字段而触发请求校验错误（422）。新创建的界面消息有 timestamp，并不能补齐此前恢复的消息。原文“刷新后可直接续聊”不能作为稳定功能承诺。

response_time 在 FAQ/RAG 分支的会话写入和缓存写入之前计算，不是完整端到端时延。降级指标在 Retrieval 的进程内类字典中累加，不存 Redis，进程重启会清空，多 worker 不会自动聚合。

路由是 async def，但内部执行同步推理/数据库/网络工作；没有把重活自动转入线程池。generate_stream 存在，API/Streamlit 未接 SSE。/stats 的 LLM 统计还会做连接探测，因此统计请求也可能产生外部模型调用。

## 第 9 讲：意图训练和 Ragas

### 意图模型

入口 `scripts/train_intent.py`：bert-base-chinese + 两类分类头，3 epochs，train/eval batch 8，warmup 500，weight_decay 0.01，seed42；按 eval_loss 选最佳模型。这个脚本没有早停 callback，不能混用 IntentClassifier.train 等其他入口的配置。

当前训练/验证文件2026-09-15重算分别4485/499条，总4984，主要由词条/症状/通用概念模板组成。构造脚本还读取 test_qa 问题，故不能把路由评测和这些训练来源当成完全独立数据。模板随机拆分还可能共享实体/问法，应补独立真实问题集。

模型目录存在权重；上一轮只核查了目录未见训练日志。准确说法是“现有材料未提供可复核的 accuracy=1.0”，不能扩张为已经搜索所有可能位置并证明从未有日志。

### 四指标在看什么

| 指标 | 关注点 | 不代表什么 |
|---|---|---|
| Faithfulness | 回答陈述能否得到上下文支持 | 不保证上下文本身正确 |
| Answer Relevancy | 回答反推的问题与原问的相关程度 | 不等于医学正确率 |
| Context Precision | 相关上下文排序表现 | 不等于数据库检索耗时 |
| Context Recall | 参考答案中的事实能否由上下文覆盖 | 不等于全库 Recall@K |

项目声明 ragas==0.2.6，环境实际安装版本仍应单独核验。评测器先规范化字段；ground_truth 或 reference_answer 可作参考答案。只有整批每条参考答案非空时才运行四项，否则整批只跑 F/AR。API EvaluateItem 只声明 ground_truth；不要默认 API 同样接收 reference_answer 别名。

Ragas 不可用或整次异常可进入自建 LLM judge fallback，它仅有 F/CP/AR，不可冒充正式四指标。逐项 NaN 保存 null，均值忽略无效值；这里还有更前置的分支：_extract 会把整列全 NaN 的指标直接排除，因此返回的 metrics 可能少于请求的四项；全部指标列都无有效值时会改走 fallback。保留下来的指标若均值聚合时无有效值，average 的兜底才是0.0。必须同时检查 engine、metrics、valid_counts 和 complete_count，不能仅凭 total=210 或一张均值表认定四项评测完整。

评测入口也有差别：evaluate_rag.py 的实时模式构造 RAGWebAPI 并调用 _handle_query(use_cache=False)，仍可能使用 FAQ 内部缓存；静态模式只评传入答案与上下文。Streamlit 评测页调用 /query 时开启缓存，提交 /evaluate 的 items 没有 ground_truth，因此不会在正式 Ragas 路径得到四项。界面还读取 context_relevance、answer_relevance、success 等旧键，与后端 context_precision、answer_relevancy 等结构不一致，可能显示默认0。判断结果应读取原始报告的 engine、metrics、average 和 valid_counts，不能直接相信该页三个指标卡片。

### 留档报告

文件：`data/test_query/eval_ragas_210_grounded_v3_ar_mixed_complete.json`。

| 指标 | 分数 | 有效数 | GLM-4.6V / DeepSeek 补缺数 |
|---|---:|---:|---:|
| F | 0.8163 | 210 | 202 / 8 |
| AR | 0.5007 | 210 | 205 / 5 |
| CP | 0.8405 | 210 | 206 / 4 |
| CR | 0.7619 | 210 | 206 / 4 |
| 四项等权平均 | 0.7299 | — | — |

这是历史完整报告，合并脚本只补 null，并记录裁判来源；它不代表本次重新生成/重测，也不是全量同一裁判配对实验。历史基线0.525与此结果的裁判和链路不同，不能把差值完全归因为某一个优化。

AR 改进流程：固定问题、参考答案与裁判 → 抽取低分样本 → 检查非承诺标记、遗漏、冗余和主题偏离 → 改提示词或检索 → 同题配对复评，同时监控 F 和有效数。要测完整系统，重新检索并生成；只对固定 contexts 重生成，仅能测生成层变化。


### 本次补核：意图、策略及评测复现

- 意图编码最多128 token；直接argmax二分类，没有低置信阈值澄清流程。没有训练权重时的本地BERT分类头不能等同于已经训练的分类器。
- 外部未知strategy会实际执行direct，但响应仍可能保留外部无效字符串。策略选择器用set遍历与子串匹配，纯标点清洗为空后也可能匹配任意策略；不是严格枚举校验。
- HyDE生成的是假设文档/答案，用于检索，不是知识库证据。HyDE与backtracking成功时替换查询；当前没有自动把原问题和改写问题双路并行融合。
- subquery只解析以“子问题”开头且含中文冒号的行；无有效行回退原问题。max_tokens=200是输出长度参数，不是明确的最大子问题数限制。
- 默认data/test_query/test_qa.json本次重算只有5条。210条留档报告来自独立答案集；运行默认命令不会自动重现210题结果。
- 本报告记录generation_models为glm-4.5-air，主裁判glm-4.6v，补缺裁判deepseek-v4-flash。它与当前Config默认生成模型deepseek-v4-flash不是同一概念；环境变量还可覆盖当前默认模型。
- fallback评测的解析失败、空输出、缺少评分字段会转成0并参与均值；这与正式Ragas的null处理不同，不得把fallback的0直接归因于答案质量差。
- 本项目直接使用langchain_openai，但pyproject.toml未把它列为直接依赖；可能通过间接依赖存在，应核验锁文件/环境。requirements.txt与pyproject.toml不完全一致，前者还未显式列openai。不能承诺两种安装方式完全等价。

## 第 10 讲：复现和排错

在项目根目录使用项目环境。下面是操作入口，写库/训练/外部评测并未在本次文档修改中执行。

```powershell
# 无业务模型调用的源码语法检查（会暴露已有脚本问题）
.venv\Scripts\python.exe -m compileall -q src scripts main.py

# mock/控制流回归
.venv\Scripts\python.exe scripts/test_quality_optimizations.py
.venv\Scripts\python.exe scripts/test_degrade_policy.py

# 在线 Milvus 诊断；加 --with-search 会加载模型并执行检索
.venv\Scripts\python.exe scripts/check_chunk_type_filter.py

# API / CLI
.venv\Scripts\python.exe scripts/run_api.py --port 8005
.venv\Scripts\python.exe main.py

# 对固定答案集重评，使用新输出名保留历史报告
.venv\Scripts\python.exe scripts/evaluate_rag.py --static data/test_query/eval_answers_210_grounded_v3_ar.json --out-json data/test_query/eval_review_20260915.json --out-csv data/test_query/eval_review_20260915.csv
```

配置通过 .env 提供 LLM_MODEL_NAME、LLM_BASE_URL、LLM_API_KEY。生成与裁判通常读取同一 Config，但历史答案可能由不同模型生成，必须看报告记录。裁判参数默认温度0、timeout180秒、max_tokens2048、重试2；可由 LLM_JUDGE_* 覆盖。

2026-09-15验证：重新确认实际分块路径和20816条统计，逻辑父块3936、子块16880，ID无重复，子块关联的父ID均存在且parent_content与父正文一致；用源码 compile 检查 `scripts/run_parallel_eval.py` 仍在第220行因 global 声明顺序报 SyntaxError。未修改业务脚本。

历史状态：2026-09-10 Milvus localhost:19530连接失败、环境未检测到CUDA、两类回归测试通过；这些是当日记录，不据此断言今天服务状态相同。

定位错答案时，依次检查：抽取原文 → 清洗后的文件 → 父子关系 → 检索 query → 过滤 → 候选 → rerank → 实际 Prompt → 输出。当前没有 request_id 全链路追踪，应保存可复现样例后再做归因。

## 第 11 讲：面试表达与追问

### 一分钟介绍

这个项目实现了医疗科普 RAG 的离线建库、在线问答和自动评测。本地知识文件本次统计2570个，当前分块文件20816条，采用400/2000字符父子分块。BGE-M3输出稠密和稀疏表示，Milvus两路融合检索；正常单次检索最多取16个子块，截前5个回溯父块，用最佳子块证据重排，最多取2条作为生成上下文。API外层还有Redis缓存和MySQL FAQ快通道。项目留有210题四指标评测，并保存缺失指标补评来源；目前主要低项是AR，下一步需要同一裁判配对验证和真实问题集。

### 常见追问

- **为什么不全用 FAQ？** 项目以FAQ处理可匹配的固定答案，未命中再进入RAG；是否减少多少调用要另行统计。
- **为什么不直接用大模型？** RAG将本地资料提供给生成器并返回来源，事实依据可检查；仍要处理检索错位与生成偏离。
- **为什么重排子块？** 正常L0用较短命中证据评分，生成仍优先使用父正文。其效果必须通过消融测量。
- **重排后一定两篇？** 最多两条，父块去重或无结果时会更少；不保证来自不同原始文档。
- **FAQ的0.85是什么？** 最多5个候选的softmax阈值，不是校准过的正确概率。
- **会话为什么存了还可能答不好？** 历史只用于最终生成，检索没有多轮问题补全，而且query缓存忽略历史。
- **指标0.7299是什么？** 留档四项等权平均；有混合裁判补缺，不是医学准确率。
- **项目是不是Multi-Agent？** 当前是RAG编排与LLM策略选择，没有多Agent调度、A2A或MCP实现。
- **有生产性能数据吗？** 当前材料不足以支持固定QPS、P95、显存或部署SLA。
- **数据授权是否核验？** 源名称来自本地材料；代码不能证明授权和隐私结论。

## 第 12 讲：代码核验索引

| 主题 | 文件与关键符号 |
|---|---|
| 参数、路径、模型默认值 | src/config/settings.py / Config.__init__ |
| CLI实际参数 | main.py / main、query_mode |
| HTML抽取 | scripts/extract_msd.py / clean_msd_html、ROOT_DIR、OUTPUT_MD_DIR |
| 加载/OCR | src/offline_pipeline/document_loader.py / DocumentLoader |
| 清洗 | src/offline_pipeline/data_cleaner.py / DataCleaner |
| 父子分块 | src/offline_pipeline/chunk_splitter.py / ChunkSplitter |
| encode参数 | src/offline_pipeline/embedding_provider.py / generate_embeddings、batch_process |
| Schema、insert、过滤 | src/offline_pipeline/milvus_store.py / child_filter_expr、add_documents |
| L0/L1、进程计数、来源过滤 | src/online_service/retrieval.py / search_child_to_parent、_METRICS、_build_source_filter |
| 多查询合并与L2 | src/online_service/rag_system.py / generate、_merge_retrieval_results |
| 精排 | src/online_service/reranker.py / rerank |
| Prompt、3个message | src/online_service/llm_generator.py / _build_user_prompt |
| API、缓存旁路、响应计时 | src/online_service/main_api.py / ChatRequest、_handle_query |
| 缓存规范化与键 | src/online_service/cache_manager.py / normalize_query、query_cache_key |
| FAQ softmax | src/online_service/faq_search.py / search_faq、search_normalized |
| 5轮存储 | src/online_service/conversation_store.py / MAX_HISTORY_TURNS |
| 客户端历史恢复 | web_demo/app.py / load_history、chat_api |
| 训练入口 | scripts/build_intent_data.py、scripts/train_intent.py |
| 正式评测、补缺 | src/online_service/rag_evaluator.py、scripts/evaluate_rag.py、scripts/merge_eval_retries.py |

## 附录 A：这次更正了什么

| 原文问题 | 校订结论 |
|---|---|
| data/processed/docs.json | 实际是 data/split_docs/docs.json |
| main.py --query / --process-data | 默认查询；入库参数 --data-processing |
| 降级计数在Redis | 在Retrieval进程内类字典 |
| 项目没有导航/版权清洗 | HTML抽取脚本有，DataCleaner没有 |
| 总是完整父块 | parent_content缺失会用子正文兜底 |
| L2无条件不调用LLM | 默认开关关闭时拦截；general另走分支 |
| 所有请求写会话 | 外层query缓存命中直接返回 |
| 关闭缓存即禁用全部缓存 | FAQ内部缓存不受use_cache控制 |
| 所有缺失均值都是null | 正式逐项无效可为null；全NaN列可能被排除、全部无效转fallback，需查引擎和有效数 |
| 16→5→2适用于所有策略整体 | 16→5是每次子查询；合并后再重排取2 |
| 存历史就是多轮检索 | 历史仅注入生成；检索前未补全指代 |
| MED alias意味着集合切换 | MED是连接alias，不是发布别名 |

## 附录 B：建议补齐的实现与验证

以下是后续工程实现与验证清单。本次已补齐其文档说明，没有把这些建议写成已完成能力：

1. 优先修正缓存键语义、FAQ缓存开关和命中后的会话写入。
2. 补多轮检索前的指代改写，修复general与grounding冲突。
3. 迁移旧chunk_type，验证完整父正文与新旧Schema的一致性。
4. 给source_filter添加校验/安全表达式构造，明确FAQ是否也应按来源过滤。
5. 校准FAQ候选数、绝对分与阈值；增加单候选、近邻疾病、剂量标点测试。
6. 为HTML抽取参数化路径，移除过宽过滤，抽检表格/标题保留。
7. 用独立问题集测意图与检索，用同一裁判测AR/F，区分生成实验和全链路实验。
8. 接通内联引用、异步执行与完整计时，实测后才写性能数字。

## 附录 C：逐步面试讲解与可复现证据

### 两分钟讲完整链路

先说“离线和在线”两部分。离线主入口读取clean_md，执行加载、清洗、元数据提取，先切2000字符父块，再切400字符子块，两级overlap配置都是60字符。父子块都向量化并写同一Milvus集合，子块冗余父正文，正常回溯不再单独查父表。这里按字符切分，与BGE的2048 token截断预算是两套单位。

在线要说明具体入口。API先查query缓存，再走FAQ缓存/手写BM25，符合0.85相对softmax阈值则读标准答案返回。只有快通道未命中才执行一次BERT二分类；general绕过检索，medical进入策略选择和检索。指定direct时跳过选策略模型调用；自动direct通常还包含一次策略选择调用。

单次正常检索先生成dense+sparse，分别在IVF_FLAT和稀疏倒排索引检索，每路最多16条，融合后仍最多16条；不是把两路相加得到32条最终候选。过滤在两路请求中下推，之后取前5个有效子块按父ID聚合，用最佳命中子块与原问题做重排，最多2条上下文供生成。一个父块的一条代表子块打分不等于所有命中子块都逐一精排。父块是较大窗口，仍不保证覆盖完整疾病章节。

最后补一句边界：L1返回子片段，L2默认不进行最终答案生成；提示词、来源返回、降级和离线评测都有实现，但不代表已经完成事实核验、医疗安全审核或生产部署。210题留档四指标均值可报，不能报成医学准确率或线上性能。

### 可量化但不编造的表述

| 可以说 | 需要额外证据才能说 |
|---|---|
| 本地2570篇、20816条，3936逻辑父块和16880子块 | 线上Milvus当前也正好这些记录 |
| 配置16→5、最终最多2 | 该参数是网格搜索得到的最优参数 |
| 保存210条混合裁判完整报告 | 同一裁判确认所有优化的因果增益 |
| FAQ与缓存命中可跳过该次生成 | 缓存零开销、毫秒SLA、命中率固定多少 |
| 当前是Python RAG编排 | 已实现Multi-Agent、A2A、MCP、LangGraph状态图 |
| 3项质量测试、降级mock断言通过 | 已验证在线可用、临床正确或高并发稳定 |

### 整改优先级与验收条件

| 优先级 | 当前实现不足 | 后续验收依据 |
|---|---|---|
| P1 | 缓存忽略上下文并抹掉数值标点 | 1.5/15、不同历史/来源/策略不会错误复用 |
| P1 | 存量3936父块类型错误 | 新产物父子字段一致；库内过滤实际复核 |
| P1 | DOCX类覆盖、清洗误删数值 | 真实DOCX解析非空；数值/表格样例保留 |
| P1 | 检索异常吞掉与部分故障透传丢失 | 故障注入后错误状态、响应、缓存一致 |
| P1 | 刷新恢复缺timestamp；评测页字段不匹配 | 恢复后chat不再422；图表读到真实指标 |
| P1 | run_parallel_eval.py语法错误 | 源码编译通过，再单独验证worker参数 |
| P2 | general提示与空context冲突 | 通用问答和医学拒答分别回归 |
| P2 | source_filter表达式直接拼接 | 引号等输入安全处理；明确来源语义 |
| P2 | FAQ单候选零分仍直答 | 独立正负样本标定、拒答与近邻题回归 |
| P2 | 多轮只影响生成，不影响检索 | 追问指代补全保留约束并测召回 |
| P2 | 评测缺项与fallback零分混淆 | 保留引擎、有效数、异常数和缺失原因 |
| P2 | 增量写入、健康探测与并发能力不足 | 幂等更新、重启恢复、真实依赖探测与压测 |

这些是代码核查发现与后续验收标准，不表示本次已修改业务实现。诊断脚本和证据见仓库artifacts/review_20260915。

### 官方原理参考

项目事实以本地代码和留档文件为准；下列链接仅用于原理核对，不作为本项目部署证据。

- [Milvus：WeightedRanker融合及分数归一化](https://milvus.io/docs/reranking.md)
- [Ragas：指标概览（版本页面）](https://docs.ragas.io/en/v0.1.21/concepts/metrics/)
- [Ragas：Context Precision实现说明](https://github.com/vibrantlabsai/ragas/blob/main/docs/concepts/metrics/available_metrics/context_precision.md)

本次未重测线上Milvus、FAQ库、模型API和端到端质量；历史2026-09-10服务状态仅作历史记录。2026-09-15现有质量测试3项通过、降级模拟断言通过；运行测试时Config报告CUDA不可用。源码检查仍发现run_parallel_eval.py第220行语法错误，不能宣称全项目检查通过。
