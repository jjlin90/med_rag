# med_rag 架构说明

> 2026-10-10 更新 FAQ 问题一致性及缓存 v3；2026-10-08 对照源码、回归和本地数据核验。详细讲解见 [学习与面试全解](med_rag_学习与面试全解.md)，本轮验收见 [项目检查记录](review_20261008.md)。

## GPU 运行环境

正式模型计算统一使用 NVIDIA CUDA GPU：BGE-M3 向量化、BGE-reranker-large 重排与 BERT 意图分类共用 `Config.DEVICE="cuda"`。配置初始化检查 `torch.cuda.is_available()`，设备未就绪则终止启动并提供环境修复说明。向量化与重排使用 fp16，意图训练沿用 CUDA 配置并启用 fp16。文档加载、清洗、分块和数据库 IO 按各组件原有实现执行。

Windows/Linux 的 uv 安装从官方 cu126 索引锁定 PyTorch；pip 需单独指定 CUDA wheel 来源。安装与实际 GPU 验证见 [快速上手](../GETTING_STARTED.md#2-校验-gpu-环境)。显存由模型、文本长度、batch 与并发共同决定；当前离线编码 batch=64，显存紧张时降低批量，并行评测需计入每个子进程独立加载的模型。

## 离线建库

HTML 抽取独立执行：`scripts/extract_msd.py` 默认从 `data/raw/MSDZHConsumerMedicalTopics` 读取，向 `data/clean_md` 输出 Markdown，支持覆盖输入与输出目录。

正式入库由 `scripts/run_offline_ingest.py` 或 CLI 数据处理模式执行：加载 → 清洗与元数据 → 父子分块 → 保存 JSON → BGE-M3 向量化 → Milvus 写入。加载器支持 PDF 文本、DOCX 段落、PPTX 文本、TXT、MD；扫描文档先转成可提取文本，旧 DOC/PPT 先转换。清洗保留 `120/80`、`1.5`、数字和单字符正文，页码按整行模式判断。

父块上限 2000 字符，子块上限 400，重叠 60，两级使用相同分隔符。子块保存 `parent_id` 和 `parent_content`。分块错误立即终止，向量化错误或输出数量不符传播异常，防止将部分产出报告为完整入库。

BGE-M3 输出 1024 维稠密向量与稀疏词权，关闭 ColBERT；编码 batch=64、max_length=2048 token。Milvus 稠密索引为 IVF_FLAT/IP，nlist=256、nprobe=16；稀疏索引为 SPARSE_INVERTED_INDEX/IP，构建与搜索 drop ratio 为 0.2。离线入口外层向量批大小为 64，`batch_process` 默认值为 32。

本地文件核验：2570 篇 Markdown，20816 条记录，3936 个父块、16880 个子块；ID 唯一、父引用和冗余父正文一致。3936 条旧父块标签已备份并修正，当前磁盘文件的顶层与元数据类型均符合父子关系。加载旧 JSON 及入库仍按 `parent_id` 推导类型。

## 在线问答

1. API 校验请求；`/chat` 取最后一条用户消息为问题，其余为 history。role 仅允许 user/assistant，正文非空，timestamp 可选。
2. 允许缓存时先查 `query:v3`，键包含问题、来源过滤、显式策略和历史，仅去首尾空白。命中也保存本轮会话。
3. 无历史、无来源过滤、无显式策略的请求进入 FAQ。允许缓存则先查 `faq:v3`，再用 MySQL/BM25；最高原始分大于零、最多 5 个候选的 softmax 分达到 0.85，且输入与索引及数据库标准问题仅首尾空白不同、答案非空时直答；缓存同样核对标准问题。带约束的请求直接进入深通道。`use_cache=false` 关闭两层缓存读写。
4. 深通道分类一次。分类故障返回 `intent_classifier_unavailable` 的 L2。general 使用独立通用提示词；medical 使用显式策略或 LLM 选择的 direct、HyDE、subquery、backtracking。选择或增强失败回退直接检索。
5. 每次稠密与稀疏两路融合，权重分别为 1.0 和 0.7，最多召回 16 个子块，取前 5 个按父 ID 聚合。父正文供生成，最高融合分子块作为 `rerank_content`。BGE-reranker-large 用原问题精排，最终取最多 2 条上下文。子查询串行检索，合并后统一重排。
6. 返回答案、sources、意图、策略、置信分数、用时、会话与降级原因。保存最近 5 轮会话；L0/L1 可缓存，L2 跳过缓存。

新 Schema 子块条件为 `chunk_type == 'child' and parent_id != ''`，旧 Schema 为 `parent_id != ''`。来源条件通过 JSON 字符串转义，再与子块条件组合。

## 降级与故障处理

| 状态 | 行为 |
|---|---|
| L0 | 子块检索、父正文恢复、子证据重排、生成 |
| L1 | 正常空召回时放宽过滤；前 5 个命中中的父块现场切为子块，最多 16 条候选，生成使用子片段 |
| L2 无召回 | 默认返回固定资料不足答复，跳过最终答案生成 |
| L2 故障 | 意图、检索、任一子查询或重排故障附对应原因；有上下文但生成失败时返回最多 2 段原文摘录，每段上限 500 字符 |

默认 `ALLOW_LLM_WHEN_NO_CONTEXT=False`；检索错误始终阻止最终生成。自动策略与查询增强在检索前，可能调用 LLM。L1 保持子块输出粒度，其质量与耗时通过独立实验衡量。模型和 Milvus 初始化属于启动阶段，请求级 L2 适用于已启动服务。

## 会话与观测

MySQL 按 `timestamp DESC, id DESC` 读取与裁剪最近 5 轮，再反转为时间正序。Streamlit 按会话 ID 获取历史并提交 `/chat`；提示词使用最后 3 条历史消息。检索输入为当前问题，追问应写明对象。

`confidence` 随路径取值：FAQ 固定 0.95，RAG 为意图分类分，L2 为 0。`used_cache` 标记外层缓存命中。`response_time` 在 FAQ/RAG 分支写会话与缓存前计算，缓存命中分支在会话写入后计算。

`/health` 提供组件状态和进程内检索降级计数；子查询分别计数，重启清零。部分状态依据初始化对象，实时连通性单独检测。API 采用同步推理与 IO，适用于本地演示；认证、会话所有权、检索指代改写、流式接口与并发调度列入后续工程计划。

## 评测与验收

质量试验开关：`RETRIEVAL_TITLE_ANCHOR` 默认false。未指定来源限制时，从本地分块目录找出问题中完整字面出现的真实标题，最长优先、最多两个；每个标题按原问题和严格来源过滤补一个子块，再在原Top-5预算内回溯父块。`topic_anchor`在去重中保留，重排先保留对应标题组，组内使用实际CrossEncoder分数。显式`source_filter`仍只检索指定来源，目录不可用时保留普通检索。各次标题补召回采用独立错误状态，失败时记录告警并跳过该路，保留主检索及其他成功补召回的资料；主检索故障仍返回L2/retrieval_error。

生成上下文与实时评测共用文档标题格式。`LLM_GROUNDING_REVIEW`默认false，启用时只对默认严格医学提示路径进行复核，通用独立提示路径不增加此调用。复核输入为原始知识、问题、辅助历史及初稿；原文段落由程序编号，模型给出关键陈述、支持判断和证据编号，再输出最终答案。代码校验JSON类型、证据编号范围及输出完整性；失败返回None，由核心进入原文摘录L2。该检查增加过程可追溯性，语义支持关系仍需要质量评测和逐题核验。

输出发布分为两层校验。非流式生成器检查`finish_reason`，仅在结束状态为`stop`时提取正文；截断、内容过滤、其他结束状态或接口异常返回None。对于`stop`响应，生成器将正文裁剪首尾空白，空内容会形成空字符串；核心再通过`if not answer`阻止空答案发布。医学分支在已有重排资料时返回原文摘录L2，原因统一为`llm_unavailable`；必需的事实复核发生调用、格式、编号或输出失败时也走同一路径，初稿不发布。该原因表示生成或复核未产出可发布答案，具体失败类型由相应日志说明；general分支无医学资料摘录，返回通用失败回复及同一L2原因。

Ragas 0.2.6 在整批都有非空参考答案时运行 F/AR/CP/CR，否则运行 F/AR。请求了但缺失的指标保留；布尔、非有限和越界分数记 null，真实零分保留。报告提供 engine、metrics、valid_counts、complete_count、average 与 scores。备用裁判独立标为 `llm_judge_fallback`。

评测保留历史210题固定答案与上下文、逐项裁判及补评来源，本轮另设固定20题开发对照。当前修订由工程回归验证，模型质量由固定题集与独立输出文件复评；公开[参考目标](quality_pilot_20261010.md)用于优化规划，实测记录在本地产物中保留。

`RAGAS_ANSWER_RELEVANCY_LANGUAGE=chinese`提供AR提示中文适配，默认仍为default；公式、严格度3及noncommittal惩罚保留。语言口径变化会改变评分，双方必须统一适配并另存结果。`LLM_JUDGE_REQUESTS_PER_SECOND`调节请求速率；长上下文评分需同时考虑供应商token配额。20题开发对照与历史全量结果分别见[质量试验记录](quality_pilot_20261010.md)和[评测与业务案例分析](Ragas评估与badcase分析.md)。

入口：`python -m unittest discover -s tests -v`、`python scripts/test_quality_optimizations.py`、`python scripts/test_degrade_policy.py`、`python scripts/audit_static.py`。线上类型与过滤检查使用 `python scripts/check_chunk_type_filter.py --with-search`；旧服务计数使用分页迭代，避免 16384 条截断。

流程图：[双通道问答](diagrams/med_rag_双通道问答流程.html)、[混合检索与父子块](diagrams/med_rag_混合检索与父子块链路.html)。
