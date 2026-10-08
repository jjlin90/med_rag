# med_rag 架构说明

> 2026-10-08 对照源码、回归和本地数据核验。详细讲解见 [学习与面试全解](med_rag_学习与面试全解.md)，本轮验收见 [项目检查记录](review_20261008.md)。

## 离线建库

HTML 抽取独立执行：`scripts/extract_msd.py` 默认从 `data/raw/MSDZHConsumerMedicalTopics` 读取，向 `data/clean_md` 输出 Markdown，支持覆盖输入与输出目录。

正式入库由 `scripts/run_offline_ingest.py` 或 CLI 数据处理模式执行：加载 → 清洗与元数据 → 父子分块 → 保存 JSON → BGE-M3 向量化 → Milvus 写入。加载器支持 PDF 文本、DOCX 段落、PPTX 文本、TXT、MD；扫描文档先转成可提取文本，旧 DOC/PPT 先转换。清洗保留 `120/80`、`1.5`、数字和单字符正文，页码按整行模式判断。

父块上限 2000 字符，子块上限 400，重叠 60，两级使用相同分隔符。子块保存 `parent_id` 和 `parent_content`。分块错误立即终止，向量化错误或输出数量不符传播异常，防止将部分产出报告为完整入库。

BGE-M3 输出 1024 维稠密向量与稀疏词权，关闭 ColBERT；编码 batch=64、max_length=2048 token。Milvus 稠密索引为 IVF_FLAT/IP，nlist=256、nprobe=16；稀疏索引为 SPARSE_INVERTED_INDEX/IP，构建与搜索 drop ratio 为 0.2。离线入口外层向量批大小为 64，`batch_process` 默认值为 32。

本地文件核验：2570 篇 Markdown，20816 条记录，3936 个父块、16880 个子块；ID 唯一、父引用和冗余父正文一致。3936 条旧父块标签已备份并修正，当前磁盘文件的顶层与元数据类型均符合父子关系。加载旧 JSON 及入库仍按 `parent_id` 推导类型。

## 在线问答

1. API 校验请求；`/chat` 取最后一条用户消息为问题，其余为 history。role 仅允许 user/assistant，正文非空，timestamp 可选。
2. 允许缓存时先查 `query:v2`，键包含问题、来源过滤、显式策略和历史，仅去首尾空白。命中也保存本轮会话。
3. 无历史、无来源过滤、无显式策略的请求进入 FAQ。允许缓存则先查 `faq:v2`，再用 MySQL/BM25；最高原始分大于零，且最多 5 个候选的 softmax 分达到 0.85 时直答。带约束的请求直接进入深通道。`use_cache=false` 关闭两层缓存读写。
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

Ragas 0.2.6 在整批都有非空参考答案时运行 F/AR/CP/CR，否则运行 F/AR。请求了但缺失的指标保留；布尔、非有限和越界分数记 null，真实零分保留。报告提供 engine、metrics、valid_counts、complete_count、average 与 scores。备用裁判独立标为 `llm_judge_fallback`。

历史 210 题原始评分复算为 F 0.8163、AR 0.5007、CP 0.8405、CR 0.7619；逐条评分等权综合为 0.729847976，四位显示 0.7298。主裁判 GLM-4.6V，DeepSeek 补齐缺失单元，逐项来源保留。当前修订使用回归验证；模型质量复评使用固定题集和独立输出文件。

入口：`python -m unittest discover -s tests -v`、`python scripts/test_quality_optimizations.py`、`python scripts/test_degrade_policy.py`、`python scripts/audit_static.py`。线上类型与过滤检查使用 `python scripts/check_chunk_type_filter.py --with-search`；旧服务计数使用分页迭代，避免 16384 条截断。

流程图：[双通道问答](diagrams/med_rag_双通道问答流程.html)、[混合检索与父子块](diagrams/med_rag_混合检索与父子块链路.html)。
