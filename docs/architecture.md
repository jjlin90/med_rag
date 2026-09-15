# med_rag 架构与运行边界

> 2026-09-15按当前工作区核查。详细解释、参数、面试追问和待整改项统一见[学习与面试全解](med_rag_学习与面试全解.md)。

## 离线建库

HTML抽取是单独脚本，仍有旧路径和固定过滤规则。正式入库默认读取data/clean_md，依次加载、清洗、提取元数据、父2000/子400字符分块（两级overlap60）、保存JSON、父子块共同向量化并insert进Milvus。不会自动先运行HTML抽取。

BGE-M3同时输出dense1024维与sparse词权，关闭ColBERT；encode batch64、max_length2048。dense索引IVF_FLAT/IP（nlist256、nprobe16），sparse倒排/IP，建库和查询drop_ratio均配置0.2。当前文件2570篇、20816块；3936逻辑父块顶层类型误标child，不能把新代码修正等同于存量库迁移完成。

加载器DOCX存在Document同名覆盖，OCR占位，旧DOC/PPT无转换器；清洗器可能把含120/80的正文误当页码删除。现有MD语料规模不能证明全部格式均可用。

## 在线API

1. query缓存命中立即返回，本轮不写会话。
2. FAQ先检查MySQL连接/游标对象，再查FAQ缓存；未命中用进程内手写BM25，对最多5候选做softmax，达到0.85读MySQL答案。FAQ命中写会话并按use_cache决定外层query缓存。
3. 未命中进入RAGSystem，执行一次general/medical意图分类；general空context调用相同生成器，默认grounding存在冲突。
4. medical使用外部指定策略或LLM选策略；direct原问题、HyDE假设答案、subquery串行拆问合并、backtracking抽象问题。
5. 每次检索两路过滤后各最多16，WeightedRanker(dense1.0,sparse0.7)融合最多16；取前5有效子块按parent_id聚合。正文优先取冗余parent_content，缺失则子正文；rerank_content用该父块最高融合分子块。
6. 用原问题重排，最多2条生成上下文；subquery限制按每次调用计算，合并后再统一重排。source_filter精确匹配来源字符串，不是权限隔离。
7. 有答案时保存会话，degrade_level>=2跳过query缓存；其余可缓存。答案与sources分开返回。

CLI直接调用RAGSystem，不经过FAQ/query缓存。缓存键只包含规范化问题，存在小数点碰撞及忽略history/source/strategy的风险；use_cache=False不关闭FAQ内部缓存。

## 降级的准确边界

- L0子块为空且无显式error时可进入L1；L1放开子块过滤，取前5命中，父块在内存切片，最多16候选，重排和生成使用子片段。
- 上层捕获到的检索error直接进入L2；默认L2且无可用文档时不调用最终生成器。此前策略与增强仍可能已调用LLM。
- 有证据但生成失败返回最多2段原文，各500字符后可能附省略号。
- 底层Milvus部分异常返回空列表、embedding异常返回空向量，故障归类不是全覆盖；多子查询部分故障可能仍以L0返回并缓存。
- 重排故障回退原排序不自动设置degraded。初始化依赖失败还可能阻断整个服务启动。

## 多轮与观测

MySQL保留最近5轮；/chat依赖请求messages，生成仅使用最后3个历史message。服务端不自动按session_id读历史，检索没有指代补全。Streamlit恢复历史缺timestamp会与API必填字段冲突。评测页字段也与后端不一致。

API为async路由内调用同步工作，没有接SSE；response_time不含后续会话/缓存写入。降级计数为进程内字典，按检索调用累计，子查询会增加分母；不是HTTP请求级比率。health部分状态仅检查对象存在，不等同于实时依赖全部可用。

## 评测与验证

默认test_qa.json为5题，历史完整报告为210题独立答案集。历史F/AR/CP/CR分别0.8163/0.5007/0.8405/0.7619，混合裁判等权0.7299。正式Ragas与fallback必须分开，检查有效数和缺失列。

本次现有3项质量测试通过、降级mock断言通过；run_parallel_eval.py第220行仍有语法错误。本次未重测线上依赖、重建知识库或执行付费评测。图示：[系统总览](diagrams/med_rag_双通道问答流程.html)、[混合检索与父子块](diagrams/med_rag_混合检索与父子块链路.html)。
