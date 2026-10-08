# Small-to-Big 缺陷与当前状态

> 首次修复记录来自 2026-08-29；2026-10-08 复核代码、回归及本地数据。当前完整流程见 [架构说明](../architecture.md)，本轮验收见 [检查记录](../review_20261008.md)。

## 1. 原始缺陷

旧实现先让父块与子块共同占用 Milvus Top-K，再在 Python 中按 `parent_id` 过滤；父块会浪费召回槽位。更严重的是，无子块时旧分支会直接返回父块，使 Small-to-Big 静默失效。

相关旧缺陷包括：

1. `Chunk.chunk_type` 没有在父块构造时显式设为 parent。
2. embedding 结果没有透传 `chunk_type`。
3. Milvus Schema 没有顶层 `chunk_type`。
4. 子块过滤后置且存在静默父块回退。
5. 旧诊断脚本绕过了正式 Small-to-Big 方法。

这些是历史问题描述；当前代码已包含相应修复。

## 2. 当前代码实现

- `chunk_splitter.py` 显式写入父/子 `chunk_type`。
- `embedding_provider.py` 透传 `parent_id`、`parent_content` 和 `chunk_type`。
- `milvus_store.py` 的新 Schema 有顶层 `chunk_type`。
- 新 Schema 联合检查 `chunk_type == "child"` 与 `parent_id != ""`；旧 Schema 回退 `parent_id != ""`，避免误标父块占用子块召回槽位。
- 来源过滤与子块过滤由 `combine_expr()` 加括号组合。
- 正常单次检索最多召回 16 个子块，再截前 5 个回溯父块；代表子块进入 `rerank_content`，最终重排保留最多 2 条上下文。父正文缺失时可回退子正文；多子查询分别召回后合并再统一重排。
- L0 无结果时进入 L1；L1 放开过滤，父块命中后在内存切为子块。L1 不返回完整父块。
- 仍无结果时进入 L2，默认关闭无上下文生成，返回固定拒答；检索、重排等故障也返回 L2，API 不缓存。这里“不调用 LLM”仅指默认拒答分支不调用最终答案生成器，此前策略选择或查询增强仍可能调用 LLM。

## 3. 存量数据与迁移边界

`data/split_docs/docs.json` 有 20,816 条记录。按 `parent_id` 统计：

| 类型 | 数量 |
|---|---:|
| 逻辑子块 | 16,880 |
| 逻辑父块 | 3,936 |
| 子块/父块 | 4.29 |

历史 JSON 曾将3936条父块标为child；2026-10-08已备份修正顶层和元数据类型，当前类型与父子关系一致。`ChunkSplitter.load_chunks` 和 `MilvusStore.add_documents` 按 `parent_id` 推导类型，继续兼容旧数据。

磁盘文件本轮已修正，在线collection单独验收。当前体检联合核对类型、父关联和实体总数；旧服务计数使用分页迭代，避免16384条上限截断。检索异常、空召回和L2均显式记录为验收问题。

## 4. 历史运行记录（2026-09-10）

2026-09-10 运行：

```powershell
.venv\Scripts\python.exe scripts\check_chunk_type_filter.py
```

当时在初始化阶段无法连接 `localhost:19530`，报 Milvus server unavailable。这是当时的连接结果，不能据此判断今天在线 collection 的字段、实体数或过滤结果。

同时：

- `scripts/test_quality_optimizations.py`：3 项测试通过。
- `scripts/test_degrade_policy.py`：断言通过。

这些 mock/静态测试不等于真实 Milvus 端到端通过。

## 5. 历史数据如何使用

历史单个查询的Top-16中曾有4个父块，过滤后16条均为子块并回溯到3个父块。这组记录用于说明过滤原理；整体召回收益采用固定多查询集评测。2026-10-08本地Milvus端口未监听，在线集合验证在服务启动后执行。

## 6. 下一次上线前验证

1. 使用已修正的本地 `docs.json`，核验父子关联及逻辑父块类型。
2. 重建或迁移 Milvus collection。
3. 运行 `scripts/check_chunk_type_filter.py` 检查 Schema 和父子数量。
4. 再运行 `--with-search`，确认两路 ANN 都应用子块过滤。
5. 对多个查询统计过滤前后 Recall@K/NDCG@K；不要用一个样本宣称总体提升。

## 7. 面试准确话术

> 旧实现把子块过滤放在 Milvus 召回之后，父块可能先占用 Top-K；我把父子类型提升为顶层标量字段，在 dense/sparse 两路联合检查类型与父 ID，并兼容旧 Schema。旧分块在加载和入库时按父子关系恢复类型，检索或重排故障显式标记 L2 并阻止缓存。存量数据迁移及真实 Milvus 过滤效果需要独立验证。
