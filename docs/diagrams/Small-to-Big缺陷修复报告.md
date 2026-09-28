# Small-to-Big 缺陷与当前状态

> 首次修复记录来自 2026-08-29，运行记录截至 2026-09-10；2026-09-28 按当前实现修订过滤、类型恢复与降级说明，未重测在线服务。当前完整流程见 [架构说明](../architecture.md)，修复与回归依据见 [工程修订与面试详解](../20260927_工程修订与面试详解.md)。

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

存量 JSON 的顶层 `chunk_type` 全部为 child，与逻辑父子关系不一致。当前 `ChunkSplitter.load_chunks` 和 `MilvusStore.add_documents` 均按 `parent_id` 推导类型，会纠正已有但错误的标签；绕过这些处理直接写原标签仍可能引入错误。

代码不会自动改写磁盘 JSON 或迁移已有 collection。迁移应核验父子关联、类型与向量记录；可通过当前加载/入库流程恢复类型，也可重新分块并向量化，不能把“已有代码修复”等同于“存量库已迁移”。

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

旧记录曾观察到某个查询的 Top-16 中有 4 个父块，即单个样本的槽位浪费率 25%。它只能作为历史单样本诊断，不能表述成全库平均或当前线上指标。旧记录还显示过滤后 16 条均为子块并回溯到 3 个父块；由于本次 Milvus 未连接，不能写成“当前全部通过”。

## 6. 下一次上线前验证

1. 用当前加载器恢复类型并核验父子关联，或重新分块生成 `docs.json`，确认逻辑父块类型为 parent。
2. 重建或迁移 Milvus collection。
3. 运行 `scripts/check_chunk_type_filter.py` 检查 Schema 和父子数量。
4. 再运行 `--with-search`，确认两路 ANN 都应用子块过滤。
5. 对多个查询统计过滤前后 Recall@K/NDCG@K；不要用一个样本宣称总体提升。

## 7. 面试准确话术

> 旧实现把子块过滤放在 Milvus 召回之后，父块可能先占用 Top-K；我把父子类型提升为顶层标量字段，在 dense/sparse 两路联合检查类型与父 ID，并兼容旧 Schema。旧分块在加载和入库时按父子关系恢复类型，检索或重排故障显式标记 L2 并阻止缓存。存量数据迁移及真实 Milvus 过滤效果需要独立验证。
