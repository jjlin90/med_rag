# Small-to-Big 缺陷与当前状态

> 首次修复记录来自 2026-08-29；本页在 2026-09-10 重新按当前代码和运行状态核验。历史实测不能冒充当前在线验证。

## 1. 原始缺陷

旧实现先让父块与子块共同占用 Milvus Top-K，再在 Python 中按 `parent_id` 过滤；父块会浪费召回槽位。更严重的是，无子块时旧分支支会直接返回父块，使 Small-to-Big 静默失效。

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
- 新 Schema 子块表达式为 `chunk_type == "child"`；旧 Schema 回退 `parent_id != ""`。
- 来源过滤与子块过滤由 `combine_expr()` 加括号组合。
- 正常路径先取 Top-16 子块，再截 Top-5，然后回溯父块；最佳子块进入 `rerank_content`，最终返回 Top-2 父块。
- L0 无结果时进入 L1；L1 放开过滤，父块命中后在内存切为子块。L1 不返回完整父块。
- 仍无结果或基础设施错误时进入 L2，无证据拒答且不调用 LLM。

## 3. 当前本地数据风险

`data/split_docs/docs.json` 有 20,816 条记录。按 `parent_id` 统计：

| 类型 | 数量 |
|---|---:|
| 逻辑子块 | 16,880 |
| 逻辑父块 | 3,936 |
| 子块/父块 | 4.29 |

但该 JSON 顶层 `chunk_type` 全部为 child。若直接把它写入带新字段的 Schema，逻辑父块会被错误标成 child。因此重建新集合前必须先用当前 `ChunkSplitter` 重新生成分块并重新向量化。

`BGEEmbeddingProvider.batch_process` 的 fallback 会优先读取对象属性；当前代码新生成的 `Chunk` 对象类型正确，但旧 JSON 不能因此自动修复。

## 4. 当前验证结果

2026-09-10 运行：

```powershell
.venv\Scripts\python.exe scripts\check_chunk_type_filter.py
```

结果是在初始化阶段无法连接 `localhost:19530`，报 Milvus server unavailable。故本次只能确认代码与本地 JSON，不能确认当前在线 collection 的字段、实体数或过滤结果。

同时：

- `scripts/test_quality_optimizations.py`：3 项测试通过。
- `scripts/test_degrade_policy.py`：断言通过。

这些 mock/静态测试不等于真实 Milvus 端到端通过。

## 5. 历史数据如何使用

旧记录曾观察到某个查询的 Top-16 中有 4 个父块，即单个样本的槽位浪费率 25%。它只能作为历史单样本诊断，不能表述成全库平均或当前线上指标。旧记录还显示过滤后 16 条均为子块并回溯到 3 个父块；由于本次 Milvus 未连接，不能写成“当前全部通过”。

## 6. 下一次上线前验证

1. 用当前分块器重建 `docs.json`，确认逻辑父块顶层 `chunk_type=parent`。
2. 重建或迁移 Milvus collection。
3. 运行 `scripts/check_chunk_type_filter.py` 检查 Schema 和父子数量。
4. 再运行 `--with-search`，确认两路 ANN 都应用子块过滤。
5. 对多个查询统计过滤前后 Recall@K/NDCG@K；不要用一个样本宣称总体提升。

## 7. 面试准确话术

> 旧实现把子块过滤放在 Milvus 召回之后，父块可能先占用 Top-K；我把父子类型提升为顶层标量字段，并在 dense/sparse 两路请求中下推过滤。为了兼容旧 Schema，代码可回退到 `parent_id != ""`。当前本地旧分块 JSON 的顶层类型仍不可信，所以重建新集合前必须重新分块。本次复核时 Milvus 未启动，代码与 mock 测试通过，但在线过滤仍需启动服务后复验。
