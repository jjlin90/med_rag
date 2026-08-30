# Small-to-Big 子块过滤缺陷修复报告

> 修复日期：2026-08-29 · 已在真实数据（20816 条）上端到端验证

---

## 一、问题：一个会静默失效的链路

Small-to-Big 的设计意图是「**用 400 字子块精确定位 → 回溯 2000 字父块完整作答**」。
但实现上，子块过滤是在 **Python 侧后置**做的——先让 Milvus 召回 Top-16（父块子块混在一起），
再在内存里挑出 `parent_id` 非空的子块。

这带来两个后果：

1. **槽位浪费**：2000 字父块语义宽泛、容易得高分，会先占用 Top-K 名额再被丢弃。
2. **静默失效**：原代码里有一条降级分支——若挑不出子块，就直接返回父块。
   这意味着最坏情况下 Small-to-Big 完全没生效，而系统不会报任何错。

### 真实规模

| 指标 | 值 |
|---|---|
| 集合实体总数 | 20816 |
| 子块（parent_id 非空） | 16880 |
| 父块（parent_id 为空） | **3936（占 18.9%）** |
| 平均每个父块切出 | 4.29 个子块 |

**实测槽位浪费率：25%**（Top-16 中有 4 个名额被父块占用后丢弃）。

---

## 二、根因：5 个相互关联的缺陷

| # | 文件 | 缺陷 |
|---|------|------|
| 1 | `src/offline_pipeline/chunk_splitter.py` | 父块构造时**未传** `chunk_type='parent'`，`Chunk.chunk_type` 恒为默认值 `'child'`。`metadata['chunk_type']` 是对的，但对象属性是错的。`_simple_split` 同样漏传 |
| 2 | `src/offline_pipeline/embedding_provider.py` | `batch_process()` 组装结果字典时**没有透传** `chunk_type`，该字段根本进不了 Milvus |
| 3 | `src/offline_pipeline/milvus_store.py` | schema **缺顶层 `chunk_type` 字段**——它只存在于 `metadata` JSON 内部，而 JSON 内部的 key 无法用于下推过滤；`add_documents()` 也从未插入该列 |
| 4 | `src/online_service/retrieval.py` | 子块过滤后置；且存在「无子块则静默返回父块」的降级分支 |
| 5 | `scripts/test_query_pipeline.py` | 诊断脚本直接调 `search()` 绕过 Small-to-Big，调试时看到的耗时/命中与线上不可比 |

> 缺陷 3 是核心：**Milvus 只能对顶层标量字段下推 `expr` 过滤**。
> `metadata` 是 JSON 字段，无法用 `metadata['chunk_type'] == 'child'` 做过滤。

---

## 三、修复方案

### 核心：把过滤下推到 Milvus

```python
# src/offline_pipeline/milvus_store.py
def child_filter_expr(self) -> str:
    if self.has_chunk_type_field():
        return "chunk_type == 'child'"   # 新库：显式字段
    return "parent_id != ''"             # 旧库：语义等价的退化方案
```

`search()` / `hybrid_search_with_rerank()` / `Retrieval.search()` 新增 `only_children` 参数，
把该表达式与来源过滤用 `combine_expr()` 组合后，传给两个 `AnnSearchRequest`。

### 关键设计：新旧库自适应，**不需要重建集合**

`has_chunk_type_field()` 惰性探测 schema 并缓存结果。旧库（无 `chunk_type` 列）自动退化为
`parent_id != ""`——**这与 `chunk_type == 'child'` 语义完全等价**，因为 `batch_process()`
已经把父块的 `parent_id` 归一为空字符串。

**这意味着现有 20816 条数据一条都不用动，过滤立即生效。**
只有当下次重建集合重新入库时，才会自动获得显式的 `chunk_type` 字段，届时代码自动切换过去。

### 移除静默降级

原「无子块则返回父块」分支已删除，改为 **记录 ERROR 并剔除异常项**。

理由：静默返回 2000 字大块会污染 LLM 上下文（稀释注意力、翻倍 token），
更重要的是它会**掩盖数据问题**。医疗场景下，失败应当显式暴露。

> **后续演进（2026-08-29 同日追加）**
>
> 删掉静默降级后，检索链路继续演进为 **L0 / L1 / L2 分层降级策略**（`src/online_service/retrieval.py`）：
>
> | 层级 | 行为 | 输出粒度 |
> |---|---|---|
> | L0 | 下推过滤只召回子块，orphan 父块记 ERROR 剔除 | 400 字子块 → 2000 字父块 |
> | L1 | 子块召回为空 → 放开过滤重查，命中父块后**现场切成 400 字子块** | **仍是 400 字子块** |
> | L2 | 仍为空 / 基础设施故障 → **安全拒答，不调用 LLM** | 无上下文 |
>
> 原则是「**降级路径必须比主路径更安全，而不是更粗糙**」——L1 换的是检索入口而不是输出质量，
> L2 拒绝让 LLM 在无依据时硬答。回归测试：`python scripts/test_degrade_policy.py`（9 项 mock 场景）。
> 面试讲法见 `docs/med_rag_面试全解.md` §3.10 与 §5 Q12 / Q23 / Q24。

---

## 四、改动文件清单

| 文件 | 改动 |
|---|---|
| `src/offline_pipeline/chunk_splitter.py` | 父块/子块构造显式传 `chunk_type`；`_simple_split` 同样修复 |
| `src/offline_pipeline/embedding_provider.py` | `batch_process()` 透传 `chunk_type`，缺失时从 metadata 回补 |
| `src/offline_pipeline/milvus_store.py` | schema 新增顶层 `chunk_type` 字段；`add_documents()` 按库类型决定是否插入该列；新增 `has_chunk_type_field()` / `child_filter_expr()` / `combine_expr()`；`search()` 与 `hybrid_search_with_rerank()` 新增 `only_children` |
| `src/online_service/retrieval.py` | `search()` 新增 `only_children`；`search_child_to_parent()` 改为下推过滤 + 数据一致性自检，删除静默降级 |
| `scripts/test_query_pipeline.py` | 改用 `search_child_to_parent()`，与线上行为对齐 |
| `scripts/check_chunk_type_filter.py` | **新增**：体检脚本 |

入库列顺序已与 schema 字段顺序做程序化校验（AST 抽取比对），10 列完全一致。

---

## 五、验证结果

### 静态体检（秒级，不加载模型）

```
python scripts/check_chunk_type_filter.py
```

```
集合: med_msd_consumer_chunk        实体总数: 20816
字段: id, text, dense_vector, sparse_vector, parent_id, parent_content, source, timestamp, metadata
[~]   schema 无顶层 chunk_type 字段（旧库）→ 退化为 parent_id != ''
子块(child): 16880      父块(parent): 3936
[OK]  父/子结构正常，平均每个父块切出 4.29 个子块
[OK]  子块 parent_id 非空，可正常回溯
[OK]  多条件组合正确（已加括号保证优先级）
```

### 端到端验证（加载 BGE-M3）

```
python scripts/check_chunk_type_filter.py --with-search --query "一型糖尿病和二型糖尿病有什么区别"
```

```
--- 未加子块过滤（修复前的行为）---
召回 16 条，其中父块 4 条、子块 12 条
[~] 有 4 个 Top-K 槽位被 2000 字父块占用后丢弃（槽位浪费率 25%）

--- 下推子块过滤（修复后的行为）---
召回 16 条，其中无 parent_id 的异常项 0 条
[OK] 全部 16 条均为子块，过滤下推生效

--- Small-to-Big 完整链路 ---
16 子块 → 3 去重父块
  [1] 糖尿病的急性并发症.md_1_parent   1923 字符  score=1.0698
  [2] 糖尿病概述.md_0_parent           1966 字符  score=1.0660
  [3] 儿童和青少年糖尿病.md_0_parent   1851 字符  score=1.0578
[OK] 回溯成功，父块平均长度 1913 字符
```

**结论：全部通过，Small-to-Big 链路工作正常。**

---

## 六、后续建议

1. **不必重建集合**——旧库的 `parent_id != ""` 退化方案语义等价且已验证。
2. 若将来因其他原因重建集合，新库会自动获得显式 `chunk_type` 字段，代码自动切换，无需改配置。
3. 目前子块召回 16 → 去重后通常只剩 3-5 个父块，精排取 Top-2。
   若希望精排有更大选择空间，可调大 `TOP_K_RETRIEVE`；若想降低延迟，16 已足够。
4. 建议把 `python scripts/check_chunk_type_filter.py` 加入CI 或部署前检查。

---

## 七、面试怎么讲这道题

> **问：你们的父子分块是怎么落地的？父块会不会参与检索？**

**答**（现在可以自信地讲）：

会，但我修掉了。最初的设计是父块和子块都进同一个 collection，
检索时先召回 Top-16 再在应用层过滤出子块。这有个隐患——
2000 字的父块语义宽泛、容易得高分，会占用 Top-K 名额，我实测槽位浪费率 25%；
更糟的是原代码有条降级分支，挑不出子块就直接返回父块，等于 Small-to-Big 静默失效。

我的修复是把过滤**下推到 Milvus**：给 schema 加了顶层 `chunk_type` 字段
（JSON 内部的 key 是没法下推过滤的），检索时直接 `chunk_type == 'child'`。
同时兼容旧库——探测到没有这个字段就退化成 `parent_id != ""`，语义等价，
所以线上两万条数据不用重建就生效了。

另外我把那条静默降级删了，改成记录 ERROR。医疗场景里，
静默返回 2000 字大块既污染上下文又掩盖数据问题，失败应该显式暴露。
