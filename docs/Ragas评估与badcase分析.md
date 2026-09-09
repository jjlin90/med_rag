# Ragas 评估与 badcase 分析

> 最后校准：2026-09-09。本文以仓库中的实际报告为准，不把旧实验、失败调用或跨裁判结果混成同一口径。

## 1. 当前可对外使用的结果

评测集共 210 组，字段为 `question / contexts / answer / ground_truth`。当前完整报告：

| 指标 | 分数 | 有效数 |
|---|---:|---:|
| Faithfulness | **0.8163** | 210/210 |
| Answer Relevancy | **0.5007** | 210/210 |
| Context Precision | **0.8405** | 210/210 |
| Context Recall | **0.7619** | 210/210 |
| 等权综合 | **0.7299** | 210/210 |

产物：

- `data/test_query/eval_answers_210_grounded_v3_ar.json`：本轮固定答案与上下文。
- `data/test_query/eval_ragas_210_grounded_v3_ar_mixed_complete.json`：完整 JSON 报告。
- `data/test_query/eval_ragas_210_grounded_v3_ar_mixed_complete.csv`：逐条 CSV。

裁判口径必须同时说明：主体分数由 **GLM-4.6V** 评出；其资源包耗尽后，9 个样本中缺失的 21 个指标单元由 **deepseek-v4-flash** 补评。合并脚本只填 `null`，不覆盖任何已有 GLM 分数，并在报告的 `metric_judge_counts` 与 `retry_fills` 中保存来源。因此这是“主裁判 + 缺失补评”，不是同题多裁判 ensemble。

各指标裁判构成：

| 指标 | GLM-4.6V | DeepSeek 补评 |
|---|---:|---:|
| Faithfulness | 202 | 8 |
| Answer Relevancy | 205 | 5 |
| Context Precision | 206 | 4 |
| Context Recall | 206 | 4 |

## 2. 版本演进

| 版本 | 答案/裁判口径 | F | AR | CP | CR | 综合 |
|---|---|---:|---:|---:|---:|---:|
| 历史路由基线 | 多模型分块路由，209 条有效 | 0.329 | 0.576 | 0.636 | 0.559 | 0.525 |
| Grounded v2 | GLM-4.5-Air 生成，GLM-4.6V 独立裁判，207 条完整 | 0.785 | 0.317 | 0.8524 | 0.746 | 0.6751 |
| AR 优化 v3 | 同一主裁判口径；最终补齐 210 条 | 0.8163 | 0.5007 | 0.8405 | 0.7619 | 0.7299 |

可比性说明：

- `0.525 → 0.7299` 能说明系统和评测流程整体迭代，但两端裁判路由不同，不应包装成严格 A/B。
- 更适合描述的同口径变化是 v2 → v3：综合约 `0.675 → 0.730`，AR 约 `0.317 → 0.501`，Faithfulness 约 `0.785 → 0.816`。
- CP 小幅波动不等于检索退化；v2、v3 的答案及少量有效样本集合不同，应结合逐条 badcase 判断。

## 3. 旧版低分为什么不再作为主结果

历史综合 0.525、Faithfulness 0.329 暴露了三个问题：

1. 生成端会在“上下文未提及”之后继续用参数知识补充，形成看似负责、实际无证据的扩写。
2. 回答常以前置套话或泛化拒答开头，降低 Answer Relevancy。
3. 早期评测把部分 API/解析失败混入统计，并使用多裁判分块路由；不同题块难度和裁判尺度相互混杂。

因此旧结果保留为历史诊断基线，不再用于简历主结论。

## 4. 本轮做了什么优化

### 4.1 生成提示词

- 首句直接回答问题核心，避免“根据资料”“关于这个问题”等模板化前缀。
- 有部分相关证据时先回答有依据的部分，再明确缺口；只有完全没有相关事实时才安全拒答。
- 拒答必须指出缺少的具体主题，避免泛化的“信息不足”。
- 继续保留 Grounding 约束：不把模型参数知识伪装成检索证据。

### 4.2 检索与重排

Small-to-Big 先召回子块，再按 `parent_id` 聚合父块。当前实现将每个父块下融合分最高的命中子块保存为 `rerank_content`，CrossEncoder 对该细粒度证据打分；排序后仍把完整父块 `content` 交给生成端。这样避免用约 2000 字父块直接重排时被截断或被无关段落稀释。

### 4.3 评测可靠性

- Ragas 返回 NaN 时保存为 `null`，均值只统计有效值，不再把“没评出来”伪装成 0 分。
- 报告记录 `valid_counts`、`complete_count`、生成模型、裁判模型与是否自评。
- 对严格温度校验的模型固定 Ragas wrapper 温度，避免框架把 `n=1` 改成 `1e-8` 后被接口拒绝。
- 支持关闭思考、JSON mode、timeout/retry、`--offset/--limit/--indices` 和缺失单元补跑。

## 5. Answer Relevancy 的判分陷阱

当前 Ragas 0.2.x 的 Answer Relevancy 会让裁判根据回答反推问题，并计算与原问题的语义相似度；同时生成 `noncommittal` 标记。多次生成中只要出现非承诺/无法回答倾向，就可能把该条分数乘成 0。

因此 AR 优化不是“塞关键词刷相似度”，而是：

- 先给直接结论，再补限定条件；
- 有依据时不要泛化拒答；
- 将真正缺失的信息具体化；
- 同时监控 Faithfulness，防止为追 AR 又恢复无依据扩写。

## 6. 如何复现

完整静态评估：

```powershell
.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --static data\test_query\eval_answers_210_grounded_v3_ar.json `
  --out-json data\test_query\eval_report.json `
  --out-csv data\test_query\ragas_evaluation_results.csv
```

只重评指定的 1-based 样本：

```powershell
.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --static data\test_query\eval_answers_210_grounded_v3_ar.json `
  --indices 18,86,92
```

配置通过 `.env` 注入：`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_NAME`。密钥不得写入文档或提交 Git。

## 7. 面试与简历口径

推荐：

> 构建 210 组 Ragas 四指标评测集，定位过度拒答、回答套话和重排内容错位等问题；在同一主裁判口径下将综合分由约 0.675 优化至 0.730，其中 Answer Relevancy 由 0.317 提升至 0.501，Faithfulness 由 0.785 提升至 0.816。

不要说：

- “0.730 是临床准确率”——它是四项自动指标的等权综合。
- “六模型 ensemble 得到 0.730”——当前不是同题 ensemble。
- “评分证明没有幻觉”——自动评测不能替代医疗人工审核。
- “DeepSeek 一定比 GLM 打分高”——现有少量同题结果不支持该结论。

## 8. 尚未完成的验证

- 当前 Milvus 服务未启动，无法在本次文档校准中复核线上集合 schema 和过滤效果；启动后应运行 `scripts/check_chunk_type_filter.py --with-search`。
- 需要补充人工分层复核，尤其是无答案、用药、急症和近邻混淆问题。
- 若要比较裁判模型，应在同一批样本上做成对评测，而不是给不同裁判分配不同题块。
