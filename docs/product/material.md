# Material: 医疗 RAG 问答系统 — 项目汇报

> 数据来源：《默沙东诊疗手册（大众版）》（Merck Manuals Consumer Version），全球权威、公开、公益性的医学科普平台。

## 1. Overview（概述）
- 项目定位：基于检索增强生成（RAG）的医疗科普问答系统，面向普通用户提供权威、可追溯的医学健康知识问答。
- 核心链路：FastAPI 后端 + Milvus 向量库 + BGE-M3 多向量嵌入 + BGE-reranker 交叉编码精排 + 大模型生成。
- 关键能力：FAQ 优先缓存、BERT 意图分类、LLM 自动检索策略、多轮对话与会话持久化。
- 数据底座：知识库 100% 来源于《默沙东诊疗手册（大众版）》公开公益医学科普资源，无商业/临床诊疗用途。

## 2. Background（背景）
- 痛点：通用大模型在医学问答中存在幻觉、引用不可靠、缺乏权威出处等问题。
- 解法：RAG 通过"召回权威知识库片段 + 大模型生成"，让每条答案"有据可依、可追溯"。
- 数据源选取：默沙东诊疗手册是全球使用最广泛的医学参考书之一，大众版免费、权威、无商业诊疗导向，天然适配 AI 科普。
- 项目采用 vibecoding 模式快速迭代，已形成可运行的端到端链路（离线建库 → 在线问答）。

## 3. Key Info（关键信息）
- 数据规模：清洗后约 **2570 篇** Markdown；分块后约 **20816 个**文本块；单块平均 **462–555** 字符，最大约 **1998** 字符。
- 向量库：Milvus 集合 `med_msd_consumer_chunk` 与分块**一一对应**。
- 离线分块：父子分块（子块 **400** / 父块 **2000** 字符），兼顾召回精度与上下文完整。
- 嵌入模型：BGE-M3 输出**稠密 + 稀疏 + 多向量**，支持 Milvus 混合检索（dense + sparse + multi-vector）。
- 在线六步：FAQ 缓存 → BERT 意图分类 → LLM 检索策略选择 → Milvus 检索合并(Top8) → BGE-reranker 精排(Top4) → LLM 生成。
- 检索策略：direct / hyde / subquery / backtracking，由 LLM **自动选择**适配问题类型。
- 缓存与 FAQ：Redis 一级缓存 + MySQL BM25 FAQ 二级；**医疗类问题跳过 FAQ 直走 RAG**。
- 多轮对话：会话历史持久化于 MySQL，支持上下文连贯问答与会话 ID 持久化。
- 评估：Ragas 四维指标（faithfulness / answer_relevancy / context_precision / context_recall）。

## 4. Evidence（证据 / 案例）
- 案例 A：用户问"1 型糖尿病"，系统返回"胰腺 β 细胞自身免疫破坏、胰岛素生成不足"等准确病理描述，来源可追溯至默沙东手册。
- 案例 B（缺陷修复）：用户问"医生，我头痛"曾误命中 FAQ 中"声带息肉"条目（BM25 阈值 0.5 失效，实际分数达 ~7），已通过"医疗问题跳过 FAQ + 关键词相关性守卫"双路径修复，现走 RAG 返回正确内容。
- 证据：Milvus 混合检索 + BGE-reranker 交叉编码精排显著提升答案相关性与可信度。

## 5. Analysis（分析）
- FAQ 优先 vs RAG：高频重复问题命中缓存直答，省时省钱；医疗问题直走 RAG 保证权威性，避免标题 BM25 抢答。
- 多向量 vs 单向量：BGE-M3 稠密+稀疏+多向量提升召回率与鲁棒性，对医学同义表述更友好。
- 意图分类价值：general（通用）直答、medical（医疗）检索，避免无关检索浪费与误答。
- 工程挑战：Windows 下 uv venv 缺 VC++ 运行时导致原生库崩溃（已用自包含 DLL 修复）；Redis 需配密码（Docker `milvus-redis`，密码 1234）。

## 6. Outlook（展望）
- 扩展专科覆盖：纳入更多慢病、罕见病及康复/营养科普内容。
- 完善 Ragas 量化评估闭环，持续监控 faithfulness / answer_relevancy 等指标。
- 前端交互优化：增强来源引用展示、检索过程可视化。
- 提升可复现性：venv 运行时自包含脚本、依赖固化（websockets 等）。

## Summary
- High-authority：默沙东诊疗手册（全球权威公益医学资源）、项目文档（README / architecture / data_source）。
- Gaps：Ragas 量化评估分数待持续采集；专科覆盖率可进一步提升。
