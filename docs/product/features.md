# PPT Outline

## Overview
本 PPT 为"医疗 RAG 问答系统"项目汇报，面向技术评审 / 项目汇报场景。核心信息（One Takeaway）：基于《默沙东诊疗手册（大众版）》权威知识库，本项目构建了"FAQ 优先 + RAG 增强 + 多轮对话"的可运行医疗科普问答系统。全篇 17 页，按金字塔结构分四章——项目概览、技术架构、核心能力、工程成果与展望。设计风格为 Tech（深色科技蓝紫），突出 AI / 工程专业感。

## Outline Content

### Page 1 · Cover
- Type: Cover
- Title: 医疗 RAG 问答系统
- Subtitle: 基于默沙东诊疗手册大众版的知识增强问答 · 项目汇报
- Content: 主标题 + 副标题 + 落款"项目汇报 · 2026.08" + 标签 FastAPI·Milvus·BGE-M3·RAG

### Page 2 · TOC
- Type: TOC
- Title: 目录
- Content: 01 项目概览（背景与数据来源）｜02 技术架构（离线建库与在线检索）｜03 核心能力（路由/多轮/评估）｜04 工程成果与展望

### Page 3 · Transition
- Type: Transition
- Title: 项目概览
- Subtitle: 01 / 背景、目标与数据来源

### Page 4 · Content
- Type: Content
- Title: 背景与目标
- Content: 痛点（幻觉/引用不可靠/缺权威出处）→ 目标（答案有据可依、可追溯）→ 解法（RAG 检索+生成）→ 价值（缓存省成本/医疗检索保权威/多轮连贯）

### Page 5 · Content
- Type: Content
- Title: 数据来源
- Content: 来源=默沙东诊疗手册大众版（权威公益）；规模=2570 篇 MD / 20816 块；粒度=462–555 字符/块，最大 1998；覆盖=内分泌/心脑血管/呼吸等；合规=仅学习演示、无隐私

### Page 6 · Transition
- Type: Transition
- Title: 技术架构
- Subtitle: 02 / 离线建库与在线检索

### Page 7 · Content
- Type: Content
- Title: 系统总体架构
- Content: 三层（知识层/服务层/交互层）；技术栈 BGE-M3·Milvus·BGE-reranker·bert-base-chinese·Redis·MySQL；主线：提问→六步处理→带来源答案

### Page 8 · Content
- Type: Content
- Title: 离线建库
- Content: 清洗（2570 篇）→ 父子分块（400/2000，20816 块）→ BGE-M3 多向量化 → 入库 Milvus（med_msd_consumer_chunk，混合索引）

### Page 9 · Content
- Type: Content
- Title: 在线检索流程
- Content: 六步——FAQ缓存→意图分类→策略选择→检索合并(Top8)→重排(Top4)→生成；输出 {答案,意图,策略,来源}

### Page 10 · Content
- Type: Content
- Title: 策略与重排
- Content: 四策略 direct/hyde/subquery/backtracking（LLM 自动选）；BGE-reranker 交叉编码精排 Top4；dense+sparse+multi-vector 混合检索提升召回

### Page 11 · Transition
- Type: Transition
- Title: 核心能力
- Subtitle: 03 / 智能路由 · 多轮对话 · 质量评估

### Page 12 · Content
- Type: Content
- Title: 智能路由
- Content: 意图分类 general/medical；医疗问题跳过 FAQ 直走 RAG；FAQ 关键词守卫防误答（"头痛"误命中"声带息肉"已修复）

### Page 13 · Content
- Type: Content
- Title: 多轮对话
- Content: 会话 ID 持久化（刷新不丢失）；历史存 MySQL 支持连贯追问；上下文窗口拼入 Prompt；体验从一问一答升级为连续问诊

### Page 14 · Content
- Type: Content
- Title: 质量评估
- Content: Ragas 四指标 faithfulness/answer_relevancy/context_precision/context_recall；闭环 离线评估→定位弱项→调参→复测；下一步建量化看板

### Page 15 · Content
- Type: Content
- Title: 工程成果
- Content: 端到端打通（API 8005 + 前端 8501）；修复 Windows VC++ 运行时崩溃（自包含 8 DLL）；修复 FAQ 误答；Milvus/Redis(1234)/MySQL 已验证；关键数字 2570/20816/BGE-M3/六步

### Page 16 · Content
- Type: Content
- Title: 后续计划
- Content: 扩展专科覆盖；落地 Ragas 量化看板；前端来源引用可视化；venv 运行时自包含与依赖固化；目标走向可评测、可维护

### Page 17 · Ending
- Type: Ending
- Title: 谢谢观看
- Subtitle: 基于默沙东诊疗手册大众版 · 让医疗科普有据可依

## Design Style
- 风格（SlideStyle）：Tech（深色科技风）
- 配色：主色 #818CF8（靛蓝），强调色 #22D3EE（青）/ #C084FC（紫）；中性色 #94A3B8；背景 #0F172A（深蓝黑）
- 字体：标题 Montserrat + Noto Sans SC；正文 Inter + Noto Sans SC
- 视觉：网格/电路/节点装饰，统一深色底 + 浅色文字，卡片化信息分区
