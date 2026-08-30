# 章节规划 — 医疗 RAG 问答系统项目汇报

> PPT 类型：Report/Summary（项目汇报）｜金字塔结构：结论先行（背景 → 架构 → 能力 → 成果/展望）
> 核心信息（One Takeaway）：基于《默沙东诊疗手册（大众版）》权威知识库，本项目构建了"FAQ 优先 + RAG 增强 + 多轮对话"的可运行医疗科普问答系统。
> 风格：Tech（科技蓝紫）｜总页数：17

---

## Page 1: 封面
- **Page Type**: Cover
- **Page Title**: 医疗 RAG 问答系统
- **Page Subtitle**: 基于默沙东诊疗手册大众版的知识增强问答 · 项目汇报
- **Selected Template**: 
- **Content Structure**: 主标题"医疗 RAG 问答系统"；副标题"基于《默沙东诊疗手册（大众版）》的知识增强问答"；落款"项目汇报 · 2026.08"；底部标签"FastAPI · Milvus · BGE-M3 · RAG"。
- **Content Density**: Light
- **Narrative Role**: 建立专业第一印象，点明项目主题与权威数据来源。
- **Image Requirements**: 无（封面以文字 + 几何装饰为主）
- **Page Weight**: 核心页

## Page 2: 目录
- **Page Type**: TOC
- **Page Title**: 目录
- **Selected Template**: 
- **Content Structure**: 四个章节导航——01 项目概览（背景与数据来源）；02 技术架构（离线建库与在线检索）；03 核心能力（路由/多轮/评估）；04 工程成果与展望。
- **Content Density**: Light
- **Narrative Role**: 给读者全局地图，建立后续逻辑预期。
- **Image Requirements**: 无
- **Page Weight**: 次要页

## Page 3: 第一章 过渡页
- **Page Type**: Transition
- **Page Title**: 项目概览
- **Page Subtitle**: 01 / 背景、目标与数据来源
- **Selected Template**: 
- **Content Structure**: 章节编号 01；章节标题"项目概览"；一句话导语"为什么做、做什么、数据从哪来"。
- **Content Density**: Light
- **Narrative Role**: 进入第一章，聚焦"为何与何据"。
- **Image Requirements**: 无
- **Page Weight**: 过渡页

## Page 4: 项目背景与目标
- **Page Type**: Content
- **Page Title**: 背景与目标
- **Selected Template**: 
- **Content Structure**: Problem/Solution 结构——
  - 痛点：通用大模型医学问答存在**幻觉、引用不可靠、缺权威出处**三大问题；用户难以判断答案是否可信。
  - 目标：构建"**答案有据可依、可追溯**"的医疗科普问答系统，让每一条回复都能回溯到默沙东手册权威内容。
  - 解法：采用 RAG（检索增强生成）——先召回权威知识片段，再由大模型组织成自然语言回答。
  - 价值：高频问题缓存直答省成本；医疗问题走检索保证权威；支持多轮连贯对话。
- **Content Density**: Medium
- **Narrative Role**: 用痛点引出项目存在的必要性，确立价值锚点。
- **Image Requirements**: 无（卡片 + 要点布局）
- **Page Weight**: 核心页
- **Content Page Selection Rationale**: 承接封面，回答"为什么做这个项目"，为后续架构与能力铺垫动机。

## Page 5: 数据来源 — 默沙东诊疗手册大众版
- **Page Type**: Content
- **Page Title**: 数据来源
- **Selected Template**: 
- **Content Structure**: Data 结构——
  - 来源：**《默沙东诊疗手册（大众版）》**，全球权威、公开、公益医学科普平台，无商业诊疗导向。
  - 规模：清洗后约 **2570 篇** Markdown；分块后约 **20816 个**文本块。
  - 粒度：单块平均 **462–555** 字符，最大约 **1998** 字符；Milvus 集合 `med_msd_consumer_chunk` 与分块一一对应。
  - 覆盖：内分泌（糖尿病/甲亢）、心脑血管、呼吸、消化、骨科、皮肤、儿科、妇科等。
  - 合规：仅用于技术学习与学术演示，无患者隐私、无侵权爬取。
- **Content Density**: Medium
- **Narrative Role**: 以权威数据来源建立可信度，这是全 PPT 的"信任基石"。
- **Image Requirements**: 无（数据卡片 + 类目标签）
- **Page Weight**: 核心页
- **Content Page Selection Rationale**: 直接响应"数据来源是默沙东诊疗手册大众版"的明确诉求，突出权威与合规。

## Page 6: 第二章 过渡页
- **Page Type**: Transition
- **Page Title**: 技术架构
- **Page Subtitle**: 02 / 离线建库与在线检索
- **Selected Template**: 
- **Content Structure**: 章节编号 02；标题"技术架构"；导语"从 20816 个文本块到一句可信回答"。
- **Content Density**: Light
- **Narrative Role**: 进入第二章，转向"如何做"。
- **Image Requirements**: 无
- **Page Weight**: 过渡页

## Page 7: 系统总体架构
- **Page Type**: Content
- **Page Title**: 系统总体架构
- **Selected Template**: 
- **Content Structure**: Concept 结构——
  - 三层划分：**知识层**（默沙东手册 → 清洗 → 分块 → Milvus 向量库）、**服务层**（FastAPI：FAQ 缓存 / 意图分类 / 检索策略 / 重排 / 生成）、**交互层**（Streamlit 前端 + 多轮会话）。
  - 技术栈：BGE-M3 嵌入、Milvus 混合检索、BGE-reranker 精排、bert-base-chinese 意图分类、Redis 缓存、MySQL 持久化。
  - 主线：用户提问 → 服务层六步处理 → 返回带来源的答案。
- **Content Density**: Medium
- **Narrative Role**: 给出系统全景，后续页面逐层展开。
- **Image Requirements**: 无（三层结构卡片）
- **Page Weight**: 核心页
- **Content Page Selection Rationale**: 承上启下，将"数据来源"映射到"系统组成"，为离线/在线两节铺垫。

## Page 8: 离线链路 — 分块与向量化
- **Page Type**: Content
- **Page Title**: 离线建库
- **Selected Template**: 
- **Content Structure**: Process 结构——
  - 步骤1 清洗：去除导航/广告/版权，得到约 **2570 篇**干净 Markdown。
  - 步骤2 父子分块：子块 **400** 字符（精准召回）、父块 **2000** 字符（完整上下文），约 **20816** 块。
  - 步骤3 向量化：BGE-M3 生成**稠密 + 稀疏 + 多向量**，表达能力更强。
  - 步骤4 入库：写入 Milvus 集合 `med_msd_consumer_chunk`，建立混合索引。
  - 成功标准：检索延迟可控、召回率满足在线问答需求。
- **Content Density**: Medium
- **Narrative Role**: 解释"知识从哪来、怎么存"，是 RAG 质量的根基。
- **Image Requirements**: 无（四步流程条）
- **Page Weight**: 核心页
- **Content Page Selection Rationale**: 解释离线侧工程，与在线检索形成完整闭环。

## Page 9: 在线六步检索流程
- **Page Type**: Content
- **Page Title**: 在线检索流程
- **Selected Template**: 
- **Content Structure**: Process 结构（六步）——
  - ① FAQ 缓存：Redis 命中即直答；未命中查 MySQL BM25。
  - ② 意图分类：BERT 区分 general / medical。
  - ③ 策略选择：LLM 自动选 direct/hyde/subquery/backtracking。
  - ④ 检索合并：Milvus 混合检索，粗排 Top-16（过滤下推，只召回子块）。
  - ⑤ Small-to-Big：Top-5 子块 → parent_id 回溯父块去重 → BGE-reranker 交叉编码精排 Top-2。
  - ⑥ 生成：医疗 Prompt + 多轮历史 → 返回 {答案, 意图, 策略, 来源}。
- **Content Density**: Heavy
- **Narrative Role**: 展示系统核心运转机制，是技术架构的高光页。
- **Image Requirements**: 无（六步竖向/横向流程）
- **Page Weight**: 核心页
- **Content Page Selection Rationale**: 在线流程是全系统最关键的链路，需逐步行清晰呈现。

## Page 10: 检索策略与重排序
- **Page Type**: Content
- **Page Title**: 策略与重排
- **Selected Template**: 
- **Content Structure**: Comparison 结构——
  - 四种检索策略（LLM 自动选）：direct（直接向量检索）、hyde（假设性文档增强）、subquery（子问题拆解）、backtracking（回溯纠错）。
  - 重排序：BGE-reranker 作为 CrossEncoder，对回溯出的候选父块做精细相关性打分，输出 Top-2。
  - 混合检索：dense（语义 1.0）+ sparse（词权 0.7）加权融合，提升医学同义表述与专名命中（ColBERT 未启用）。
  - 收益：相比单向量 + 无重排，答案相关性/可信度显著提升。
- **Content Density**: Medium
- **Narrative Role**: 解释"为什么检索得准"，突出策略与精排的工程价值。
- **Image Requirements**: 无（策略对比卡片 + 重排说明）
- **Page Weight**: 核心页
- **Content Page Selection Rationale**: 深入在线流程的关键环节，体现 RAG 检索的专业性。

## Page 11: 第三章 过渡页
- **Page Type**: Transition
- **Page Title**: 核心能力
- **Page Subtitle**: 03 / 智能路由 · 多轮对话 · 质量评估
- **Selected Template**: 
- **Content Structure**: 章节编号 03；标题"核心能力"；导语"让系统更聪明、更连贯、更可信"。
- **Content Density**: Light
- **Narrative Role**: 进入第三章，聚焦差异化能力。
- **Image Requirements**: 无
- **Page Weight**: 过渡页

## Page 12: 意图分类与 FAQ 智能路由
- **Page Type**: Content
- **Page Title**: 智能路由
- **Selected Template**: 
- **Content Structure**: Concept 结构——
  - 意图分类：bert-base-chinese 微调，输出 general / medical 及置信度。
  - 路由规则：**所有问题先过 FAQ 快通道**（Redis → MySQL+BM25，softmax 归一化阈值 0.85），未命中才降级 RAG 深通道；深通道内再由 BERT 判定 general（直答）/ medical（检索）。
  - FAQ 守卫：查询须与 FAQ 问题共享有效关键词，否则回退 RAG，杜绝标题 BM25 误答（如"头痛"误命中"声带息肉"已修复）。
  - 收益：高频问题缓存直答省成本，医疗问题检索保权威，错误命中趋近于零。
- **Content Density**: Medium
- **Narrative Role**: 体现系统的"判断力"，区别于朴素 RAG。
- **Image Requirements**: 无（路由决策图 + 守卫说明）
- **Page Weight**: 核心页
- **Content Page Selection Rationale**: 路由是本项目关键设计，且包含真实 bug 修复案例，说服力强。

## Page 13: 多轮对话与会话持久化
- **Page Type**: Content
- **Page Title**: 多轮对话
- **Selected Template**: 
- **Content Structure**: Concept 结构——
  - 会话管理：每次对话分配会话 ID，写入网址，刷新不丢失。
  - 历史持久化：多轮上下文存储于 MySQL，支持上下文连贯追问（如先问"1 型糖尿病"再问"饮食注意"）。
  - 上下文窗口：最近若干轮拼入 Prompt，兼顾连贯与成本。
  - 价值：从"一问一答"升级为"连续问诊式"交互，体验更自然。
- **Content Density**: Medium
- **Narrative Role**: 展示交互层面的产品成熟度。
- **Image Requirements**: 无（会话流卡片）
- **Page Weight**: 次要页
- **Content Page Selection Rationale**: 多轮能力是医疗咨询场景的刚需，呼应"会话持久化"特性。

## Page 14: 评估体系 — Ragas 四指标
- **Page Type**: Content
- **Page Title**: 质量评估
- **Selected Template**: 
- **Content Structure**: Data 结构——
  - 评估框架：Ragas 自动评估生成质量。
  - 四指标：faithfulness（忠实度，答案不杜撰）、answer_relevancy（回答相关性）、context_precision（上下文精确度）、context_recall（上下文召回率）。
  - 闭环：离线评估 → 定位弱项 → 调参（分块/策略/重排）→ 复测。
  - 下一步：将评估分数持续采集，建立量化看板，指导迭代。
- **Content Density**: Medium
- **Narrative Role**: 用评估维度证明系统"可度量、可改进"。
- **Image Requirements**: 无（四指标卡片）
- **Page Weight**: 核心页
- **Content Page Selection Rationale**: 评估是 RAG 项目专业度的体现，回应"如何知道答得好不好"。

## Page 15: 工程实践与成果
- **Page Type**: Content
- **Page Title**: 工程成果
- **Selected Template**: 
- **Content Structure**: Summary 结构——
  - 可运行：离线建库 + 在线 API（端口 8005）+ Streamlit 前端（8501）端到端打通。
  - 稳定性：修复 Windows uv venv 缺 VC++ 运行时导致的原生库崩溃（自包含 DLL 8 个）。
  - 可用性：修复 FAQ 误答缺陷（BM25 softmax 归一化 + 阈值 0.85，从评分尺度根因解决）；检索失败时分层降级（L0/L1/L2），不再让 LLM 凭空作答。
  - 中间件：Milvus / Redis（Docker，密码 1234）/ MySQL 已联通并验证。
  - 关键数字：2570 篇文档、20816 块、BGE-M3 多向量、六步流程。
- **Content Density**: Medium
- **Narrative Role**: 汇总"已经做成了什么"，形成阶段成果结论。
- **Image Requirements**: 无（成果清单 + 关键数字）
- **Page Weight**: 核心页
- **Content Page Selection Rationale**: 收束前三章，给出可交付成果清单。

## Page 16: 后续计划
- **Page Type**: Content
- **Page Title**: 后续计划
- **Selected Template**: 
- **Content Structure**: Trend 结构——
  - 数据扩展：纳入更多专科（罕见病、营养/康复科普），提升覆盖。
  - 评估闭环：落地 Ragas 量化看板，持续监控四指标。
  - 前端增强：来源引用可视化、检索过程透明化。
  - 可复现性：venv 运行时自包含脚本、依赖固化（websockets 等）。
  - 目标：从"可用 Demo"走向"可评测、可维护"的成熟系统。
- **Content Density**: Medium
- **Narrative Role**: 指向未来，体现项目可持续演进。
- **Image Requirements**: 无（四方向卡片）
- **Page Weight**: 次要页
- **Content Page Selection Rationale**: 给汇报一个前瞻收尾，呼应"展望"章节诉求。

## Page 17: 结尾
- **Page Type**: Ending
- **Page Title**: 谢谢观看
- **Page Subtitle**: 基于默沙东诊疗手册大众版 · 让医疗科普有据可依
- **Selected Template**: 
- **Content Structure**: 致谢语"谢谢观看"；一句价值主张"权威知识库 + RAG = 可信医疗科普"；底部"FastAPI · Milvus · BGE-M3"。
- **Content Density**: Light
- **Narrative Role**: 收尾并强化核心信息。
- **Image Requirements**: 无
- **Page Weight**: 次要页
