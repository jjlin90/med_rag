# 医疗 RAG 项目汇报素材（代码核验版）

## 项目事实

- 入口：FastAPI 后端、Streamlit 问答界面、CLI；`frontend/` 是 PPT 演示项目。
- 数据：2570 个清洗 Markdown；20,816 条分块记录，其中逻辑子块 16,880、父块 3,936。
- 分块：child 400、parent 2000、overlap 60。
- 向量：BGE-M3 dense+sparse，ColBERT 关闭，dense 维度 1024。
- 检索：Milvus 配置 sparse 0.7 + dense 1.0；正常 L0 为 Top-16 子块 → 截 Top-5 → 父块回溯 → 子证据重排 → Top-2 父块。
- 快通道：通用 Redis query cache → MySQL FAQ/BM25；FAQ softmax 阈值 0.85。
- 深通道：general/medical 二分类；direct/HyDE/subquery/backtracking 四策略。
- 降级：L1 放宽过滤并保持子块粒度；L2 无证据拒答；有证据而 LLM 失败时返回来源摘录。
- 评测：210/210；F=0.8163、AR=0.5007、CP=0.8405、CR=0.7619，四项等权平均 0.7299。GLM-4.6V 主裁判，DeepSeek 只补缺失指标单元。

## 演示时必须说明的边界

- 2026-09-10 本次诊断无法连接 Milvus，因此索引和 Schema 是代码配置，不是当前在线实测。
- MySQL 历史不会自动读回生成链路；`/chat` 依赖调用方提交历史，且 Prompt 只取最后 3 个 message。
- query cache key 不包含会话、历史、来源过滤或策略。
- OCR、答案内联引用、SSE、RBAC、多租户、影子集合/灰度、生产压测均未实现或未验证。
- 代码不能证明数据授权和隐私合规。
- 0.7299 不是临床准确率；AR 0.5007 仍是当前主要低项。

## 可展示的改进方向

扩展缓存键、修复 general+grounding 空上下文冲突、接通服务端历史、增加内联引用、建立真实意图测试集、用标注召回集调检索权重、在同一裁判下做配对评测、启动 Milvus 后重建并在线验证。
