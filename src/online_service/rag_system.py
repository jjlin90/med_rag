"""
RAG System Module
核心 RAG 系统，对齐 EduRAG 主流程图中的「RAG 生成答案」六步：

    意图分类 → 策略选择 → 检索与合并 → 重排序 → 构建上下文 → LLM 生成

本模块不处理缓存、FAQ、会话管理等外围逻辑，只封装纯 RAG 生成流程，
便于被 FastAPI、命令行、测试脚本等多种入口复用。
"""

import logging
from typing import List, Dict, Any, Optional

from ..config.settings import Config
from .retrieval import Retrieval, RetrievalResult, DEGRADE_OK, DEGRADE_L1, DEGRADE_L2
from .reranker import Reranker
from .llm_generator import LLMGenerator
from .intent_classifier import IntentClassifier
from .strategy_selector import StrategySelector
from .query_augmenter import QueryAugmenter

logger = logging.getLogger(__name__)

# L2 安全拒答话术。检索不到依据时**不调用 LLM** —— 医疗场景下让模型
# 用参数知识硬答是整条链路风险最高的行为：它会编造，且语气权威。
# 宁可少答，不可错答。
NO_CONTEXT_ANSWER = (
    "抱歉，我在现有医学资料中没有找到关于这个问题的可靠依据。"
    "为避免给出不准确的信息，我无法作答。建议咨询专业医生获取针对性意见。"
)
SERVICE_UNAVAILABLE_ANSWER = (
    "抱歉，医学知识检索服务暂时不可用，请稍后再试。"
    "如情况紧急，请直接就医或咨询专业医生。"
)


class RAGSystem:
    """
    RAG 系统核心类（对齐 EduRAG 的 rag_qa.core.rag_system.RAGSystem）。

    允许外部传入已初始化的组件（避免在 FastAPI 等场景中重复加载大模型），
    如果未传入，则由本类自行按 Config 初始化。
    """

    def __init__(
        self,
        config: Config,
        retrieval: Optional[Retrieval] = None,
        reranker: Optional[Reranker] = None,
        llm_generator: Optional[LLMGenerator] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        strategy_selector: Optional[StrategySelector] = None,
        query_augmenter: Optional[QueryAugmenter] = None,
    ):
        self.config = config

        # 复用外部组件或自行初始化
        self.retrieval = retrieval or Retrieval(config)
        self.reranker = reranker or Reranker(config)
        self.llm_generator = llm_generator or LLMGenerator(config)
        self.intent_classifier = intent_classifier or IntentClassifier(config)
        # strategy_selector 与 query_augmenter 依赖 llm_generator
        self.strategy_selector = strategy_selector or StrategySelector(config, self.llm_generator)
        self.query_augmenter = query_augmenter or QueryAugmenter(config, self.llm_generator)

    def generate(
        self,
        query: str,
        source_filter: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        RAG 生成答案并返回完整元信息（answer / intent / strategy / sources）。

        med_rag 的 FastAPI 需要返回 sources、strategy 等字段，因此提供此接口；
        如果只想拿到答案字符串，请使用 generate_answer()。

        Args:
            query: 用户查询
            source_filter: 学科/来源过滤（可选）
            history: 多轮对话历史（可选）
            strategy: 强制指定检索策略（可选）。未指定时由大模型自动判断。
        """
        logger.info(f"RAGSystem 开始处理查询: '{query}', source_filter={source_filter}")

        # 1. 意图分类：通用知识 / 专业咨询
        intent_result = self.intent_classifier.predict(query)
        intent = intent_result.get("intent", "medical")
        confidence = intent_result.get("confidence", 0.0)
        logger.info(f"查询分类结果: {intent}")

        # 2. 通用知识：直接调用 LLM，不检索医学知识库
        if intent == "general":
            logger.info("通用知识查询，直接调用 LLM")
            answer = self.llm_generator.generate_with_context(
                query, context="", history=history,
                system_prompt='你是一个通用知识助手。回答用户的一般问题；无法确认时说明不确定性。不要提供个体化医疗诊断或治疗方案。',
            )
            return {
                "answer": answer or "抱歉，无法生成回答。",
                "intent": intent,
                "strategy": strategy or "direct",
                "sources": [],
                "confidence": confidence,
                "degraded": not bool(answer),
                "degrade_level": DEGRADE_L2 if not answer else DEGRADE_OK,
                "degrade_reason": "llm_unavailable" if not answer else "",
            }

        # 3. 专业咨询：大模型自动选择检索策略（外部指定时优先使用）
        if strategy:
            strategy = strategy.lower().strip()
            logger.info(f"使用外部指定检索策略: {strategy}")
        else:
            strategy = self.strategy_selector.select_strategy(query)
        logger.info(f"检索策略: {strategy}")

        # 4. 检索与合并（对齐 EduRag: 混合检索 → Small-to-Big 子块→父块 → 去重）
        #    使用 search_child_to_parent 而非普通 search，确保：
        #    - 粗排召回的是细粒度子块（~400字，定位精准）
        #    - 通过 parent_id 回溯到父块（~2000字，上下文完整）
        #    - 父块去重后交给 CrossEncoder 精排
        #    返回 RetrievalResult，携带降级水位（L0 正常 / L1 同粒度降级 / L2 无召回）
        retrieval_result = self._retrieve_and_merge(query, source_filter, strategy)

        # 4.5 L2 拦截：无召回或基础设施故障时，不进入 LLM。
        #     这是医疗场景的红线 —— 没有检索依据却让 LLM 作答，等同于
        #     用模型参数知识编造医疗建议，且输出语气与有据可依时一样权威，
        #     用户无从分辨。宁可少答，不可错答。
        if not retrieval_result.is_usable:
            if retrieval_result.error or not getattr(self.config, 'ALLOW_LLM_WHEN_NO_CONTEXT', False):
                answer = (SERVICE_UNAVAILABLE_ANSWER if retrieval_result.error
                          else NO_CONTEXT_ANSWER)
                logger.warning(
                    "RAGSystem: L2 拒答 (reason=%s, error=%s, query=%r)",
                    retrieval_result.degrade_reason, retrieval_result.error, query
                )
                return {
                    "answer": answer,
                    "intent": intent,
                    "strategy": strategy,
                    "sources": [],
                    "confidence": 0.0,
                    "degraded": True,
                    "degrade_level": DEGRADE_L2,
                    "degrade_reason": retrieval_result.degrade_reason,
                }

        # 5. 重排序（BGE-reranker CrossEncoder 精排，对齐 EduRag "Reranker 精排 Top-2"）
        try:
            reranked_results = self.reranker.rerank(
                query, retrieval_result.documents, top_k=self.config.TOP_K_RERANK
            )
        except Exception:
            logger.exception('Reranking unavailable')
            return {
                'answer': SERVICE_UNAVAILABLE_ANSWER, 'intent': intent, 'strategy': strategy,
                'sources': [], 'confidence': 0.0, 'degraded': True,
                'degrade_level': DEGRADE_L2, 'degrade_reason': 'reranker_unavailable',
            }
        reranked_results = [doc for doc in reranked_results if (doc.get('content') or '').strip()]
        if not reranked_results and not getattr(self.config, 'ALLOW_LLM_WHEN_NO_CONTEXT', False):
            return {
                'answer': NO_CONTEXT_ANSWER, 'intent': intent, 'strategy': strategy,
                'sources': [], 'confidence': 0.0, 'degraded': True,
                'degrade_level': DEGRADE_L2, 'degrade_reason': 'no_context_after_rerank',
            }

        # 6. 构建上下文
        context = self._build_context(reranked_results)

        # 7. LLM 生成答案
        answer = self.llm_generator.generate_with_context(
            query, context, history=history
        )

        # LLM 不可用时（返回空），有检索结果也要降级为「只给原文 + 就医建议」，
        # 而不是抛一个空答案——用户至少能看到检索到的原始资料。
        if not answer:
            logger.error("RAGSystem: LLM 生成失败，降级为原文摘录 (query=%r)", query)
            return {
                "answer": self._build_source_only_answer(reranked_results),
                "intent": intent,
                "strategy": strategy,
                "sources": reranked_results,
                "confidence": 0.0,
                "degraded": True,
                "degrade_level": DEGRADE_L2,
                "degrade_reason": "llm_unavailable",
            }

        return {
            "answer": answer,
            "intent": intent,
            "strategy": strategy,
            "sources": reranked_results,
            "confidence": confidence,
            "degraded": retrieval_result.degraded,
            "degrade_level": retrieval_result.degrade_level,
            "degrade_reason": retrieval_result.degrade_reason,
        }

    def generate_answer(
        self,
        query: str,
        source_filter: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        strategy: Optional[str] = None,
    ) -> str:
        """
        RAG 生成答案主入口（对齐 EduRAG 的 RAGSystem.generate_answer，返回字符串）。

        Args:
            query: 用户查询
            source_filter: 学科/来源过滤（可选）
            history: 多轮对话历史（可选），格式 [{"role": "user/assistant", "content": "..."}, ...]
            strategy: 强制指定检索策略（可选）

        Returns:
            生成的回答文本
        """
        return self.generate(query, source_filter, history, strategy)["answer"]

    def _retrieve_and_merge(
        self,
        query: str,
        source_filter: Optional[str],
        strategy: Optional[str],
    ) -> RetrievalResult:
        """
        根据检索策略执行检索并合并结果（对齐 EduRag 深通道）。

        所有策略统一走 Small-to-Big 链路：
          混合检索(dense+sparse) → Top-K子块 → 父块回溯去重 → 供精排

        支持的策略：
        - direct:      直接检索（默认）
        - hyde:        假设问题检索（Hypothetical Document Embedding）
        - subquery:    子查询检索，多路合并去重
        - backtracking: 回溯问题检索，将复杂问题简化后检索

        Returns:
            RetrievalResult，携带降级水位供上层 decide 是否拒答。
        """
        strategy = (strategy or "direct").lower().strip()

        if strategy == "direct":
            logger.info(f"Small-to-Big 直接检索 (查询: '{query}')")
            return self.retrieval.search_child_to_parent(query, source_filter)

        if strategy == "hyde":
            logger.info(f"Small-to-Big + HyDE 策略 (查询: '{query}')")
            hypo_answer = self.query_augmenter.hyde_retrieval(query)
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer[:200]}...'")
            return self.retrieval.search_child_to_parent(hypo_answer, source_filter)

        if strategy == "subquery":
            logger.info(f"Small-to-Big + 子查询策略 (查询: '{query}')")
            subqueries = self.query_augmenter.subquery_retrieval(query)
            logger.info(f"生成的子查询: {subqueries}")
            # 多查询合并：每个子查询分别走 Small-to-Big，再按文档 ID 去重
            per_query = [self.retrieval.search_child_to_parent(sq, source_filter)
                         for sq in subqueries]
            return self._merge_retrieval_results(per_query)

        if strategy == "backtracking":
            logger.info(f"Small-to-Big + 回溯策略 (查询: '{query}')")
            simplified_query = self.query_augmenter.backtracking_retrieval(query)
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            return self.retrieval.search_child_to_parent(simplified_query, source_filter)

        logger.warning(f"未知策略 '{strategy}'，回退到 Small-to-Big 直接检索")
        return self.retrieval.search_child_to_parent(query, source_filter)

    @staticmethod
    def _merge_retrieval_results(results: List[RetrievalResult]) -> RetrievalResult:
        """
        合并多路检索结果（子查询策略），并正确传播降级水位。

        水位规则：
          - 任一路出现 error → 整体 L2，禁止把部分服务故障掩盖为正常结果。
          - 无 error 且合并后有文档 → 取有产出分支的最高水位。
            某路正常空召回可忽略，但 L1 产出不能被另一条 L0 隐藏。
          - 合并后无文档 → 取最严重的水位（无召回 / 故障）。
        """
        best_by_id: Dict[str, Dict] = {}
        for r in results:
            for doc in r.documents:
                rid = doc.get('id')
                if rid is None:
                    continue
                if rid not in best_by_id or doc.get('score', 0) > best_by_id[rid].get('score', 0):
                    best_by_id[rid] = doc

        documents = sorted(best_by_id.values(), key=lambda x: x.get('score', 0), reverse=True)

        if documents:
            productive = [r for r in results if r.documents]
            level = max(r.degrade_level for r in productive)
            reason = next(
                (r.degrade_reason for r in productive if r.degrade_level == level), ""
            )
        else:
            level = max((r.degrade_level for r in results), default=DEGRADE_L2)
            reason = next(
                (r.degrade_reason for r in results if r.degrade_level == level), "no_recall"
            )

        # 任意一路出现基础设施故障都要透传，不能因为有其他路成功就吞掉
        error = next((r.error for r in results if r.error), None)
        if error:
            level = DEGRADE_L2
            reason = 'retrieval_error'

        return RetrievalResult(
            documents=documents,
            degraded=level > DEGRADE_OK,
            degrade_level=level,
            degrade_reason=reason,
            orphan_count=sum(r.orphan_count for r in results),
            error=error,
        )

    @staticmethod
    def _build_source_only_answer(documents: List[Dict]) -> str:
        """
        LLM 不可用时的降级：直接给出检索到的原文摘录 + 就医提示。

        同样是降级，这条路径是安全的——它没有新增任何模型生成内容，
        只是把已有依据原样呈现，并明确告知用户当前无法生成解读。
        """
        if not documents:
            return SERVICE_UNAVAILABLE_ANSWER

        parts = ["抱歉，答案生成服务暂时不可用。以下是检索到的相关资料原文，仅供参考：\n"]
        for i, doc in enumerate(documents[:2], 1):
            snippet = (doc.get('content') or '').strip()
            if len(snippet) > 500:
                snippet = snippet[:500] + "…"
            source = doc.get('source') or '未知来源'
            parts.append(f"【资料{i}｜来源：{source}】\n{snippet}\n")

        parts.append("\n以上为原始资料摘录，未经解读。如需专业意见请咨询医生。")
        return "\n".join(parts)

    def _build_context(self, documents: List[Dict]) -> str:
        """
        将排序后的父块拼接为 LLM 上下文（对齐 EduRag "Top-2 父块 → 拼上下文"）。

        Small-to-Big 链路保证传入的是父块（~1200-2000字，上下文完整），
        默认取 Top-2 精排后的父块拼装即可。
        """
        if not documents:
            return "未找到相关医学知识。"

        top_k = self.config.TOP_K_RERANK  # 对齐 EduRag: 默认 2
        context_parts = []
        for i, doc in enumerate(documents[:top_k], 1):
            context_parts.append(f"【知识来源{i}】\n{doc['content']}\n")

        return "\n".join(context_parts)
