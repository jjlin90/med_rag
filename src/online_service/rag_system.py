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
from .retrieval import Retrieval
from .reranker import Reranker
from .llm_generator import LLMGenerator
from .intent_classifier import IntentClassifier
from .strategy_selector import StrategySelector
from .query_augmenter import QueryAugmenter

logger = logging.getLogger(__name__)


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
                query, context="", history=history
            )
            return {
                "answer": answer or "抱歉，无法生成回答。",
                "intent": intent,
                "strategy": strategy or "direct",
                "sources": [],
                "confidence": confidence,
            }

        # 3. 专业咨询：大模型自动选择检索策略（外部指定时优先使用）
        if strategy:
            strategy = strategy.lower().strip()
            logger.info(f"使用外部指定检索策略: {strategy}")
        else:
            strategy = self.strategy_selector.select_strategy(query)
        logger.info(f"检索策略: {strategy}")

        # 4. 检索与合并（按策略执行）
        retrieval_results = self._retrieve_and_merge(query, source_filter, strategy)

        # 5. 重排序（BGE-reranker 精排）
        reranked_results = self.reranker.rerank(
            query, retrieval_results, top_k=self.config.TOP_K_RERANK
        )

        # 6. 构建上下文
        context = self._build_context(reranked_results)

        # 7. LLM 生成答案
        answer = self.llm_generator.generate_with_context(
            query, context, history=history
        )

        return {
            "answer": answer or "抱歉，无法生成回答。",
            "intent": intent,
            "strategy": strategy,
            "sources": reranked_results,
            "confidence": confidence,
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
    ) -> List[Dict]:
        """
        根据检索策略执行检索并合并结果。

        支持的策略与 EduRAG 一致：
        - direct:      直接检索
        - hyde:        假设问题检索（Hypothetical Document Embedding）
        - subquery:    子查询检索，多路合并去重
        - backtracking: 回溯问题检索，将复杂问题简化后检索
        """
        strategy = (strategy or "direct").lower().strip()

        if strategy == "direct":
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            return self.retrieval.search(query, source_filter)

        if strategy == "hyde":
            logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
            hypo_answer = self.query_augmenter.hyde_retrieval(query)
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer[:200]}...'")
            return self.retrieval.search(hypo_answer, source_filter)

        if strategy == "subquery":
            logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
            subqueries = self.query_augmenter.subquery_retrieval(query)
            logger.info(f"生成的子查询: {subqueries}")
            return self.retrieval.search_multi_queries(subqueries, source_filter)

        if strategy == "backtracking":
            logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
            simplified_query = self.query_augmenter.backtracking_retrieval(query)
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            return self.retrieval.search(simplified_query, source_filter)

        logger.warning(f"未知策略 '{strategy}'，回退到直接检索")
        return self.retrieval.search(query, source_filter)

    def _build_context(self, documents: List[Dict]) -> str:
        """
        将排序后的文档拼接为 LLM 上下文。

        如果检索结果为空，返回明确的提示，便于 LLM 给出兜底回答。
        """
        if not documents:
            return "未找到相关医学知识。"

        context_parts = []
        top_k = self.config.TOP_K_RERANK
        for i, doc in enumerate(documents[:top_k], 1):
            context_parts.append(f"【知识来源{i}】\n{doc['content']}\n")

        return "\n".join(context_parts)
