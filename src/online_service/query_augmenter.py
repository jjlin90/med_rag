"""
Query Augmenter Module
四种Query增强：原生/HyDE/子查询/回溯抽象
"""

import logging
from typing import List, Optional, Union

from ..config.settings import Config
from .llm_generator import LLMGenerator

logger = logging.getLogger(__name__)

class QueryAugmenter:
    """查询增强器"""

    def __init__(self, config: Config, llm_generator: LLMGenerator):
        self.config = config
        self.llm = llm_generator

        # 增强提示词模板
        self.hyde_prompt = """假设你是医疗专家，请针对以下问题生成一个详细的回答：

问题：{query}

回答："""

        self.subquery_prompt = """请将以下复杂问题分解为多个相关的子问题，每个子问题针对问题的不同方面。

问题：{query}

请按以下格式输出：
子问题1：[第一个子问题]
子问题2：[第二个子问题]
..."""

        self.backtracking_prompt = """请将以下具体的问题抽象化、简化，生成一个更基础、更通用的版本。

问题：{query}

简化的基础问题："""

    def direct_retrieval(self, query: str) -> str:
        """直接检索"""
        return query

    def hyde_retrieval(self, query: str) -> Optional[str]:
        """HyDE假设问题检索"""
        try:
            # 生成假设答案
            hypothesis = self.llm.generate(
                self.hyde_prompt.format(query=query),
                max_tokens=500
            )

            if hypothesis and len(hypothesis.strip()) > 0:
                logger.info("HyDE hypothesis generated successfully")
                return hypothesis
            else:
                logger.warning("HyDE generation failed, returning original query")
                return query

        except Exception as e:
            logger.error(f"HyDE retrieval failed: {str(e)}")
            return query

    def subquery_retrieval(self, query: str) -> List[str]:
        """子查询检索"""
        try:
            # 生成子查询
            subquery_text = self.llm.generate(
                self.subquery_prompt.format(query=query),
                max_tokens=200
            )

            if not subquery_text or len(subquery_text.strip()) == 0:
                logger.warning("Subquery generation failed")
                return [query]

            # 解析子查询
            subqueries = []
            lines = subquery_text.strip().split('\n')

            for line in lines:
                line = line.strip()
                if line.startswith('子问题') and '：' in line:
                    subquery = line.split('：', 1)[1].strip()
                    if subquery:
                        subqueries.append(subquery)

            if not subqueries:
                logger.warning("No valid subqueries extracted")
                return [query]

            logger.info(f"Generated {len(subqueries)} subqueries")
            return subqueries

        except Exception as e:
            logger.error(f"Subquery retrieval failed: {str(e)}")
            return [query]

    def backtracking_retrieval(self, query: str) -> Optional[str]:
        """回溯问题检索"""
        try:
            # 生成简化问题
            simplified_query = self.llm.generate(
                self.backtracking_prompt.format(query=query),
                max_tokens=100
            )

            if simplified_query and len(simplified_query.strip()) > 0:
                logger.info("Backtracking query generated successfully")
                return simplified_query.strip()
            else:
                logger.warning("Backtracking generation failed, returning original query")
                return query

        except Exception as e:
            logger.error(f"Backtracking retrieval failed: {str(e)}")
            return query

    def augment_query(self, query: str, strategy: str = 'direct') -> Union[str, List[str]]:
        """
        查询增强

        Args:
            query: 原始查询
            strategy: 增强策略 ('direct', 'hyde', 'subquery', 'backtracking')

        Returns:
            增强后的查询（字符串或查询列表）
        """
        logger.info(f"Applying {strategy} strategy to query: {query}")

        if strategy == 'direct':
            return self.direct_retrieval(query)
        elif strategy == 'hyde':
            return self.hyde_retrieval(query)
        elif strategy == 'subquery':
            return self.subquery_retrieval(query)
        elif strategy == 'backtracking':
            return self.backtracking_retrieval(query)
        else:
            logger.warning(f"Unknown strategy: {strategy}, using direct")
            return self.direct_retrieval(query)

    def batch_augment(self, queries: List[str], strategy: str = 'direct') -> List[Union[str, List[str]]]:
        """批量查询增强"""
        results = []
        for query in queries:
            result = self.augment_query(query, strategy)
            results.append(result)
        return results

    def get_strategy_description(self, strategy: str) -> str:
        """获取策略描述"""
        descriptions = {
            'direct': '直接检索，使用原始查询进行检索',
            'hyde': 'HyDE检索，先生成假设答案，用假设答案检索',
            'subquery': '子查询检索，将复杂问题分解为多个子问题分别检索',
            'backtracking': '回溯检索，将具体问题抽象为更基础的问题进行检索'
        }
        return descriptions.get(strategy, '未知策略')

    def get_available_strategies(self) -> List[str]:
        """获取可用策略列表"""
        return ['direct', 'hyde', 'subquery', 'backtracking']