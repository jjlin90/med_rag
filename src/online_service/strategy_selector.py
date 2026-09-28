"""
Strategy Selector Module
由大模型自动判断检索策略

参考 EduRag 项目的 strategy_selector 设计，但保持 med_rag 现有英文策略键一致：
- direct      : 直接检索
- hyde        : 假设问题检索（Hypothetical Document Embedding）
- subquery    : 子查询检索
- backtracking: 回溯问题检索
"""

import logging
import re
from typing import Optional

from ..config.settings import Config
from .llm_generator import LLMGenerator

logger = logging.getLogger(__name__)

# 合法策略集合，与 query_augmenter.py 保持一致
VALID_STRATEGIES = {"direct", "hyde", "subquery", "backtracking"}

# 策略名称映射（中文 -> 英文），用于兼容 LLM 可能返回的中文策略名
STRATEGY_NAME_MAP = {
    "直接检索": "direct",
    "假设问题检索": "hyde",
    "hyde检索": "hyde",
    "子查询检索": "subquery",
    "子问题检索": "subquery",
    "回溯问题检索": "backtracking",
    "回溯检索": "backtracking",
}


class StrategySelector:
    """检索策略选择器：根据用户查询自动选择最合适的 RAG 检索策略"""

    def __init__(self, config: Config, llm_generator: LLMGenerator):
        self.config = config
        self.llm = llm_generator

        self.strategy_prompt = """你是一位智能检索策略专家。请分析下面的用户查询，并从以下四种策略中选择最适合的一种。直接返回策略名称（英文小写），不要解释过程。

可选策略：
1. direct      - 直接检索：查询意图明确，适合用原始问题直接检索医学知识库。例如："高血压的正常范围是多少？"、"1型糖尿病是什么？"
2. hyde        - 假设问题检索：查询较抽象或需要推理，先生成假设答案再用其检索。例如："糖尿病如果不控制会有什么后果？"
3. subquery    - 子查询检索：复杂查询涉及多个实体或方面，需要拆成多个子问题分别检索。例如："比较1型糖尿病和2型糖尿病的病因与治疗"
4. backtracking- 回溯问题检索：查询较复杂或具体，需要先抽象成更基础、更易检索的问题。例如："我家族有糖尿病史，最近口渴多尿，是不是得了糖尿病？"

用户查询：{query}

请只返回以下之一：direct / hyde / subquery / backtracking"""

    def select_strategy(self, query: str, default: str = "direct") -> str:
        """
        根据查询自动选择检索策略

        Args:
            query: 用户查询
            default: LLM 解析失败时的兜底策略

        Returns:
            策略名称（direct / hyde / subquery / backtracking）
        """
        if not self.llm or not self.llm.client:
            logger.warning("LLM 客户端不可用，使用默认策略 direct")
            return default

        try:
            raw = self.llm.generate(
                self.strategy_prompt.format(query=query),
                system_prompt="你只会从 direct/hyde/subquery/backtracking 中选择一个策略返回，不输出任何解释。",
                temperature=0.1,
                max_tokens=20,
            )

            strategy = self._normalize_strategy(raw)
            logger.info(f"策略选择：查询='{query}' -> strategy='{strategy}'")
            return strategy

        except Exception as e:
            logger.error(f"策略选择失败，使用默认策略 {default}：{e}")
            return default

    def _normalize_strategy(self, raw: Optional[str]) -> str:
        """
        将 LLM 返回的原始文本归一化为标准策略键
        """
        if not raw:
            return "direct"

        # 去掉代码块、引号、首尾空白，转小写
        cleaned = raw.strip().strip("`\"'").lower()
        # 若被 JSON 包裹，只取第一个合法策略词
        cleaned = re.sub(r'[^a-z\u4e00-\u9fa5]', '', cleaned)

        # 中文映射
        if cleaned in STRATEGY_NAME_MAP:
            return STRATEGY_NAME_MAP[cleaned]

        # 直接匹配英文
        for s in sorted(VALID_STRATEGIES):
            if cleaned and s in cleaned:
                return s

        # 兜底
        logger.warning(f"无法识别的策略 '{raw}'，使用默认策略 direct")
        return "direct"
