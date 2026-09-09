"""不依赖外部服务的 RAG 质量优化回归测试。"""

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.online_service.llm_generator import LLMGenerator
from src.online_service.reranker import Reranker
from src.online_service.retrieval import Retrieval


class _RecordingModel:
    def __init__(self):
        self.pairs = None

    def compute_score(self, pairs):
        self.pairs = pairs
        return [0.1, 0.9]


class QualityOptimizationTests(unittest.TestCase):
    def test_parent_keeps_best_matched_child_for_reranking(self):
        retrieval = Retrieval.__new__(Retrieval)
        retrieval.top_k_children = 5
        children = [
            {
                "id": "c-low", "parent_id": "p1", "content": "较弱命中",
                "parent_content": "完整父块", "score": 0.2,
            },
            {
                "id": "c-high", "parent_id": "p1", "content": "最相关子块",
                "parent_content": "完整父块", "score": 0.8,
            },
        ]

        parents = retrieval._backtrack_to_parents(children)

        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0]["content"], "完整父块")
        self.assertEqual(parents[0]["rerank_content"], "最相关子块")

    def test_reranker_scores_child_but_returns_parent(self):
        reranker = Reranker.__new__(Reranker)
        reranker.model = _RecordingModel()
        docs = [
            {"id": "p1", "content": "父块一", "rerank_content": "子块一"},
            {"id": "p2", "content": "父块二", "rerank_content": "子块二"},
        ]

        ranked = reranker.rerank("查询", docs, top_k=1)

        self.assertEqual(reranker.model.pairs, [("查询", "子块一"), ("查询", "子块二")])
        self.assertEqual(ranked[0]["content"], "父块二")

    def test_grounding_prompt_preserves_partial_answers(self):
        generator = LLMGenerator.__new__(LLMGenerator)
        generator.config = SimpleNamespace(LLM_GROUNDING=True)

        system_prompt = generator._build_system_prompt()
        user_prompt = generator._build_user_prompt(
            "药物多久服用一次？", "资料只说明按标签服用。",
            history=[{"role": "assistant", "content": "每四小时一次"}],
        )

        self.assertIn("能回答一部分时，先直接给出该部分", system_prompt)
        self.assertIn("完全没有任何能回应问题的事实", system_prompt)
        self.assertIn("仅用于理解指代，不是医学事实依据", user_prompt)
        self.assertIn("首句给结论", user_prompt)


if __name__ == "__main__":
    unittest.main()
