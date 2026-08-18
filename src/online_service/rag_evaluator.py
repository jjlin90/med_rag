"""
RAG 评估模块（基于 Ragas 0.2.x，对齐 EduRag 的 ragas_evaluate.py）

四项核心指标（与参考脚本一致）：
1. faithfulness（忠实度）       —— 回答是否完全基于给定上下文，无幻觉/编造。
2. answer_relevancy（答案相关性）—— 回答是否切题、完整、有针对性地回应问题。
3. context_precision（上下文精确度）—— 检索到的上下文是否聚焦、仅含相关信息。
4. context_recall（上下文召回率）  —— 上下文是否覆盖回答/标准答案所需的全部关键信息。

数据格式（与参考脚本 ragas_evaluate.py 对齐，采用 Ragas v1 字段名，会自动转换为 v2）：
    question, answer, contexts(list[str]), ground_truth

LLM 与 Embeddings 配置：
- LLM：复用项目既有的 DashScope 兼容接口（LLM_BASE_URL / LLM_API_KEY / LLM_MODEL_NAME），
  经由 langchain_openai.ChatOpenAI + ragas.LangchainLLMWrapper 接入，与参考脚本一致。
- Embeddings：复用项目本地 BGE-M3 模型（BGEEmbeddingProvider），
  不依赖任何外部 embedding 服务，保证与检索阶段同分布、且无需额外密钥。

降级策略：若 Ragas 运行失败（如依赖缺失 / BGE 不可用），自动回退到内置
LLM-as-judge（faithfulness / context_precision / answer_relevancy），保证评估接口始终可用。
"""

import json
import logging
import math
import re
from typing import List, Dict, Any, Optional

from ..config.settings import Config
from ..offline_pipeline.embedding_provider import BGEEmbeddingProvider

# Ragas 0.2.x 输出的规范字段名（evaluate 会把 v1 的 question/answer/contexts/ground_truth
# 转换成以下规范列，提取指标时需排除这些非指标列）。
RAGAS_NON_METRIC_COLS = {"user_input", "response", "retrieved_contexts", "reference"}

logger = logging.getLogger(__name__)


class _BgeEmbeddings:
    """把项目本地 BGE-M3 包装成 langchain / Ragas 兼容的 Embeddings 接口。"""

    def __init__(self, provider: BGEEmbeddingProvider):
        self.provider = provider

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        dense, _ = self.provider.generate_embeddings(list(texts))
        return dense.tolist()

    def embed_query(self, text: str) -> List[float]:
        dense, _ = self.provider.generate_embeddings([text])
        if len(dense) == 0:
            return []
        return dense[0].tolist()


class RAGEvaluator:
    """RAG 质量评估器（基于 Ragas）"""

    def __init__(self, config: Config,
                 llm_generator=None,
                 embedding_provider: Optional[BGEEmbeddingProvider] = None):
        self.config = config
        self.embedding_provider = embedding_provider
        self._ragas = self._probe_ragas()

    # --------------------------- 环境探测 ---------------------------
    def _probe_ragas(self) -> bool:
        try:
            import ragas  # noqa: F401
            from ragas.metrics import (Faithfulness, AnswerRelevancy,
                                       ContextPrecision, ContextRecall)  # noqa: F401
            from langchain_openai import ChatOpenAI  # noqa: F401
            from ragas.evaluation import (LangchainLLMWrapper,  # noqa: F401
                                          LangchainEmbeddingsWrapper)  # noqa: F401
            from langchain_core.embeddings import Embeddings  # noqa: F401
            return True
        except Exception as e:
            logger.warning(f"Ragas 依赖不可用，将降级内置 LLM-as-judge: {e}")
            return False

    # --------------------------- Ragas 构件 ---------------------------
    def _build_llm(self):
        from langchain_openai import ChatOpenAI
        from ragas.evaluation import LangchainLLMWrapper
        llm = ChatOpenAI(
            model=self.config.LLM_MODEL_NAME,
            openai_api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL,
            temperature=0.0,
        )
        return LangchainLLMWrapper(llm)

    def _build_embeddings(self):
        from ragas.evaluation import LangchainEmbeddingsWrapper
        if self.embedding_provider is None:
            logger.info("评估未传入 embedding_provider，加载本地 BGE-M3 用于语义打分 ...")
            self.embedding_provider = BGEEmbeddingProvider(self.config)
        emb = _BgeEmbeddings(self.embedding_provider)
        return LangchainEmbeddingsWrapper(emb)

    # --------------------------- 数据集评估 ---------------------------
    def evaluate_dataset(self, items: List[Dict[str, Any]],
                         show_progress: bool = False) -> Dict[str, Any]:
        """批量评估 QA 数据集。

        items 中每条可含：question, answer, contexts(list[str]),
        ground_truth（或别名 reference_answer）。
        返回结构：{engine, total, metrics, has_ground_truth, average, scores}
        """
        norm = self._normalize(items)
        if self._ragas:
            try:
                return self._evaluate_ragas(norm, show_progress=show_progress)
            except Exception as e:
                logger.error(f"Ragas 评估失败，降级 LLM-as-judge: {e}")
        return self._evaluate_fallback(norm)

    @staticmethod
    def _normalize(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for it in items:
            gt = it.get("ground_truth") or it.get("reference_answer") or ""
            ctx = it.get("contexts") or []
            if isinstance(ctx, str):
                ctx = [ctx]
            out.append({
                "question": (it.get("question") or "").strip(),
                "answer": (it.get("answer") or "").strip(),
                "contexts": [str(c) for c in ctx],
                "ground_truth": (gt or "").strip(),
            })
        return out

    def _evaluate_ragas(self, items: List[Dict[str, Any]],
                        show_progress: bool = False) -> Dict[str, Any]:
        from ragas import evaluate
        from ragas.metrics import (Faithfulness, AnswerRelevancy,
                                   ContextPrecision, ContextRecall)
        from datasets import Dataset

        # ground_truth 缺失时，依赖它的两项上下文指标无法计算，仅跑 LLM 类两项
        has_gt = all(bool(it["ground_truth"]) for it in items)
        if has_gt:
            metrics = [Faithfulness(), AnswerRelevancy(),
                       ContextPrecision(), ContextRecall()]
        else:
            metrics = [Faithfulness(), AnswerRelevancy()]

        dataset = Dataset.from_dict({
            "question": [it["question"] for it in items],
            "answer": [it["answer"] for it in items],
            "contexts": [it["contexts"] for it in items],
            "ground_truth": [it["ground_truth"] for it in items],
        })

        llm = self._build_llm()
        embeddings = self._build_embeddings()

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=show_progress,
            raise_exceptions=False,
        )

        metric_keys, rows = self._extract(result)
        if not metric_keys:
            logger.warning("Ragas 未返回有效指标，转内置 LLM-as-judge 降级。")
            return self._evaluate_fallback(items)

        scores = []
        for i, it in enumerate(items):
            row = {"question": it["question"]}
            for k in metric_keys:
                v = rows[i].get(k) if i < len(rows) else None
                row[k] = round(float(v), 4) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0
            scores.append(row)

        avg = {}
        for k in metric_keys:
            vals = [r[k] for r in scores
                    if isinstance(r[k], (int, float)) and math.isfinite(r[k])]
            avg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0

        return {
            "engine": "ragas",
            "total": len(items),
            "metrics": metric_keys,
            "has_ground_truth": has_gt,
            "average": avg,
            "scores": scores,
        }

    @staticmethod
    def _extract(result) -> (List[str], List[Dict[str, Any]]):
        """从 EvaluationResult 提取 (指标列名, 逐条分数dict列表)。

        Ragas 0.2.x 通过 to_pandas() 返回 DataFrame，指标列之外还包含规范字段
        （user_input / response / retrieved_contexts / reference）。这里剔除非数值列，
        仅保留 0~1 的指标列，避免把规范字段误判为指标。
        """
        try:
            df = result.to_pandas()
        except Exception as e:
            logger.warning(f"to_pandas 失败: {e}")
            return [], []

        metric_keys = []
        for c in df.columns:
            if c in RAGAS_NON_METRIC_COLS:
                continue
            vals = df[c].dropna().tolist()
            if vals and all(isinstance(v, (int, float)) for v in vals):
                metric_keys.append(c)

        rows = df[metric_keys].to_dict(orient="records") if metric_keys else []
        return metric_keys, rows

    # --------------------------- 内置降级（LLM-as-judge） ---------------------------
    def _evaluate_fallback(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        from .llm_generator import LLMGenerator
        llm = LLMGenerator(self.config)
        scores = []
        for it in items:
            res = self._judge_one(llm, it["question"], it["answer"], it["contexts"])
            scores.append({"question": it["question"], **res})

        avg = self._aggregate(scores)
        return {
            "engine": "llm_judge_fallback",
            "total": len(items),
            "metrics": ["faithfulness", "context_precision", "answer_relevancy"],
            "average": avg,
            "scores": scores,
        }

    def _judge_one(self, llm, question: str, answer: str,
                   contexts: List[str]) -> Dict[str, Any]:
        if not answer:
            return {"faithfulness": 0.0, "context_precision": 0.0,
                    "answer_relevancy": 0.0, "rationale": "系统未生成回答"}
        ctx_text = "\n\n".join(
            f"【上下文 {i+1}】\n{c}" for i, c in enumerate(contexts)
        ) or "（无检索上下文）"
        prompt = self._prompt(question, answer, ctx_text)
        raw = llm.generate(
            prompt,
            system_prompt="你是一个严谨的 RAG 质量评估专家，只输出 JSON，不要任何额外解释。",
            temperature=0.0,
            max_tokens=512,
        )
        return self._parse(raw)

    @staticmethod
    def _prompt(question: str, answer: str, ctx_text: str) -> str:
        return f"""请基于以下材料，对 RAG 系统的回答质量进行三项评分。

【用户问题】
{question}

【检索到的上下文】
{ctx_text}

【系统回答】
{answer}

请仅依据上下文判断，对以下三个指标各打 0~1 之间的小数（1 为最好）：
1. faithfulness（忠实度）：回答中的事实陈述是否都能被上下文支撑，有无编造/幻觉。
2. context_precision（上下文精确度）：检索到的上下文是否包含回答问题所需的关键信息、是否聚焦。
3. answer_relevancy（答案相关性）：回答是否切题、完整、有帮助地回应了问题。

请严格按如下 JSON 格式输出（不要包含 ``` 代码块标记）：
{{
  "faithfulness": 0.0,
  "context_precision": 0.0,
  "answer_relevancy": 0.0,
  "rationale": "简要说明评分理由"
}}"""

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(),
                             flags=re.IGNORECASE)
            s, e = cleaned.find("{"), cleaned.rfind("}")
            if s != -1 and e != -1 and e > s:
                cleaned = cleaned[s:e + 1]
            data = json.loads(cleaned)

            def to_score(v):
                try:
                    return max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    return 0.0

            return {
                "faithfulness": to_score(data.get("faithfulness")),
                "context_precision": to_score(data.get("context_precision")),
                "answer_relevancy": to_score(data.get("answer_relevancy")),
                "rationale": str(data.get("rationale", "")),
            }
        except Exception as e:
            logger.warning(f"降级评估 JSON 解析失败: {e}")
            return {"faithfulness": 0.0, "context_precision": 0.0,
                    "answer_relevancy": 0.0, "rationale": "解析失败"}

    @staticmethod
    def _aggregate(scores: List[Dict[str, Any]]) -> Dict[str, float]:
        if not scores:
            return {"faithfulness": 0.0, "context_precision": 0.0,
                    "answer_relevancy": 0.0}
        n = len(scores)
        return {
            "faithfulness": round(sum(s["faithfulness"] for s in scores) / n, 4),
            "context_precision": round(sum(s["context_precision"] for s in scores) / n, 4),
            "answer_relevancy": round(sum(s["answer_relevancy"] for s in scores) / n, 4),
        }
