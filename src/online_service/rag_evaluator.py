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
import copy
import logging
import math
import os
import re
from numbers import Real
from typing import List, Dict, Any, Optional

from ..config.settings import Config
from ..offline_pipeline.embedding_provider import BGEEmbeddingProvider

# Ragas 0.2.x 输出的规范字段名（evaluate 会把 v1 的 question/answer/contexts/ground_truth
# 转换成以下规范列，提取指标时需排除这些非指标列）。
RAGAS_NON_METRIC_COLS = {"user_input", "response", "retrieved_contexts", "reference"}


def valid_metric_score(value) -> bool:
    """A real score is finite, in [0, 1], and never a boolean."""
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)

logger = logging.getLogger(__name__)


def answer_relevancy_metric(language='default'):
    """Keep the Ragas formula/strictness; optionally translate its prompt to Chinese."""
    from ragas.metrics import AnswerRelevancy
    from ragas.metrics._answer_relevance import ResponseRelevanceInput, ResponseRelevanceOutput
    metric = AnswerRelevancy()
    if language == 'default':
        return metric
    if language != 'chinese':
        raise ValueError('RAGAS_ANSWER_RELEVANCY_LANGUAGE must be default or chinese')
    prompt = copy.deepcopy(metric.question_generation)
    prompt.language = 'chinese'
    prompt.instruction = (
        '根据给出的回答生成一个问题，并判断回答是否含糊。'
        '含糊的回答将noncommittal设为1，明确的回答设为0。'
        '含糊的回答指回避、模糊或含义不清的回答，例如“我不知道”或“我不确定”。'
        '生成的问题使用中文。'
    )
    # Direct translation of the two upstream examples; labels are unchanged.
    prompt.examples = [
        (ResponseRelevanceInput(response='阿尔伯特·爱因斯坦出生于德国。'),
         ResponseRelevanceOutput(question='阿尔伯特·爱因斯坦出生在哪里？',noncommittal=0)),
        (ResponseRelevanceInput(response='我不了解2023年发明的智能手机的突破性功能，因为我不知道2022年之后的信息。'),
         ResponseRelevanceOutput(question='2023年发明的智能手机的突破性功能是什么？',noncommittal=1)),
    ]
    metric.question_generation = prompt
    return metric


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
        from langchain_core.rate_limiters import InMemoryRateLimiter

        # 限流 + 重试：DashScope qwen 系列有 RPM 限制，Ragas 会并发发起数百次
        # judge 调用，不加限流会被 429 兜底成 0 分（2026-08-29 评估事故：210 条中
        # 142 条四项全 0）。用内存限流器把速率压到 ~1 次/秒，超出部分阻塞等待而非
        # 失败；max_retries 兜底偶发 429（注意：本机 langchain_core 版本的
        # InMemoryRateLimiter 不支持 max_bucket 参数，仅用 rps + check_every_n_seconds）。
        judge_rps = float(os.getenv("LLM_JUDGE_REQUESTS_PER_SECOND", "1.0"))
        if not math.isfinite(judge_rps) or judge_rps <= 0:
            raise ValueError('LLM_JUDGE_REQUESTS_PER_SECOND must be finite and positive')
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=judge_rps,
            check_every_n_seconds=0.1,
        )
        # temperature：judge 默认用 0.0 保证可复现。但部分模型（如 TokenHub 上的
        # deepseek-v4-pro）会拒绝非 1 的 temperature，返回 400
        # "invalid temperature: only 1 is allowed for this model"。
        # 因此做成环境变量可配：LLM_JUDGE_TEMPERATURE=1 即可适配这类模型。
        # 代价是评分确定性下降，需在报告中注明该裁判的 temperature 设置。
        try:
            temperature = float(os.getenv("LLM_JUDGE_TEMPERATURE", "0.0"))
        except ValueError:
            logger.warning("LLM_JUDGE_TEMPERATURE 不是合法数字，回退为 0.0")
            temperature = 0.0

        judge_timeout = float(os.getenv("LLM_JUDGE_TIMEOUT", "180"))
        judge_max_tokens = int(os.getenv("LLM_JUDGE_MAX_TOKENS", "2048"))
        judge_max_retries = int(os.getenv("LLM_JUDGE_MAX_RETRIES", "2"))
        disable_thinking = os.getenv(
            "LLM_JUDGE_DISABLE_THINKING", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}

        # GLM-4.5-Air 默认可能启用深度思考。Ragas 的 Faithfulness 提示较长，
        # 开启思考时会大量消耗推理 token，甚至到达 max_tokens 仍未输出 JSON。
        # 智谱的 OpenAI 兼容接口通过 extra_body.thinking 显式关闭该模式。
        extra_body = (
            {"thinking": {"type": "disabled"}} if disable_thinking else None
        )
        force_json_mode = os.getenv(
            "LLM_JUDGE_JSON_MODE", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        model_kwargs = (
            {"response_format": {"type": "json_object"}}
            if force_json_mode
            else {}
        )

        llm = ChatOpenAI(
            model=self.config.LLM_MODEL_NAME,
            openai_api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL,
            temperature=temperature,
            max_tokens=judge_max_tokens,
            timeout=judge_timeout,
            max_retries=judge_max_retries,
            rate_limiter=rate_limiter,
            extra_body=extra_body,
            model_kwargs=model_kwargs,
        )
        class _FixedTemperatureWrapper(LangchainLLMWrapper):
            """Keep providers with strict sampling validation compatible with Ragas."""

            def get_temperature(self, n: int) -> float:
                # Ragas normally substitutes 1e-8 for n=1. GLM-4.6V rejects
                # that value because its API accepts at most two decimals.
                return temperature

        return _FixedTemperatureWrapper(llm)

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
        if not norm:
            raise ValueError('Evaluation items cannot be empty')
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
        from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall
        from ragas.run_config import RunConfig
        from datasets import Dataset

        # ground_truth 缺失时，依赖它的两项上下文指标无法计算，仅跑 LLM 类两项
        has_gt = all(bool(it["ground_truth"]) for it in items)
        if has_gt:
            metrics = [Faithfulness(), answer_relevancy_metric(os.getenv('RAGAS_ANSWER_RELEVANCY_LANGUAGE','default')),
                       ContextPrecision(), ContextRecall()]
        else:
            metrics = [Faithfulness(), answer_relevancy_metric(os.getenv('RAGAS_ANSWER_RELEVANCY_LANGUAGE','default'))]

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
            run_config=RunConfig(
                timeout=int(os.getenv("LLM_JUDGE_TIMEOUT", "180")),
                max_retries=int(os.getenv("LLM_JUDGE_MAX_RETRIES", "2")),
                max_wait=30,
                max_workers=8,
                seed=42,
            ),
            show_progress=show_progress,
            raise_exceptions=False,
        )

        _, rows = self._extract(result)
        # Report every requested metric even if the provider omitted a whole column.
        metric_keys = ['faithfulness', 'answer_relevancy']
        if has_gt:
            metric_keys.extend(['context_precision', 'context_recall'])

        scores = []
        for i, it in enumerate(items):
            row = {"question": it["question"]}
            for k in metric_keys:
                v = rows[i].get(k) if i < len(rows) else None
                # Ragas 的 API/解析失败通常表现为 NaN。保留为 null，不能把
                # “没评出来”伪装成质量 0 分并拖低均值。
                row[k] = (round(float(v), 4)
                          if valid_metric_score(v)
                          else None)
            scores.append(row)

        avg = {}
        valid_counts = {}
        for k in metric_keys:
            vals = [r[k] for r in scores
                     if valid_metric_score(r[k])]
            valid_counts[k] = len(vals)
            avg[k] = round(sum(vals) / len(vals), 4) if vals else None

        complete_count = sum(
            1 for r in scores
            if all(valid_metric_score(r[k])
                   for k in metric_keys)
        )

        return {
            "engine": "ragas",
            "total": len(items),
            "metrics": metric_keys,
            "has_ground_truth": has_gt,
            "average": avg,
            "valid_counts": valid_counts,
            "complete_count": complete_count,
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
            if c in {'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'}:
                metric_keys.append(c)
            elif vals and all(isinstance(v, (int, float)) for v in vals):
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
        metrics = ['faithfulness', 'context_precision', 'answer_relevancy']
        valid_counts = {key: sum(row.get(key) is not None for row in scores) for key in metrics}
        return {
            "engine": "llm_judge_fallback",
            "total": len(items),
            "metrics": ["faithfulness", "context_precision", "answer_relevancy"],
            "has_ground_truth": all(bool(it['ground_truth']) for it in items),
            "average": avg,
            "valid_counts": valid_counts,
            "complete_count": sum(all(row.get(k) is not None for k in metrics) for row in scores),
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
                if isinstance(v, bool):
                    return None
                try:
                    value = float(v)
                    return value if valid_metric_score(value) else None
                except (TypeError, ValueError):
                    return None

            return {
                "faithfulness": to_score(data.get("faithfulness")),
                "context_precision": to_score(data.get("context_precision")),
                "answer_relevancy": to_score(data.get("answer_relevancy")),
                "rationale": str(data.get("rationale", "")),
            }
        except Exception as e:
            logger.warning(f"降级评估 JSON 解析失败: {e}")
            return {"faithfulness": None, "context_precision": None,
                    "answer_relevancy": None, "rationale": "解析失败"}

    @staticmethod
    def _aggregate(scores: List[Dict[str, Any]]) -> Dict[str, float]:
        result = {}
        for key in ('faithfulness', 'context_precision', 'answer_relevancy'):
            values = [row.get(key) for row in scores
                      if valid_metric_score(row.get(key))]
            result[key] = round(sum(values) / len(values), 4) if values else None
        return result
