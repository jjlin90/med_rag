"""
Main API Module
FastAPI接口入口
"""

# 支持直接运行文件；包导入时不修改搜索路径。
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.online_service"

import logging
import time
from typing import List, Dict, Any, Optional, Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, AliasChoices

from ..config.settings import Config
from .cache_manager import CacheManager, query_cache_key
from .faq_search import FAQSearch
from .intent_classifier import IntentClassifier
from .query_augmenter import QueryAugmenter
from .retrieval import Retrieval
from .reranker import Reranker
from .llm_generator import LLMGenerator
from .strategy_selector import StrategySelector
from .conversation_store import ConversationStore
from .rag_evaluator import RAGEvaluator
from .rag_system import RAGSystem as CoreRAGSystem

logger = logging.getLogger(__name__)

# Pydantic模型
class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    source_filter: Optional[str] = None
    use_cache: bool = True
    # 检索策略改由大模型自动判断；保留字段用于兼容与调试，None 表示自动
    strategy: Optional[Literal['direct', 'hyde', 'subquery', 'backtracking']] = None
    # 会话 ID：用于跨会话持久化历史（对齐 EduRag）；为空则后端自动生成
    session_id: Optional[str] = None

    @field_validator('question')
    @classmethod
    def validate_question(cls, value):
        if not value.strip():
            raise ValueError('question cannot be blank')
        return value

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    response_time: float
    used_cache: bool
    intent: str
    strategy: str
    session_id: str
    # 降级透明化：任何一次降级都必须让调用方可见。
    # 静默降级比直接失败更危险——前端和用户会以为这是正常质量的答案。
    degraded: bool = False
    degrade_level: int = 0     # 0=正常 1=备用检索且保持子块粒度 2=无召回或服务故障
    degrade_reason: str = ""

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    services: Dict[str, bool]
    # 降级打点快照：l1_rate 持续偏高说明子块过滤或数据有问题；
    # l2_error > 0 说明基础设施故障，必须告警。
    degradation: Dict[str, Any] = {}

class ChatHistoryItem(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1)
    timestamp: Optional[float] = None

    @field_validator('content')
    @classmethod
    def validate_content(cls, value):
        if not value.strip():
            raise ValueError('content cannot be blank')
        return value

class ChatRequest(BaseModel):
    messages: List[ChatHistoryItem]
    source_filter: Optional[str] = None
    use_cache: bool = True
    # 会话 ID：多轮对话跨会话持久化（对齐 EduRag）
    session_id: Optional[str] = None

class EvaluateItem(BaseModel):
    question: str
    answer: str
    contexts: Optional[List[str]] = None
    # 标准答案（可选）。提供后 Ragas 可额外计算 context_precision / context_recall；
    # 不提供则仅评估 faithfulness / answer_relevancy。兼容别名 reference_answer。
    ground_truth: Optional[str] = Field(default=None, validation_alias=AliasChoices('ground_truth', 'reference_answer'))

class EvaluateRequest(BaseModel):
    items: List[EvaluateItem]

class RAGWebAPI:
    """RAG Web API 服务封装（底层调用 CoreRAGSystem 执行 RAG 生成流程）"""

    def __init__(self, config: Config):
        self.config = config

        # 初始化各个组件
        self.cache = CacheManager(config)
        self.faq_search = FAQSearch(config, self.cache)
        self.intent_classifier = IntentClassifier(config)
        self.llm_generator = LLMGenerator(config)
        self.query_augmenter = QueryAugmenter(config, self.llm_generator)
        self.strategy_selector = StrategySelector(config, self.llm_generator)
        self.retrieval = Retrieval(config)
        self.reranker = Reranker(config)
        # 核心 RAG 系统：对齐 EduRAG 的 RAGSystem.generate_answer 六步流程，
        # 复用已初始化的组件，避免重复加载大模型。
        self.rag_core = CoreRAGSystem(
            config,
            retrieval=self.retrieval,
            reranker=self.reranker,
            llm_generator=self.llm_generator,
            intent_classifier=self.intent_classifier,
            strategy_selector=self.strategy_selector,
            query_augmenter=self.query_augmenter,
        )
        # MySQL 会话历史存储（对齐 EduRag conversations 表 + 会话 ID 管理机制）
        self.conversation_store = ConversationStore(config)
        # RAG 评估器（基于 Ragas；复用检索阶段的 BGE-M3 作为 embedding，避免重复加载）
        self.evaluator = RAGEvaluator(config, embedding_provider=self.retrieval.embedding_provider)

        # 创建FastAPI应用
        self.app = FastAPI(
            title="Medical RAG API",
            description="医疗知识问答系统API",
            version="1.0.0"
        )

        # 添加CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册路由"""

        @self.app.get("/", response_model=dict)
        async def root():
            return {"message": "Medical RAG API", "version": "1.0.0"}

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """健康检查"""
            status = {
                'redis': self.cache.is_connected(),
                'mysql': self.faq_search.connection is not None,
                # 会话历史表依赖 MySQL，单独标识便于排查
                'conversation': self.conversation_store.is_available(),
                # 检查Milvus集合是否真实可用
                'milvus': self._check_milvus(),
                'llm': self.llm_generator.client is not None,
                # BERT模型已加载则可用
                'intent_classifier': self.intent_classifier.model is not None,
                'reranker': self.reranker.model is not None
            }

            all_healthy = all(status.values())

            # 降级打点：进程内计数器，重启清零。生产建议改为 Prometheus Counter。
            # 关注两个比率：
            #   l1_rate 持续偏高 → 子块过滤过严或数据分布异常，需排查入库链路
            #   l2_error > 0     → 基础设施故障，应立即告警
            degradation = Retrieval.get_degrade_metrics()

            return HealthResponse(
                status="healthy" if all_healthy else "degraded",
                timestamp=time.time(),
                services=status,
                degradation=degradation,
            )

        @self.app.post("/query", response_model=QueryResponse)
        async def query(request: QueryRequest):
            """
            单一问答入口。检索策略由后端大模型自动判断（用户无需指定）。
            具体编排见 `self._handle_query`（对齐 EduRag 主流程）。
            """
            return self._handle_query(
                request.question,
                request.source_filter,
                request.use_cache,
                history=None,
                strategy=request.strategy,
                session_id=request.session_id,
            )

        @self.app.post("/chat", response_model=QueryResponse)
        async def chat(request: ChatRequest):
            """多轮对话入口：把上下文历史注入大模型（对齐 EduRag 的 history 注入）"""
            if not request.messages:
                raise HTTPException(status_code=400, detail="Messages cannot be empty")

            last_message = request.messages[-1]
            if last_message.role != 'user':
                raise HTTPException(status_code=400, detail="Last message must be from user")

            # 除最后一条（当前问题）外，其余全部作为多轮历史传入 LLM
            history = [
                {'role': m.role, 'content': m.content}
                for m in request.messages[:-1]
            ]

            return self._handle_query(
                last_message.content,
                request.source_filter,
                request.use_cache,
                history=history if history else None,
                strategy=None,
                session_id=request.session_id,
            )

        @self.app.get("/stats")
        async def get_stats():
            """获取系统统计信息"""
            stats = {
                'faq_stats': self.faq_search.get_faq_stats(),
                'retrieval_stats': self.retrieval.get_retrieval_stats(),
                'llm_stats': self.llm_generator.get_llm_stats(),
                'reranker_stats': self.reranker.get_reranker_stats(),
                'cache_stats': self.cache.get_memory_usage() if self.cache.is_connected() else {},
                'timestamp': time.time()
            }
            return stats

        @self.app.get("/available_strategies")
        async def get_strategies():
            """获取可用检索策略（由大模型自动选择，前端不再要求用户手动选择）"""
            strategies = self.query_augmenter.get_available_strategies()
            descriptions = {
                strategy: self.query_augmenter.get_strategy_description(strategy)
                for strategy in strategies
            }
            return {
                'strategies': strategies,
                'descriptions': descriptions,
                'auto_select': True,
                'note': '检索策略现在由大模型根据查询自动判断，无需用户手动选择。'
            }

        @self.app.get("/intent_example")
        async def get_intent_examples():
            """获取意图分类示例"""
            examples = {
                'general': [
                    "什么是人工智能？",
                    "怎么学习编程？",
                    "什么是云计算？"
                ],
                'medical': [
                    "头痛怎么办？",
                    "发烧需要吃药吗？",
                    "血压正常范围是多少？"
                ]
            }
            return examples

        @self.app.get("/conversation/{session_id}")
        async def get_conversation(session_id: str):
            """获取指定会话最近的历史（对齐 EduRag get_session_history）"""
            history = self.conversation_store.get_session_history(session_id)
            return {"session_id": session_id, "history": history}

        @self.app.delete("/conversation/{session_id}")
        async def delete_conversation(session_id: str):
            """清空指定会话历史"""
            ok = self.conversation_store.clear_session_history(session_id)
            return {"session_id": session_id, "cleared": ok}

        @self.app.post("/evaluate")
        async def evaluate(request: EvaluateRequest):
            """
            RAG 评估（基于 Ragas，与 EduRag 的 ragas_evaluate.py 对齐）。
            四项指标（0~1）：faithfulness / answer_relevancy / context_precision / context_recall。
            每条 item 可含 ground_truth；缺失时自动跳过依赖它的两项上下文指标。
            """
            if not request.items:
                raise HTTPException(status_code=400, detail="items 不能为空")

            try:
                # EvaluateItem 是 Pydantic 模型，先转 dict 再交给评估器
                items = [it.model_dump() for it in request.items]
                result = self.evaluator.evaluate_dataset(items)
                return result
            except Exception as e:
                logger.error(f"RAG 评估失败: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    def _check_milvus(self) -> bool:
        """检查Milvus集合是否可用"""
        try:
            from pymilvus import utility
            # 必须与 MilvusStore 注册连接时使用的 alias 一致，否则默认找 alias="default"
            return utility.has_collection(
                self.config.MILVUS_COLLECTION_NAME,
                using=self.retrieval.milvus_store.alias)
        except Exception:
            return False

    def _handle_query(self, question: str,
                      source_filter: Optional[str],
                      use_cache: bool,
                      history: Optional[List[Dict[str, Any]]] = None,
                      strategy: Optional[str] = None,
                      session_id: Optional[str] = None) -> QueryResponse:
        """
        统一查询编排（对齐 EduRag 双通道流程）：

        外层查询缓存：Redis 查 query:v3:<SHA-256>，键包含问题、来源、策略和历史；
          有答案且降级等级低于L2的缓存结果可直接返回。

        通道① 条件快通道 · FAQ 高频问答（无历史、无来源过滤、无显式策略）：
          Redis 查 faq:v3:<MD5> FAQ缓存 → 标准问题一致才返回
          → jieba 分词 BM25Okapi → softmax 归一化 → best_score
          → best_score ≥ 0.85 且标准问题一致？→ 取答案、回填缓存、返回
          → 否 → 降级到 RAG

        通道② 深通道 · 专业知识问答（BGE-M3 + Milvus + Reranker）：
          BERT 意图分类：通用知识 / 专业咨询
          → 通用知识？→ 直接 LLM，不检索
          → LLM 策略选择：直接检索 / 回溯 / 子查询 / HyDE
          → 混合检索(dense+sparse) → Small-to-Big(子块→父块去重) → Reranker 精排(Top-2)
          → Prompt 组装 → 配置的 LLM 非流式生成

        按路径写 MySQL 会话历史，L0/L1结果可写查询缓存，返回响应。
        """
        start_time = time.time()

        # 会话 ID：未提供则由后端生成（UUID4），用于跨会话持久化历史
        if session_id is None:
            session_id = ConversationStore.new_session_id()

        # ===== 外层查询缓存：适用于 FAQ 与深通道结果 =====
        # Step 1: 查询缓存 query:v3:<SHA-256>，键包含问题、来源、策略与历史。
        cache_key = query_cache_key(question, source_filter, strategy, history)
        if use_cache and self.cache.is_connected():
            cached_result = self.cache.get(cache_key)
            if (isinstance(cached_result, dict) and cached_result.get('answer')
                    and cached_result.get('degrade_level', 0) < 2):
                logger.info(f"通用缓存命中: {question}")
                self.conversation_store.update_session_history(
                    session_id, question, cached_result['answer'])
                response_time = time.time() - start_time
                return QueryResponse(
                    answer=cached_result['answer'],
                    sources=cached_result.get('sources', []),
                    confidence=cached_result.get('confidence', 0.0),
                    response_time=response_time,
                    used_cache=True,
                    intent=cached_result.get('intent', 'unknown'),
                    strategy=cached_result.get('strategy', strategy or 'auto'),
                    session_id=session_id,
                    degraded=cached_result.get('degraded', False),
                    degrade_level=cached_result.get('degrade_level', 0),
                    degrade_reason=cached_result.get('degrade_reason', ''),
                )

        # Step 2: FAQ BM25 候选门槛 + 标准问题一致性校验
        # 无历史、来源或策略约束时先尝试 FAQ；只有标准问题一致才直答。
        # BM25/softmax 分数不能证明适用人群或医学含义相同。
        # FAQ has no conversation context or source/strategy filtering.
        faq_answer, need_rag = (None, True)
        if not history and not source_filter and not strategy:
            faq_answer, need_rag = self.faq_search.search_faq(question, use_cache=use_cache)

        if not need_rag and faq_answer:
            # FAQ 快通道命中：写会话历史 + 写缓存 + 返回
            response_time = time.time() - start_time
            self.conversation_store.update_session_history(session_id, question, faq_answer)

            if use_cache and self.cache.is_connected():
                self.cache.set(cache_key, {
                    'type': 'faq',
                    'answer': faq_answer,
                    'sources': [],
                    'confidence': 0.95,
                    'intent': 'faq',
                    'strategy': 'faq_fast_channel',
                })

            logger.info(f"FAQ 快通道命中，跳过 RAG: {question}")
            return QueryResponse(
                answer=faq_answer,
                sources=[],
                confidence=0.95,
                response_time=response_time,
                used_cache=False,
                intent='faq',
                strategy='faq_fast_channel',
                session_id=session_id,
            )

        # ===== 通道② 深通道：RAG 降级处理 =====#
        # Step 3: 调用 CoreRAGSystem.generate() 执行完整 RAG 流程：
        #         意图分类 → 通用知识直通/策略选择 → 混合检索+Small-to-Big → 精排 → 生成
        rag_result = self.rag_core.generate(
            question, source_filter=source_filter, history=history, strategy=strategy
        )

        final_answer = rag_result['answer']
        intent = rag_result['intent']
        selected_strategy = rag_result['strategy']
        sources = rag_result['sources']
        confidence = rag_result['confidence']
        degraded = bool(rag_result.get('degraded', False))
        degrade_level = int(rag_result.get('degrade_level', 0))
        degrade_reason = rag_result.get('degrade_reason', '')
        response_time = time.time() - start_time

        if degraded:
            logger.warning(
                "RAG 深通道降级返回 (level=%d, reason=%s, query=%r)",
                degrade_level, degrade_reason, question
            )

        # Step 4: 写 MySQL 会话历史（对齐 EduRag conversations 表，保留最近 5 轮）
        if final_answer:
            self.conversation_store.update_session_history(session_id, question, final_answer)

        # Step 5: 写缓存 —— 降级结果区别对待
        #   L1（同粒度降级）：结果由真实检索数据产出且稳定，可以缓存。
        #   L2 无召回：不缓存。知识库随时会入库新数据，缓存拒答会让
        #              新数据上线后用户仍拿到"没找到"，直到 TTL 过期。
        #   L2 基础设施故障：**严禁缓存**。否则 Milvus 恢复后用户仍会吃到
        #              拒答长达一个 TTL——把一次分钟级故障放大成小时级。
        if use_cache and self.cache.is_connected():
            if degrade_level >= 2:
                logger.info(
                    "跳过缓存：降级水位 %d (%s)，避免延长故障影响 (query=%r)",
                    degrade_level, degrade_reason, question
                )
            else:
                self.cache.set(cache_key, {
                    'type': 'rag',
                    'answer': final_answer,
                    'sources': sources,
                    'confidence': confidence,
                    'intent': intent,
                    'strategy': selected_strategy,
                    'degraded': degraded,
                    'degrade_level': degrade_level,
                    'degrade_reason': degrade_reason,
                })

        return QueryResponse(
            answer=final_answer,
            sources=sources,
            confidence=confidence,
            response_time=response_time,
            used_cache=False,
            intent=intent,
            strategy=selected_strategy,
            session_id=session_id,
            degraded=degraded,
            degrade_level=degrade_level,
            degrade_reason=degrade_reason,
        )

    def run(self, host: str = "0.0.0.0", port: int = 8005, debug: bool = False):
        """运行API服务（延迟导入 uvicorn，避免未安装时影响模块导入）"""
        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn 未安装，无法启动服务。请执行: uv add uvicorn")
            raise
        logger.info(f"Starting Medical RAG API on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            # Uvicorn reload requires an import string, not this initialized app object.
            reload=False,
            log_level="debug" if debug else "info"
        )

def create_app(config: Config) -> FastAPI:
    """创建FastAPI应用"""
    rag_web = RAGWebAPI(config)
    return rag_web.app

if __name__ == "__main__":
    # 所有命令行入口共用 host/port/debug 参数与默认值。
    from scripts.run_api import main

    main()
