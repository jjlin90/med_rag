"""
Main API Module
FastAPI接口入口
"""

import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    question: str
    source_filter: Optional[str] = None
    use_cache: bool = True
    # 检索策略改由大模型自动判断；保留字段用于兼容与调试，None 表示自动
    strategy: Optional[str] = None
    # 会话 ID：用于跨会话持久化历史（对齐 EduRag）；为空则后端自动生成
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    response_time: float
    used_cache: bool
    intent: str
    strategy: str
    session_id: str

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    services: Dict[str, bool]

class ChatHistoryItem(BaseModel):
    role: str
    content: str
    timestamp: float

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
    ground_truth: Optional[str] = None

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

            return HealthResponse(
                status="healthy" if all_healthy else "degraded",
                timestamp=time.time(),
                services=status
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
        统一查询编排（对齐 EduRag 的 IntegratedQASystem.query 主流程）：

        1. 缓存命中 → 直接返回（最快路径）
        2. **FAQ 优先**（Redis 一级 → MySQL 二级 BM25）→ 命中直接返回，
           不命中才进入下一步（与 EduRag「先 BM25/FAQ 命中即返」一致）
        3. 调用 CoreRAGSystem.generate() 执行 RAG 生成：
           意图分类 → 策略选择 → 检索与合并 → 重排序 → 构建上下文 → LLM 生成
           （严格对齐 EduRag 主流程图中的 RAG 生成答案六步）
        4. 写会话历史（MySQL conversations 表，对齐 EduRag update_session_history）
        5. 写缓存并返回（响应携带 session_id，供前端跨会话持久化）

        任一环节失败均可优雅降级（FAQ 不可用 → RAG；LLM 不可用 → 兜底文案）。
        """
        start_time = time.time()

        # 会话 ID：未提供则由后端生成（UUID4），用于跨会话持久化历史
        if session_id is None:
            session_id = ConversationStore.new_session_id()

        # 1. 缓存命中（稳定键，跨重启可命中；RAG/LLM 回答缓存在 query: 命名空间）
        cache_key = query_cache_key(question)
        if use_cache and self.cache.is_connected():
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Cache hit")
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
                )

        confidence = 0.0
        intent = 'medical'
        selected_strategy = strategy or 'auto'
        sources = []
        final_answer = None

        # 2. FAQ 优先（对齐 EduRag：先 BM25/FAQ，命中即返）
        #    但先做意图预判：医疗类问题不应被 FAQ 抢答。FAQ 是基于 MySQL 文章标题的
        #    BM25 粗检索，会误把“头痛”匹配到“声带息肉”等无关条目并返回整篇文章
        #    （阈值 0.5 对 ~7 量级的原始 BM25 分数形同虚设）。医疗问题统一走 RAG
        #    （稠密检索 + 重排，准确性远高于标题 BM25），避免答非所问。
        pre_intent = 'medical'
        try:
            pre_intent = self.intent_classifier.predict(question).get('intent', 'medical')
        except Exception:
            # 分类失败则保守地走 RAG（不冒险用可能错配的 FAQ）
            pre_intent = 'medical'

        faq_answer, need_llm = (None, True)
        if pre_intent != 'medical':
            faq_answer, need_llm = self.faq_search.search_faq(question, intent=pre_intent)

        if not need_llm:
            final_answer = faq_answer
            intent = 'faq'
            selected_strategy = 'faq'
            confidence = 0.95
        else:
            # 3. 统一走 CoreRAGSystem 生成（内部严格对齐 EduRAG 六步流程：
            #    意图分类 → 策略选择 → 检索与合并 → 重排序 → 构建上下文 → LLM 生成）
            rag_result = self.rag_core.generate(
                question, source_filter=source_filter, history=history, strategy=strategy
            )
            final_answer = rag_result['answer']
            intent = rag_result['intent']
            selected_strategy = rag_result['strategy']
            sources = rag_result['sources']
            confidence = rag_result['confidence']

        response_time = time.time() - start_time

        # 6. 写会话历史（FAQ 命中与 RAG 命中都记录，对齐 EduRag 双向写入）
        if final_answer:
            self.conversation_store.update_session_history(session_id, question, final_answer)

        # 7. 写缓存
        if use_cache and self.cache.is_connected():
            self.cache.set(cache_key, {
                'type': 'rag',
                'answer': final_answer,
                'sources': sources,
                'confidence': confidence,
                'intent': intent,
                'strategy': selected_strategy,
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
        )

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
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
            reload=debug,
            log_level="info"
        )

def create_app(config: Config) -> FastAPI:
    """创建FastAPI应用"""
    rag_web = RAGWebAPI(config)
    return rag_web.app

if __name__ == "__main__":
    # 直接运行API服务
    from ..config.settings import Config

    config = Config()
    rag_web = RAGWebAPI(config)
    rag_web.run(debug=True)