"""
Online Service Module
在线问答服务链路
"""

from .cache_manager import CacheManager
from .faq_search import FAQSearch, BM25Index
from .intent_classifier import IntentClassifier
from .query_augmenter import QueryAugmenter
from .retrieval import Retrieval
from .reranker import Reranker
from .llm_generator import LLMGenerator
from .conversation_store import ConversationStore
from .rag_evaluator import RAGEvaluator
from .rag_system import RAGSystem


def __getattr__(name):
    # 保留包级导出，但避免 python -m ...main_api 前提前加载目标模块。
    if name in ('RAGWebAPI', 'create_app'):
        from importlib import import_module
        value = getattr(import_module('.main_api', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = [
    'CacheManager',
    'FAQSearch',
    'BM25Index',
    'IntentClassifier',
    'QueryAugmenter',
    'Retrieval',
    'Reranker',
    'LLMGenerator',
    'ConversationStore',
    'RAGEvaluator',
    'RAGWebAPI',
    'RAGSystem',
    'create_app'
]
