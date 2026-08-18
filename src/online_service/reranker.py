"""
Reranker Module
BGE-reranker-large重排 + 父块上下文恢复
"""

import logging
import numpy as np
from typing import List, Dict, Any

from ..config.settings import Config

# 兼容不同版本的 FlagEmbedding
# FlagEmbedding >= 1.3 提供 FlagReranker（底层基于 BGE-reranker CrossEncoder 架构）
try:
    from FlagEmbedding import FlagReranker
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    FlagReranker = None

logger = logging.getLogger(__name__)

class Reranker:
    """重排序器"""

    def __init__(self, config: Config):
        self.config = config
        self.model_name = config.RERANK_MODEL_NAME
        self.device = config.RERANK_DEVICE

        # 初始化重排模型
        self.model = None
        if FLAG_EMBEDDING_AVAILABLE:
            self._init_model()

    def _init_model(self):
        """初始化BGE-Reranker模型"""
        try:
            model_path = self.config.BASE_DIR / "src/models" / self.model_name.lower()
            # FlagEmbedding >= 1.3: FlagReranker 直接实例化, 参数 devices
            self.model = FlagReranker(
                str(model_path),
                use_fp16=(self.device == 'cuda'),
                devices=self.device
            )
            logger.info("BGE-Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {str(e)}")
            self.model = None

    def rerank(self, query: str, documents: List[Dict], top_k: int = 2) -> List[Dict]:
        """
        对文档进行重排序（CrossEncoder 精排）

        原理：与向量检索的「双塔」粗排不同，CrossEncoder 将 query 和文档
        拼接后一起送入模型做交互式编码，相关性判断更精准，但速度慢，
        因此只对粗排召回的少量候选（如 Top 8）做精排。

        Args:
            query: 查询文本
            documents: 文档列表（每项需含 'content' 字段）
            top_k: 返回的Top-K数量

        Returns:
            重排序后的文档列表（每项新增 'rerank_score' 字段）
        """
        if not documents:
            return []

        # 如果模型不可用，按原顺序返回
        if not self.model:
            logger.warning("Reranker model not available, returning original order")
            return documents[:top_k]

        try:
            # 准备输入对：(query, doc) 二元组列表
            pairs = [(query, doc['content']) for doc in documents]

            # 计算分数 (FlagReranker 使用 compute_score)
            scores = self.model.compute_score(pairs)
            # compute_score 对单个 pair 返回标量，对多个返回 list，统一为 list
            if not isinstance(scores, (list, tuple, np.ndarray)):
                scores = [scores]

            # 添加分数到文档
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)

            # 按分数排序
            documents.sort(key=lambda x: x['rerank_score'], reverse=True)

            # 返回Top-K
            top_scores_str = ', '.join(f"{d.get('rerank_score', 0):.3f}" for d in documents[:top_k])
            logger.info(f"Reranked {len(documents)} documents, top {top_k} scores: [{top_scores_str}]")

            return documents[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return documents[:top_k]

    def rerank_with_parent_context(self, query: str, child_results: List[Dict],
                                   parent_map: Dict[str, Dict], top_k: int = 2) -> List[Dict]:
        """
        重排序并恢复父块上下文

        Args:
            query: 查询文本
            child_results: 子块检索结果
            parent_map: 父块映射字典
            top_k: 返回的Top-K数量

        Returns:
            重排序后的父块列表
        """
        # 先按父块分组
        parent_groups = {}
        for child in child_results:
            parent_id = child.get('parent_id')
            if parent_id and parent_id in parent_map:
                if parent_id not in parent_groups:
                    parent_groups[parent_id] = {
                        'parent': parent_map[parent_id],
                        'children': [],
                        'max_score': 0
                    }
                parent_groups[parent_id]['children'].append(child)
                parent_groups[parent_id]['max_score'] = max(
                    parent_groups[parent_id]['max_score'],
                    child.get('score', 0)
                )

        # 构造父块文档列表
        parent_docs = []
        for parent_id, group in parent_groups.items():
            parent = group['parent']
            # 使用父块内容进行重排
            parent_doc = {
                'id': parent['id'],
                'content': parent['content'],
                'metadata': parent.get('metadata', {}),
                'score': group['max_score'],  # 使用子块的最高分数
                'parent_id': parent_id,
                'children_count': len(group['children'])
            }
            parent_docs.append(parent_doc)

        # 重排父块
        reranked_parents = self.rerank(query, parent_docs, top_k)

        return reranked_parents

    def cross_encoder_rerank(self, query: str, documents: List[Dict],
                           top_k: int = 2) -> List[Dict]:
        """
        使用CrossEncoder进行重排（备选方案）

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回的Top-K数量

        Returns:
            重排序后的文档列表
        """
        if not FLAG_EMBEDDING_AVAILABLE:
            logger.warning("CrossEncoder not available")
            return documents[:top_k]

        try:
            # 准备输入对
            pairs = []
            for doc in documents:
                pairs.append((query, doc['content']))

            # 计算分数
            scores = self.model.compute_score(pairs)

            # 添加分数并排序
            for doc, score in zip(documents, scores):
                doc['cross_encoder_score'] = float(score)

            documents.sort(key=lambda x: x['cross_encoder_score'], reverse=True)

            return documents[:top_k]

        except Exception as e:
            logger.error(f"CrossEncoder rerank failed: {str(e)}")
            return documents[:top_k]

    def batch_rerank(self, queries: List[str], document_lists: List[List[Dict]],
                    top_k_list: List[int] = None) -> List[List[Dict]]:
        """批量重排"""
        if top_k_list is None:
            top_k_list = [2] * len(queries)

        results = []
        for query, documents, top_k in zip(queries, document_lists, top_k_list):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)

        return results

    def score_documents(self, query: str, documents: List[Dict]) -> List[float]:
        """计算文档分数"""
        if not self.model:
            return [doc.get('score', 0) for doc in documents]

        try:
            pairs = [(query, doc['content']) for doc in documents]
            scores = self.model.compute_score(pairs)
            return [float(score) for score in scores]

        except Exception as e:
            logger.error(f"Scoring failed: {str(e)}")
            return [doc.get('score', 0) for doc in documents]

    def get_reranker_stats(self) -> Dict[str, Any]:
        """获取重排器统计信息"""
        stats = {
            'model_available': self.model is not None,
            'model_name': self.model_name,
            'device': self.device,
            'supported_strategies': ['cross_encoder'] if self.model else []
        }

        if self.model:
            try:
                # 获取模型信息
                stats.update({
                    'model_type': type(self.model).__name__,
                    'max_length': 512,  # BGE-Reranker默认
                    'batch_size': 32
                })
            except:
                pass

        return stats

    def validate_reranker(self) -> bool:
        """验证重排器是否正常工作"""
        if not self.model:
            return False

        try:
            # 测试查询和文档
            test_query = "什么是高血压？"
            test_doc = "高血压是一种常见的慢性疾病，指动脉血压持续升高。"
            test_pair = [(test_query, test_doc)]

            # 预测分数
            scores = self.model.compute_score(test_pair)
            # compute_score 对单个 pair 返回 float，对 list 返回 list，统一兼容
            if isinstance(scores, (list, tuple, np.ndarray)):
                return isinstance(scores[0], (int, float, np.floating))
            return isinstance(scores, (int, float, np.floating))

        except Exception as e:
            logger.error(f"Reranker validation failed: {str(e)}")
            return False