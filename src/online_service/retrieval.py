"""
Retrieval Module
Milvus混合稠密+稀疏检索
"""

import logging
from typing import List, Dict, Any, Optional

from ..config.settings import Config
from ..offline_pipeline.milvus_store import MilvusStore
from ..offline_pipeline.embedding_provider import BGEEmbeddingProvider

logger = logging.getLogger(__name__)

class Retrieval:
    """检索器"""

    def __init__(self, config: Config):
        self.config = config
        self.milvus_store = MilvusStore(config)
        self.embedding_provider = BGEEmbeddingProvider(config)

        # 检索参数
        self.top_k_retrieve = config.TOP_K_RETRIEVE
        self.bm25_weight = config.BM25_WEIGHT
        self.dense_weight = config.DENSE_WEIGHT

    def search(self, query: str, source_filter: Optional[str] = None,
               use_hybrid: bool = True) -> List[Dict]:
        """
        执行搜索

        Args:
            query: 查询文本
            source_filter: 来源过滤
            use_hybrid: 是否使用混合检索

        Returns:
            搜索结果列表
        """
        try:
            # 生成查询向量
            dense_vec, sparse_vec = self.embedding_provider.generate_embeddings([query])
            query_dense = dense_vec[0].tolist() if len(dense_vec) > 0 else []
            query_sparse = sparse_vec[0] if len(sparse_vec) > 0 else {}

            # 构建过滤表达式
            expr = self._build_source_filter(source_filter) if source_filter else ""

            # 执行检索
            if use_hybrid:
                results = self.milvus_store.hybrid_search_with_rerank(
                    query_dense, query_sparse,
                    limit=self.top_k_retrieve,
                    expr=expr
                )
            else:
                results = self.milvus_store.search(
                    query_dense, query_sparse,
                    limit=self.top_k_retrieve,
                    expr=expr
                )

            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_result = {
                    'id': result['id'],
                    'content': result['entity']['text'],
                    'metadata': result['entity']['metadata'],
                    'score': result['score'],
                    'distance': result['distance'],
                    'source': result['entity']['source'],
                    'parent_id': result['entity'].get('parent_id'),
                    'parent_content': result['entity'].get('parent_content')
                }
                formatted_results.append(formatted_result)

            logger.info(f"Retrieved {len(formatted_results)} chunks for query: {query}")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def search_with_parent_recovery(self, query: str, source_filter: Optional[str] = None) -> List[Dict]:
        """
        搜索并恢复父块

        Args:
            query: 查询文本
            source_filter: 来源过滤

        Returns:
            包含父块信息的搜索结果
        """
        # 先检索子块
        child_results = self.search(query, source_filter)

        if not child_results:
            return []

        # 恢复父块
        parent_results = self._recover_parents(child_results)

        # 去重
        deduped_results = self._deduplicate_results(parent_results)

        logger.info(f"Retrieved {len(deduped_results)} unique parent chunks")
        return deduped_results

    def _recover_parents(self, child_results: List[Dict]) -> List[Dict]:
        """从子块恢复父块"""
        parent_ids = set()
        for result in child_results:
            if result.get('parent_id'):
                parent_ids.add(result['parent_id'])

        parents = []
        for parent_id in parent_ids:
            # 这里简化处理，实际应该从Milvus检索父块
            # 临时从子块信息中构造父块
            parent = self._construct_parent_from_child(parent_id, child_results)
            if parent:
                parents.append(parent)

        return parents

    def _construct_parent_from_child(self, parent_id: str, child_results: List[Dict]) -> Optional[Dict]:
        """从子块构造父块"""
        # 找到属于该父块的所有子块
        children = [r for r in child_results if r.get('parent_id') == parent_id]

        if not children:
            return None

        # 使用第一个子块的基本信息
        first_child = children[0]

        parent = {
            'id': parent_id,
            'content': first_child.get('parent_content', first_child['content']),
            'metadata': first_child.get('metadata', {}),
            'score': max(child['score'] for child in children),
            'source': first_child.get('source'),
            'children_count': len(children)
        }

        return parent

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """去重结果"""
        seen_ids = set()
        deduped = []

        for result in results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                deduped.append(result)

        return deduped

    def _build_source_filter(self, source: str) -> str:
        """构建来源过滤表达式"""
        return f"source == '{source}'"

    def get_source_stats(self) -> Dict[str, int]:
        """获取各来源的文档统计"""
        try:
            # 这里简化处理，实际应该查询Milvus
            stats = {}
            logger.info("Source stats not fully implemented")
            return stats

        except Exception as e:
            logger.error(f"Failed to get source stats: {str(e)}")
            return {}

    def batch_search(self, queries: List[str], source_filter: Optional[str] = None) -> Dict[str, List[Dict]]:
        """批量搜索"""
        results = {}

        for query in queries:
            results[query] = self.search(query, source_filter)

        return results

    def search_multi_queries(self, queries: List[str], source_filter: Optional[str] = None) -> List[Dict]:
        """
        多查询合并检索（用于子查询策略）

        对每个子查询分别检索，合并结果后按分数去重、排序，返回综合最相关的文档。

        Args:
            queries: 子查询列表
            source_filter: 来源过滤

        Returns:
            合并去重后的结果列表
        """
        if not queries:
            return []

        all_results = []
        for query in queries:
            results = self.search(query, source_filter)
            all_results.extend(results)
            logger.info(f"子查询 '{query}' 检索到 {len(results)} 个候选")

        # 按 id 去重，保留最高 score 的记录
        best_by_id = {}
        for result in all_results:
            rid = result.get('id')
            if rid is None:
                continue
            if rid not in best_by_id or result.get('score', 0) > best_by_id[rid].get('score', 0):
                best_by_id[rid] = result

        # 按 score 降序排序，限制数量
        merged = sorted(best_by_id.values(), key=lambda x: x.get('score', 0), reverse=True)
        final_results = merged[:self.top_k_retrieve]

        logger.info(
            f"子查询合并完成：原始 {len(all_results)} 条，去重后 {len(merged)} 条，最终返回 {len(final_results)} 条"
        )
        return final_results

    def similarity_search(self, query: str, threshold: float = 0.5,
                         source_filter: Optional[str] = None) -> List[Dict]:
        """相似度搜索（带阈值）"""
        results = self.search(query, source_filter)

        # 过滤低相似度结果
        filtered_results = [r for r in results if r.get('score', 0) >= threshold]

        logger.info(f"Filtered {len(results)} to {len(filtered_results)} results above threshold {threshold}")
        return filtered_results

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        try:
            collection_info = self.milvus_store.get_collection_info()

            stats = {
                'collection_name': self.config.MILVUS_COLLECTION_NAME,
                'total_entities': collection_info.get('num_entities', 0),
                'available_strategies': ['dense', 'sparse', 'hybrid'],
                'current_strategy': 'hybrid',
                'top_k': self.top_k_retrieve,
                'bm25_weight': self.bm25_weight,
                'dense_weight': self.dense_weight
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {str(e)}")
            return {}