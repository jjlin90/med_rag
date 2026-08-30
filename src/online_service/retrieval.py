"""
Retrieval Module
Milvus混合稠密+稀疏检索
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ..config.settings import Config
from ..offline_pipeline.milvus_store import MilvusStore
from ..offline_pipeline.embedding_provider import BGEEmbeddingProvider
from ..offline_pipeline.chunk_splitter import CHINESE_SEPARATORS

logger = logging.getLogger(__name__)


# 降级水位：0=正常，1=同粒度降级（安全），2=无召回/故障（需拒答或告警）
DEGRADE_OK = 0
DEGRADE_L1 = 1
DEGRADE_L2 = 2


@dataclass
class RetrievalResult:
    """
    检索结果 + 降级元信息。

    实现序列代理（__iter__/__len__/__bool__/__getitem__）以兼容既有的
    List[Dict] 用法，老调用点（如 scripts/test_query_pipeline.py）无需改动。
    """
    documents: List[Dict] = field(default_factory=list)
    degraded: bool = False
    degrade_level: int = DEGRADE_OK
    degrade_reason: str = ""
    orphan_count: int = 0     # 缺失 parent_id 的条数（数据质量异常）
    error: Optional[str] = None  # 基础设施异常（Milvus/Embedding 故障）

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)

    def __bool__(self):
        return bool(self.documents)

    def __getitem__(self, item):
        return self.documents[item]

    @property
    def is_usable(self) -> bool:
        """是否有可交给 LLM 的上下文。"""
        return bool(self.documents) and self.error is None


class Retrieval:
    """检索器"""

    def __init__(self, config: Config):
        self.config = config
        self.milvus_store = MilvusStore(config)
        self.embedding_provider = BGEEmbeddingProvider(config)

        # 检索参数（对齐 EduRag 混合检索与父子块链路）
        self.top_k_retrieve = config.TOP_K_RETRIEVE
        self.top_k_children = config.TOP_K_CHILDREN      # Small-to-Big: 子块召回数
        self.sparse_weight = config.SPARSE_WEIGHT         # 稀疏权重 (EduRag: 0.7)
        self.dense_weight = config.DENSE_WEIGHT           # 稠密权重 (EduRag: 1.0)

    def search(self, query: str, source_filter: Optional[str] = None,
               use_hybrid: bool = True,
               only_children: bool = False,
               status: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        执行搜索

        Args:
            query: 查询文本
            source_filter: 来源过滤
            use_hybrid: 是否使用混合检索
            only_children: True 时把「只召回子块」的过滤下推到 Milvus。
                Small-to-Big 链路必须开启，否则 2000 字父块会挤占 Top-K 槽位。
            status: 可选的状态输出字典（out 参数）。调用方传入空 dict，本方法会写入
                {'error': str|None}。之所以用 out 参数而非抛异常，是因为
                「Milvus 异常」与「确实没有相关数据」必须被区分开：
                两者都返回空列表，但前者要触发告警，后者是正常的拒答。
                混为一谈会让基础设施故障静默消失。

        Returns:
            搜索结果列表（异常时为空列表，错误写入 status['error']）
        """
        if status is not None:
            status.setdefault('error', None)
        try:
            # 生成查询向量
            dense_vec, sparse_vec = self.embedding_provider.generate_embeddings([query])
            query_dense = dense_vec[0].tolist() if len(dense_vec) > 0 else []
            query_sparse = sparse_vec[0] if len(sparse_vec) > 0 else {}

            # 构建过滤表达式
            expr = self._build_source_filter(source_filter) if source_filter else ""

            # 执行检索（对齐 EduRag: sparse 0.7 : dense 1.0）
            if use_hybrid:
                results = self.milvus_store.hybrid_search_with_rerank(
                    query_dense, query_sparse,
                    limit=self.top_k_retrieve,
                    expr=expr,
                    sparse_weight=self.sparse_weight,
                    dense_weight=self.dense_weight,
                    only_children=only_children,
                )
            else:
                results = self.milvus_store.search(
                    query_dense, query_sparse,
                    limit=self.top_k_retrieve,
                    expr=expr,
                    only_children=only_children,
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
            if status is not None:
                status['error'] = f"{type(e).__name__}: {e}"
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

    def search_child_to_parent(self, query: str, source_filter: Optional[str] = None) -> RetrievalResult:
        """
        Small-to-Big 子块→父块检索链路（对齐 EduRag 混合检索与父子块链路）。

        正常路径（L0）：
          1. BGE-M3 一次前向，同时输出 dense(1024维) + sparse(词权) 两种向量
          2. dense → Milvus IVF_FLAT IP (nprobe=16) 语义粗排
          3. sparse → SPARSE_INVERTED_INDEX IP (drop_ratio=0.2) 关键词精确命中
          4. WeightedRanker 加权融合 (sparse 0.7 : dense 1.0)
          5. 取 Top-K **子块**（~400字，检索粒度细）
          6. 通过 parent_id 回溯到**父块**（~2000字，上下文完整）
          7. 父块去重（多个子块可能映射到同一父块）

        降级策略（核心原则：降级路径必须比主路径更安全，而不是更粗糙）：

          L0 严格模式  → 子块过滤下推。数据异常（子块缺 parent_id）只打点剔除，
                         **绝不退化为「直接返回 2000 字父块」**。
                         理由：那不是降级，是质量下降——稀释 LLM 注意力、
                         翻倍 token，且让数据问题静默消失。该路径已永久移除，
                         不提供开关。

          L1 同粒度降级 → 子块召回为空时，放开过滤重查；若命中父块，
                         在内存中按 **与入库完全一致的参数** 切成 400 字子块再返回。
                         输出粒度没有退化，只是检索入口从「子块层」换到「父块层」。
                         风险不上升，默认开启（ENABLE_CHILD_FILTER_FALLBACK）。

          L2 无召回    → 返回空结果并标记，由上层决定是否拒答。
                         **检索不到依据时不调用 LLM**（见 rag_system.generate），
                         医疗场景下让 LLM 用参数知识硬答是整条链路风险最高的行为。

        Args:
            query: 用户查询文本
            source_filter: 学科/来源过滤

        Returns:
            RetrievalResult。documents 为去重后的父块（L0）或内存切片子块（L1），
            按 fusion score 降序，供 CrossEncoder 精排使用。
        """
        # ==================== L0：严格模式 ====================
        status: Dict[str, Any] = {}
        children = self.search(query, source_filter, use_hybrid=True,
                               only_children=True, status=status)

        # 基础设施故障（Milvus 不可达 / Embedding 推理失败）与「没有相关数据」
        # 必须严格区分：前者要告警，后者是正常拒答。混为一谈会让故障静默消失。
        if status.get('error'):
            self._bump('l2_error')
            logger.error(
                "Small-to-Big: 检索基础设施异常，进入 L2 (query=%r, error=%s)",
                query, status['error']
            )
            return RetrievalResult(
                degraded=True, degrade_level=DEGRADE_L2,
                degrade_reason='retrieval_error', error=status['error']
            )

        # 数据一致性自检：过滤已下推，理论上不应再出现无 parent_id 的块。
        # 若出现，说明入库时 parent_id 未正确写入，或旧库过滤表达式失效。
        orphan = [c for c in children if not c.get('parent_id')]
        if orphan:
            self._bump('orphan', len(orphan))
            # 取过滤表达式仅用于日志，属于错误处理路径的一部分。
            # 这里必须防御：若它自身抛异常，会把一次「可恢复的数据告警」
            # 升级成「整个检索失败」——日志绝不该成为故障源。
            try:
                filter_expr = self.milvus_store.child_filter_expr()
            except Exception:
                filter_expr = "<unavailable>"
            logger.error(
                "Small-to-Big: 子块过滤已下推，但仍有 %d/%d 条结果缺失 parent_id，"
                "请检查入库数据（parent_id 是否为空）与过滤表达式 %r",
                len(orphan), len(children), filter_expr
            )
            children = [c for c in children if c.get('parent_id')]

        if children:
            self._bump('l0_ok')
            return RetrievalResult(
                documents=self._backtrack_to_parents(children),
                orphan_count=len(orphan)
            )

        # ==================== L1：同粒度降级 ====================
        if not getattr(self.config, 'ENABLE_CHILD_FILTER_FALLBACK', True):
            self._bump('l2_empty')
            logger.warning(
                "Small-to-Big: L0 子块召回为空且 L1 降级已关闭，进入 L2 (query=%r)", query
            )
            return RetrievalResult(
                degraded=True, degrade_level=DEGRADE_L2,
                degrade_reason='child_recall_empty', orphan_count=len(orphan)
            )

        logger.warning(
            "Small-to-Big: L0 子块召回为空，进入 L1 同粒度降级 (query=%r)", query
        )
        relaxed = self.search(query, source_filter, use_hybrid=True,
                              only_children=False, status=status)

        if status.get('error'):
            self._bump('l2_error')
            return RetrievalResult(
                degraded=True, degrade_level=DEGRADE_L2,
                degrade_reason='retrieval_error', orphan_count=len(orphan),
                error=status['error']
            )

        # 放开过滤后命中的块分两类：
        #   - 有 parent_id → 本身即子块（说明过滤表达式可能过严），直接采用
        #   - 无 parent_id → 父块，需在内存中切成 400 字子块，保证粒度不退化
        # 候选总量与 L0 的 top_k_retrieve 对齐：一个 2000 字父块可切出 5-7 个子块，
        # 若命中多个父块，候选集会线性膨胀，导致 CrossEncoder 精排开销不可控。
        # 上限保证「降级后 reranker 的输入规模不比正常路径更大」——
        # 这也是「降级路径不能比主路径更贵」的一部分。
        l1_docs: List[Dict] = []
        for hit in relaxed[:self.top_k_children]:
            if hit.get('parent_id'):
                l1_docs.append(hit)
            else:
                l1_docs.extend(self._split_parent_in_memory(hit))
            if len(l1_docs) >= self.top_k_retrieve:
                l1_docs = l1_docs[:self.top_k_retrieve]
                logger.info(
                    "L1 候选达到上限 %d 条，截断以保证精排开销恒定", self.top_k_retrieve
                )
                break

        if l1_docs:
            self._bump('l1_fallback')
            alert = getattr(self.config, 'DEGRADE_ALERT_LEVEL', 1)
            log_fn = logger.warning if alert <= DEGRADE_L1 else logger.info
            log_fn(
                "Small-to-Big: L1 降级成功，%d 条命中 → %d 条子块（粒度未退化）(query=%r)",
                len(relaxed[:self.top_k_children]), len(l1_docs), query
            )
            return RetrievalResult(
                documents=l1_docs, degraded=True, degrade_level=DEGRADE_L1,
                degrade_reason='child_recall_empty_relaxed', orphan_count=len(orphan)
            )

        # ==================== L2：无召回 ====================
        self._bump('l2_empty')
        logger.warning("Small-to-Big: L1 降级后仍无召回，进入 L2 (query=%r)", query)
        return RetrievalResult(
            degraded=True, degrade_level=DEGRADE_L2,
            degrade_reason='no_recall', orphan_count=len(orphan)
        )

    def _backtrack_to_parents(self, children: List[Dict]) -> List[Dict]:
        """
        Step 6-7: 子块 → 父块映射 + 去重（L0 正常路径使用）。

        多个子块可能映射到同一父块，按 parent_id 聚合并保留最高分。
        """
        top_children = children[:self.top_k_children]
        logger.info(
            f"Small-to-Big: 下推过滤后召回 {len(children)} 条子块，"
            f"截取 Top-{self.top_k_children}"
        )

        parent_map = {}  # parent_id -> best_parent_dict
        for child in top_children:
            pid = child.get('parent_id')
            if not pid:
                continue

            if pid not in parent_map:
                # 构造父块：优先用冗余存储的 parent_content，否则用子块 content 兜底
                parent_map[pid] = {
                    'id': pid,
                    'content': child.get('parent_content') or child.get('content', ''),
                    'metadata': child.get('metadata', {}),
                    'score': child.get('score', 0),
                    'source': child.get('source', 'unknown'),
                    'children_ids': [child.get('id')],
                }
            else:
                # 同一父块的多个子块：保留最高分，累积 children_ids
                existing = parent_map[pid]
                if child.get('score', 0) > existing['score']:
                    existing['score'] = child.get('score', 0)
                existing['children_ids'].append(child.get('id'))

        deduped_parents = list(parent_map.values())
        # 按 score 降序排列（融合分数高的父块排前面，供 reranker 精排参考）
        deduped_parents.sort(key=lambda x: x.get('score', 0), reverse=True)

        logger.info(
            f"Small-to-Big: {len(top_children)} 子块 → {len(deduped_parents)} 去重父块"
        )
        return deduped_parents

    def _split_parent_in_memory(self, parent_hit: Dict) -> List[Dict]:
        """
        L1 降级专用：把命中的父块就地切成子块。

        切分参数（chunk_size / chunk_overlap / separators）必须与入库时
        ChunkSplitter 完全一致，否则线上切片与库内子块不同构，
        reranker 打分分布会漂移——这类偏差极难排查。
        """
        text = parent_hit.get('content') or ''
        if not text:
            return []

        size = self.config.CHILD_CHUNK_SIZE
        overlap = self.config.CHUNK_OVERLAP

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                length_function=len,
                separators=CHINESE_SEPARATORS,
            )
            pieces = splitter.split_text(text)
        except ImportError:
            # 无 langchain 时退化为定长滑窗：粒度仍不退化，只是边界不如语义切分自然
            step = max(size - overlap, 1)
            pieces = [text[i:i + size] for i in range(0, len(text), step)] or [text]

        out = []
        for i, piece in enumerate(pieces):
            out.append({
                'id': f"{parent_hit.get('id')}_{i}_l1child",
                'content': piece,
                'metadata': {
                    **parent_hit.get('metadata', {}),
                    'chunk_type': 'child',
                    'l1_split': True,      # 标记来源，便于离线分析降级影响
                },
                'score': parent_hit.get('score', 0),
                'source': parent_hit.get('source', 'unknown'),
                'parent_id': parent_hit.get('id'),
                'parent_content': text,
            })

        logger.info(
            "L1 内存切片: 父块 %s (%d 字) → %d 个子块",
            parent_hit.get('id'), len(text), len(out)
        )
        return out

    # ---------------- 降级打点（接入告警用） ----------------

    _METRICS: Dict[str, int] = {
        'l0_ok': 0,        # L0 正常命中
        'l1_fallback': 0,  # L1 同粒度降级命中
        'l2_empty': 0,     # L2 无召回
        'l2_error': 0,     # L2 基础设施故障
        'orphan': 0,       # 缺失 parent_id 的数据异常条数
    }

    @classmethod
    def _bump(cls, key: str, n: int = 1) -> None:
        cls._METRICS[key] = cls._METRICS.get(key, 0) + n

    @classmethod
    def get_degrade_metrics(cls) -> Dict[str, Any]:
        """
        返回降级打点快照，供 /health 或监控系统采集。

        生产环境应接入 Prometheus（Counter），这里用进程内计数器保持零依赖。
        关键看两个比率：
          - l1 占比过高 → 子块过滤阈值/数据本身有问题，需排查入库
          - l2_error > 0 → 基础设施故障，必须立即告警
        """
        m = dict(cls._METRICS)
        total = m['l0_ok'] + m['l1_fallback'] + m['l2_empty'] + m['l2_error']
        m['total'] = total
        m['l1_rate'] = round(m['l1_fallback'] / total, 4) if total else 0.0
        m['l2_rate'] = round((m['l2_empty'] + m['l2_error']) / total, 4) if total else 0.0
        return m

    @classmethod
    def reset_degrade_metrics(cls) -> None:
        for k in cls._METRICS:
            cls._METRICS[k] = 0

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
                'available_strategies': ['dense', 'sparse', 'hybrid', 'small_to_big'],
                'current_strategy': 'hybrid + small_to_big',
                'top_k_retrieve': self.top_k_retrieve,
                'top_k_children': self.top_k_children,
                'sparse_weight': self.sparse_weight,
                'dense_weight': self.dense_weight
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {str(e)}")
            return {}