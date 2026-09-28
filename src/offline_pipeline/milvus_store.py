"""
Milvus Store Module
Milvus集合创建、索引构建、批量入库
"""

import logging
import time
from typing import List, Dict, Any

from ..config.settings import Config

logger = logging.getLogger(__name__)

try:
    from pymilvus import (connections, utility, FieldSchema, CollectionSchema,
                          DataType, Collection, MilvusException, MilvusClient,
                          AnnSearchRequest, WeightedRanker)
except ImportError:
    logger.warning(
        "pymilvus not installed. Please install with: pip install pymilvus")
    raise


class MilvusStore:
    """Milvus向量存储管理器"""

    def __init__(self, config: Config):
        self.config = config
        self.collection_name = config.MILVUS_COLLECTION_NAME
        self.index_type = config.MILVUS_INDEX_TYPE
        self.nlist = config.MILVUS_NLIST
        self.nprobe = config.MILVUS_NPROBE

        # Milvus连接配置
        self.host = getattr(config, "MILVUS_HOST", "localhost")
        self.port = getattr(config, "MILVUS_PORT", "19530")
        self.db_name = getattr(config, "MILVUS_DB_NAME", "MED")
        self.alias = getattr(config, "MILVUS_CONN_ALIAS", "MED")
        # MilvusClient 用于混合检索（hybrid_search）
        self.client = None
        # 集合是否含顶层 chunk_type 字段（None=未探测，首次用到时惰性探测并缓存）
        self._has_chunk_type: Any = None

        # 初始化连接
        self._init_connection()

    def has_chunk_type_field(self) -> bool:
        """当前集合 schema 是否包含顶层 chunk_type 字段。

        旧库没有该字段（chunk_type 只存在于 metadata JSON 内部），此时必须退化用
        `parent_id != ""` 区分子块——父块的 parent_id 在 batch_process 中已被
        归一为空字符串，因此该条件与 chunk_type=='child' 等价。
        """
        if self._has_chunk_type is not None:
            return bool(self._has_chunk_type)

        try:
            if not utility.has_collection(self.collection_name, using=self.alias):
                self._has_chunk_type = False
                return False
            collection = Collection(self.collection_name, using=self.alias)
            field_names = {f.name for f in collection.schema.fields}
            self._has_chunk_type = "chunk_type" in field_names
            if not self._has_chunk_type:
                logger.warning(
                    "集合 %s 缺少顶层 chunk_type 字段（旧库），子块过滤将退化为 "
                    "parent_id != ''。效果等价，但建议重建集合以获得显式字段。",
                    self.collection_name)
        except Exception as e:  # 探测失败不应阻断主流程
            logger.warning("探测 chunk_type 字段失败(%s)，退化为 parent_id 过滤", e)
            self._has_chunk_type = False

        return bool(self._has_chunk_type)

    def child_filter_expr(self) -> str:
        """返回「只召回子块」的 Milvus 过滤表达式。

        新库用显式字段 chunk_type=='child'；旧库退化为 parent_id != ''。
        两者语义等价：chunk_type=='parent' 的块 parent_id 恒为空字符串。
        """
        if self.has_chunk_type_field():
            return "chunk_type == 'child' and parent_id != ''"
        return "parent_id != ''"

    @staticmethod
    def combine_expr(*exprs: str) -> str:
        """把多个过滤表达式用 and 连接，自动跳过空串并加括号保证优先级。"""
        parts = [e for e in exprs if e and e.strip()]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return " and ".join(f"({p})" for p in parts)

    def _ensure_database(self):
        """确保目标数据库存在；不存在则自动创建（需要 default 库权限）。"""
        tmp_client = None
        try:
            tmp_client = MilvusClient(
                uri=f"http://{self.host}:{self.port}",
                db_name="default",
            )
            dbs = tmp_client.list_databases()
            if self.db_name not in dbs:
                logger.info(
                    f"数据库 {self.db_name} 不存在，自动创建...")
                tmp_client.create_database(self.db_name)
                logger.info(f"数据库 {self.db_name} 创建成功")
            else:
                logger.info(f"数据库 {self.db_name} 已存在")
        finally:
            if tmp_client is not None:
                try:
                    tmp_client.close()
                except Exception:
                    pass

    def _init_connection(self):
        """初始化Milvus连接"""
        try:
            # 先确保目标数据库存在
            self._ensure_database()

            # 若同名 alias 已存在（如重复初始化），先断开避免报错
            try:
                connections.disconnect(self.alias)
            except Exception:
                pass

            # 连接到Milvus服务器（ORM 风格，用于 Collection 操作）
            connections.connect(alias=self.alias,
                                host=self.host,
                                port=self.port,
                                db_name=self.db_name)
            # MilvusClient（新 SDK 风格，用于 hybrid_search 混合检索）
            self.client = MilvusClient(
                uri=f"http://{self.host}:{self.port}",
                db_name=self.db_name,
            )
            logger.info(
                f"Connected to Milvus server at {self.host}:{self.port}, "
                f"database={self.db_name}, alias={self.alias}")
        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    def create_or_load_collection(self, dim: int = 1024) -> Collection:
        """创建或加载集合"""
        try:
            # 检查集合是否已存在
            if utility.has_collection(self.collection_name, using=self.alias):
                logger.info(
                    f"Collection {self.collection_name} already exists, loading..."
                )
                collection = Collection(self.collection_name, using=self.alias)
                # 幂等地确保索引存在：兼容上次建库中断导致的"有集合但无索引"状态，
                # 否则 Milvus 会拒绝 load 无索引的集合 (code=700 index not found)
                self._create_indexes(collection)
                collection.load()
                return collection
            else:
                logger.info(f"Creating collection {self.collection_name}...")
                return self._create_collection(dim)

        except MilvusException as e:
            logger.error(f"Failed to create/load collection: {str(e)}")
            raise

    def _create_collection(self, dim: int) -> Collection:
        """
        创建新的集合（Collection）

        集合 schema 设计说明：
        - 采用「父子分块」结构入库：检索命中子块(child)，返回时带上父块(parent)上下文
        - 同时存储稠密向量(dense_vector)和稀疏向量(sparse_vector)，
          以支持「语义检索 + 关键词检索」的混合检索（Hybrid Search）
        - enable_dynamic_field=True 允许插入 schema 之外的动态字段

        Args:
            dim: 稠密向量维度（BGE-M3 为 1024）

        Returns:
            创建并加载好的 Collection 对象
        """
        # 定义字段schema
        fields = [
            # 主键字段：每个分块的唯一标识（父子块 id 规则见 chunk_splitter）
            FieldSchema(name="id",
                        dtype=DataType.VARCHAR,
                        description="Document chunk ID",
                        max_length=256,
                        is_primary=True),
            # 文本内容字段：分块后的正文（子块内容，用于检索展示）
            FieldSchema(name="text",
                        dtype=DataType.VARCHAR,
                        description="Text content",
                        max_length=65535),
            # 稠密向量字段：BGE-M3 输出的 1024 维语义向量，承载语义相似度
            FieldSchema(name="dense_vector",
                        dtype=DataType.FLOAT_VECTOR,
                        description="Dense embedding vector",
                        dim=dim),
            # 稀疏向量字段：BGE-M3 输出的词权重向量（{token_id: weight}），
            # 类似 BM25，承载关键词匹配能力
            FieldSchema(name="sparse_vector",
                        dtype=DataType.SPARSE_FLOAT_VECTOR,
                        description="Sparse embedding vector"),
            # 父块ID字段：当前子块所属的父块 id，用于「子块命中后回溯父块」
            FieldSchema(name="parent_id",
                        dtype=DataType.VARCHAR,
                        description="Parent chunk ID",
                        max_length=256),
            # 父块内容字段：冗余存储父块完整正文，避免二次查询即可返回上下文
            FieldSchema(name="parent_content",
                        dtype=DataType.VARCHAR,
                        description="Parent chunk content",
                        max_length=65535),
            # 分块类型字段：'parent' | 'child'。
            # 顶层标量字段（而非 metadata 里的 JSON key）才能在检索时下推过滤表达式，
            # 让 Milvus 只召回子块，避免 2000 字父块抢占 Top-K 槽位。
            FieldSchema(name="chunk_type",
                        dtype=DataType.VARCHAR,
                        description="Chunk type: parent or child",
                        max_length=16),
            # 来源字段：学科/科室来源（如 心血管、神经科），用于按来源过滤
            FieldSchema(name="source",
                        dtype=DataType.VARCHAR,
                        description="Document source",
                        max_length=128),
            # 时间戳字段：文档写入时间（Unix 秒），便于排序与增量更新
            FieldSchema(name="timestamp",
                        dtype=DataType.INT64,
                        description="Creation timestamp"),
            # 元数据字段：JSON 类型，存放标题、关键词、字数等附加信息
            FieldSchema(name="metadata",
                        dtype=DataType.JSON,
                        description="Additional metadata")
        ]

        # 创建schema
        schema = CollectionSchema(
            fields=fields,
            description="Medical RAG knowledge base chunks",
            enable_dynamic_field=True)

        # 创建集合
        collection = Collection(name=self.collection_name, schema=schema, using=self.alias)

        # 创建索引
        self._create_indexes(collection)

        # 加载集合
        collection.load()

        logger.info(f"Collection {self.collection_name} created successfully")
        return collection

    def _create_indexes(self, collection: Collection):
        """创建索引"""
        try:
            # 1. 稠密向量索引 - IVF_FLAT
            index_params_dense = {
                "index_type": self.index_type,
                "params": {
                    "nlist": self.nlist
                },
                "metric_type": "IP"  # 内积相似度
            }

            # 检查是否已存在索引 (pymilvus >= 2.x: 用 collection.indexes 判断)
            existing_index_fields = {
                idx.field_name
                for idx in collection.indexes
            }
            if "dense_vector" not in existing_index_fields:
                collection.create_index(field_name="dense_vector",
                                        index_params=index_params_dense)
                logger.info("Created dense vector index")

            # 2. 稀疏向量索引 - SPARSE_INVERTED_INDEX
            index_params_sparse = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {
                    "drop_ratio_build": 0.2
                },
                "metric_type": "IP"
            }

            if "sparse_vector" not in existing_index_fields:
                collection.create_index(field_name="sparse_vector",
                                        index_params=index_params_sparse)
                logger.info("Created sparse vector index")

        except MilvusException as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict], batch_size: int = 1000):
        """
        批量添加文档到Milvus

        Args:
            documents: 文档列表，每个文档包含各种字段
            batch_size: 批处理大小
        """
        if not documents:
            logger.warning("No documents to add")
            return 0

        collection = Collection(self.collection_name, using=self.alias)
        total_added = 0

        # 旧集合没有 chunk_type 列，此时必须省略该列，
        # 否则按列位置插入会因列数与 schema 不匹配而报错。
        with_chunk_type = self.has_chunk_type_field()
        if not with_chunk_type:
            logger.warning(
                "集合 %s 无 chunk_type 字段，本次入库将跳过该列（子块过滤退化为 parent_id != ''）",
                self.collection_name)

        try:
            # 按batch处理
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]

                # 准备批量数据
                ids = []
                texts = []
                dense_vectors = []
                sparse_vectors = []
                parent_ids = []
                parent_contents = []
                chunk_types = []
                sources = []
                timestamps = []
                metadata_list = []

                for doc in batch_docs:
                    ids.append(doc.get('id', ''))
                    texts.append(doc.get('content', ''))
                    dense_vectors.append(doc.get('dense_embedding', []))
                    sparse_vectors.append(doc.get('sparse_embedding', {}))
                    # 父块 parent_id 为 None 时归一为空串：空串是「我是父块」的判定依据
                    parent_ids.append(doc.get('parent_id') or '')
                    parent_contents.append(doc.get('parent_content') or '')
                    # 按父子关系写类型，纠正旧数据中已有但错误的 chunk_type。
                    chunk_types.append('child' if doc.get('parent_id') else 'parent')
                    sources.append(
                        doc.get('metadata', {}).get('source', 'unknown'))
                    # timestamp 在 schema 中为 int64，但源头可能是 float（如 st_mtime），统一转 int 兜底
                    _ts = doc.get('metadata', {}).get('timestamp',
                                                      int(time.time()))
                    timestamps.append(int(_ts))
                    metadata_list.append(doc.get('metadata', {}))

                # 构造entities（列顺序必须与 _create_collection 的 fields 一致）
                entities = [
                    ids, texts, dense_vectors, sparse_vectors, parent_ids,
                    parent_contents
                ]
                if with_chunk_type:
                    entities.append(chunk_types)
                entities.extend([sources, timestamps, metadata_list])

                # 批量插入
                insert_result = collection.insert(entities)
                batch_count = len(batch_docs)
                total_added += batch_count

                logger.info(
                    f"Inserted batch {i//batch_size + 1}: {batch_count} documents"
                )

        except MilvusException as e:
            logger.error(f"Failed to insert documents: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

        logger.info(
            f"Successfully added {total_added} documents to collection {self.collection_name}"
        )
        return total_added

    def search(self,
               query_dense: List[float],
               query_sparse: Dict,
               limit: int = 10,
               expr: str = "",
               sparse_weight: float = 0.7,
               dense_weight: float = 1.0,
               only_children: bool = False) -> List[Dict]:
        """
        执行混合搜索（稠密向量 + 稀疏向量）

        使用 pymilvus 2.x 的 MilvusClient.hybrid_search：
        - 为每个向量字段构造一个 AnnSearchRequest
        - 用 WeightedRanker 按权重融合两路结果

        Args:
            query_dense: 查询稠密向量 (list of float)
            query_sparse: 查询稀疏向量 ({token_id: weight} 字典)
            limit: 返回结果数量
            expr: 过滤表达式（如 source == 'xxx'）
            sparse_weight: 稀疏权重（对齐 EduRag: 0.7）
            dense_weight: 稠密权重（对齐 EduRag: 1.0）
            only_children: True 时下推「只召回子块」过滤。

        Returns:
            搜索结果列表，每项包含 id/distance/score/entity
        """
        # 确保集合已加载到内存
        collection = Collection(self.collection_name, using=self.alias)
        collection.load()

        # 子块过滤下推：在 Milvus 侧生效，父块不占用 Top-K 槽位。
        # 若在应用层再做后置过滤，2000 字父块（语义宽泛、易高分）会挤占子块名额，
        # 极端情况下 Top-K 全为父块，Small-to-Big 静默失效。
        if only_children:
            expr = self.combine_expr(expr, self.child_filter_expr())
            logger.info("子块过滤下推生效，expr = %s", expr)

        # 稠密向量检索请求
        dense_req = AnnSearchRequest(data=[query_dense],
                                     anns_field="dense_vector",
                                     param={
                                         "metric_type": "IP",
                                         "params": {
                                             "nprobe": self.nprobe
                                         }
                                     },
                                     limit=limit,
                                     expr=expr or None)
        # 稀疏向量检索请求
        sparse_req = AnnSearchRequest(data=[query_sparse],
                                      anns_field="sparse_vector",
                                      param={
                                          "metric_type": "IP",
                                          "params": {
                                              "drop_ratio_search": 0.2
                                          }
                                      },
                                      limit=limit,
                                      expr=expr or None)

        # 加权融合器：权重顺序需与 reqs 顺序一致 [dense, sparse]（对齐 EduRag: sparse 0.7 : dense 1.0）
        ranker = WeightedRanker(dense_weight, sparse_weight)

        # 需要返回的标量字段
        output_fields = [
            "text", "parent_id", "parent_content", "source", "metadata"
        ]

        try:
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_req, sparse_req],
                ranker=ranker,
                limit=limit,
                output_fields=output_fields)

            # hybrid_search 返回 List[List[dict]]，单查询取第 0 个
            formatted_results = []
            for hits in results:
                for hit in hits:
                    entity = hit.get('entity', {})
                    formatted_results.append({
                        'id': hit.get('id'),
                        'distance': hit.get('distance'),
                        # hybrid_search 结果用 distance 作为得分
                        'score': hit.get('distance'),
                        'entity': entity
                    })

            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results

        except MilvusException as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def hybrid_search_with_rerank(self,
                                  query_dense: List[float],
                                  query_sparse: Dict,
                                  limit: int = 10,
                                  expr: str = "",
                                  sparse_weight: float = 0.7,
                                  dense_weight: float = 1.0,
                                  only_children: bool = False) -> List[Dict]:
        """
        混合搜索（稠密+稀疏），结果已由 WeightedRanker 加权融合。

        说明：精确重排序（BGE-reranker）由 online_service.reranker 在更上层完成，
        此处直接返回混合检索的融合结果，并按 id 去重。

        Args:
            query_dense: 查询稠密向量
            query_sparse: 查询稀疏向量
            limit: 返回数量
            expr: 过滤表达式
            sparse_weight: 稀疏权重（对齐 EduRag: 0.7）
            dense_weight: 稠密权重（对齐 EduRag: 1.0）
            only_children: True 时只在子块中召回（Small-to-Big 必需）

        Returns:
            融合后的结果列表
        """
        search_results = self.search(
            query_dense, query_sparse, limit, expr,
            sparse_weight=sparse_weight, dense_weight=dense_weight,
            only_children=only_children
        )

        if not search_results:
            return []

        # 按 id 去重（混合检索两路可能命中同一文档）
        seen_ids = set()
        unique_results = []
        for result in search_results:
            rid = result.get('id')
            if rid not in seen_ids:
                seen_ids.add(rid)
                unique_results.append(result)

        return unique_results

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            if utility.has_collection(self.collection_name, using=self.alias):
                collection = Collection(self.collection_name, using=self.alias)

                info = {
                    'name': self.collection_name,
                    'description': collection.description,
                    'num_entities': collection.num_entities,
                    'schema': collection.schema.to_dict(),
                    'indexes': utility.list_indexes(self.collection_name, using=self.alias)
                }

                return info
            else:
                return {'error': 'Collection does not exist'}

        except MilvusException as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {'error': str(e)}
