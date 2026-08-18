"""
Embedding Provider Module
BGE-M3 稠密+稀疏向量生成
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json
import hashlib

try:
    import torch  # 用于在 GPU 上每批后释放显存缓存
except ImportError:
    torch = None

from ..config.settings import Config

logger = logging.getLogger(__name__)


class BGEEmbeddingProvider:
    """BGE-M3 Embedding Provider"""

    def __init__(self, config: Config):
        self.config = config
        self.model_name = config.EMBED_MODEL_NAME
        self.device = config.EMBED_DEVICE
        self.embed_dim = config.EMBED_DIM

        # 模型路径
        self.model_path = config.BASE_DIR / "src/models" / self.model_name.lower(
        )

        # 初始化模型
        self.dense_model = None
        self.sparse_model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """加载BGE-M3模型"""
        try:
            try:
                from FlagEmbedding import BGEM3FlagModel
                logger.info("Using FlagEmbedding BGEM3FlagModel")
            except ImportError:
                from FlagEmbedding import BGE_M3_FlagModel as BGEM3FlagModel
                logger.info("Using BGE_M3_FlagModel from FlagEmbedding")

            logger.info(f"Loading BGE-M3 model from {self.model_path}...")

            # 加载模型

            self.dense_model = BGEM3FlagModel(
                str(self.model_path),
                devices=self.device,
                # 仅在 GPU 上开启 fp16：BGE-M3 约 2.3GB 权重，fp16 下约 1.1GB，
                # 既能塞进 8G 显存又能显著提速；CPU 不支持 fp16，保持 fp32
                use_fp16=(self.device == 'cuda'))

            logger.info("BGE-M3 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {str(e)}")
            raise

    def generate_embeddings(self,
                            texts: List[str]) -> Tuple[np.ndarray, List[Dict]]:
        """
        生成稠密向量和稀疏向量（BGE-M3 一次前向同时产出两种向量）

        原理说明：
        - 稠密向量(dense)：1024 维连续向量，编码文本语义，用于语义相似检索
        - 稀疏向量(sparse)：{token_id: weight} 字典，类似 BM25 词项权重，
          编码关键词匹配信息，用于精确词面检索
        两者结合即「混合检索」，兼顾语义理解与关键词精确命中。

        Args:
            texts: 输入文本列表

        Returns:
            dense_embeddings: 稠密向量矩阵 (n_samples, embed_dim)
            sparse_embeddings: 稀疏向量列表，每个元素是 {idx: weight} 字典
        """
        if not texts:
            return np.array([]), []

        logger.info(f"Generating embeddings for {len(texts)} texts...")

        try:
            # 使用BGE-M3同时生成稠密和稀疏向量
            # return_dense/return_sparse 控制输出哪些向量；colbert 向量本项目不用
            results = self.dense_model.encode(texts,
                                              # GPU 前向批大小。注意：128 在持续跑时会因
                                              # 文本长短不一造成显存碎片、顶到 8G 上限而空转，
                                              # 故用 64 保留足够显存余量
                                              batch_size=64,
                                              # 块文本最长约 1333 token，2048 留足余量；
                                              # 之前设 8192 会浪费大量显存/内存，GPU 上易 OOM
                                              max_length=2048,
                                              return_dense=True,
                                              return_sparse=True,
                                              return_colbert_vecs=False)

            # 提取稠密向量，shape = (len(texts), 1024)
            dense_embeddings = results['dense_vecs']

            # 提取稀疏向量：lexical_weights 是 List[Dict[token_id_str, weight]]
            sparse_embeddings = []
            for sparse_vec in results['lexical_weights']:
                # 转换为 {idx: weight} 格式，只保留正权重项（Milvus 稀疏向量要求）
                sparse_dict = {}
                for idx, weight in sparse_vec.items():
                    if weight > 0:  # 只保留非零权重
                        sparse_dict[idx] = weight
                sparse_embeddings.append(sparse_dict)

            logger.info(
                f"Generated embeddings: dense shape {dense_embeddings.shape}, "
                f"sparse vectors {len(sparse_embeddings)}")

            return dense_embeddings, sparse_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            return np.array([]), []

    def compute_hash(self, text: str) -> str:
        """计算文本的MD5哈希值作为ID"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def save_embeddings(self, texts: List[str], dense_embeddings,
                        sparse_embeddings: List[Dict], output_path: Path):
        """保存向量数据

        dense_embeddings 兼容两种输入：
        - numpy 数组 (n, dim)：调用 .tolist() 序列化
        - list[list]（batch_process 收集的结果，每个 chunk 已是 tolist 后的 list）：
          直接就是可序列化格式，无需再转
        """
        # 统一转成可序列化的 list，避免对 list 误调用 .size/.tolist() 报错
        if isinstance(dense_embeddings, np.ndarray):
            dense_list = dense_embeddings.tolist(
            ) if dense_embeddings.size > 0 else []
        else:
            dense_list = dense_embeddings  # 已是 list，直接使用

        # 转换为可序列化的格式
        serializable_data = {
            'texts': texts,
            'dense_embeddings': dense_list,
            'sparse_embeddings': sparse_embeddings,
            'embed_dim': self.embed_dim,
            'model_name': self.model_name
        }

        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Saved embeddings for {len(texts)} texts to {output_path}")

    def batch_process(self, chunks: List, batch_size: int = 32) -> List[Dict]:
        """
        批量处理分块数据，生成向量

        Args:
            chunks: 分块列表，每个分块应包含content字段
            batch_size: 批处理大小

        Returns:
            包含向量的分块列表，每个分块包含：
            - id: 分块ID
            - content: 内容
            - metadata: 元数据
            - dense_embedding: 稠密向量
            - sparse_embedding: 稀疏向量
        """
        results = []

        # 按batch处理
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]

            try:
                # 生成向量
                dense_vecs, sparse_vecs = self.generate_embeddings(batch_texts)

                # 组合结果
                for j, (chunk, dense_vec, sparse_vec) in enumerate(
                        zip(batch_chunks, dense_vecs, sparse_vecs)):
                    result = {
                        'id':
                        chunk.id,
                        'content':
                        chunk.content,
                        'metadata':
                        chunk.metadata,
                        'dense_embedding':
                        dense_vec.tolist() if isinstance(
                            dense_vec, np.ndarray) else dense_vec,
                        'sparse_embedding':
                        sparse_vec,
                        # 父子分块关联：子块命中后需经 parent_id 回溯父块，
                        # 并直接返回 parent_content（冗余存储的父块完整正文）。
                        # 这两个字段若在此丢弃，Milvus 中 parent_id/parent_content
                        # 将恒为空，父子检索功能实际失效。
                        'parent_id': chunk.parent_id or '',
                        'parent_content': chunk.parent_content or '',
                        'hash_id':
                        self.compute_hash(chunk.content)
                    }
                    results.append(result)

                logger.info(
                    f"Processed batch {i//batch_size + 1}: {len(batch_chunks)} chunks"
                )

            except Exception as e:
                logger.error(
                    f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
            finally:
                # 每批结束后释放 PyTorch 缓存的显存。因文本长短不一，
                # 否则 reserved 显存会持续爬升直至顶到 8G 上限，
                # 触发分配器空转(thrashing)导致每批骤降十几倍
                if torch is not None and self.device == 'cuda':
                    torch.cuda.empty_cache()

        logger.info(
            f"Successfully processed {len(results)} out of {len(chunks)} chunks"
        )
        return results
