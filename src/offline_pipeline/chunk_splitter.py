"""
Chunk Splitter Module
父子分层分块器（父块2000｜子块400）
"""

import logging
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from dataclasses import dataclass

from ..config.settings import Config
from .document_loader import Document

logger = logging.getLogger(__name__)

# 中文优化分隔符（入库切分与在线 L1 降级切分共用）。
# 必须与线上保持完全一致：L1 降级会在内存中把父块临时切成子块，
# 若分隔符/参数与入库时不同，检索到的片段与库内子块不同构，会引入难以排查的偏差。
CHINESE_SEPARATORS = [
    "\n\n",  # 双换行
    "\n",    # 单换行
    "。",    # 中文句号
    "！",    # 中文感叹号
    "？",    # 中文问号
    "；",    # 中文分号
    "：",    # 中文冒号
    ",",     # 英文逗号
    " ",     # 空格
    ""       # 字符级别
]


@dataclass
class Chunk:
    """分块数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    parent_content: Optional[str] = None
    chunk_type: str = 'child'  # 'parent' or 'child'

class ChunkSplitter:
    """分块器"""

    def __init__(self, config: Config):
        self.config = config
        self.parent_chunk_size = config.PARENT_CHUNK_SIZE
        self.child_chunk_size = config.CHILD_CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        if not 0 <= self.chunk_overlap < min(self.parent_chunk_size, self.child_chunk_size):
            raise ValueError('chunk overlap must be nonnegative and smaller than both chunk sizes')

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        将文档分割为「父子双层」块

        父子分块（Parent-Child Chunking）策略：
        - 父块(parent, ~2000字)：保留完整上下文，用于最终给 LLM 提供背景
        - 子块(child, ~400字)：粒度细，用于向量检索时精确命中
        检索时用子块匹配（精度高），返回时通过 parent_id 回溯到父块（上下文全），
        兼顾「检索精度」与「上下文完整性」。

        Args:
            documents: 清洗后的文档列表

        Returns:
            所有父块 + 子块的扁平列表（通过 chunk_type 字段区分）
        """
        all_chunks = []

        for doc in documents:
            try:
                # 先分父块（大块，保留上下文）
                parent_chunks = self._split_into_parent_chunks(doc)

                # 再把每个父块细分为子块（小块，用于检索）
                child_chunks = []
                for parent_chunk in parent_chunks:
                    children = self._split_into_child_chunks(parent_chunk, parent_chunk.id)
                    child_chunks.extend(children)

                # 保存父块和子块
                all_chunks.extend(parent_chunks)
                all_chunks.extend(child_chunks)

                logger.info(f"Split document {doc.metadata.get('file_path', 'unknown')} into "
                          f"{len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks")

            except Exception as e:
                logger.error(f"Error splitting document {doc.metadata.get('file_path', 'unknown')}: {str(e)}")
                continue

        return all_chunks

    def _split_into_parent_chunks(self, document: Document) -> List[Chunk]:
        """将文档分割为父块"""
        parent_chunks = []

        # 使用RecursiveCharacterTextSplitter进行智能分割
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            # 中文优化分割器（分隔符与在线 L1 降级切分共用，勿单独修改）
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.parent_chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=CHINESE_SEPARATORS
            )

            # 分割文本
            split_texts = text_splitter.split_text(document.page_content)

            # 创建父块
            for i, text in enumerate(split_texts):
                chunk_id = f"{document.metadata.get('file_path', 'unknown')}_{i}_parent"

                parent_chunk = Chunk(
                    id=chunk_id,
                    content=text,
                    metadata={
                        **document.metadata,
                        'chunk_index': i,
                        'chunk_type': 'parent',
                        'total_chunks': len(split_texts)
                    },
                    # 必须显式传入：Chunk.chunk_type 默认值为 'child'，
                    # 不传会导致父块的 chunk_type 被错误标记为 'child'，
                    # 使下游任何基于 chunk_type 的过滤/统计全部失效。
                    chunk_type='parent'
                )
                parent_chunks.append(parent_chunk)

        except ImportError:
            # 如果没有langchain，使用简单的分割方法
            parent_chunks = self._simple_split(document, self.parent_chunk_size, 'parent')

        return parent_chunks

    def _split_into_child_chunks(self, parent_chunk: Chunk, parent_id: str) -> List[Chunk]:
        """将父块分割为子块"""
        child_chunks = []

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            # 子块使用更小的chunk_size（分隔符与在线 L1 降级切分共用，勿单独修改）
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.child_chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=CHINESE_SEPARATORS
            )

            # 分割父块内容
            split_texts = text_splitter.split_text(parent_chunk.content)

            # 创建子块
            for i, text in enumerate(split_texts):
                chunk_id = f"{parent_id}_{i}_child"

                child_chunk = Chunk(
                    id=chunk_id,
                    content=text,
                    metadata={
                        **parent_chunk.metadata,
                        'chunk_index': i,
                        'chunk_type': 'child',
                        'total_chunks': len(split_texts)
                    },
                    parent_id=parent_id,
                    parent_content=parent_chunk.content,
                    chunk_type='child'
                )
                child_chunks.append(child_chunk)

        except ImportError:
            # 简单分割方法
            child_chunks = self._simple_split(parent_chunk, self.child_chunk_size, 'child')
            # 更新父块ID
            for chunk in child_chunks:
                chunk.parent_id = parent_id
                chunk.parent_content = parent_chunk.content

        return child_chunks

    def _simple_split(self, document: Document, chunk_size: int, chunk_type: str) -> List[Chunk]:
        """简单的文本分割（作为备选方案）"""
        chunks = []
        content = document.content if isinstance(document, Chunk) else document.page_content
        metadata = document.metadata.copy()

        # 添加chunk类型信息
        metadata['chunk_type'] = chunk_type

        # 简单按字符分割
        for index, i in enumerate(range(0, len(content), chunk_size - self.chunk_overlap)):
            chunk_content = content[i:i + chunk_size]
            base_id = document.id if isinstance(document, Chunk) else metadata.get('file_path', 'unknown')
            chunk_id = f"{base_id}_{index}_{chunk_type}"

            chunk = Chunk(
                id=chunk_id,
                content=chunk_content,
                metadata={
                    **metadata,
                    'chunk_index': index,
                    'total_chunks': len(range(0, len(content), chunk_size - self.chunk_overlap))
                },
                chunk_type=chunk_type
            )

            chunks.append(chunk)

        return chunks

    def save_chunks(self, chunks: List[Chunk], output_path: Path):
        """保存分块结果"""
        # 转换为可序列化的格式
        serializable_chunks = []
        for chunk in chunks:
            serializable_chunk = {
                'id': chunk.id,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'parent_id': chunk.parent_id,
                'parent_content': chunk.parent_content,
                'chunk_type': chunk.chunk_type
            }
            serializable_chunks.append(serializable_chunk)

        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: Path) -> List[Chunk]:
        """加载分块结果"""
        chunks = []

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            chunk = Chunk(
                id=item['id'],
                content=item['content'],
                metadata=item['metadata'],
                parent_id=item.get('parent_id'),
                parent_content=item.get('parent_content'),
                chunk_type='child' if item.get('parent_id') else 'parent'
            )
            chunk.metadata['chunk_type'] = chunk.chunk_type
            chunks.append(chunk)

        logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
        return chunks

    def get_parent_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """获取所有父块"""
        return [chunk for chunk in chunks if chunk.chunk_type == 'parent']

    def get_child_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """获取所有子块"""
        return [chunk for chunk in chunks if chunk.chunk_type == 'child']

    def get_children_by_parent(self, chunks: List[Chunk], parent_id: str) -> List[Chunk]:
        """根据父ID获取所有子块"""
        return [chunk for chunk in chunks if chunk.parent_id == parent_id]

    def get_chunk_by_id(self, chunks: List[Chunk], chunk_id: str) -> Optional[Chunk]:
        """根据ID获取特定块"""
        for chunk in chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def remove_duplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """移除重复的块"""
        seen_ids = set()
        unique_chunks = []

        for chunk in chunks:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_chunks.append(chunk)

        logger.info(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
        return unique_chunks
