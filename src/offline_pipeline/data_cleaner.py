"""
Data Cleaner Module
文本清洗、抽取标题/来源元数据
"""

import re
import logging
from typing import List, Optional
import json
from pathlib import Path

from ..config.settings import Config
from .document_loader import Document

logger = logging.getLogger(__name__)

class DataCleaner:
    """数据清洗器"""

    def __init__(self, config: Config):
        self.config = config
        # 清洗配置
        self.repeated_line_threshold = 3  # 行重复阈值
        self.min_line_length = 2  # 最小行长度
        self.page_number_patterns = [
            r'第\s*\d+\s*页',
            r'-\s*\d+\s*-',
            r'Page\s*\d+',
            r'page\s*\d+',
            r'\d+\s*/\s*\d+'
        ]

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """批量清洗文档"""
        cleaned_docs = []

        for doc in documents:
            try:
                cleaned_content = self._clean_text(doc.page_content)
                cleaned_doc = Document(
                    cleaned_content,
                    doc.metadata
                )
                cleaned_docs.append(cleaned_doc)
            except Exception as e:
                logger.error(f"Error cleaning document {doc.metadata.get('file_path', 'unknown')}: {str(e)}")
                # 保留原始文档作为备选
                cleaned_docs.append(doc)

        logger.info(f"Cleaned {len(documents)} documents, kept {len(cleaned_docs)}")
        return cleaned_docs

    def _clean_text(self, text: str) -> str:
        """清洗单个文本"""
        if not text:
            return ""

        # 1. 字符级清洗
        text = self._clean_characters(text)

        # 2. 行级清洗
        lines = text.split('\n')
        lines = self._clean_lines(lines)

        # 3. 段落级清洗
        paragraphs = self._clean_paragraphs(lines)

        # 4. 空白规范化
        text = self._normalize_whitespace(paragraphs)

        return text

    def _clean_characters(self, text: str) -> str:
        """字符级清洗"""
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 删除控制字符（保留基本空白字符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # 删除BOM（字节顺序标记）
        text = text.replace('﻿', '')

        # 删除零宽字符
        text = re.sub(r'[​-‍﻿]', '', text)

        # 删除软连字符
        text = text.replace('­', '')

        return text

    def _clean_lines(self, lines: List[str]) -> List[str]:
        """行级清洗"""
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 删除页码行
            if self._is_page_number(line):
                continue

            # 删除高频短行（可能是页眉页脚）
            if self._is_repeated_short_line(line, cleaned_lines):
                continue

            # 删除纯符号行
            if self._is_symbols_only(line):
                continue

            cleaned_lines.append(line)

        return cleaned_lines

    def _is_page_number(self, line: str) -> bool:
        """判断是否为页码"""
        for pattern in self.page_number_patterns:
            if re.search(pattern, line):
                return True
        return False

    def _is_repeated_short_line(self, line: str, existing_lines: List[str]) -> bool:
        """判断是否为高频短行（页眉页脚）"""
        # 过滤掉太短的行
        if len(line) < self.min_line_length:
            return True

        # 统计相同行出现的次数
        count = 0
        for existing_line in existing_lines:
            if line == existing_line:
                count += 1

        # 如果出现次数超过阈值，认为是页眉页脚
        return count >= self.repeated_line_threshold

    def _is_symbols_only(self, line: str) -> bool:
        """判断是否为纯符号行"""
        # 移除空白字符
        clean_line = re.sub(r'\s+', '', line)

        # 如果不含中文字符和英文字符，认为是纯符号
        if not re.search(r'[一-鿿a-zA-Z]', clean_line):
            return True

        return False

    def _clean_paragraphs(self, lines: List[str]) -> List[str]:
        """段落级清洗：去重连续出现的相同段落"""
        if not lines:
            return []

        cleaned_paragraphs = []
        prev_paragraph = None

        for line in lines:
            # 如果当前段落与上一个段落相同，跳过
            if line == prev_paragraph:
                continue

            cleaned_paragraphs.append(line)
            prev_paragraph = line

        return cleaned_paragraphs

    def _normalize_whitespace(self, lines: List[str]) -> str:
        """规范化空白"""
        # 压缩行内连续空白
        normalized_lines = []
        for line in lines:
            # 压缩连续空格，保留单个空格
            normalized_line = re.sub(r' {2,}', ' ', line)
            normalized_lines.append(normalized_line)

        # 连接段落，用两个换行符分隔
        text = '\n\n'.join(normalized_lines)

        # 删除开头和结尾的空白
        text = text.strip()

        return text

    def extract_metadata(self, documents: List[Document]) -> List[Document]:
        """提取元数据"""
        enhanced_docs = []

        for doc in documents:
            metadata = doc.metadata.copy()

            # 提取标题
            title = self._extract_title(doc.page_content)
            if title:
                metadata['title'] = title

            # 提取关键词（简化版）
            keywords = self._extract_keywords(doc.page_content)
            if keywords:
                metadata['keywords'] = keywords

            # 计算字符数和词数
            metadata['char_count'] = len(doc.page_content)
            metadata['word_count'] = len(doc.page_content.split())

            enhanced_doc = Document(
                doc.page_content,
                metadata
            )
            enhanced_docs.append(enhanced_doc)

        return enhanced_docs

    def _extract_title(self, text: str) -> Optional[str]:
        """提取标题"""
        lines = text.split('\n')

        # 尝试从标题行提取
        for line in lines[:10]:  # 只检查前10行
            line = line.strip()

            # Markdown标题
            if line.startswith('# ') and len(line) < 100:
                return line[2:].strip()

            # 大写字母开头的短行（可能是标题）
            if (len(line) < 50 and
                line and
                line[0].isupper() and
                not line.endswith('.') and
                len(line.split()) < 10):
                return line

        # 如果没找到，使用文件名作为标题
        return None

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简化版）"""
        # 这里使用简单的词频统计，实际项目中可以使用jieba等工具
        words = text.split()

        # 过滤掉常见停用词
        stop_words = {'的', '了', '和', '是', '在', '我', '有', '就', '不', '人', '都', '一', '个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

        # 统计词频
        word_freq = {}
        for word in words:
            if len(word) > 1 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # 取前5个高频词
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        return [word for word, freq in keywords]

    def save_cleaned_documents(self, documents: List[Document], output_path: Path):
        """保存清洗后的文档"""
        # 转换为可序列化的格式
        serializable_docs = []
        for doc in documents:
            serializable_docs.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })

        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(documents)} cleaned documents to {output_path}")

    def load_cleaned_documents(self, input_path: Path) -> List[Document]:
        """加载清洗后的文档"""
        documents = []

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            doc = Document(
                item['content'],
                item['metadata']
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {input_path}")
        return documents