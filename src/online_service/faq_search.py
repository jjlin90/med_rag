"""
FAQ Search Module
MySQL FAQ + BM25检索、置信度判断
"""

import logging
try:
    import jieba
except ImportError:
    jieba = None
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# 项目使用 PyMySQL（纯Python驱动，见 pyproject.toml），而非 mysql-connector-python
try:
    import pymysql
    from pymysql.cursors import DictCursor
    MYSQL_AVAILABLE = True
except ImportError:
    pymysql = None
    DictCursor = None
    MYSQL_AVAILABLE = False

from ..config.settings import Config
from .cache_manager import CacheManager, faq_cache_key, normalize_query

logger = logging.getLogger(__name__)


def faq_question_matches(query: str, question: str) -> bool:
    """Only trim outer whitespace; never infer medical equivalence from overlap."""
    return (isinstance(question, str) and bool(normalize_query(query))
            and normalize_query(query) == normalize_query(question))

class FAQSearch:
    """FAQ搜索器"""

    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager

        # MySQL连接配置（统一从 settings 读取，不再写死 localhost/root/空密码）
        self.db_config = {
            'host': config.MYSQL_HOST,
            'port': config.MYSQL_PORT,
            'user': config.MYSQL_USER,
            'password': config.MYSQL_PASSWORD,
            'database': config.MYSQL_DATABASE,
            'charset': 'utf8mb4'
        }

        # BM25参数（对齐 EduRag 快通道：softmax 归一化 → [0,1] 区间 → 阈值 0.85）
        self.bm25_k1 = 1.2
        self.bm25_b = 0.75
        # 候选相对分布阈值，不是正确概率；直答还必须校验标准问题一致。
        self.bm25_threshold = config.FAQ_NORMALIZED_THRESHOLD  # 对齐 EduRag: 0.85

        # 初始化数据库连接
        self.connection = None
        self.cursor = None
        self.bm25_index = None
        self.faq_id_map = {}  # index -> db id 映射，用于把 BM25 结果转回 MySQL 主键

        # 初始化
        self._init_db()
        self._init_bm25_index()

    def _init_db(self):
        """初始化数据库连接"""
        # MySQL驱动未安装时优雅降级
        if not MYSQL_AVAILABLE:
            logger.warning("pymysql 未安装，FAQ功能不可用。请执行: uv add pymysql")
            self.connection = None
            self.cursor = None
            return

        try:
            # PyMySQL 连接，cursorclass 指定返回字典格式
            self.connection = pymysql.connect(**self.db_config)
            self.cursor = self.connection.cursor(DictCursor)

            # 创建FAQ表（如果不存在）
            self._create_faq_table()

            logger.info("MySQL database connected successfully")
        except pymysql.Error as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            # 如果连接失败，设置为None，后续操作会优雅降级
            self.connection = None
            self.cursor = None

    def _create_faq_table(self):
        """创建FAQ表"""
        if not self.connection or not self.cursor:
            return

        create_table_query = """
        CREATE TABLE IF NOT EXISTS faq (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question VARCHAR(1000) NOT NULL,
            answer TEXT NOT NULL,
            category VARCHAR(100),
            keywords VARCHAR(500),
            confidence_score FLOAT DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_question (question(255)),
            INDEX idx_category (category)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("FAQ table created successfully")
        except pymysql.Error as e:
            logger.error(f"Failed to create FAQ table: {str(e)}")

    def _init_bm25_index(self):
        """初始化BM25索引"""
        if not self.connection or not self.cursor:
            return

        try:
            # 从数据库加载FAQ数据
            self.cursor.execute("SELECT id, question FROM faq")
            faq_data = self.cursor.fetchall()

            # 记录 index -> db id 的映射（BM25Index 返回的是列表下标，不是主键）
            self.faq_id_map = {i: faq['id'] for i, faq in enumerate(faq_data)}

            # 构建BM25索引
            self.bm25_index = BM25Index()
            self.bm25_index.build_index([faq['question'] for faq in faq_data])

            logger.info(f"BM25 index built with {len(faq_data)} FAQ questions")

        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {str(e)}")
            self.bm25_index = None
            self.faq_id_map = {}

    def search_faq(self, query: str, intent: str = 'unknown', use_cache: bool = True) -> Tuple[Optional[str], bool]:
        """
        FAQ 快通道检索（对齐 EduRag ①快通道：BM25 + Redis + MySQL）。

        流程：
          1. 查 faq:v3:<MD5> 缓存，校验其标准问题与输入一致后返回
          2. jieba 分词 → BM25Okapi 计算原始相关性
          3. softmax 归一化 → argmax → best_score ∈ (0,1]
          4. 分数达到阈值且标准问题仅首尾空白不同 → 取答案、回填缓存
          5. 未命中 → 返回 (None, True)，交给上游 RAG 深通道处理

        Args:
            query: 用户问题
            intent: 意图分类结果（仅用于缓存回写标记）
            use_cache: 是否允许 FAQ 缓存读写

        Returns:
            (answer, need_rag): 答案和是否需要走 RAG
        """
        # v3 隔离旧模糊匹配答案；缓存必须保留匹配问题作为接受依据。
        faq_key = faq_cache_key(query)
        cached = self.cache.get(faq_key) if use_cache else None
        if (isinstance(cached, dict) and cached.get('type') == 'faq'
                and isinstance(cached.get('answer'), str) and cached['answer'].strip()
                and faq_question_matches(query, cached.get('question'))):
            logger.info(f"FAQ Redis 缓存命中，直接返回: {query}")
            return cached.get('answer'), False

        if not self.connection or not self.cursor or self.bm25_index is None:
            return None, True

        # 2. 二级库：jieba 分词 → BM25 → softmax 归一化
        try:
            faq_scores = self.bm25_index.search_normalized(query, k=5)

            if not faq_scores:
                return None, True

            # 3. 取归一化后的最佳匹配
            best_score, best_index = faq_scores[0]
            best_faq_id = self.faq_id_map.get(best_index)
            if best_faq_id is None:
                logger.error(f"FAQ index {best_index} has no corresponding db id")
                return None, True

            # 4. softmax 只表示候选优势，不能证明同义或适用人群相同。
            if best_score < self.bm25_threshold:
                logger.info(f"FAQ 候选softmax相对分数 {best_score:.3f} < 阈值 {self.bm25_threshold}，转入深通道")
                return None, True

            candidate_question = self.bm25_index.documents[best_index]
            if not faq_question_matches(query, candidate_question):
                logger.info("FAQ 候选问题不一致，转入深通道: ID=%s", best_faq_id)
                return None, True

            # 5. 再核对数据库当前问题，防止索引与记录发生偏离。
            self.cursor.execute("SELECT question, answer FROM faq WHERE id = %s", (best_faq_id,))
            result = self.cursor.fetchone()

            if not result or not faq_question_matches(query, result.get('question')):
                return None, True

            answer = result['answer']
            if not isinstance(answer, str) or not answer.strip():
                return None, True

            # 6. 写回 Redis 一级缓存（对齐 EduRag "取答案 + 回填缓存"）
            if use_cache:
                self.cache.set(faq_key, {
                    'type': 'faq',
                    'question': result['question'],
                    'answer': answer,
                    'sources': [],
                    'confidence': best_score,
                    'intent': intent,
                    'strategy': 'faq',
                    'faq_id': best_faq_id,
                    'need_rag': False,
                }, ttl=self.config.FAQ_CACHE_TTL)

            logger.info(
                f"FAQ 快通道命中: ID={best_faq_id}, "
                f"normalized_score={best_score:.3f}, threshold={self.bm25_threshold}"
            )
            return answer, False

        except Exception as e:
            logger.error(f"FAQ search failed: {str(e)}")
            return None, True

    def add_faq(self, question: str, answer: str, category: str = None,
                keywords: str = None) -> bool:
        """添加FAQ"""
        if not self.connection or not self.cursor:
            return False

        try:
            query = """
            INSERT INTO faq (question, answer, category, keywords)
            VALUES (%s, %s, %s, %s)
            """
            self.cursor.execute(query, (question, answer, category, keywords))
            self.connection.commit()

            # 重建BM25索引
            self._init_bm25_index()

            logger.info(f"Added FAQ: {question}")
            return True

        except pymysql.Error as e:
            logger.error(f"Failed to add FAQ: {str(e)}")
            return False

    def batch_add_faq(self, faq_list: List[Dict]) -> int:
        """批量添加FAQ"""
        if not self.connection or not self.cursor:
            return 0

        try:
            query = """
            INSERT INTO faq (question, answer, category, keywords)
            VALUES (%s, %s, %s, %s)
            """

            # 准备批量数据
            data = [(faq['question'], faq['answer'],
                    faq.get('category'), faq.get('keywords'))
                    for faq in faq_list]

            self.cursor.executemany(query, data)
            self.connection.commit()

            # 重建BM25索引
            self._init_bm25_index()

            added_count = len(faq_list)
            logger.info(f"Batch added {added_count} FAQs")
            return added_count

        except pymysql.Error as e:
            logger.error(f"Failed to batch add FAQs: {str(e)}")
            return 0

    def get_faq_by_category(self, category: str) -> List[Dict]:
        """获取指定类别的FAQ"""
        if not self.connection or not self.cursor:
            return []

        try:
            query = "SELECT * FROM faq WHERE category = %s ORDER BY created_at DESC"
            self.cursor.execute(query, (category,))
            return self.cursor.fetchall()

        except pymysql.Error as e:
            logger.error(f"Failed to get FAQ by category: {str(e)}")
            return []

    def update_faq(self, faq_id: int, question: str = None,
                   answer: str = None, category: str = None,
                   keywords: str = None) -> bool:
        """更新FAQ"""
        if not self.connection or not self.cursor:
            return False

        try:
            # 构建更新语句
            updates = []
            params = []

            if question:
                updates.append("question = %s")
                params.append(question)

            if answer:
                updates.append("answer = %s")
                params.append(answer)

            if category:
                updates.append("category = %s")
                params.append(category)

            if keywords:
                updates.append("keywords = %s")
                params.append(keywords)

            if not updates:
                return True  # 无需更新

            query = f"UPDATE faq SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
            params.append(faq_id)

            self.cursor.execute(query, params)
            self.connection.commit()

            # 重建BM25索引
            self._init_bm25_index()

            logger.info(f"Updated FAQ ID {faq_id}")
            return True

        except pymysql.Error as e:
            logger.error(f"Failed to update FAQ: {str(e)}")
            return False

    def delete_faq(self, faq_id: int) -> bool:
        """删除FAQ"""
        if not self.connection or not self.cursor:
            return False

        try:
            query = "DELETE FROM faq WHERE id = %s"
            self.cursor.execute(query, (faq_id,))
            self.connection.commit()

            # 重建BM25索引
            self._init_bm25_index()

            logger.info(f"Deleted FAQ ID {faq_id}")
            return True

        except pymysql.Error as e:
            logger.error(f"Failed to delete FAQ: {str(e)}")
            return False

    def get_faq_stats(self) -> Dict[str, Any]:
        """获取FAQ统计信息"""
        if not self.connection or not self.cursor:
            return {}

        try:
            # 总数
            self.cursor.execute("SELECT COUNT(*) as total FROM faq")
            total = self.cursor.fetchone()['total']

            # 按类别统计
            self.cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM faq
                GROUP BY category
                ORDER BY count DESC
            """)
            categories = self.cursor.fetchall()

            # 最近添加
            self.cursor.execute("SELECT COUNT(*) as recent FROM faq WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)")
            recent = self.cursor.fetchone()['recent']

            return {
                'total': total,
                'categories': categories,
                'recent_7_days': recent,
                'bm25_ready': self.bm25_index is not None
            }

        except pymysql.Error as e:
            logger.error(f"Failed to get FAQ stats: {str(e)}")
            return {}

    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("MySQL connection closed")

class BM25Index:
    """BM25索引实现"""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.documents = []
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.tokenizer = None
        # BM25 超参数：k1 控制词频饱和度，b 控制文档长度归一化程度
        self.bm25_k1 = k1
        self.bm25_b = b

    def build_index(self, documents: List[str]):
        """构建BM25索引"""
        self.documents = documents
        n_docs = len(documents)

        # 计算文档频率和平均文档长度
        doc_freqs = []
        total_length = 0

        for doc in documents:
            tokens = self._tokenize(doc)
            # 统计词频
            freq = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1

            doc_freqs.append(freq)
            total_length += len(tokens)

        self.doc_freqs = doc_freqs
        self.avgdl = total_length / n_docs if n_docs > 0 else 0

        # 计算IDF
        idf = {}
        for freq in doc_freqs:
            for token in freq:
                idf[token] = idf.get(token, 0) + 1

        for token, freq in idf.items():
            idf[token] = np.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))

        self.idf = idf

    def search(self, query: str, k: int = 5) -> List[Tuple[float, int]]:
        """搜索查询（返回原始 BM25 分数）"""
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, (doc, freq) in enumerate(zip(self.documents, self.doc_freqs)):
            score = self._score(query_tokens, freq)
            scores.append((score, i))

        # 排序并返回Top-K
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]

    def search_normalized(self, query: str, k: int = 5) -> List[Tuple[float, int]]:
        """
        搜索查询并返回 **softmax 归一化** 后的分数（对齐 EduRag 快通道）。

        原始 BM25 分数量级取决于语料（当前 ~7.x），阈值难以跨场景通用。
        softmax 将任意量级的分数压缩到 (0,1] 区间：
          - 最高分接近 1.0（远超其他候选时）
          - 多个候选分数相近时，大家均分概率质量
          - 阈值 0.85 表示「最佳匹配显著优于其余候选」

        Returns:
            [(normalized_score, doc_index), ...]，按归一化分数降序
        """
        raw_results = self.search(query, k=k)
        if not raw_results or raw_results[0][0] <= 0:
            return []

        raw_scores = np.array([s for s, _ in raw_results], dtype=np.float64)

        # softmax（数值稳定：减去最大值防止溢出）
        shifted = raw_scores - np.max(raw_scores)
        exp_scores = np.exp(shifted)
        softmax_probs = exp_scores / np.sum(exp_scores)

        # 取 argmax 的归一化分数作为 best_score（对齐 EduRag "argmax → best_score"）
        normalized_results = [
            (float(softmax_probs[i]), idx) for i, (_, idx) in enumerate(raw_results)
        ]
        normalized_results.sort(key=lambda x: x[0], reverse=True)

        return normalized_results

    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if not text:
            return []

        # 使用jieba进行中文分词
        if jieba:
            try:
                tokens = jieba.lcut(text.lower())
                return [token for token in tokens if any(c.isalnum() for c in token)]
            except:
                pass

        # 如果jieba不可用，使用简单分割
        return text.lower().split()

    def _score(self, query_tokens: List[str], doc_freq: Dict) -> float:
        """
        计算单个文档对查询的 BM25 分数

        BM25 公式：score = Σ IDF(t) * [f(t,D)*(k1+1)] / [f(t,D) + k1*(1-b+b*dl/avgdl)]
        - IDF(t)：词 t 的逆文档频率，越稀有的词权重越高
        - f(t,D)：词 t 在文档 D 中的词频
        - dl/avgdl：文档长度 / 平均文档长度，用于长度归一化（长文档不因词多而占优）
        - k1：词频饱和参数，控制词频对分数的贡献上限
        - b：长度归一化强度，b=0 不做长度归一化，b=1 完全归一化

        Args:
            query_tokens: 查询分词结果
            doc_freq: 该文档的词频字典 {token: count}

        Returns:
            BM25 相关性分数（越高越相关）
        """
        score = 0.0

        for token in query_tokens:
            if token in doc_freq and token in self.idf:
                # 词频 f(t,D)
                f = doc_freq[token]
                # 文档长度 dl（该文档总词数）
                dl = sum(doc_freq.values())
                # BM25 公式的分子与分母
                numerator = f * (self.bm25_k1 + 1)
                denominator = f + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * dl / self.avgdl)
                # 累加该词项的贡献：IDF * 词频饱和度
                score += self.idf[token] * numerator / denominator

        return score
