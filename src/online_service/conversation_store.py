"""
MySQL 会话历史存储模块（会话表 + 会话 ID 管理）

严格对齐 EduRag 的 IntegratedQASystem 会话能力：
- init_conversation_table：建 conversations 表（session_id / question / answer / timestamp）
- update_session_history：写入本轮问答，并裁剪到最近 N 轮
- get_session_history / _fetch_recent_history：读取最近 N 轮（时间正序）
- clear_session_history：清空指定会话

区别在于：EduRag 用自研 MySQLClient，这里改用项目统一的 PyMySQL + settings 配置；
并新增 new_session_id() 静态方法供前端/路由生成会话标识。
"""

import logging
import uuid
from typing import List, Dict, Optional

try:
    import pymysql
    from pymysql.cursors import DictCursor
    MYSQL_AVAILABLE = True
except ImportError:
    pymysql = None
    DictCursor = None
    MYSQL_AVAILABLE = False

from ..config.settings import Config

logger = logging.getLogger(__name__)

# 与 EduRag 一致：每个会话仅保留最近 5 轮，避免上下文无限膨胀
MAX_HISTORY_TURNS = 5


class ConversationStore:
    """MySQL 会话历史存储（对齐 EduRag 的 conversations 表机制）"""

    def __init__(self, config: Config):
        self.config = config
        self.db_config = {
            'host': config.MYSQL_HOST,
            'port': config.MYSQL_PORT,
            'user': config.MYSQL_USER,
            'password': config.MYSQL_PASSWORD,
            'database': config.MYSQL_DATABASE,
            'charset': 'utf8mb4',
        }

        self.connection = None
        self.cursor = None

        self._init_db()
        if self.connection:
            self.init_conversation_table()

    def _init_db(self):
        """初始化 MySQL 连接（与 FAQSearch 同套配置来源）"""
        if not MYSQL_AVAILABLE:
            logger.warning("pymysql 未安装，会话历史功能不可用。请执行: uv add pymysql")
            self.connection = None
            self.cursor = None
            return

        try:
            self.connection = pymysql.connect(**self.db_config)
            self.cursor = self.connection.cursor(DictCursor)
            logger.info("ConversationStore MySQL 连接成功")
        except pymysql.Error as e:
            logger.error(f"ConversationStore MySQL 连接失败: {str(e)}")
            self.connection = None
            self.cursor = None

    def init_conversation_table(self):
        """初始化 conversations 表，用于存储对话历史（对齐 EduRag）"""
        if not self.connection or not self.cursor:
            return

        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer LONGTEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            self.connection.commit()
            logger.info("会话历史表 conversations 初始化成功")
        except pymysql.Error as e:
            logger.error(f"初始化对话历史表失败: {str(e)}")

    def _fetch_recent_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取最近 5 轮对话历史（时间正序返回）"""
        if not self.connection or not self.cursor:
            return []

        try:
            self.cursor.execute("""
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC, id DESC
                LIMIT %s
            """, (session_id, MAX_HISTORY_TURNS))

            rows = self.cursor.fetchall()
            history = [{"question": r["question"], "answer": r["answer"]} for r in rows]
            # 反转：按时间正序（旧→新）返回，方便注入 prompt
            return history[::-1]
        except pymysql.Error as e:
            logger.error(f"获取对话历史失败: {str(e)}")
            return []

    def update_session_history(self, session_id: str, question: str, answer: str) -> List[Dict[str, str]]:
        """
        写入本轮问答到 MySQL，并裁剪到最近 MAX_HISTORY_TURNS 轮。
        返回更新后的最近历史（时间正序），供调用方使用。
        """
        if not self.connection or not self.cursor:
            return []

        # MySQL LONGTEXT 上限约 4GB，这里仅做基础保护，避免异常超长内容
        answer = answer or ""

        try:
            # 插入本轮问答
            self.cursor.execute("""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))

            # 读取更新后的最近历史
            history = self._fetch_recent_history(session_id)

            # 删除超出 MAX_HISTORY_TURNS 轮的旧记录
            self.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s
                  AND id NOT IN (
                    SELECT id FROM (
                        SELECT id FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC, id DESC
                        LIMIT %s
                    ) AS keep
                  )
            """, (session_id, session_id, MAX_HISTORY_TURNS))

            self.connection.commit()
            logger.info(f"会话 {session_id} 历史更新成功（保留最近 {MAX_HISTORY_TURNS} 轮）")
            return history
        except pymysql.Error as e:
            logger.error(f"更新会话历史失败: {str(e)}")
            try:
                self.connection.rollback()
            except Exception:
                pass
            return []

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """从 MySQL 获取会话历史"""
        return self._fetch_recent_history(session_id)

    def clear_session_history(self, session_id: str) -> bool:
        """清除指定会话历史"""
        if not self.connection or not self.cursor:
            return False

        try:
            self.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s
            """, (session_id,))
            self.connection.commit()
            logger.info(f"会话 {session_id} 历史已清除")
            return True
        except pymysql.Error as e:
            logger.error(f"清除会话历史失败: {str(e)}")
            try:
                self.connection.rollback()
            except Exception:
                pass
            return False

    @staticmethod
    def new_session_id() -> str:
        """生成新的会话 ID（UUID4，与 EduRag CLI 用法一致）"""
        return str(uuid.uuid4())

    def is_available(self) -> bool:
        """MySQL 会话存储是否可用"""
        return self.connection is not None
