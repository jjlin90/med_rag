"""
Cache Manager Module
Redis缓存工具
"""

import redis
import json
import logging
import hashlib
import re
from typing import Any, Optional, Dict
import time

from ..config.settings import Config

logger = logging.getLogger(__name__)

def normalize_query(text: str) -> str:
    """把用户问题规范化为稳定的缓存键。

    仅去除首尾空白，保留数字、标点、大小写和内部空格，避免合并不同语义。
    """
    if not text:
        return ""
    # Preserve units, decimal points, signs and token boundaries.
    return text.strip()


def _digest(text: str) -> str:
    """对规范化后的文本求稳定摘要（跨进程/重启一致）。"""
    return hashlib.md5(normalize_query(text).encode("utf-8")).hexdigest()


def query_cache_key(text: str, source_filter=None, strategy=None, history=None) -> str:
    """问题、来源、策略和历史共同构成稳定的 v2 缓存键。"""
    payload = json.dumps([normalize_query(text), source_filter, strategy, history],
                         ensure_ascii=False, sort_keys=True)
    return f"query:v2:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def faq_cache_key(text: str) -> str:
    """FAQ 一级缓存键：MySQL 命中后写入，下次查询优先查这里。"""
    return f"faq:v2:{_digest(text)}"

class CacheManager:
    """Redis缓存管理器"""

    def __init__(self, config: Config):
        self.config = config

        # Redis连接配置（从 config 读取，便于统一管理与覆盖）
        self.host = getattr(config, "REDIS_HOST", "localhost")
        self.port = getattr(config, "REDIS_PORT", 6379)
        self.password = getattr(config, "REDIS_PASSWORD", None)
        self.db = getattr(config, "REDIS_DB", 0)
        self.decode_responses = True

        # 连接池
        self.pool = None
        self.client = None

        # 缓存配置
        self.default_ttl = 3600  # 默认1小时过期

        # 初始化连接
        self._init_connection()

    def _init_connection(self):
        """初始化Redis连接"""
        try:
            # 创建连接池
            # 注意：lib_name/lib_version 设为空字符串，可跳过连接时自动发送的
            # "CLIENT SETINFO LIB-NAME/LIB-VER" 命令。redis-py 5.3.x 在 Redis 不可达时，
            # 该命令会触发 "on_connect -> send_command -> 重连 -> on_connect" 的无限递归，
            # 最终抛 RecursionError 导致整个程序崩溃（原 except redis.ConnectionError 抓不到）。
            self.pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=self.decode_responses,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=False,
                health_check_interval=0,
                lib_name="",
                lib_version="",
            )

            # 创建Redis客户端
            self.client = redis.Redis(connection_pool=self.pool)

            # 测试连接
            self.client.ping()
            logger.info("Connected to Redis successfully")

        except Exception as e:
            logger.warning(
                f"Redis 不可用，已降级为无缓存模式（client=None）：{type(e).__name__}: {e}"
            )
            # 如果Redis不可用，设置为None，后续操作会优雅降级
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.client:
            return None

        try:
            value = self.client.get(key)
            if value:
                # 尝试解析JSON
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # 如果不是JSON，返回原始值
                    return value
            return None

        except redis.RedisError as e:
            logger.error(f"Redis GET error for key {key}: {str(e)}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        if not self.client:
            return False

        # 序列化值
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            else:
                value = str(value)
        except Exception as e:
            logger.error(f"Failed to serialize value for key {key}: {str(e)}")
            return False

        try:
            # 设置过期时间
            if ttl is None:
                ttl = self.default_ttl

            if ttl > 0:
                result = self.client.setex(key, ttl, value)
            else:
                result = self.client.set(key, value)

            return bool(result)

        except redis.RedisError as e:
            logger.error(f"Redis SET error for key {key}: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """删除缓存"""
        if not self.client:
            return False

        try:
            result = self.client.delete(key)
            return bool(result)

        except redis.RedisError as e:
            logger.error(f"Redis DELETE error for key {key}: {str(e)}")
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self.client:
            return False

        try:
            return bool(self.client.exists(key))

        except redis.RedisError as e:
            logger.error(f"Redis EXISTS error for key {key}: {str(e)}")
            return False

    def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        if not self.client:
            return False

        try:
            return bool(self.client.expire(key, ttl))

        except redis.RedisError as e:
            logger.error(f"Redis EXPIRE error for key {key}: {str(e)}")
            return False

    def ttl(self, key: str) -> int:
        """获取剩余过期时间"""
        if not self.client:
            return -2

        try:
            return self.client.ttl(key)

        except redis.RedisError as e:
            logger.error(f"Redis TTL error for key {key}: {str(e)}")
            return -2

    def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """递增计数器"""
        if not self.client:
            return None

        try:
            if amount == 1:
                return self.client.incr(key)
            else:
                return self.client.incrby(key, amount)

        except redis.RedisError as e:
            logger.error(f"Redis INCR error for key {key}: {str(e)}")
            return None

    def decr(self, key: str, amount: int = 1) -> Optional[int]:
        """递减计数器"""
        if not self.client:
            return None

        try:
            if amount == 1:
                return self.client.decr(key)
            else:
                return self.client.decrby(key, amount)

        except redis.RedisError as e:
            logger.error(f"Redis DECR error for key {key}: {str(e)}")
            return None

    def hget(self, key: str, field: str) -> Optional[Any]:
        """获取哈希字段"""
        if not self.client:
            return None

        try:
            value = self.client.hget(key, field)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None

        except redis.RedisError as e:
            logger.error(f"Redis HGET error for key {key}, field {field}: {str(e)}")
            return None

    def hset(self, key: str, field: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置哈希字段"""
        if not self.client:
            return False

        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            else:
                value = str(value)

            result = self.client.hset(key, field, value)

            # 设置过期时间
            if ttl and ttl > 0:
                self.client.expire(key, ttl)

            return bool(result)

        except redis.RedisError as e:
            logger.error(f"Redis HSET error for key {key}, field {field}: {str(e)}")
            return False

    def hgetall(self, key: str) -> Dict[str, Any]:
        """获取哈希所有字段"""
        if not self.client:
            return {}

        try:
            data = self.client.hgetall(key)
            result = {}
            for field, value in data.items():
                try:
                    result[field] = json.loads(value)
                except json.JSONDecodeError:
                    result[field] = value
            return result

        except redis.RedisError as e:
            logger.error(f"Redis HGETALL error for key {key}: {str(e)}")
            return {}

    def hmget(self, key: str, fields: list) -> list:
        """批量获取哈希字段"""
        if not self.client:
            return [None] * len(fields)

        try:
            data = self.client.hmget(key, fields)
            result = []
            for value in data:
                if value:
                    try:
                        result.append(json.loads(value))
                    except json.JSONDecodeError:
                        result.append(value)
                else:
                    result.append(None)
            return result

        except redis.RedisError as e:
            logger.error(f"Redis HMGET error for key {key}, fields {fields}: {str(e)}")
            return [None] * len(fields)

    def lpush(self, key: str, *values) -> Optional[int]:
        """列表左推"""
        if not self.client:
            return None

        try:
            # 序列化values
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value, ensure_ascii=False))
                else:
                    serialized_values.append(str(value))

            return self.client.lpush(key, *serialized_values)

        except redis.RedisError as e:
            logger.error(f"Redis LPUSH error for key {key}: {str(e)}")
            return None

    def rpop(self, key: str) -> Optional[Any]:
        """列表右弹"""
        if not self.client:
            return None

        try:
            value = self.client.rpop(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None

        except redis.RedisError as e:
            logger.error(f"Redis RPOP error for key {key}: {str(e)}")
            return None

    def llen(self, key: str) -> Optional[int]:
        """获取列表长度"""
        if not self.client:
            return None

        try:
            return self.client.llen(key)

        except redis.RedisError as e:
            logger.error(f"Redis LLEN error for key {key}: {str(e)}")
            return None

    def set_with_prefix(self, prefix: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """使用前缀设置缓存"""
        full_key = f"{prefix}:{key}"
        return self.set(full_key, value, ttl)

    def get_with_prefix(self, prefix: str, key: str) -> Optional[Any]:
        """使用前缀获取缓存"""
        full_key = f"{prefix}:{key}"
        return self.get(full_key)

    def delete_with_prefix(self, prefix: str, pattern: str = "*") -> int:
        """删除前缀匹配的缓存"""
        if not self.client:
            return 0

        try:
            keys = self.client.keys(f"{prefix}:{pattern}")
            if keys:
                return self.client.delete(*keys)
            return 0

        except redis.RedisError as e:
            logger.error(f"Redis delete with prefix error for {prefix}: {str(e)}")
            return 0

    def set_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """批量设置缓存"""
        if not self.client:
            return False

        try:
            # 使用pipeline批量操作
            pipe = self.client.pipeline()
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    pipe.set(key, json.dumps(value, ensure_ascii=False))
                else:
                    pipe.set(key, str(value))
                if ttl and ttl > 0:
                    pipe.expire(key, ttl)

            pipe.execute()
            return True

        except redis.RedisError as e:
            logger.error(f"Redis SET_MANY error: {str(e)}")
            return False

    def get_many(self, keys: list) -> Dict[str, Any]:
        """批量获取缓存"""
        if not self.client:
            return {}

        try:
            values = self.client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value
                else:
                    result[key] = None
            return result

        except redis.RedisError as e:
            logger.error(f"Redis GET_MANY error: {str(e)}")
            return {}

    def clear_all(self) -> bool:
        """清空所有缓存"""
        if not self.client:
            return False

        try:
            self.client.flushdb()
            return True

        except redis.RedisError as e:
            logger.error(f"Redis FLUSHDB error: {str(e)}")
            return False

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        if not self.client:
            return {}

        try:
            info = self.client.info('memory')
            return {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'used_memory_peak': info.get('used_memory_peak', 0),
                'used_memory_peak_human': info.get('used_memory_peak_human', '0B'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0)
            }

        except redis.RedisError as e:
            logger.error(f"Redis memory info error: {str(e)}")
            return {}

    def is_connected(self) -> bool:
        """检查Redis连接状态"""
        if not self.client:
            return False

        try:
            self.client.ping()
            return True
        except:
            return False

    def close(self):
        """关闭连接"""
        if self.pool:
            self.pool.disconnect()
            logger.info("Redis connection closed")
