"""
Common Utilities
通用工具函数
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

def compute_md5(text: str) -> str:
    """计算文本的MD5哈希值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def save_json(data: Any, file_path: Path, indent: int = 2, ensure_ascii: bool = False):
    """保存数据为JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

def load_json(file_path: Path) -> Any:
    """从JSON文件加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """清理文本"""
    if not text:
        return ""

    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 删除控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # 压缩连续空白
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_file_info(file_path: Path) -> Dict[str, Any]:
    """提取文件信息"""
    stat = file_path.stat()
    return {
        'name': file_path.name,
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'created': stat.st_ctime,
        'extension': file_path.suffix.lower()
    }

def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_number(num: int, suffix: str = "") -> str:
    """格式化数字"""
    for unit in ['', 'K', 'M', 'G', 'T']:
        if abs(num) < 1000:
            return f"{num}{unit}{suffix}"
        num /= 1000
    return f"{num:.1f}P{suffix}"

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def validate_file_path(file_path: Path, extensions: Optional[List[str]] = None) -> bool:
    """验证文件路径"""
    if not file_path.exists():
        return False

    if file_path.is_dir():
        return False

    if extensions:
        return file_path.suffix.lower() in extensions

    return True

def get_file_size_mb(file_path: Path) -> float:
    """获取文件大小（MB）"""
    return file_path.stat().st_size / (1024 * 1024)

def create_backup(file_path: Path, suffix: str = ".bak") -> Path:
    """创建文件备份"""
    backup_path = file_path.with_suffix(file_path.suffix + suffix)
    backup_path.write_text(file_path.read_text(encoding='utf-8'))
    return backup_path

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法"""
    try:
        return a / b if b != 0 else default
    except:
        return default

def normalize_string(text: str) -> str:
    """标准化字符串"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    """将列表分割成小块"""
    return [lst[i:i + size] for i in range(0, len(lst), size)]

def merge_dicts(dict1: Dict, dict2: Dict, overwrite: bool = True) -> Dict:
    """合并字典"""
    result = dict1.copy()
    for key, value in dict2.items():
        if overwrite or key not in result:
            result[key] = value
    return result

def is_chinese(text: str) -> bool:
    """检查是否包含中文"""
    return bool(re.search(r'[一-鿿]', text))

def is_english(text: str) -> bool:
    """检查是否包含英文"""
    return bool(re.search(r'[a-zA-Z]', text))

def count_tokens(text: str) -> int:
    """简单估算token数量"""
    if is_chinese(text):
        return len(text) // 1.3
    else:
        return len(text.split()) * 0.75

def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
                    continue
            raise last_exception
        return wrapper
    return decorator

class Timer:
    """计时器"""
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.name} 耗时: {format_time(elapsed)}")

    def elapsed(self) -> float:
        """获取已用时间"""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time