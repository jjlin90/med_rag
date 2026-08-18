"""
启动在线问答 API 服务

为什么不能直接 `python src/online_service/main_api.py`：
    main_api.py 内部使用相对导入（from ..config.settings import Config），
    直接以脚本方式运行会失去包上下文，报
    "attempted relative import with no known parent package"。
本脚本先把项目根目录加入 sys.path，再以「包内模块」方式导入并启动，等效于
    python -m src.online_service.main_api
但更直观、且能复用统一的启动参数。

用法（在项目根目录）：
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8000 --debug
"""

import os

# 双保险：在任何 torch / numpy / FlagEmbedding 导入之前设置，避免 Windows 上
# Intel OpenMP(libiomp5md.dll) 与微软 OpenMP(vcomp140.dll) 重复加载导致 0xC0000005 崩溃。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，使 src 包可被导入
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Config  # noqa: E402
from src.online_service.main_api import RAGWebAPI  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="启动医疗 RAG 问答 API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8005, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="开启热重载调试模式")
    args = parser.parse_args()

    logger.info("初始化配置...")
    config = Config()
    rag_web = RAGWebAPI(config)
    logger.info(f"启动 API: http://{args.host}:{args.port}  (文档: /docs)")
    rag_web.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
