"""
Offline Ingestion Script
一键执行完整离线入库流程
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# 必须在任何 CUDA 初始化之前设置：
# expandable_segments 让 PyTorch 按需扩展显存段，大幅减少「文本长短不一」
# 造成的显存碎片化，避免显存被顶到上限后分配器空转（thrashing）导致的骤降
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Config
from src.offline_pipeline.document_loader import DocumentLoader
from src.offline_pipeline.data_cleaner import DataCleaner
from src.offline_pipeline.chunk_splitter import ChunkSplitter
from src.offline_pipeline.embedding_provider import BGEEmbeddingProvider
from src.offline_pipeline.milvus_store import MilvusStore

# 配置日志
(project_root / 'logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs/offline_ingest.log')
    ])
logger = logging.getLogger(__name__)


def setup_directories(config: Config):
    """设置必要的目录"""
    config.init_dirs()

    # 创建日志目录
    (project_root / 'logs').mkdir(exist_ok=True)


def validate_config(config: Config, data_dir: Optional[Path] = None) -> bool:
    """验证配置"""
    # 入库读取清洗文档或用户指定目录；HTML 原始目录只供抽取脚本使用。
    source_dir = data_dir if data_dir is not None else config.CLEAN_MD_DIR
    if not source_dir.is_dir():
        logger.error(f"文档目录不存在或不是目录: {source_dir}")
        return False

    # 检查模型目录
    model_dir = project_root / "src/models"
    if not model_dir.exists():
        logger.error(f"模型目录不存在: {model_dir}")
        return False

    # 检查必要的模型
    required_models = ['bge-m3']
    for model_name in required_models:
        model_path = model_dir / model_name
        if not model_path.exists():
            logger.error(f"模型目录不存在: {model_path}")
            return False

    return True


def process_documents(config: Config, data_dir: Optional[Path] = None):
    """处理文档并写入 Milvus。

    Args:
        config: 全局配置
        data_dir: 待处理的本地文档目录。默认使用 config.CLEAN_MD_DIR。
    """
    data_dir = data_dir or config.CLEAN_MD_DIR
    logger.info(f"开始离线知识入库流程，数据源: {data_dir}")

    # 1. 初始化组件
    logger.info("初始化组件...")
    loader = DocumentLoader(config)
    cleaner = DataCleaner(config)
    splitter = ChunkSplitter(config)
    embedder = BGEEmbeddingProvider(config)
    milvus_store = MilvusStore(config)

    # 2. 创建或加载Milvus集合
    logger.info("创建/加载Milvus集合...")
    collection = milvus_store.create_or_load_collection(config.EMBED_DIM)

    # 3. 加载文档
    logger.info("加载文档...")
    start_time = time.time()
    documents = loader.load_documents_from_directory(data_dir)
    load_time = time.time() - start_time
    logger.info(f"加载了 {len(documents)} 个文档，耗时 {load_time:.2f}秒")

    if not documents:
        logger.error("没有找到任何文档，请检查数据目录")
        raise ValueError(f'No supported documents found in {data_dir}')

    # 4. 清洗文档
    logger.info("清洗文档...")
    start_time = time.time()
    cleaned_docs = cleaner.clean_documents(documents)
    clean_time = time.time() - start_time
    logger.info(f"清洗完成，耗时 {clean_time:.2f}秒")

    # 5. 提取元数据
    logger.info("提取元数据...")
    enhanced_docs = cleaner.extract_metadata(cleaned_docs)

    # 6. 分块
    logger.info("文档分块...")
    start_time = time.time()
    chunks = splitter.split_documents(enhanced_docs)
    split_time = time.time() - start_time
    logger.info(f"分块完成，生成 {len(chunks)} 个块，耗时 {split_time:.2f}秒")

    # 7. 保存分块结果
    logger.info("保存分块结果...")
    splitter.save_chunks(chunks, config.CHUNK_SAVE_PATH)

    # 8. 批量生成向量
    logger.info("生成向量...")
    start_time = time.time()
    # batch_size=64：配合 expandable_segments 防止显存顶格导致的空转降速
    # （128 在持续跑时会因显存碎片顶到 8G 上限而严重变慢）
    vectorized_chunks = embedder.batch_process(chunks, batch_size=64)
    vector_time = time.time() - start_time
    logger.info(f"向量生成完成，耗时 {vector_time:.2f}秒")

    # 9. 保存向量数据到 JSON 缓存（可选，默认关闭）
    # 该缓存全项目无读取方，且 20816×1024 维写 JSON 约 380MB、慢且占内存。
    # 由 config.SAVE_EMBEDDING_CACHE 控制，默认 False 直接跳过，避免拖慢/崩溃。
    if getattr(config, 'SAVE_EMBEDDING_CACHE', False):
        logger.info("保存向量数据到 JSON 缓存...")
        # 提取向量和文本
        texts = [chunk['content'] for chunk in vectorized_chunks]
        dense_embeddings = [
            chunk['dense_embedding'] for chunk in vectorized_chunks
        ]
        sparse_embeddings = [
            chunk['sparse_embedding'] for chunk in vectorized_chunks
        ]
        embedder.save_embeddings(texts, dense_embeddings, sparse_embeddings,
                                 config.EMBED_CACHE_DIR / "embeddings.json")
    else:
        logger.info("跳过向量 JSON 缓存（SAVE_EMBEDDING_CACHE=False），直接进入 Milvus 入库")

    # 10. 批量插入Milvus
    logger.info("批量插入Milvus...")
    start_time = time.time()
    total_added = milvus_store.add_documents(vectorized_chunks,
                                             batch_size=1000)
    insert_time = time.time() - start_time
    logger.info(f"Milvus插入完成，添加 {total_added} 个文档，耗时 {insert_time:.2f}秒")

    # 11. 获取统计信息
    stats = milvus_store.get_collection_info()
    logger.info(f"入库完成！集合信息: {stats}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="医疗 RAG 离线知识入库")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="待处理的本地文档目录（默认使用 config.CLEAN_MD_DIR）",
    )
    args = parser.parse_args()

    try:
        # 1. 初始化配置
        logger.info("初始化配置...")
        config = Config()

        # 2. 设置目录
        setup_directories(config)

        # 3. 验证配置
        if not validate_config(config, data_dir=args.data_dir):
            logger.error("配置验证失败，退出")
            raise SystemExit(1)

        # 4. 处理文档
        process_documents(config, data_dir=args.data_dir)

        logger.info("离线知识入库流程完成！")

    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"离线入库流程失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()
