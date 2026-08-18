"""
Simple Offline Ingestion Script
简化版离线入库脚本，用于测试基础功能
"""

import logging
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Config
from src.offline_pipeline.document_loader import DocumentLoader
from src.offline_pipeline.data_cleaner import DataCleaner
from src.offline_pipeline.chunk_splitter import ChunkSplitter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs/simple_ingest.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_directories(config: Config):
    """设置必要的目录"""
    config.init_dirs()
    (project_root / 'logs').mkdir(exist_ok=True)

def validate_config(config: Config) -> bool:
    """验证配置"""
    # 检查原始数据目录
    if not config.RAW_MSD_DIR.exists():
        logger.error(f"原始数据目录不存在: {config.RAW_MSD_DIR}")
        return False

    logger.info(f"原始数据目录: {config.RAW_MSD_DIR}")

    # 列出支持的文件
    loader = DocumentLoader(config)
    supported_extensions = list(loader.supported_extensions)
    logger.info(f"支持的文件格式: {supported_extensions}")

    # 检查文件数量
    files = []
    for ext in supported_extensions:
        files.extend(list(config.RAW_MSD_DIR.rglob(f"*{ext}")))

    logger.info(f"找到 {len(files)} 个文档文件")
    return len(files) > 0

def process_documents(config: Config):
    """处理文档（简化版，不使用向量模型）"""
    logger.info("开始简化版离线知识入库流程...")

    # 1. 初始化组件
    logger.info("初始化组件...")
    loader = DocumentLoader(config)
    cleaner = DataCleaner(config)
    splitter = ChunkSplitter(config)

    # 2. 加载文档
    logger.info("加载文档...")
    start_time = time.time()
    documents = loader.load_documents_from_directory(config.RAW_MSD_DIR)
    load_time = time.time() - start_time
    logger.info(f"加载了 {len(documents)} 个文档，耗时 {load_time:.2f}秒")

    if not documents:
        logger.error("没有找到任何文档，请检查数据目录")
        return

    # 3. 显示文档信息
    logger.info("文档信息:")
    for i, doc in enumerate(documents[:5]):  # 只显示前5个
        logger.info(f"  文档{i+1}: {doc.metadata.get('file_path', 'unknown')}, "
                   f"长度: {len(doc.page_content)} 字符")
    if len(documents) > 5:
        logger.info(f"  ... 还有 {len(documents)-5} 个文档")

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

    # 7. 统计信息
    parent_chunks = [c for c in chunks if c.chunk_type == 'parent']
    child_chunks = [c for c in chunks if c.chunk_type == 'child']

    logger.info("分块统计:")
    logger.info(f"  父块数量: {len(parent_chunks)}")
    logger.info(f"  子块数量: {len(child_chunks)}")
    logger.info(f"  平均父块长度: {sum(len(c.content) for c in parent_chunks) / len(parent_chunks) if parent_chunks else 0:.1f} 字符")
    logger.info(f"  平均子块长度: {sum(len(c.content) for c in child_chunks) / len(child_chunks) if child_chunks else 0:.1f} 字符")

    # 8. 保存分块结果
    logger.info("保存分块结果...")
    splitter.save_chunks(chunks, config.CHUNK_SAVE_PATH)
    logger.info(f"分块结果已保存到: {config.CHUNK_SAVE_PATH}")

    # 9. 模拟向量生成（实际项目中使用真实的向量模型）
    logger.info("模拟向量生成...")
    start_time = time.time()

    # 这里只是模拟，实际项目中应该使用BGE-M3
    simulated_vectors = []
    for i, chunk in enumerate(chunks):
        # 生成模拟向量（实际项目中使用BGE-M3）
        simulated_vectors.append({
            'id': chunk.id,
            'content': chunk.content,
            'metadata': chunk.metadata,
            'dense_embedding': [0.1] * 1024,  # 模拟1024维向量
            'sparse_embedding': {str(i % 1000): 0.5}  # 模拟稀疏向量
        })

        if i % 100 == 0:
            logger.info(f"  已处理 {i} 个分块...")

    vector_time = time.time() - start_time
    logger.info(f"向量模拟完成，耗时 {vector_time:.2f}秒")

    # 10. 保存模拟向量数据
    import json
    vector_data_path = config.EMBED_CACHE_DIR / "simulated_embeddings.json"
    config.EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    vector_data = {
        'texts': [v['content'] for v in simulated_vectors],
        'dense_embeddings': [v['dense_embedding'] for v in simulated_vectors],
        'sparse_embeddings': [v['sparse_embedding'] for v in simulated_vectors],
        'embed_dim': 1024,
        'model_name': 'simulated'
    }

    with open(vector_data_path, 'w', encoding='utf-8') as f:
        json.dump(vector_data, f, ensure_ascii=False, indent=2)

    logger.info(f"模拟向量数据已保存到: {vector_data_path}")

    logger.info("简化版入库流程完成！")
    logger.info("\n注意：这是简化版本，没有使用真实的向量模型和Milvus数据库。")
    logger.info("要完成完整的入库流程，请确保：")
    logger.info("1. 下载了BGE-M3模型")
    logger.info("2. 安装并启动了Milvus服务")
    logger.info("3. 使用完整的 run_offline_ingest.py")

def main():
    """主函数"""
    try:
        # 1. 初始化配置
        logger.info("初始化配置...")
        config = Config()

        # 2. 设置目录
        setup_directories(config)

        # 3. 验证配置
        if not validate_config(config):
            logger.error("配置验证失败，退出")
            return

        # 4. 处理文档
        process_documents(config)

    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"入库流程失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()