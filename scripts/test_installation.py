"""
Test Installation Script
验证关键依赖是否正确安装
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试关键模块导入"""
    print("Testing module imports...")

    try:
        # 配置
        from src.config.settings import Config
        print("✓ Config imported")

        # 基础工具
        from src.utils.logger import setup_logger
        print("✓ Logger imported")

        # 离线流水线
        from src.offline_pipeline.document_loader import DocumentLoader
        print("✓ DocumentLoader imported")

        from src.offline_pipeline.data_cleaner import DataCleaner
        print("✓ DataCleaner imported")

        from src.offline_pipeline.chunk_splitter import ChunkSplitter
        print("✓ ChunkSplitter imported")

        # 在线服务
        from src.online_service.cache_manager import CacheManager
        print("✓ CacheManager imported")

        from src.online_service.intent_classifier import IntentClassifier
        print("✓ IntentClassifier imported")

        from src.online_service.llm_generator import LLMGenerator
        print("✓ LLMGenerator imported")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_dependencies():
    """测试关键依赖包"""
    print("\nTesting dependencies...")

    deps = [
        ('numpy', 'np'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('pymilvus', 'milvus'),
        ('redis', 'redis'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
    ]

    all_ok = True
    for module_name, alias in deps:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError:
            print(f"✗ {module_name} - not installed")
            all_ok = False

    return all_ok

def test_models():
    """测试模型文件"""
    print("\nTesting model files...")

    model_dir = project_root / "src" / "models"
    if not model_dir.exists():
        print("✗ Model directory does not exist")
        return False

    required_models = [
        'bge-m3',
        'bert-base-chinese',
        'bge-reranker-large'
    ]

    all_ok = True
    for model_name in required_models:
        model_path = model_dir / model_name
        if model_path.exists():
            print(f"✓ {model_name} directory exists")
        else:
            print(f"✗ {model_name} directory missing")
            all_ok = False

    return all_ok

def test_directories():
    """测试必要目录"""
    print("\nTesting directories...")

    required_dirs = [
        'data/raw',
        'data/clean_md',
        'data/split_docs',
        'data/test_query',
        'cache/milvus_local',
        'cache/embedding_cache',
        'logs'
    ]

    all_ok = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}")
        else:
            print(f"✗ {dir_name} - creating...")
            dir_path.mkdir(parents=True, exist_ok=True)
            all_ok = False

    return all_ok

def main():
    """主函数"""
    print("=" * 50)
    print("Medical RAG System - Installation Test")
    print("=" * 50)

    results = []

    # 测试模块导入
    results.append(("Module Imports", test_imports()))

    # 测试依赖
    results.append(("Dependencies", test_dependencies()))

    # 测试模型
    results.append(("Model Files", test_models()))

    # 测试目录
    results.append(("Directories", test_directories()))

    # 汇总结果
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! Ready to use.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure API keys")
        print("2. Run: python scripts/run_offline_ingest.py")
        print("3. Run: python scripts/test_query_pipeline.py")
    else:
        print("✗ Some tests failed. Please check the output above.")

    print("=" * 50)

if __name__ == "__main__":
    main()