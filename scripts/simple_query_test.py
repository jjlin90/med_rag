"""
Simple Query Test Script
简化版问答测试脚本，不依赖LLM API
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Config
from src.online_service.cache_manager import CacheManager
from src.online_service.intent_classifier import IntentClassifier
from src.online_service.retrieval import Retrieval

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs/simple_query.log')
    ]
)
logger = logging.getLogger(__name__)

class SimpleRAGTester:
    """简化版RAG测试类"""

    def __init__(self, config: Config):
        self.config = config

        # 初始化组件
        self.cache = CacheManager(config)
        self.intent_classifier = IntentClassifier(config)
        self.retrieval = Retrieval(config)

        # 加载模拟数据
        self.load_mock_data()

    def load_mock_data(self):
        """加载模拟数据"""
        # 模拟FAQ数据
        self.mock_faq_data = [
            {
                'id': 1,
                'question': '什么是高血压？',
                'answer': '高血压是指血压持续高于140/90mmHg的疾病。',
                'category': '心血管',
                'confidence': 0.95
            },
            {
                'id': 2,
                'question': '头痛怎么办？',
                'answer': '头痛时可以休息、按摩、服用止痛药，严重时请就医。',
                'category': '神经科',
                'confidence': 0.90
            }
        ]

        # 模拟文档数据
        self.mock_documents = [
            {
                'id': 'doc1_0_parent',
                'content': '高血压是一种常见的慢性疾病，指动脉血压持续升高。正常血压值低于120/80mmHg，而高血压的诊断标准是血压持续高于140/90mmHg。',
                'metadata': {
                    'source': '心血管',
                    'chunk_type': 'parent',
                    'file_path': 'data/raw/cardiology/hypertension.md'
                }
            },
            {
                'id': 'doc1_1_child',
                'content': '高血压的症状包括头痛、头晕、心悸等，但有些患者可能没有明显症状。',
                'metadata': {
                    'source': '心血管',
                    'chunk_type': 'child',
                    'parent_id': 'doc1_0_parent',
                    'file_path': 'data/raw/cardiology/hypertension.md'
                }
            }
        ]

    def test_intent_classification(self):
        """测试意图分类"""
        print("\n=== 测试意图分类 ===")

        test_queries = [
            "什么是人工智能？",
            "头痛应该怎么处理？",
            "如何学习编程？",
            "感冒了吃什么药？"
        ]

        for query in test_queries:
            result = self.intent_classifier.predict(query)
            print(f"查询: {query}")
            print(f"  意图: {result['intent']}")
            print(f"  置信度: {result['confidence']:.3f}")
            print()

    def test_faq_search(self):
        """测试FAQ搜索"""
        print("\n=== 测试FAQ搜索 ===")

        test_queries = [
            "什么是高血压？",
            "头痛怎么办？",
            "糖尿病是什么？"
        ]

        for query in test_queries:
            # 模拟FAQ搜索
            answer = None
            need_llm = True

            for faq in self.mock_faq_data:
                if query in faq['question'] or faq['question'] in query:
                    answer = faq['answer']
                    need_llm = False
                    break

            if answer:
                print(f"查询: {query}")
                print(f"  FAQ命中: ✓")
                print(f"  答案: {answer}")
            else:
                print(f"查询: {query}")
                print(f"  FAQ命中: ✗ (需要RAG)")
            print()

    def test_retrieval(self):
        """测试检索"""
        print("\n=== 测试检索 ===")

        # 模拟向量
        mock_query_vector = [0.1] * 1024

        # 模拟检索结果
        mock_results = [
            {
                'id': 'doc1_0_parent',
                'content': '高血压是一种常见的慢性疾病，指动脉血压持续升高。',
                'metadata': {
                    'source': '心血管',
                    'chunk_type': 'parent'
                },
                'score': 0.85
            },
            {
                'id': 'doc1_1_child',
                'content': '高血压的症状包括头痛、头晕、心悸等。',
                'metadata': {
                    'source': '心血管',
                    'chunk_type': 'child'
                },
                'score': 0.72
            }
        ]

        print(f"模拟查询: 什么是高血压？")
        print(f"检索结果:")
        for i, result in enumerate(mock_results[:2]):
            print(f"  {i+1}. [{result['metadata']['source']}] {result['content'][:50]}...")
            print(f"     分数: {result['score']:.3f}")
        print()

    def test_full_pipeline(self):
        """测试完整流程"""
        print("\n=== 测试完整问答流程 ===")

        test_query = "什么是高血压？"
        start_time = time.time()

        # 1. 意图分类
        intent_result = self.intent_classifier.predict(test_query)
        intent = intent_result['intent']
        confidence = intent_result['confidence']
        print(f"1. 意图分类: {intent} (置信度: {confidence:.3f})")

        # 2. FAQ搜索
        answer = None
        need_llm = True

        for faq in self.mock_faq_data:
            if test_query in faq['question'] or faq['question'] in test_query:
                answer = faq['answer']
                need_llm = False
                break

        if answer:
            print(f"2. FAQ命中: ✓")
            print(f"3. 最终答案: {answer}")
        else:
            print(f"2. FAQ命中: ✗ (需要RAG)")
            print(f"3. 模拟RAG检索...")

            # 模拟检索结果
            mock_results = [
                {
                    'content': '高血压是一种常见的慢性疾病，指动脉血压持续升高。正常血压值低于120/80mmHg，而高血压的诊断标准是血压持续高于140/90mmHg。',
                    'metadata': {'source': '心血管'}
                }
            ]

            # 模拟生成回答
            context = "\n".join([f"[{i+1}] {r['content']}" for i, r in enumerate(mock_results)])
            mock_answer = f"基于医学知识，{context}"

            print(f"4. 生成答案: {mock_answer}")

        # 5. 总耗时
        total_time = time.time() - start_time
        print(f"\n总耗时: {total_time:.3f}秒")

    def run_interactive_test(self):
        """运行交互式测试"""
        print("\n" + "="*50)
        print("简化版RAG系统测试")
        print("="*50)
        print("1. 意图分类测试")
        print("2. FAQ搜索测试")
        print("3. 检索测试")
        print("4. 完整流程测试")
        print("5. 所有测试")
        print("quit - 退出")

        while True:
            choice = input("\n请选择测试项 (1-5, quit): ").strip()

            if choice == 'quit':
                break
            elif choice == '1':
                self.test_intent_classification()
            elif choice == '2':
                self.test_faq_search()
            elif choice == '3':
                self.test_retrieval()
            elif choice == '4':
                self.test_full_pipeline()
            elif choice == '5':
                self.test_intent_classification()
                self.test_faq_search()
                self.test_retrieval()
                self.test_full_pipeline()
            else:
                print("无效选择，请重新输入")

def main():
    """主函数"""
    try:
        print("初始化配置...")
        config = Config()

        # 初始化测试器
        tester = SimpleRAGTester(config)

        # 检查系统状态
        print("\n系统状态:")
        print(f"意图分类器: ✓")
        print(f"缓存: {'✓' if tester.cache.is_connected() else '✗'}")
        print(f"检索器: ✓ (模拟数据)")

        # 运行测试
        tester.run_interactive_test()

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        logger.error(f"测试脚本失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()