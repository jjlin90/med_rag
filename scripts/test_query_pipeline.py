"""
Query Pipeline Test Script
问答链路本地调试脚本
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Config
from src.online_service.cache_manager import CacheManager, query_cache_key
from src.online_service.faq_search import FAQSearch
from src.online_service.intent_classifier import IntentClassifier
from src.online_service.query_augmenter import QueryAugmenter
from src.online_service.retrieval import Retrieval
from src.online_service.reranker import Reranker
from src.online_service.llm_generator import LLMGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs/query_test.log')
    ]
)
logger = logging.getLogger(__name__)

class RAGTester:
    """RAG测试类"""

    def __init__(self, config: Config):
        self.config = config

        # 初始化组件
        self.cache = CacheManager(config)
        self.faq_search = FAQSearch(config, self.cache)
        self.intent_classifier = IntentClassifier(config)
        self.llm_generator = LLMGenerator(config)
        self.query_augmenter = QueryAugmenter(config, self.llm_generator)
        self.retrieval = Retrieval(config)
        self.reranker = Reranker(config)

        # 测试计数器
        self.test_count = 0
        self.success_count = 0

    def test_single_query(self, query: str, strategy: str = "direct",
                          source_filter: Optional[str] = None) -> Dict[str, Any]:
        """测试单个查询"""
        start_time = time.time()
        result = {
            'query': query,
            'strategy': strategy,
            'source_filter': source_filter,
            'steps': {},
            'total_time': 0,
            'success': False
        }

        try:
            # 1. 缓存检查（稳定键，跨重启可命中）
            cache_key = query_cache_key(query)
            if self.cache.is_connected():
                cached = self.cache.get(cache_key)
                if cached:
                    result['steps']['cache'] = {
                        'hit': True,
                        'answer': cached.get('answer', ''),
                        'time': 0
                    }
                    result['answer'] = cached['answer']
                    result['used_cache'] = True
                    result['success'] = True
                    result['total_time'] = time.time() - start_time
                    return result

            # 2. 意图分类
            intent_start = time.time()
            intent_result = self.intent_classifier.predict(query)
            intent_time = time.time() - intent_start
            result['steps']['intent'] = {
                'predicted': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'time': intent_time
            }

            # 3. FAQ搜索（search_faq 内部：先查 Redis(faq:) → 未命中查 MySQL → 命中写回 Redis）
            faq_start = time.time()
            faq_answer, need_llm = self.faq_search.search_faq(query, intent_result['intent'])
            faq_time = time.time() - faq_start
            result['steps']['faq'] = {
                'hit': not need_llm,
                'answer': faq_answer if not need_llm else '',
                'time': faq_time
            }

            # 如果FAQ命中
            if not need_llm:
                result['answer'] = faq_answer
                result['success'] = True
                result['total_time'] = time.time() - start_time
                # 注：命中结果已由 search_faq 写入 Redis(faq:)，无需在此重复缓存
                return result

            # 4. RAG处理（需要LLM）
            rag_steps = []

            # 4.1 查询增强
            augment_start = time.time()
            augmented_query = self.query_augmenter.augment_query(query, strategy)
            augment_time = time.time() - augment_start
            rag_steps.append({
                'augmentation': {
                    'strategy': strategy,
                    'result': augmented_query,
                    'time': augment_time
                }
            })

            # 4.2 检索
            retrieve_start = time.time()
            retrieval_results = self.retrieval.search(augmented_query, source_filter)
            retrieve_time = time.time() - retrieve_start
            rag_steps.append({
                'retrieval': {
                    'results_count': len(retrieval_results),
                    'time': retrieve_time
                }
            })

            # 4.3 重排序
            rerank_start = time.time()
            reranked_results = self.reranker.rerank(query, retrieval_results, top_k=2)
            rerank_time = time.time() - rerank_start
            rag_steps.append({
                'reranking': {
                    'results_count': len(reranked_results),
                    'top_score': reranked_results[0].get('rerank_score', 0) if reranked_results else 0,
                    'time': rerank_time
                }
            })

            # 4.4 构建上下文
            context_start = time.time()
            context = self._build_context(reranked_results)
            context_time = time.time() - context_start
            rag_steps.append({
                'context_building': {
                    'context_length': len(context),
                    'time': context_time
                }
            })

            result['steps']['rag'] = rag_steps

            # 5. LLM生成
            llm_start = time.time()
            answer = self.llm_generator.generate_with_context(query, context)
            llm_time = time.time() - llm_start
            result['steps']['llm'] = {
                'answer_length': len(answer) if answer else 0,
                'time': llm_time
            }

            if answer:
                result['answer'] = answer
                result['success'] = True

                # 缓存结果
                if self.cache.is_connected():
                    self.cache.set(cache_key, {
                        'answer': answer,
                        'intent': intent_result['intent'],
                        'strategy': strategy,
                        'sources': reranked_results
                    })

            result['total_time'] = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Query test failed for '{query}': {str(e)}")
            result['error'] = str(e)
            result['total_time'] = time.time() - start_time
            return result

    def _build_context(self, documents: List[Dict]) -> str:
        """构建上下文"""
        if not documents:
            return "未找到相关医学知识。"

        context_parts = []
        for i, doc in enumerate(documents[:2]):  # 使用前2个最相关的文档
            context_parts.append(f"【知识来源{i+1}】\n{doc['content']}\n")

        return "\n".join(context_parts)

    def test_batch_queries(self, queries: List[str], strategies: List[str] = None) -> Dict[str, Any]:
        """批量测试查询"""
        if strategies is None:
            strategies = ["direct", "hyde", "subquery", "backtracking"]

        results = {
            'total_tests': len(queries) * len(strategies),
            'successful_tests': 0,
            'strategy_results': {},
            'total_time': 0
        }

        for strategy in strategies:
            strategy_results = {
                'name': strategy,
                'tests': [],
                'success_count': 0,
                'avg_time': 0,
                'total_time': 0
            }

            for query in queries:
                result = self.test_single_query(query, strategy)
                strategy_results['tests'].append(result)
                strategy_results['total_time'] += result['total_time']

                if result['success']:
                    results['successful_tests'] += 1
                    strategy_results['success_count'] += 1

            strategy_results['avg_time'] = strategy_results['total_time'] / len(strategy_results['tests'])
            results['strategy_results'][strategy] = strategy_results

        results['total_time'] = sum(r['total_time'] for r in results['strategy_results'].values())

        return results

    def print_test_result(self, result: Dict[str, Any]):
        """打印测试结果"""
        status = "✓" if result['success'] else "✗"
        print(f"\n{status} 测试 #{self.test_count + 1}: {result['query']}")
        print(f"  策略: {result['strategy']}")
        print(f"  总耗时: {result['total_time']:.3f}秒")

        if result['success']:
            print(f"  回答: {result['answer'][:100]}...")
        else:
            print(f"  失败原因: {result.get('error', 'Unknown')}")

        # 打印详细步骤时间
        if 'steps' in result:
            total_step_time = 0
            for step_name, step_data in result['steps'].items():
                if isinstance(step_data, dict) and 'time' in step_data:
                    step_time = step_data['time']
                    total_step_time += step_time
                    print(f"  {step_name}: {step_time:.3f}秒")

            print(f"  步骤总耗时: {total_step_time:.3f}秒")

    def run_interactive_test(self):
        """运行交互式测试"""
        print("\n" + "="*50)
        print("RAG问答系统测试")
        print("="*50)
        print("输入查询进行测试，输入 'quit' 退出，输入 'batch' 批量测试")

        while True:
            query = input("\n请输入查询: ").strip()

            if query.lower() == 'quit':
                break
            elif query.lower() == 'batch':
                self.run_batch_test()
                continue
            elif not query:
                print("请输入有效的查询")
                continue

            # 测试所有策略
            strategies = ["direct", "hyde", "subquery", "backtracking"]
            results = {}

            for strategy in strategies:
                self.test_count += 1
                result = self.test_single_query(query, strategy)
                results[strategy] = result
                self.print_test_result(result)

            # 汇总结果
            successful_strategies = [s for s, r in results.items() if r['success']]
            print(f"\n汇总: {len(successful_strategies)}/{len(strategies)} 策略成功")

    def run_batch_test(self):
        """运行批量测试"""
        print("\n批量测试模式")
        print("1. 使用预设测试集")
        print("2. 手动输入多个查询")

        choice = input("请选择 (1/2): ").strip()

        if choice == '1':
            # 使用预设测试集
            test_queries = [
                "什么是高血压？",
                "头痛应该怎么处理？",
                "如何预防糖尿病？",
                "感冒了吃什么药？",
                "什么是人工智能？"
            ]
            strategies = ["direct", "hyde"]
        elif choice == '2':
            # 手动输入
            print("\n请输入多个查询，每行一个，空行结束:")
            test_queries = []
            while True:
                query = input().strip()
                if query:
                    test_queries.append(query)
                else:
                    break
            strategies = ["direct", "hyde", "subquery", "backtracking"]
        else:
            print("无效选择")
            return

        # 执行批量测试
        self.test_count = 0
        results = self.test_batch_queries(test_queries, strategies)

        # 打印结果汇总
        print("\n" + "="*50)
        print("批量测试结果汇总")
        print("="*50)
        print(f"总测试数: {results['total_tests']}")
        print(f"成功数: {results['successful_tests']}")
        print(f"成功率: {results['successful_tests']/results['total_tests']*100:.1f}%")
        print(f"总耗时: {results['total_time']:.3f}秒")

        # 各策略详情
        for strategy_name, strategy_result in results['strategy_results'].items():
            success_rate = strategy_result['success_count'] / len(strategy_result['tests']) * 100
            print(f"\n策略 {strategy_name}:")
            print(f"  成功率: {success_rate:.1f}%")
            print(f"  平均耗时: {strategy_result['avg_time']:.3f}秒")

def main():
    """主函数"""
    try:
        # 初始化配置
        config = Config()

        # 初始化测试器
        tester = RAGTester(config)

        # 检查系统状态
        print("系统状态检查:")
        print(f"LLM可用: {'✓' if tester.llm_generator.client else '✗'}")
        print(f"Redis可用: {'✓' if tester.cache.is_connected() else '✗'}")
        print(f"意图分类器可用: ✓")
        print(f"检索器可用: ✓")
        print(f"重排序器可用: {'✓' if tester.reranker.model else '✗'}")

        # 运行测试
        tester.run_interactive_test()

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        logger.error(f"测试脚本失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()