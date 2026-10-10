"""用正常响应和内存变异验证输出发布门控，不调用模型接口或数据库。"""
import ast
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import textwrap
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT/'tests/test_quality_pilot.py').is_file():
    raise RuntimeError('请从包含tests目录的项目源码或解包后的源码包运行门控检查')
sys.path[:0] = [str(ROOT/'tests'), str(ROOT)]
from src.online_service.llm_generator import LLMGenerator

REGRESSION = 'test_quality_pilot.QualityPilotTests.test_completion_status_controls_core_publication'


class RemoveCompletionGate(ast.NodeTransformer):
    """只移除finish_reason与stop比较的完整分支。"""

    def __init__(self):
        self.removed = 0

    def visit_If(self, node):
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Attribute)
                and test.left.attr == 'finish_reason' and len(test.ops) == 1
                and isinstance(test.ops[0], ast.NotEq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == 'stop'):
            self.removed += 1
            return None
        return self.generic_visit(node)


def run_regression():
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName(REGRESSION)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    return result, output.getvalue()


def main():
    source_path = ROOT/'src/online_service/llm_generator.py'
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    original = LLMGenerator.generate
    baseline, details = run_regression()
    if baseline.testsRun != 1 or not baseline.wasSuccessful():
        raise RuntimeError('正常代码的发布门控回归未通过：\n' + details)

    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    transformer = RemoveCompletionGate()
    tree = transformer.visit(tree)
    if transformer.removed != 1:
        raise RuntimeError('必须且只能识别到一个结束状态门控分支')
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<completion-gate-mutation>', 'exec'), namespace)
    with patch.object(LLMGenerator, 'generate', namespace['generate']):
        mutated, details = run_regression()

    reasons = [case.params.get('finish_reason') for case, _ in mutated.failures]
    if (mutated.testsRun != 1 or mutated.errors or len(mutated.failures) != 3
            or set(reasons) != {'length', 'content_filter', None}
            or not all('AssertionError: 0 != 2' in failure for _, failure in mutated.failures)):
        raise RuntimeError('回归未按预期检出三个非正常结束的核心发布错误：\n' + details)
    if (LLMGenerator.generate is not original
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hash):
        raise RuntimeError('验证结束后生成方法或生产源码发生了变化')

    print(json.dumps({
        'baseline_regression_passed': True,
        'removed_gate_count': transformer.removed,
        'expected_mutation_failures': reasons,
        'normal_stop_and_empty_content_controls_passed': True,
        'production_source_unchanged': True,
        'source_sha256': source_hash,
        'description': '三种非正常结束均被核心回归检出；变异仅发生在内存中。',
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
