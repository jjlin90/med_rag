"""Regression coverage for failed judges, frozen inputs and partial answer reuse."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.eval_score_utils import (parse_judge_score, make_pair_row, summarize_pairs,
                                     validate_frozen_contexts, require_independent_judge)


class EvaluationComparisonTests(unittest.TestCase):
    def test_judge_rejects_invalid_values_and_keeps_genuine_zero(self):
        self.assertEqual(parse_judge_score('```json\n{"score": 0}\n```'), 0.0)
        for raw in ('{}', 'broken', '{"score": -0.2}', '{"score": 1.2}',
                    '{"score": true}', '{"score": "0.5"}', '{"score": NaN}',
                    '{"score": Infinity}', '{"score": 0.5', '[]'):
            with self.subTest(raw=raw), self.assertRaises((ValueError, TypeError)):
                parse_judge_score(raw)

    def test_complete_pair_means_exclude_unmatched_and_preserve_precision(self):
        rows = [make_pair_row('zero', 0, 0, 0, 0),
                make_pair_row('precise', .123456, .5, .876543, .75),
                make_pair_row('failed', 1, 1, None, 1)]
        summary, n = summarize_pairs(rows)
        self.assertEqual(n, 2)
        self.assertEqual(summary['orig']['F'], .0617)
        self.assertEqual(summary['delta']['F'], .3765)
        self.assertIsNone(rows[2]['dAR'])
        self.assertEqual(rows[2]['oF'], 1)
        self.assertIsNone(summarize_pairs([rows[2]])[0]['orig']['F'])

    def test_answer_only_comparison_rejects_changed_evidence(self):
        originals = [{'question': 'q', 'contexts': ['original']}]
        validate_frozen_contexts(originals, [{'question': 'q', 'answer': 'cached'}])
        with self.assertRaises(ValueError):
            validate_frozen_contexts(originals, [{'question': 'q', 'contexts': ['changed']}])
        with self.assertRaises(ValueError):
            require_independent_judge('model-a', [{'question': 'q', '_gen_model': 'model-a'}])

    def test_direct_judge_failure_and_truncation_do_not_become_zero(self):
        from scripts.eval_f_ar_direct import score
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError('offline')
        with redirect_stdout(io.StringIO()):
            self.assertIsNone(score(client, 'prompt'))
            client.chat.completions.create.side_effect = None
            client.chat.completions.create.return_value = SimpleNamespace(choices=[
                SimpleNamespace(finish_reason='length', message=SimpleNamespace(content='{"score": 1}'))])
            self.assertIsNone(score(client, 'prompt'))

    def test_single_judge_missing_answer_retains_partial_scores_and_all_contexts(self):
        from scripts import eval_f_ar_single_judge_210 as module
        contexts = ['first', 'second', 'third', 'fourth']
        with patch.object(module, 'score', side_effect=[0.0, .5]) as judge:
            row = module.eval_one(None, 'q', 'answer', '', contexts)
        self.assertTrue(row['failed'])
        self.assertEqual(row['oF'], 0.0)
        self.assertIsNone(row['gF'])
        self.assertIn('fourth', judge.call_args_list[0].args[1])

    def test_partial_cached_answers_never_shift_generated_answers_to_another_question(self):
        from scripts import eval_f_ar_paired as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = [{'question': f'q{i}', 'answer': f'orig{i}', 'contexts': [f'ctx{i}']}
                         for i in range(3)]
            orig, source, cached = root/'orig.json', root/'source.json', root/'cached.json'
            orig.write_text(json.dumps(originals), encoding='utf-8')
            source.write_text('[]', encoding='utf-8')
            cached.write_text(json.dumps([{'question': 'q1', 'answer': 'cached1', 'contexts': ['ctx1']}]),
                              encoding='utf-8')
            def judge(items, *args, **kwargs):
                return {it['question']: {'faithfulness': .5, 'answer_relevancy': .5} for it in items}
            with patch.object(module, 'PROJECT_ROOT', root), patch.object(module, 'ORIG_ANSWERS', orig), \
                 patch.object(module, 'SRC', source), patch.object(module, 'get_zhipu_client'), \
                 patch.object(module, 'generate_grounding_zhipu', side_effect=lambda c, q, ctx: 'generated-'+q), \
                 patch.object(module, 'eval_f_ar_ragas', side_effect=judge), \
                 patch('sys.argv', ['paired', '--answers', str(cached)]), redirect_stdout(io.StringIO()):
                module.main()
            saved = json.loads((root/'data/test_query/_paired_grounding_answers.json').read_text(encoding='utf-8'))
            self.assertEqual([r['grounded_answer'] for r in saved], ['generated-q0', 'cached1', 'generated-q2'])
            report = json.loads((root/'data/test_query/_eval_f_ar_paired.json').read_text(encoding='utf-8'))
            self.assertEqual(report['n_complete_pairs'], 3)
            self.assertNotIn('composite_new', report['summary'])

    def test_ragas_comparison_rejects_fallback_and_restores_environment(self):
        from scripts.eval_f_ar_paired import eval_f_ar_ragas
        keys = ('LLM_BASE_URL', 'LLM_MODEL_NAME', 'LLM_JUDGE_TEMPERATURE')
        with patch.dict(os.environ, dict(zip(keys, ('original-url', 'original-model', '0.2')))):
            with patch('src.config.settings.Config'), patch('src.online_service.rag_evaluator.RAGEvaluator') as cls:
                cls.return_value.evaluate_dataset.return_value = {'engine': 'llm_judge_fallback'}
                with self.assertRaises(RuntimeError):
                    eval_f_ar_ragas([], 'judge')
            self.assertEqual([os.environ[k] for k in keys], ['original-url', 'original-model', '0.2'])

    def test_ragas_paired_missing_scores_stay_missing_and_exclude_incomplete_pairs(self):
        from scripts.eval_f_ar_paired import eval_f_ar_ragas
        raw_scores = [
            {'question': 'missing', 'faithfulness': None, 'answer_relevancy': None},
            {'question': 'absent', 'faithfulness': .7},
            {'question': 'zero', 'faithfulness': 0, 'answer_relevancy': 0},
            {'question': 'valid', 'faithfulness': .2, 'answer_relevancy': .4},
        ]
        invalid_values = (float('nan'), float('inf'), -.1, 1.1, True, '0.5')
        raw_scores.extend({'question': f'invalid-{i}', 'faithfulness': value,
                           'answer_relevancy': value} for i, value in enumerate(invalid_values))
        with patch('src.config.settings.Config'), patch('src.online_service.rag_evaluator.RAGEvaluator') as cls:
            cls.return_value.evaluate_dataset.return_value = {'engine': 'ragas', 'scores': raw_scores}
            scores = eval_f_ar_ragas([], 'independent-judge')
        self.assertEqual(scores['missing'], {'faithfulness': None, 'answer_relevancy': None})
        self.assertEqual(scores['absent'], {'faithfulness': .7, 'answer_relevancy': None})
        self.assertEqual(scores['zero'], {'faithfulness': 0, 'answer_relevancy': 0})
        for i in range(len(invalid_values)):
            with self.subTest(invalid_value=i):
                self.assertEqual(scores[f'invalid-{i}'], {'faithfulness': None, 'answer_relevancy': None})
        rows = [make_pair_row(q, s['faithfulness'], s['answer_relevancy'],
                              0 if q == 'zero' else .8, 0 if q == 'zero' else .6)
                for q, s in scores.items()]
        summary, complete = summarize_pairs(rows)
        self.assertEqual(complete, 2)
        self.assertEqual(summary, {'orig': {'F': .1, 'AR': .2},
                                   'grnd': {'F': .4, 'AR': .3},
                                   'delta': {'F': .3, 'AR': .1}})
        self.assertEqual(sum(row['failed'] for row in rows), len(raw_scores) - 2)
        json.dumps(rows, allow_nan=False)

    def test_csv_missing_metric_stays_blank_and_nested_output_is_created(self):
        from scripts.evaluate_rag import print_and_save
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {'engine': 'ragas', 'total': 1, 'metrics': ['faithfulness'],
                      'average': {'faithfulness': None}, 'scores': [{'question': 'q'}]}
            with redirect_stdout(io.StringIO()):
                print_and_save(report, str(root/'report.json'), str(root/'nested/report.csv'))
            import csv
            with (root/'nested/report.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['faithfulness'], '')


if __name__ == '__main__':
    unittest.main()
