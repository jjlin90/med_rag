"""Compare FAQ acceptance on the local CSV using real BM25 and isolated IO.

No MySQL/Redis/LLM calls: the cursor returns the actual selected CSV record.
The report checks routing acceptance, not the correctness of downstream answers.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.online_service.faq_search import BM25Index, FAQSearch
from src.online_service.cache_manager import faq_cache_key, query_cache_key

SUBSTITUTIONS = [('一型','二型'), ('二型','一型'), ('左侧','右侧'), ('右侧','左侧'),
                 ('儿童','成人'), ('成人','儿童'), ('男性','女性'), ('女性','男性'),
                 ('急性','慢性'), ('慢性','急性'), ('低血压','高血压'), ('高血压','低血压'),
                 ('1型','2型'), ('2型','1型'), ('升高','降低'), ('降低','升高')]

ACCEPTANCE_POLICY = (
    'Exact standard question after trimming outer whitespace; '
    'BM25/softmax gate and nonempty database answer also required.'
)


class CSVCursor:
    def __init__(self, rows):
        self.rows = rows
        self.selected = None

    def execute(self, statement, parameters):
        self.selected = parameters[0]

    def fetchone(self):
        return self.rows[self.selected]


def verify(dataset):
    with dataset.open(encoding='utf-8-sig', newline='') as file:
        rows = [{'question': row['question'].strip(), 'answer': row.get('answer','').strip()}
                for row in csv.DictReader(file) if row.get('question','').strip()]
    index = BM25Index()
    index.build_index([row['question'] for row in rows])
    faq = FAQSearch.__new__(FAQSearch)
    faq.bm25_index, faq.faq_id_map = index, dict(enumerate(range(len(rows))))
    faq.connection, faq.cursor = True, CSVCursor(rows)
    faq.bm25_threshold = .85
    faq.config = SimpleNamespace(FAQ_CACHE_TTL=3600)
    records = []
    for row in rows:
        question = row['question']
        for old, new in SUBSTITUTIONS:
            if old in question and new not in question:
                query = question.replace(old, new, 1)
                scores = index.search_normalized(query, k=5)
                matched = rows[scores[0][1]]['question'] if scores else None
                before = bool(scores and scores[0][0] >= .85 and old in matched)
                answer, need_rag = faq.search_faq(query, use_cache=False)
                after = not need_rag and old in matched
                records.append({'old_term':old, 'new_term':new,
                                'before_opposite_acceptance':before,
                                'after_opposite_acceptance':after,
                                'after_direct_answer':bool(answer and not need_rag),
                                'after_candidate_is_exact':matched == query})
                break
    # Also test every original question as a control to report the recall tradeoff.
    positives = Counter()
    for row in rows:
        scores = index.search_normalized(row['question'],k=5)
        positives['before_direct'] += bool(scores and scores[0][0] >= .85)
        answer, need_rag = faq.search_faq(row['question'],use_cache=False)
        positives['after_direct'] += bool(answer and not need_rag)
    resolved_dataset = dataset.resolve()
    dataset_label = (resolved_dataset.relative_to(ROOT).as_posix()
                     if resolved_dataset.is_relative_to(ROOT) else str(resolved_dataset))
    return {'generated_at':datetime.now(timezone.utc).isoformat(),
            'dataset':dataset_label, 'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'faq_count':len(rows), 'perturbation_count':len(records),
            'before_opposite_acceptances':sum(r['before_opposite_acceptance'] for r in records),
            'after_opposite_acceptances':sum(r['after_opposite_acceptance'] for r in records),
            'after_direct_answers':sum(r['after_direct_answer'] for r in records),
            'after_nonexact_direct_answers':sum(r['after_direct_answer'] and not r['after_candidate_is_exact'] for r in records),
            'original_question_controls':{'count':len(rows), **dict(positives)},
            'scope':'Real local FAQ corpus and BM25; CSV cursor replaces MySQL, cache disabled; no deep RAG or clinical answer validation.',
            'acceptance_policy':ACCEPTANCE_POLICY,
            'cache_namespaces':[':'.join(key.split(':')[:2]) for key in
                                (query_cache_key('verification'), faq_cache_key('verification'))],
            'cases':records}


def run_automated_tests():
    """Return the actual passed regression count; stop publication on failure."""
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    with redirect_stdout(sys.stderr):
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Regression tests failed; verification report was not published')
    return result.testsRun - len(result.skipped) - len(result.expectedFailures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/msd_faq_clean.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/faq_guard_verification.json')
    parser.add_argument('--with-tests',action='store_true',
                        help='运行完整回归测试，并在报告中记录实际通过数量')
    parser.add_argument('--summary-only',action='store_true',
                        help='省略逐条测试样例，生成公开验证摘要')
    args = parser.parse_args()
    passed = run_automated_tests() if args.with_tests else None
    report = verify(args.dataset)
    if passed is not None:
        report['automated_tests_passed'] = passed
    if args.summary_only:
        report.pop('cases')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2))
    return int(bool(report['after_opposite_acceptances'] or report['after_nonexact_direct_answers']))


if __name__ == '__main__':
    raise SystemExit(main())
