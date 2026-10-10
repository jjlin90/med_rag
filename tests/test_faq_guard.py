"""FAQ answer acceptance with real BM25 and isolated database/cache boundaries."""
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.online_service.cache_manager import faq_cache_key, query_cache_key
from src.online_service.faq_search import BM25Index, FAQSearch
from src.online_service.main_api import RAGWebAPI


def make_faq(question, answer='standard answer'):
    faq = FAQSearch.__new__(FAQSearch)
    faq.config = SimpleNamespace(FAQ_CACHE_TTL=3600)
    faq.cache = Mock()
    faq.cache.get.return_value = None
    faq.connection = Mock()
    faq.cursor = Mock()
    faq.cursor.fetchone.return_value = {'question': question, 'answer': answer}
    faq.bm25_threshold = .85
    faq.faq_id_map = {0: 122}  # Index and MySQL primary key are different.
    faq.bm25_index = BM25Index()
    faq.bm25_index.build_index([question])
    return faq


class FAQGuardTests(unittest.TestCase):
    def test_single_candidate_softmax_one_cannot_override_question_mismatch(self):
        cases = [
            ('HDL 胆固醇升高', 'HDL 胆固醇降低'),
            ('促进儿童保持最佳健康和发育状态', '促进成人保持最佳健康和发育状态'),
            ('COVID-19 和急性肾损伤', 'COVID-19 和慢性肾损伤'),
            ('男性头痛', '女性头痛'), ('左侧疼痛', '右侧疼痛'),
            ('1型糖尿病', '2型糖尿病'), ('儿童健康', '不要儿童健康，只查成人健康'),
            ('药物 1.5mg', '药物 15mg'), ('血压 120/80', '血压 12080'),
            ('体温 -5', '体温 5'), ('剂量 mL', '剂量 ML'),
            ('儿童健康', '儿童健康未知新词'),
        ]
        for question, query in cases:
            with self.subTest(query=query):
                faq = make_faq(question)
                # This really reproduces the old gate, not a mocked similarity.
                self.assertEqual(faq.bm25_index.search_normalized(query)[0][0], 1.0)
                self.assertEqual(faq.search_faq(query), (None, True))
                faq.cursor.execute.assert_not_called()
                faq.cache.set.assert_not_called()

    def test_exact_question_still_answers_and_caches_acceptance_evidence(self):
        faq = make_faq('HDL 胆固醇升高')
        self.assertEqual(faq.search_faq('  HDL 胆固醇升高\n'), ('standard answer', False))
        faq.cursor.execute.assert_called_once_with(
            'SELECT question, answer FROM faq WHERE id = %s', (122,))
        args, kwargs = faq.cache.set.call_args
        self.assertEqual(args[0], faq_cache_key('HDL 胆固醇升高'))
        self.assertEqual(args[1]['question'], 'HDL 胆固醇升高')
        self.assertEqual(kwargs['ttl'], 3600)

    def test_database_question_must_still_match_index_and_query(self):
        faq = make_faq('儿童健康')
        faq.cursor.fetchone.return_value = {'question': '成人健康', 'answer': 'adult answer'}
        self.assertEqual(faq.search_faq('儿童健康'), (None, True))
        faq.cache.set.assert_not_called()

    def test_empty_or_missing_database_answers_do_not_enter_cache(self):
        for row in (None, {'question':'儿童健康','answer':''},
                    {'question':'儿童健康','answer':'  '},
                    {'question':'儿童健康','answer':None}):
            with self.subTest(row=row):
                faq = make_faq('儿童健康')
                faq.cursor.fetchone.return_value = row
                self.assertEqual(faq.search_faq('儿童健康'), (None, True))
                faq.cache.set.assert_not_called()

    def test_offline_cache_requires_matching_standard_question(self):
        faq = make_faq('儿童健康')
        faq.connection = faq.cursor = faq.bm25_index = None
        for row in ({'type':'faq','answer':'old answer'},
                    {'type':'faq','question':'成人健康','answer':'wrong answer'},
                    {'type':'faq','question':'儿童健康','answer':'  '}):
            faq.cache.get.return_value = row
            self.assertEqual(faq.search_faq('儿童健康'), (None, True))
        faq.cache.get.return_value = {'type':'faq','question':'儿童健康','answer':'safe answer'}
        self.assertEqual(faq.search_faq('儿童健康'), ('safe answer', False))
        self.assertEqual(faq.search_faq('儿童健康', use_cache=False), (None, True))

    def test_both_cache_namespaces_isolate_pre_guard_answers(self):
        self.assertTrue(query_cache_key('q').startswith('query:v3:'))
        self.assertTrue(faq_cache_key('q').startswith('faq:v3:'))
        old_key = 'faq:v2:' + hashlib.md5('q'.encode()).hexdigest()
        faq = make_faq('q')
        faq.connection = faq.cursor = faq.bm25_index = None
        old_cache = {old_key:{'type':'faq','answer':'old unsafe answer'}}
        faq.cache.get.side_effect = old_cache.get
        self.assertEqual(faq.search_faq('q'), (None, True))
        faq.cache.get.assert_called_once_with(faq_cache_key('q'))

    def test_api_rejected_faq_runs_deep_channel_and_exact_faq_skips_it(self):
        for query, expected_strategy, deep_calls in [
                ('HDL 胆固醇降低','direct',1),
                ('HDL 胆固醇升高','faq_fast_channel',0)]:
            with self.subTest(query=query):
                api = RAGWebAPI.__new__(RAGWebAPI)
                api.cache, api.conversation_store, api.rag_core = Mock(), Mock(), Mock()
                api.cache.get.return_value = None
                api.faq_search = make_faq('HDL 胆固醇升高')
                api.rag_core.generate.return_value = dict(
                    answer='deep answer',sources=[],confidence=.9,intent='medical',strategy='direct')
                result = api._handle_query(query,None,False,session_id='test')
                self.assertEqual(result.strategy,expected_strategy)
                self.assertEqual(api.rag_core.generate.call_count,deep_calls)
                api.conversation_store.update_session_history.assert_called_once()

    def test_api_does_not_read_old_outer_query_cache(self):
        import json
        query = 'HDL 胆固醇降低'
        payload = json.dumps([query,None,None,None],ensure_ascii=False,sort_keys=True)
        old_key = 'query:v2:' + hashlib.sha256(payload.encode()).hexdigest()
        api = RAGWebAPI.__new__(RAGWebAPI)
        api.cache, api.conversation_store, api.rag_core = Mock(),Mock(),Mock()
        api.cache.get.side_effect = {old_key:{'answer':'old wrong FAQ'}}.get
        api.faq_search = make_faq('HDL 胆固醇升高')
        api.rag_core.generate.return_value = dict(
            answer='deep answer',sources=[],confidence=.9,intent='medical',strategy='direct')
        result = api._handle_query(query,None,True,session_id='test')
        self.assertFalse(result.used_cache)
        self.assertEqual(result.answer,'deep answer')
        api.cache.get.assert_called_once_with(query_cache_key(query))


if __name__ == '__main__':
    unittest.main()
