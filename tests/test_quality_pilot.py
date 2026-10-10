"""Quality experiment integrity and fact-review failure boundaries."""
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock

import numpy as np

from src.online_service.llm_generator import LLMGenerator, format_knowledge_document, validated_review_answer
from src.online_service.rag_evaluator import answer_relevancy_metric
from src.online_service.retrieval import Retrieval
from src.online_service.reranker import Reranker
from src.online_service.rag_system import RAGSystem
from scripts.run_quality_pilot import validate_arm_rows


class QualityPilotTests(unittest.TestCase):
    def make_medical_core(self, retrieval, generator):
        system = RAGSystem.__new__(RAGSystem)
        system.config = SimpleNamespace(TOP_K_RERANK=2, ALLOW_LLM_WHEN_NO_CONTEXT=False)
        system.intent_classifier = Mock(predict=Mock(return_value={'intent':'medical','confidence':.9}))
        system._retrieve_and_merge = Mock(return_value=retrieval)
        system.reranker = Mock(rerank=Mock(return_value=retrieval.documents))
        system.llm_generator = generator
        return system

    def make_title_retrieval(self, responses):
        retrieval = Retrieval.__new__(Retrieval)
        retrieval.config = SimpleNamespace(RETRIEVAL_TITLE_ANCHOR=True)
        retrieval.top_k_retrieve, retrieval.top_k_children = 16, 5
        retrieval.sparse_weight, retrieval.dense_weight = .7, 1.
        retrieval._bump = Mock()
        retrieval._matching_source_titles = Mock(return_value=['疾病甲','疾病乙'])
        retrieval.embedding_provider = Mock(generate_embeddings=Mock(
            return_value=(np.array([[1.,2.]]),[{1:.5}])))
        retrieval.milvus_store = Mock(hybrid_search_with_rerank=Mock(side_effect=responses))
        return retrieval

    @staticmethod
    def child_hit(name):
        return {'id':name+'-child','score':1.,'distance':1.,'entity':{
            'text':name+' child evidence','metadata':{},'source':name,
            'parent_id':name+'-parent','parent_content':name+' parent evidence'}}

    def test_duplicate_cannot_replace_missing_question(self):
        manifest = {'indices_1based':list(range(20)), 'arms':{'baseline':{},'candidate':{}}}
        rows = [{'index':i,'arm':a} for i in range(20) for a in manifest['arms']]
        validate_arm_rows(rows,manifest)
        rows[-1] = rows[0].copy()
        with self.assertRaises(ValueError): validate_arm_rows(rows,manifest)

    def test_required_review_failure_does_not_release_draft(self):
        generator = LLMGenerator.__new__(LLMGenerator)
        generator.config = SimpleNamespace(LLM_GROUNDING=True,LLM_GROUNDING_REVIEW=True)
        generator.generate = Mock(side_effect=['an unreviewed medical draft',None])
        self.assertIsNone(generator.generate_with_context('q','evidence'))
        from src.online_service.retrieval import RetrievalResult
        evidence = RetrievalResult(documents=[{'id':'p','source':'source','content':'original evidence'}])
        generator.generate.reset_mock(side_effect=True)
        generator.generate.side_effect = ['an unreviewed medical draft',None]
        result = self.make_medical_core(evidence,generator).generate('q',strategy='direct')
        self.assertEqual(result['degrade_level'],2)
        self.assertEqual(result['degrade_reason'],'llm_unavailable')
        self.assertIn('original evidence',result['answer'])
        self.assertNotIn('unreviewed medical draft',result['answer'])

    def test_nonexistent_review_quote_is_rejected(self):
        raw = json.dumps({'checks':[{'claim':'a medical claim','evidence':'invented evidence',
                                     'supported':True}], 'final_answer':'medical answer'})
        self.assertIsNone(validated_review_answer(raw,'actual evidence'))
        payload = json.loads(raw)
        payload['checks'][0]['evidence'] = 'actual\n evidence'
        self.assertEqual(validated_review_answer(json.dumps(payload),'actual evidence'),'medical answer')
        payload['checks'][0] = {'claim':'supported medical fact','evidence_ids':[1],'supported':True}
        self.assertEqual(validated_review_answer(json.dumps(payload),'actual evidence'),'medical answer')
        for invalid in [0,2,True,'1']:
            payload['checks'][0]['evidence_ids'] = [invalid]
            self.assertIsNone(validated_review_answer(json.dumps(payload),'actual evidence'))

    def test_completion_status_controls_core_publication(self):
        generator = LLMGenerator.__new__(LLMGenerator)
        generator.model_name = 'mock-model'
        generator.client = Mock()
        # Exercise the defaults used by generate_with_context, rather than
        # falling into the exception handler before the completion is read.
        generator.temperature = 0.0
        generator.max_tokens = 3072
        generator.config = SimpleNamespace(LLM_GROUNDING=True,LLM_GROUNDING_REVIEW=False)
        from src.online_service.retrieval import RetrievalResult
        evidence = RetrievalResult(documents=[{'id':'p','source':'source','content':'original evidence'}])
        cases = [('stop','complete supported answer',0),
                 ('length','unfinished',2),('content_filter','unfinished',2),
                 (None,'unfinished',2),('stop','   ',2),('stop',None,2)]
        for finish_reason, content, expected_level in cases:
            with self.subTest(finish_reason=finish_reason,content=content):
                generator.client.reset_mock()
                generator.client.chat.completions.create.return_value = SimpleNamespace(choices=[
                    SimpleNamespace(finish_reason=finish_reason,message=SimpleNamespace(content=content))])
                result = self.make_medical_core(evidence,generator).generate('q',strategy='direct')
                generator.client.chat.completions.create.assert_called_once()
                request = generator.client.chat.completions.create.call_args.kwargs
                self.assertEqual(request['temperature'],generator.temperature)
                self.assertEqual(request['max_tokens'],generator.max_tokens)
                self.assertEqual(result['degrade_level'],expected_level)
                self.assertEqual(result['sources'],evidence.documents)
                if expected_level == 0:
                    self.assertEqual(result['answer'],content)
                    self.assertFalse(result['degraded'])
                    self.assertEqual(result['degrade_reason'],'')
                else:
                    self.assertTrue(result['degraded'])
                    self.assertEqual(result['degrade_reason'],'llm_unavailable')
                    self.assertIn('original evidence',result['answer'])
                    self.assertNotIn('unfinished',result['answer'])

    def test_custom_general_prompt_skips_medical_review(self):
        generator = LLMGenerator.__new__(LLMGenerator)
        generator.config = SimpleNamespace(LLM_GROUNDING=True,LLM_GROUNDING_REVIEW=True)
        generator.generate = Mock(return_value='general answer')
        self.assertEqual(generator.generate_with_context('q','',system_prompt='general'),'general answer')
        self.assertEqual(generator.generate.call_count,1)

    def test_chinese_prompt_keeps_formula_and_default_prompt(self):
        from ragas.metrics._answer_relevance import ResponseRelevanceOutput
        from ragas.metrics import AnswerRelevancy
        original = AnswerRelevancy().question_generation.instruction
        metrics = [answer_relevancy_metric(),answer_relevancy_metric('chinese')]
        for metric in metrics:
            self.assertEqual(metric.strictness,3)
            metric.embeddings = SimpleNamespace(embed_query=lambda q:[1.,0.],
                embed_documents=lambda qs:[[1.,0.] for q in qs])
            responses = [ResponseRelevanceOutput(question='同一问题',noncommittal=0)]*3
            self.assertEqual(metric._calculate_score(responses,{'user_input':'同一问题'}),1.)
            responses[0] = ResponseRelevanceOutput(question='同一问题',noncommittal=1)
            self.assertEqual(metric._calculate_score(responses,{'user_input':'同一问题'}),0.)
        self.assertEqual(AnswerRelevancy().question_generation.instruction,original)

    def test_explicit_source_restriction_is_never_broadened(self):
        retrieval = Retrieval.__new__(Retrieval)
        retrieval.config = SimpleNamespace(RETRIEVAL_TITLE_ANCHOR=True)
        retrieval.top_k_children = 5
        retrieval._bump = Mock()
        retrieval._matching_source_titles = Mock(return_value=['another source'])
        retrieval.search = Mock(return_value=[{'id':'c','parent_id':'p','content':'child',
            'parent_content':'parent','source':'restricted','score':1}])
        result = retrieval.search_child_to_parent('another source','restricted')
        self.assertTrue(result.is_usable)
        retrieval._matching_source_titles.assert_not_called()
        self.assertEqual(retrieval.search.call_count,1)
        self.assertEqual(retrieval.search.call_args.args[1],'restricted')
        self.assertTrue(retrieval.search.call_args.kwargs['only_children'])

    def test_optional_title_failure_preserves_primary_evidence(self):
        primary = [self.child_hit('primary')]
        for title_results in [[RuntimeError('anchor offline'),[]],
                              [RuntimeError('anchor offline'),[self.child_hit('疾病乙')]],
                              [[self.child_hit('疾病甲')],RuntimeError('anchor offline')]]:
            with self.subTest(title_results=title_results):
                retrieval = self.make_title_retrieval([primary,*title_results])
                with self.assertLogs('src.online_service.retrieval',level='WARNING') as logs:
                    result = retrieval.search_child_to_parent('疾病甲与疾病乙')
                self.assertTrue(result.is_usable)
                self.assertEqual(result.degrade_level,0)
                self.assertIsNone(result.error)
                self.assertEqual(result.degrade_reason,'')
                self.assertIn('primary-parent',[row['id'] for row in result.documents])
                self.assertEqual(retrieval.milvus_store.hybrid_search_with_rerank.call_count,3)
                retrieval._bump.assert_called_once_with('l0_ok')
                self.assertTrue(any(record.levelname == 'WARNING' for record in logs.records))
                for rows in title_results:
                    if isinstance(rows,list) and rows:
                        parent = next(row for row in result.documents if row['id'] == rows[0]['entity']['parent_id'])
                        self.assertTrue(parent['topic_anchor'])
                generator = Mock(generate_with_context=Mock(return_value='supported answer'))
                answer = self.make_medical_core(result,generator).generate('q',strategy='direct')
                self.assertEqual(answer['answer'],'supported answer')
                self.assertEqual(answer['degrade_level'],0)
                generator.generate_with_context.assert_called_once()

    def test_primary_failure_with_title_anchor_still_refuses(self):
        retrieval = self.make_title_retrieval([RuntimeError('primary offline')])
        result = retrieval.search_child_to_parent('疾病甲与疾病乙')
        self.assertFalse(result.is_usable)
        self.assertEqual(result.degrade_level,2)
        self.assertEqual(result.degrade_reason,'retrieval_error')
        self.assertIn('primary offline',result.error)
        retrieval._matching_source_titles.assert_not_called()
        retrieval.milvus_store.hybrid_search_with_rerank.assert_called_once()
        generator = Mock()
        self.make_medical_core(result,generator).generate('q',strategy='direct')
        generator.generate_with_context.assert_not_called()

    def test_title_matching_preserves_disease_numbers(self):
        retrieval = Retrieval.__new__(Retrieval)
        retrieval._source_titles = ['1型糖尿病','2型糖尿病','糖尿病','病']
        self.assertEqual(retrieval._matching_source_titles('1型糖尿病如何处理？'),
                         ['1型糖尿病','糖尿病'])

    def test_broken_title_catalog_retains_ordinary_retrieval(self):
        retrieval = Retrieval.__new__(Retrieval)
        retrieval.config = SimpleNamespace(CHUNK_SAVE_PATH=Mock(read_text=Mock(return_value='{broken')))
        self.assertEqual(retrieval._matching_source_titles('some query'),[])

    def test_parent_dedup_preserves_title_anchor_and_strongest_child(self):
        retrieval = Retrieval.__new__(Retrieval)
        retrieval.top_k_children = 5
        rows = [{'id':'c1','parent_id':'p','content':'named child','source':'named',
                 'score':.1,'topic_anchor':True},
                {'id':'c2','parent_id':'p','content':'strongest child','source':'named','score':.9}]
        parents = retrieval._backtrack_to_parents(rows)
        self.assertEqual(len(parents),1)
        self.assertTrue(parents[0]['topic_anchor'])
        self.assertEqual(parents[0]['rerank_content'],'strongest child')
        self.assertEqual(parents[0]['children_ids'],['c1','c2'])

    def test_named_disease_survives_higher_scoring_different_disease(self):
        reranker = Reranker.__new__(Reranker)
        reranker.model = Mock(compute_score=Mock(return_value=[1.,5.,3.]))
        rows = [{'id':'named','content':'named disease','topic_anchor':True},
                {'id':'other','content':'different disease'},
                {'id':'third','content':'third disease'}]
        self.assertEqual([r['id'] for r in reranker.rerank('named disease',rows,2)],['named','other'])
        self.assertEqual(rows[0]['rerank_score'],1.)

    def test_generation_and_judging_share_document_title(self):
        system = RAGSystem.__new__(RAGSystem)
        system.config = SimpleNamespace(TOP_K_RERANK=2)
        document = {'source':'疾病甲','content':'正文'}
        judge_context = format_knowledge_document(document)
        self.assertEqual(judge_context,'【文档标题：疾病甲】\n正文')
        self.assertIn(judge_context,system._build_context([document]))


if __name__ == '__main__':
    unittest.main()
