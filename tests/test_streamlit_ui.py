"""Streamlit interaction test with deterministic HTTP fixtures, no external API calls."""
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, 'Install the demo extra to test Streamlit')
class StreamlitTests(unittest.TestCase):
    def test_degraded_answer_has_visible_status(self):
        response = Mock()
        response.json.return_value = {'status': 'healthy', 'services': {}, 'history': []}
        answer = Mock()
        answer.json.return_value = {'answer': '原文摘录', 'sources': [], 'degraded': True,
                                     'degrade_level': 2, 'confidence': 0.0}
        path = Path(__file__).resolve().parents[1] / 'web_demo/app.py'
        with patch('requests.get', return_value=response), patch('requests.post', return_value=answer):
            app = AppTest.from_file(str(path)).run(timeout=20)
            app.chat_input[0].set_value('问题').run(timeout=20)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('降级状态' in warning.value for warning in app.warning))

    def test_evaluation_renders_actual_metric_keys_and_missing_values(self):
        def get(url, **kwargs):
            response = Mock()
            response.json.return_value = {'status': 'healthy', 'services': {}, 'history': []}
            return response

        def post(url, **kwargs):
            response = Mock()
            response.status_code = 200
            response.json.return_value = ({'engine': 'ragas', 'total': 1, 'complete_count': 0,
                'metrics': ['faithfulness', 'answer_relevancy'],
                'average': {'faithfulness': None, 'answer_relevancy': .75},
                'valid_counts': {'faithfulness': 0, 'answer_relevancy': 1},
                'scores': [{'question': 'q', 'faithfulness': None, 'answer_relevancy': .75}]}
                if url.endswith('/evaluate') else {'answer': 'answer', 'sources': [{'content': 'context'}]})
            return response

        path = Path(__file__).resolve().parents[1] / 'web_demo/app.py'
        with patch('requests.get', side_effect=get), patch('requests.post', side_effect=post):
            app = AppTest.from_file(str(path)).run(timeout=20)
            self.assertEqual(len(app.exception), 0)
            next(r for r in app.radio if r.label == '功能页面').set_value('RAG 评估').run()
            app.text_area[0].set_value('q')
            # Saving a report is not part of this UI assertion; preserve local reports.
            with patch.object(Path, 'write_text', return_value=0):
                next(b for b in app.button if '开始评估' in b.label).click().run(timeout=20)
            self.assertEqual(len(app.exception), 0)
            values = {metric.label: metric.value for metric in app.metric}
            self.assertEqual(values['忠实度'], '未评出')
            self.assertEqual(values['答案相关性'], '0.750')


if __name__ == '__main__':
    unittest.main()
