import unittest
from unittest.mock import patch

from src.config.settings import Config
from src.offline_pipeline.data_cleaner import DataCleaner
from src.offline_pipeline.document_loader import Document
from src.offline_pipeline.markdown_normalizer import normalize_markdown


class MarkdownNormalizerTests(unittest.TestCase):
    def test_balanced_links_images_and_references(self):
        source = '[病[型]](https://x/a_(b) "说明") ![症状图](assets/a.png)\n\n[剂量][rx]\n\n[rx]: https://x/long\n'
        self.assertEqual(normalize_markdown(source), '病[型] 症状图\n\n剂量')

    def test_undefined_references_and_medical_notation(self):
        source = '分级[1]，[未知][ref]，<HLA-B27>，剂量<5 mg，浓度≥2.5 mmol/L'
        self.assertEqual(normalize_markdown(source), source)

    def test_superscripts_do_not_merge_numeric_values(self):
        self.assertEqual(normalize_markdown('1.5 × 10<sup>9</sup>/L；CO<sub>2</sub>'), '1.5 × 10^9/L；CO_2')

    def test_inline_and_fenced_code_are_literal(self):
        source = '`[例](url)`\n\n``[例](url) with ` inside``\n\n```html\n<a href="x">例</a>\n```'
        self.assertEqual(normalize_markdown(source), source)

    def test_blank_lines_inside_protected_code_remain_literal(self):
        source = '```text\nfirst\n\n\nsecond\n```\n\n`line one\n\n\nline two`'
        self.assertEqual(normalize_markdown(source), source)

    def test_index_cleaner_preserves_code_blank_lines_and_trailing_spaces(self):
        source = '# Example\n\n```python\nvalue = "literal"   \n\n\nprint(value)\n```'
        cleaner = DataCleaner(Config())
        doc = Document(source, {'file_type':'markdown'})
        self.assertEqual(cleaner.clean_documents([doc])[0].page_content, source)

    def test_html_labels_headings_and_lists(self):
        source = '<a id="anchor"></a><h2>治疗</h2><p><a href="x">5 mg</a>，<span class="x">每日2次</span></p><ul><li>监测<ul><li>血糖</li></ul></li></ul><!--noise-->'
        result = normalize_markdown(source)
        self.assertIn('## 治疗', result)
        self.assertIn('5 mg，每日2次', result)
        self.assertIn('- 监测', result)
        self.assertIn('  - 血糖', result)
        self.assertNotIn('anchor', result)
        self.assertNotIn('noise', result)

    def test_table_caption_rowspan_colspan_and_cell_pipes(self):
        source = '<table><caption>治疗剂量</caption><tr><th>药物</th><th colspan="2">用量</th></tr><tr><td rowspan="2">A|B</td><td>5 mg</td><td>2次/日</td></tr><tr><td>10 mg</td><td>1次/日</td></tr></table>'
        result = normalize_markdown(source)
        self.assertIn('治疗剂量', result)
        self.assertIn('| 药物 | 用量 | 用量 |\n| --- | --- | --- |', result)
        self.assertIn('| A\\|B | 5 mg | 2次/日 |', result)
        self.assertIn('| A\\|B | 10 mg | 1次/日 |', result)
        self.assertEqual(result, normalize_markdown(result))

    def test_html_table_without_header_retains_first_row(self):
        result = normalize_markdown('<table><tr><td>疾病</td><td>3%</td></tr></table>')
        self.assertIn('| 列1 | 列2 |', result)
        self.assertIn('| 疾病 | 3% |', result)

    def test_index_cleaner_preserves_markdown_structure(self):
        value = '# 标题\n\n| 指标 | 值 |\n| --- | --- |\n| 血糖 | 5 mmol/L |\n\n- 第一项\n  - 第二项'
        cleaner = DataCleaner(Config())
        doc = Document(value, {'file_type': 'markdown', 'source': '测试'})
        result = cleaner.clean_documents([doc])[0]
        self.assertEqual(result.page_content, value)
        self.assertEqual(result.metadata, doc.metadata)

    def test_plain_text_path_unchanged(self):
        cleaner = DataCleaner(Config())
        doc = Document('[5 mg](url)\nPage 2', {'file_type': 'txt'})
        self.assertEqual(cleaner.clean_documents([doc])[0].page_content, '[5 mg](url)')

    def test_normalization_failure_does_not_index_raw_markup(self):
        with patch('src.offline_pipeline.data_cleaner.normalize_markdown', side_effect=ValueError('invalid')):
            with self.assertRaises(ValueError):
                DataCleaner(Config()).clean_documents([Document('<table>', {'file_type': 'markdown'})])


if __name__ == '__main__':
    unittest.main()
