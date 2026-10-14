"""Extraction contracts; the real MinerU check runs in the isolated environment."""

from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import PackageNotFoundError
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup
from scripts import extract_msd


HTML = """<html><body><nav>外部导航</nav><main>
<h1>示例疾病</h1>
<div class="TopicHead_topic__authors_main_cont__123">作者测试 MD</div>
<div class="TopicSubNavigation_topicSubNavigation__123">目录残留</div>
<div class="TestYourKnowledge_testYourKnowledgeContainer__123">Take a Quiz</div>
<h2>药物治疗</h2><h3>注意事项</h3>
<p>剂量 1.5mg，血压 120/80，[保留医学说明]；另见本段的具体条件。</p>
<p>请查看<button role="link">Safety 101</button>中的说明。</p>
<ul><li>5</li><li>应保留列表正文</li></ul>
<table><tr><th>药物</th><th>剂量</th></tr>
<tr><td>示例药物</td><td>1.5mg</td></tr></table>
<p>""" + "用于验证正文长度与完整性的原始文档说明。" * 10 + """</p>
<script>不应进入解析器</script>
</main></body></html>"""

MARKDOWN = '## 药物治疗\n\n### 注意事项\n\n剂量 1.5mg，血压 120/80，5，Safety 101。\n\n' + '正文。' * 50


class MinerUExtractionTests(unittest.TestCase):
    def make_source(self, root, titles):
        folder = root / 'source'
        index = folder / 'Json'
        index.mkdir(parents=True)
        (index / 'sections.json').write_text(
            json.dumps({'sections': [{'SectionId': '{chapter}'}]}), encoding='utf-8')
        topics = [{'TopicId': '{' + topic + '}', 'Title': title}
                  for topic, title in titles.items()]
        (index / '{chapter}.json').write_text(json.dumps({'topics': topics}), encoding='utf-8')
        for topic, title in titles.items():
            html = HTML.replace('<h1>示例疾病</h1>', f'<h1>{title}</h1>')
            (folder / ('{' + topic + '}.html')).write_text(html, encoding='utf-8')
        return folder

    def run_extraction(self, folder, output, mineru, *extra):
        with patch.object(extract_msd, 'create_mineru_parser', return_value=mineru), \
             patch.object(extract_msd, 'version', return_value='4.0.12'), \
             redirect_stdout(io.StringIO()):
            return extract_msd.main(['--input-dir', str(folder), '--output-dir', str(output), *extra])

    def test_dom_filter_preserves_medical_structure(self):
        prepared = extract_msd.prepare_msd_html(HTML, '示例疾病')
        soup = BeautifulSoup(prepared, 'html.parser')
        self.assertIsNone(soup.find('h1'))
        self.assertEqual(soup.h2.get_text(), '药物治疗')
        self.assertEqual(soup.h3.get_text(), '注意事项')
        self.assertEqual(len(soup.find_all('tr')), 2)
        self.assertEqual(len(soup.find_all('li')), 2)
        for value in ['1.5mg', '120/80', '[保留医学说明]', '另见本段', '>5<', 'Safety 101']:
            self.assertIn(value, prepared)
        for value in ['外部导航', '作者测试', '目录残留', 'Take a Quiz', '不应进入解析器']:
            self.assertNotIn(value, prepared)

    def test_article_fallback_and_no_body(self):
        self.assertIn('<table>', extract_msd.prepare_msd_html(
            '<article><h2>诊断</h2><table><tr><td>5</td></tr></table></article>', '疾病'))
        self.assertEqual(extract_msd.prepare_msd_html('<nav>仅导航</nav>', '疾病'), '')

    def test_render_has_single_title_and_keeps_markup(self):
        content = '# 示例疾病\n\n## 治疗\n\n<table><tr><td>1.5mg</td></tr></table>'
        result = extract_msd.render_msd_markdown(content, '示例疾病')
        self.assertEqual(result.count('# 示例疾病'), 1)
        self.assertIn('## 治疗', result)
        self.assertIn('<table>', result)

    def test_integrity_guard_rejects_missing_headings_or_values(self):
        prepared = extract_msd.prepare_msd_html(HTML, '示例疾病')
        extract_msd.validate_mineru_content(prepared, MARKDOWN)
        for incomplete in [MARKDOWN.replace('注意事项', ''), MARKDOWN.replace('1.5mg', '')]:
            with self.assertRaisesRegex(ValueError, '完整性检查失败'):
                extract_msd.validate_mineru_content(prepared, incomplete)

    def test_default_mineru_path_and_traceable_manifest(self):
        parsed_inputs = []

        def parse(path, **kwargs):
            parsed_inputs.append(Path(path).read_text(encoding='utf-8'))
            return SimpleNamespace(markdown=lambda: MARKDOWN)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, {'aaa': '示例疾病'})
            (source / 'unindexed.html').write_text(HTML, encoding='utf-8')
            output = root / 'output'
            self.assertEqual(self.run_extraction(source, output, SimpleNamespace(parse=parse)), 0)
            report = json.loads((output / 'extraction_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(report['parser'], 'mineru')
            self.assertEqual(report['parser_version'], '4.0.12')
            self.assertEqual(report['success'], 1)
            self.assertEqual(report['skipped'], 1)
            self.assertEqual(report['documents'][0]['topic_id'], 'aaa')
            self.assertEqual(Path(report['documents'][0]['source_html']), source / '{aaa}.html')
            self.assertTrue((output / '示例疾病.md').read_text(encoding='utf-8').startswith('# 示例疾病\n'))
            self.assertIn('<table>', parsed_inputs[0])
            self.assertNotIn('作者测试', parsed_inputs[0])

    def test_duplicate_sanitized_titles_and_htm_keep_both_topics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, {'aaa': '疾病/说明', 'bbb': '疾病:说明'})
            (source / '{bbb}.html').rename(source / '{bbb}.htm')
            output = root / 'output'
            mineru = SimpleNamespace(parse=lambda _, **kwargs: SimpleNamespace(markdown=lambda: MARKDOWN))
            self.assertEqual(self.run_extraction(source, output, mineru), 0)
            report = json.loads((output / 'extraction_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(report['success'], 2)
            self.assertEqual({p.name for p in output.glob('*.md')}, {'疾病_说明.md', '疾病_说明_bbb.md'})

    def test_parser_failure_records_error_and_returns_failure(self):
        def parse(path, **kwargs):
            if Path(path).stem == '{aaa}':
                raise RuntimeError('parse failed')
            return SimpleNamespace(markdown=lambda: MARKDOWN)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, {'aaa': '失败', 'bbb': '成功'})
            output = root / 'output'
            self.assertEqual(self.run_extraction(source, output, SimpleNamespace(parse=parse)), 1)
            report = json.loads((output / 'extraction_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(report['success'], 1)
            self.assertEqual(report['failures'][0]['error'], 'parse failed')
            self.assertFalse((output / '失败.md').exists())

    def test_limit_is_applied_to_indexed_topics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, {'aaa': '甲', 'bbb': '乙'})
            output = root / 'output'
            mineru = SimpleNamespace(parse=lambda _, **kwargs: SimpleNamespace(markdown=lambda: MARKDOWN))
            self.assertEqual(self.run_extraction(source, output, mineru, '--limit', '1'), 0)
            report = json.loads((output / 'extraction_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(report['processed'], 1)
            self.assertEqual(report['success'], 1)

    def test_dependency_failure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, {'aaa': '示例疾病'})
            output = root / 'output'
            with patch.object(extract_msd, 'version', side_effect=PackageNotFoundError), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                extract_msd.main(['--input-dir', str(source), '--output-dir', str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_legacy_parser_is_explicit_and_needs_no_mineru(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, {'aaa': '示例疾病'})
            output = root / 'output'
            with patch.object(extract_msd, 'create_mineru_parser') as factory, redirect_stdout(io.StringIO()):
                result = extract_msd.main(['--input-dir', str(source), '--output-dir', str(output),
                                           '--parser', 'beautifulsoup'])
            factory.assert_not_called()
            self.assertEqual(result, 0)

    def test_real_mineru_native_html(self):
        try:
            parser = extract_msd.create_mineru_parser()
        except RuntimeError as exc:
            if '未安装' in str(exc):
                self.skipTest('Run in .venv-mineru for the real SDK check')
            raise
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'article.html'
            path.write_text(extract_msd.prepare_msd_html(HTML, '示例疾病'), encoding='utf-8')
            result = parser.parse(str(path)).markdown()
            for value in ['药物治疗', '注意事项', '1.5mg', '120/80', '[保留医学说明]', '示例药物', 'Safety 101']:
                self.assertIn(value, result)
            self.assertTrue('<table' in result or '| 药物' in result)
            self.assertNotIn('作者测试', result)
            extract_msd.validate_mineru_content(path.read_text(encoding='utf-8'), result)
            nested = '<main><div><section><h2>开始</h2><p>剂量100mg。</p></section>'
            nested += '<section><h2>主体</h2><div><p>' + '很长的护理说明。' * 100 + '</p></div></section>'
            nested += '<section><h2>末尾</h2><p>随访20天，血压120/80。</p></section></div></main>'
            path.write_text(extract_msd.prepare_msd_html(nested, '测试正文选择'), encoding='utf-8')
            result = parser.parse(str(path)).markdown()
            extract_msd.validate_mineru_content(path.read_text(encoding='utf-8'), result)


if __name__ == '__main__':
    unittest.main()
