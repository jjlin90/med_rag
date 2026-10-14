"""Extract indexed MSD HTML with MinerU's local native parser.

Use the separate MinerU environment; RAG model dependencies are not needed.
The old text extractor remains available for explicit comparison.
"""

from pathlib import Path
import json
from bs4 import BeautifulSoup
import re
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import PackageNotFoundError, version
import sys
from tempfile import TemporaryDirectory

# =====================【核对路径！】=====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_ROOT / 'data/raw/MSDZHConsumerMedicalTopics'
OUTPUT_MD_DIR = PROJECT_ROOT / 'data/clean_md_mineru'
# =======================================================
JSON_FOLDER = ROOT_DIR / "Json"


def build_uuid_title_mapping(json_folder=None):
    json_folder = Path(json_folder) if json_folder is not None else JSON_FOLDER
    uuid_2_title = {}
    sections_path = json_folder / "sections.json"
    sections_data = json.loads(sections_path.read_text(encoding="utf-8-sig"))

    for section in sections_data.get("sections", []):
        # SectionId自带大括号，直接拿来拼接文件名
        section_id = section["SectionId"]
        chapter_file = json_folder / f"{section_id}.json"
        if not chapter_file.exists():
            print(f"警告：章节文件不存在 {chapter_file.name}")
            continue
        try:
            chap_data = json.loads(
                chapter_file.read_text(encoding="utf-8-sig"))

            # 递归遍历整个json，抓取所有 TopicId + Title（防止层级变化）
            def scan_node(node):
                if isinstance(node, dict):
                    if "TopicId" in node and "Title" in node:
                        tid_raw = node["TopicId"]
                        # TopicId同样带大括号，统一去除
                        tid = tid_raw.strip("{}").lower()
                        title = node["Title"].strip()
                        uuid_2_title[tid] = title
                    for v in node.values():
                        scan_node(v)
                elif isinstance(node, list):
                    for item in node:
                        scan_node(item)

            scan_node(chap_data)
        except Exception as e:
            print(f"读取 {chapter_file.name} 失败：{str(e)}")

    print(f"✅ 加载目录索引完成，共 {len(uuid_2_title)} 篇疾病文档")
    return uuid_2_title


def prepare_msd_html(html_content: str, title: str) -> str:
    """Keep the article's native markup, removing MSD page chrome first."""
    soup = BeautifulSoup(html_content, "html.parser")
    drop_selectors = [
        "script", "style", "iframe", "noscript", "header", "footer", "nav", "aside",
        ".header", ".footer", ".sidebar", ".navigation", ".test-your-knowledge",
        ".quiz", ".toc", ".related-articles-heading", ".copyright", ".disclaimer",
        ".author-block", ".review-block", ".pub-date",
        '[class*="TopicHead_topic__authors_main_cont"]',
        '[class*="TopicHead_rightBlockContainer"]',
        '[class*="TopicSubNavigation_"]',
        '[class*="TestYourKnowledge_"]',
    ]
    for selector in drop_selectors:
        for tag in soup.select(selector):
            tag.decompose()

    main_block = soup.find("main") or soup.find("article")
    if main_block is None:
        return ""

    # Remove standalone navigation labels, never sentences containing them.
    trash_text = {
        "Test your Knowledge", "Take a Quiz", "小知识", "了解更多信息", "另见",
        "以下是可能对您有帮助的英文资料", "本手册对该资料中的内容不承担责任",
    }
    for text in list(main_block.find_all(string=True)):
        if text.strip() in trash_text:
            text.extract()
    # The authoritative disease title is supplied by the directory index.
    for heading in list(main_block.find_all("h1")):
        if heading.get_text(" ", strip=True) == title:
            heading.decompose()
        else:
            heading.name = "h2"
    # MSD's layout div/section wrappers can make MinerU's article selector pick
    # just one long subsection. We have already selected the article, so unwrap
    # only layout containers while keeping semantic headings/lists/tables.
    # MSD also renders inline topic names as buttons. Native HTML parsers drop
    # interactive buttons, so unwrap the control while preserving its label.
    for container in list(main_block.find_all(['div', 'section', 'article', 'main', 'button'])):
        container.unwrap()
    main_block.attrs = {}
    return ('<!doctype html><html lang="zh"><head><meta charset="utf-8"></head><body>'
            + str(main_block) + '</body></html>')


class MinerUHtmlParser:
    """Keep original source paths when parsing a cleaned temporary document."""

    def __init__(self, parser, source_context_type):
        self.parser = parser
        self.source_context_type = source_context_type

    def parse(self, staged_path, *, source_path=None):
        source_path = Path(source_path or staged_path).resolve()
        context = self.source_context_type(
            source_uri=source_path.as_uri(),
            local_resource_root=source_path.parent,
            transport_encoding='utf-8',
        )
        return self.parser.parse(staged_path, source_context=context)


def create_mineru_parser():
    """Load MinerU on demand, using native HTML parsing without model inference."""
    try:
        installed_version = version("mineru")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "MinerU 未安装。请按 docs/mineru.md 创建独立环境，并使用 "
            ".venv-mineru/Scripts/python.exe scripts/extract_msd.py 运行。"
        ) from exc
    if installed_version.split(".", 1)[0] != "4":
        raise RuntimeError(f"需要 MinerU 4.x，当前为 {installed_version}；安装说明见 docs/mineru.md")
    from mineru.parser import MinerUParser
    from docvortex.document.contracts import HtmlSourceContext
    return MinerUHtmlParser(
        MinerUParser(tier="flash", parse_mode="txt", image_analysis=False),
        HtmlSourceContext,
    )


def render_msd_markdown(content: str, title: str) -> str:
    """Prefix an indexed document title without flattening tables/headings."""
    lines = content.strip().splitlines()
    if lines and re.fullmatch(r"#\s+" + re.escape(title), lines[0].strip()):
        lines = lines[1:]
    return f"# {title}\n\n" + '\n'.join(lines).strip() + '\n'


def validate_mineru_content(prepared_html: str, markdown: str):
    """Reject silent losses of article headings or numeric values."""
    source = BeautifulSoup(prepared_html, 'html.parser')
    # Markdown targets/anchor IDs can interrupt visible text comparisons.
    visible_markdown = re.sub(r'!?\[([^\]]*)\]\([^\n]*?\)', r'\1', markdown)
    visible_markdown = BeautifulSoup(visible_markdown, 'html.parser').get_text(' ', strip=True)
    normalize = lambda text: re.sub(r'[\W_]+', '', text)
    normalized = normalize(visible_markdown)
    headings = [h.get_text(' ', strip=True) for h in source.find_all(re.compile(r'^h[1-6]$'))]
    missing_headings = [heading for heading in headings
                        if normalize(heading) and normalize(heading) not in normalized]
    source_numbers = set(re.findall(r'(?<![\d.])[+-]?\d+(?:\.\d+)?(?:/\d+)?',
                                    source.get_text(' ', strip=True)))
    missing_numbers = sorted(value for value in source_numbers if value not in visible_markdown)
    if missing_headings or missing_numbers:
        raise ValueError(f'MinerU 正文完整性检查失败：缺失标题={missing_headings}，缺失数值={missing_numbers}')


def clean_msd_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. 彻底删除所有页面侧边栏、导航、测验、版权、作者区块
    drop_selectors = [
        "script", "style", "iframe", "noscript", ".header", ".footer",
        ".sidebar", ".navigation", ".test-your-knowledge", ".quiz", ".toc",
        ".related-articles-heading", ".copyright", ".disclaimer",
        ".author-block", ".review-block", ".pub-date"
    ]
    for sel in drop_selectors:
        for tag in soup.select(sel):
            tag.decompose()

    # 清除纯测验文字、小知识标签、外部资料跳转文字
    trash_text = [
        "Test your Knowledge", "Take a Quiz", "小知识", "了解更多信息", "另见",
        "以下是可能对您有帮助的英文资料", "本手册对该资料中的内容不承担责任"
    ]
    for bad_word in trash_text:
        for elem in soup.find_all(string=lambda t: bad_word in t):
            elem.extract()

    main_block = soup.find("main") or soup.find("article")
    if not main_block:
        return ""

    # 2. 识别原生小标题，转为二级Markdown标题，区分章节
    chapter_keywords = [
        "1 型糖尿病的病因", "1 型糖尿病的筛查和预防", "1 型糖尿病的症状", "1 型糖尿病的诊断", "1 型糖尿病的治疗",
        "1 型糖尿病的治疗监测", "1 型糖尿病的并发症"
    ]
    raw_lines = main_block.get_text(separator="\n").splitlines()
    processed_lines = []

    for line in raw_lines:
        strip_line = line.strip()
        # 过滤垃圾行：作者、医院、版本、纯数字日期、重复主标题
        filter_words = [
            "MD", "Medical College", "Cedars-Sinai",
            "New York Medical College", "12月 2025", "1 型糖尿病 (DM)"
        ]
        skip_line = False
        for kw in filter_words:
            if kw == strip_line:
                skip_line = True
                break
        if skip_line or not strip_line:
            continue

        # 匹配章节，自动添加二级标题##
        match_chapter = False
        for chap in chapter_keywords:
            if strip_line.startswith(chap):
                processed_lines.append(f"\n## {strip_line}")
                match_chapter = True
                break
        if match_chapter:
            continue

        processed_lines.append(strip_line)

    # 3. 压缩连续空行，段落之间仅保留一行分隔
    final_text = []
    empty_flag = False
    for ln in processed_lines:
        if ln == "":
            if not empty_flag:
                final_text.append("")
                empty_flag = True
        else:
            final_text.append(ln)
            empty_flag = False

    # 拼接文本
    clean_content = "\n".join(final_text).strip()

    # 4. 精简重复冗余格式：清理多余换行、重复词汇造成的割裂感
    import re
    # 去除段落内过多连续空格
    clean_content = re.sub(r"\s{3,}", " ", clean_content)
    # 清理跳转类括号标记（不删除医学括号释义，只删纯跳转标识）
    # 方括号可能包含有效医学信息，不按标点形式整段删除。
    return clean_content


def parse_topic(html_path, title, output_dir, mineru_parser):
    """Return a per-file result; failures remain visible to the parent process."""
    try:
        html_txt = html_path.read_text(encoding='utf-8-sig')
        if mineru_parser is None:
            content = clean_msd_html(html_txt)
        else:
            prepared = prepare_msd_html(html_txt, title)
            if not prepared:
                return html_path, title, '', None
            with TemporaryDirectory(prefix='msd-mineru-', dir=output_dir) as temp_dir:
                staged_path = Path(temp_dir) / html_path.name
                staged_path.write_text(prepared, encoding='utf-8')
                content = mineru_parser.parse(str(staged_path), source_path=html_path).markdown()
                validate_mineru_content(prepared, content)
        return html_path, title, content, None
    except Exception as exc:
        return html_path, title, '', str(exc)


_worker_parser = None


def initialize_worker(parser_name):
    global _worker_parser
    _worker_parser = create_mineru_parser() if parser_name == 'mineru' else None


def parse_worker_topic(job):
    return parse_topic(*job, _worker_parser)


def iter_parsed_topics(topics, output_dir, parser_name, mineru_parser, workers):
    if workers == 1:
        for html_path, title in topics:
            yield parse_topic(html_path, title, output_dir, mineru_parser)
    else:
        # One parser per process; keep input order so filename collision handling
        # and the manifest are deterministic regardless of completion order.
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                 initargs=(parser_name,)) as pool:
            jobs = ((html_path, title, output_dir) for html_path, title in topics)
            yield from pool.map(parse_worker_topic, jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Extract MSD HTML into Markdown with local MinerU 4.x')
    parser.add_argument('--input-dir', type=Path, default=ROOT_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_MD_DIR)
    parser.add_argument('--parser', choices=('mineru', 'beautifulsoup'), default='mineru',
                        help='默认 mineru；beautifulsoup 仅用于显式旧流程对照')
    parser.add_argument('--limit', type=int, default=None, help='最多处理多少篇索引中的正文，用于抽样验证')
    parser.add_argument('--workers', type=int, default=1, help='原生 HTML 解析进程数；批量抽取可设为 4')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.workers < 1:
        parser.error('--workers must be positive')
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    json_folder = input_dir / 'Json'
    if not (json_folder / 'sections.json').is_file():
        parser.error(f'Missing input index: {json_folder / "sections.json"}')
    # Dependency errors must fail before writing anything, never silently fall back.
    try:
        mineru_parser = create_mineru_parser() if args.parser == 'mineru' else None
    except (RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    uuid_map = build_uuid_title_mapping(json_folder)
    html_list = sorted(path for path in input_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in {'.html', '.htm'})
    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    topics = [(path, uuid_map[path.stem.strip('{}').lower()]) for path in html_list
              if path.stem.strip('{}').lower() in uuid_map]
    skip = len(html_list) - len(topics)
    if args.limit is not None:
        topics = topics[:args.limit]
    processed = 0
    failures = []
    manifest = []
    seen_outputs = set()

    for html_path, title, content, error in iter_parsed_topics(
            topics, output_dir, args.parser, mineru_parser, args.workers):
        pure_uuid = html_path.stem.strip("{}").lower()
        processed += 1
        if error is not None:
            failures.append({'source_html': str(html_path), 'error': error})
            print(f"❌ 处理失败 {html_path.name} : {error}", flush=True)
            continue
        try:
            if len(content) < 100:
                skip += 1
                continue

            safe_name = re.sub(r'[\\/:*?"<>|]', "_", title).rstrip(' .') or pure_uuid
            # Distinct UUIDs may share a title or a sanitized filename.
            output_name = f"{safe_name}.md"
            if output_name.casefold() in seen_outputs:
                output_name = f"{safe_name}_{pure_uuid}.md"
            seen_outputs.add(output_name.casefold())
            md_file = output_dir / output_name
            markdown = render_msd_markdown(content, title)
            temp_output = md_file.with_suffix('.md.tmp')
            temp_output.write_text(markdown, encoding="utf-8")
            temp_output.replace(md_file)
            manifest.append({
                'topic_id': pure_uuid, 'title': title, 'source_html': str(html_path),
                'markdown_file': output_name, 'parser': args.parser,
                'characters': len(markdown),
            })
            success += 1
            if success % 50 == 0:
                print(f"已导出：{success} 篇", flush=True)
        except Exception as err:
            failures.append({'source_html': str(html_path), 'error': str(err)})
            print(f"❌ 处理失败 {html_path.name} : {str(err)}")

    report = {
        'parser': args.parser,
        'parser_version': version('mineru') if mineru_parser is not None else version('beautifulsoup4'),
        'input_dir': str(input_dir), 'output_dir': str(output_dir),
        'processed': processed, 'success': success, 'skipped': skip,
        'workers': args.workers,
        'documents': manifest, 'failures': failures,
    }
    (output_dir / 'extraction_manifest.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n===== 导出结束 =====")
    print(f"成功导出文档：{success}")
    print(f"跳过无效页面：{skip}")
    print(f"失败文档：{len(failures)}；输出目录：{output_dir}")
    return 1 if failures or not success else 0


if __name__ == "__main__":
    sys.exit(main())
