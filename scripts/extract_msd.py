from pathlib import Path
import json
from bs4 import BeautifulSoup
import re
import argparse

# =====================【核对路径！】=====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_ROOT / 'data/raw/MSDZHConsumerMedicalTopics'
OUTPUT_MD_DIR = PROJECT_ROOT / 'data/clean_md'
# =======================================================
JSON_FOLDER = ROOT_DIR / "Json"


def build_uuid_title_mapping():
    uuid_2_title = {}
    sections_path = JSON_FOLDER / "sections.json"
    sections_data = json.loads(sections_path.read_text(encoding="utf-8-sig"))

    for section in sections_data.get("sections", []):
        # SectionId自带大括号，直接拿来拼接文件名
        section_id = section["SectionId"]
        chapter_file = JSON_FOLDER / f"{section_id}.json"
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


def main():
    global ROOT_DIR, OUTPUT_MD_DIR, JSON_FOLDER
    parser = argparse.ArgumentParser(description='Extract MSD HTML into Markdown')
    parser.add_argument('--input-dir', type=Path, default=ROOT_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_MD_DIR)
    args = parser.parse_args()
    ROOT_DIR, OUTPUT_MD_DIR = args.input_dir, args.output_dir
    JSON_FOLDER = ROOT_DIR / 'Json'
    if not (JSON_FOLDER / 'sections.json').is_file():
        parser.error(f'Missing input index: {JSON_FOLDER / "sections.json"}')
    OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)
    uuid_map = build_uuid_title_mapping()
    html_list = list(ROOT_DIR.glob("*.html"))
    success = 0
    skip = 0

    for html_path in html_list:
        stem_name = html_path.stem
        pure_uuid = stem_name.strip("{}").lower()

        if pure_uuid not in uuid_map:
            skip += 1
            continue

        title = uuid_map[pure_uuid]
        try:
            html_txt = html_path.read_text(encoding="utf-8")
            content = clean_msd_html(html_txt)
            if len(content) < 100:
                skip += 1
                continue

            safe_name = re.sub(r'[\/:*?"<>|]', "_", title)
            md_file = OUTPUT_MD_DIR / f"{safe_name}.md"
            md_file.write_text(f"# {title}\n\n{content}", encoding="utf-8")
            success += 1
            if success % 50 == 0:
                print(f"已导出：{success} 篇")
        except Exception as err:
            print(f"❌ 处理失败 {html_path.name} : {str(err)}")

    print(f"\n===== 导出结束 =====")
    print(f"成功导出文档：{success}")
    print(f"跳过无效页面：{skip}")


if __name__ == "__main__":
    main()
