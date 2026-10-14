# 用 MinerU 抽取 MSD HTML

2026-10-14追加复核：完整工程回归114项运行、113项通过、1项SDK环境跳过；独立MinerU环境11项专项全部通过。新增审计覆盖未跟踪且未忽略的文件，并修复归一化和后续入库清洗对代码空行/行尾空格的改动。历史评测与当前工程验证分开说明，见[本次核查](review_20261014.md)。

MSD 原始 HTML 现在默认使用 **MinerU 4.0.12** 的本地 Flash 原生解析。HTML 不使用 PDF OCR/VLM 推理；脚本关闭图片分析，每个进程复用同一个解析器处理文件。SDK 用法见 [官方文档](https://opendatalab.github.io/MinerU/usage/sdk_api/)。

流程为：目录 JSON 的 TopicId/Title 映射 → MSD 页面导航、作者栏和测验过滤 → MinerU HTML 解析 → 原始 Markdown → 索引文本归一化 → 父子分块、BGE-M3 向量化和 Milvus 入库。BeautifulSoup 负责抽取前 DOM 过滤和入库前 HTML 结构读取；正文抽取仍由 MinerU 完成。

## 安装独立环境

RAG 环境的 Python、Torch、Transformers 和 FastAPI 有固定版本。MinerU 使用独立环境，避免两套依赖互相覆盖。以下 PowerShell 命令在项目根目录执行：

```powershell
uv venv --python 3.12 .venv-mineru
uv pip install --python .venv-mineru/Scripts/python.exe --default-index https://pypi.org/simple -r requirements-mineru.txt
```

项目要求 Python 3.10，创建这个独立 Python 3.12 环境时 uv 可能提示版本不匹配；该环境仅用于抽取，不运行 RAG，也不在其中执行项目的 uv sync。Linux 对应的 Python 路径为 .venv-mineru/bin/python。

## 抽样与批量运行

```powershell
# 先抽样；输出不与全量语料混用
./.venv-mineru/Scripts/python.exe scripts/extract_msd.py --limit 20 --output-dir data/mineru_sample

# 完整抽取，默认输出 data/clean_md_mineru
./.venv-mineru/Scripts/python.exe scripts/extract_msd.py --workers 4

# 在原 RAG 环境里入库，明确选择新语料目录
./.venv/Scripts/python.exe scripts/run_offline_ingest.py --data-dir data/clean_md_mineru
```

也可指定 --input-dir 和 --output-dir。--workers 默认 1，设为 4 时每个进程使用独立解析器，输出顺序和重名处理仍按原始文件排序。默认输出到新目录，保留已有 data/clean_md、分块缓存和 Milvus 数据；运行抽取本身不触发向量化或数据库写入。重新入库时应按 [数据更新顺序](data_source.md#数据使用与更新) 核对目标集合。

每篇 Markdown 仍以索引中的疾病标题作为一级标题，保留正文标题层级、列表、表格以及数值和单位。过滤页面布局容器后再解析，避免正文选择器只保留某个长章节；正文中的按钮链接保留文字，原始文件位置通过来源上下文传给 MinerU。相同标题或清理文件名后发生冲突时，以 TopicId 后缀区分，不覆盖另一篇正文。extraction_manifest.json 记录 UUID、标题、原始 HTML、输出文件、解析器版本及失败明细；入库加载器只读取正文文件，忽略此 JSON。

缺少 MinerU 或版本不是 4.x 时，脚本在写入前退出；单篇解析失败或标题、数值保留检查失败会记录并继续其他文件，整批返回非零退出码。失败批次应先查看报告并处理问题，不直接入库。检查只能发现特定文字丢失，不能证明医学语义完全正确。没有静默回退。旧流程对照必须显式指定：

```powershell
./.venv/Scripts/python.exe scripts/extract_msd.py --parser beautifulsoup --output-dir data/clean_md_bs4_compare
```

## 验证边界

抽取验收要对照原文检查疾病标题、正文顺序、剂量与单位、列表、表格和导航残留。更换解析器后的语料数量与内容可能变化，需要重新分块和向量化后再评估召回及回答质量。

本次切换覆盖 MSD HTML 抽取与 Markdown 索引清洗。`DataCleaner` 自动去除 Markdown 链接目标、HTML 锚点和属性，保留可见文字、标题和列表；复杂 HTML 表格转为 Markdown 表格，跨行/跨列单元格展开，标题与单元格文字保留。上标/下标使用 `^`/`_` 表达，代码片段保留原文；表格分隔行与列表缩进不会被原来的纯符号行清理删除。归一化失败会抛错，阻止带标记原文静默入库。

原始抽取文件保留；`data/clean_md_mineru_index/` 是本次试验生成的核对副本。DocumentLoader 直接加载 PDF/DOCX/PPTX 的路径仍沿用现有实现，PDF 扫描页 OCR 仍未接入。现有字符分块仍不保证大表格完整保留在同一块内。全量清洗核对和固定 20 题重测见 [2026-10-13 索引清洗记录](mineru_index_20261013.md)。

2026-10-13 本地验证：2572 篇索引正文全部抽取成功，失败 0；跳过 1429 个非索引 HTML 页面。2572 个 UUID 与 2572 个输出文件一一对应，文件标题、字符数和原始文件位置核对通过。“自杀行为”和“膝关节脱位”各有两个不同 UUID，均保留为独立文档，避免旧流程同名覆盖。来源清单位于 data/clean_md_mineru/extraction_manifest.json。

抽取阶段曾运行 97 项回归，96 项通过、1 项因原 RAG 环境未安装 MinerU 而跳过；独立 MinerU 环境中的 11 项专项测试全部通过。索引清洗后完整工程回归运行 107 项，106 项通过、1 项为同一 SDK 环境跳过；新增用例覆盖嵌套链接、引用式链接、代码、表格跨度、缩进、科学计数和失败传播。抽取验证报告位于 artifacts/mineru_20261013/verification.json；清洗与 RAGAS 对照另存于 artifacts/mineru_normalized_20261013/。线上 Milvus 尚未用这套语料重新验收。
