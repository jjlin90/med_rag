"""核查源码工作区的本地文档链接、Git忽略规则及本地凭据值混入。

在含Git索引的源码目录运行，无GPU、数据库或模型接口调用。
默认只读；仅指定--output时写入报告。凭据检查不输出凭据值。
检查已跟踪与未跟踪且未忽略的文件，不需要先暂存新文件。
--require-tracked-links额外要求已跟踪Markdown使用可移植的本地链接，文件目标在Git索引中。
"""
import argparse
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PATHS = (
    '.env', '.env.production', '.env.test', 'web_demo/.env.staging',
    'config.env', 'tmp-credentials.env', '.venv/a', 'data/docs.json',
    'artifacts/a.json', 'src/models/bge-m3/model.safetensors',
    'frontend/node_modules/a', 'frontend/dist/a', 'frontend/dist-slides/a',
    'logs/a.log', 'cache/a.json', '.workbuddy/a', 'docs/pages.json',
    '__pycache__/a.pyc',
    'output/pdf/a.pdf', 'tmp/a.md', '.venv-mineru/a',
)
ALLOWED_PATHS = (
    '.env.example', 'README.md', 'requirements.txt', 'pyproject.toml',
    'MANIFEST.in', 'uv.lock', 'tests/test_quality_pilot.py',
    'scripts/verify_faq_services.py', 'scripts/check_completion_gate.py',
    'docs/faq-guard.md', 'docs/quality_pilot_20261010.json',
    'docs/faq_guard_verification_20261010.json',
    'docs/quality_focus_20261011.json',
    'src/models/bge-m3/tokenizer.json', 'src/models/bge-m3/config.json',
    'src/models/bge-m3/sentencepiece.bpe.model', 'frontend/public/favicon.svg',
    'scripts/audit_repository.py',
    'requirements-mineru.txt', 'docs/mineru.md', 'docs/mineru_index_20261013.md',
    'docs/review_20261012.md', 'src/offline_pipeline/markdown_normalizer.py',
    'tests/test_intent_data.py', 'tests/test_markdown_normalizer.py',
    'tests/test_mineru_extraction.py', 'tests/test_repository_audit.py',
    'docs/review_20261014.md',
)
LINK = re.compile(r'!?\[[^\]]*\]\((<[^>]+>|(?:[^\s()]|\([^()]*\))+)(?:\s+["\x27][^\n]*?["\x27])?\)')
RESTRICTED_SUFFIXES = {
    '.pyc', '.log', '.docx', '.safetensors', '.pt', '.pth', '.gguf',
    '.bin', '.onnx', '.h5', '.msgpack', '.tflite',
}


class AuditError(RuntimeError):
    """只携带审计器定义的安全错误分类与说明，不携带输入内容。"""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class LocalLinkError(ValueError):
    """只携带固定的本地链接格式说明，不回显底层异常。"""


def git(root, *arguments, allowed_codes=(0,)):
    try:
        result = subprocess.run(['git', *arguments], cwd=root, capture_output=True)
    except OSError:
        raise AuditError('git_unavailable', '无法启动Git；请检查Git安装及执行权限。') from None
    if result.returncode not in allowed_codes:
        raise AuditError('git_check_failed', 'Git核查失败；请确认--root指向可访问的Git源码工作区。')
    return result


def unfenced(source):
    return re.sub(r'(?ms)^ {0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^ {0,3}(?P=fence)[ \t]*$', '', source)


def anchors(path):
    source = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.md':
        source = unfenced(source)
    result = set(re.findall(r'\b(?:id|name)=["\x27]([^"\x27]+)', source))
    if path.suffix.lower() != '.md':
        return result
    counts = {}
    for heading in re.findall(r'^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$', source, re.M):
        heading = html.unescape(re.sub(r'<[^>]+>', '', heading))
        heading = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', heading)
        heading = heading.replace('`', '').replace('*', '').lower()
        slug = ''.join(char for char in heading
                       if char in ' _-' or unicodedata.category(char)[0] in 'LNM')
        slug = slug.replace(' ', '-')
        count = counts.get(slug, 0)
        result.add(slug if not count else f'{slug}-{count}')
        counts[slug] = count + 1
    return result


def local_link_target(document, target):
    """解析本地目标；网络/其他协议返回None，不访问远程文件系统。"""
    file_uri = target.lower().startswith('file:')
    if file_uri:
        uri = urlsplit(target)
        if (uri.netloc.lower() not in ('', 'localhost') or uri.query
                or not uri.path.startswith('/')):
            raise LocalLinkError('unsupported file URI')
        filename, fragment = unquote(uri.path), unquote(uri.fragment)
        if re.match(r'^/[A-Za-z]:', filename):
            filename = filename[1:]
    else:
        # Recognize Windows drive paths before the generic URI scheme rule.
        if not re.match(r'^[A-Za-z]:', target) and (
                re.match(r'[A-Za-z][A-Za-z0-9+.-]*:', target) or target.startswith('//')):
            return None
        filename, _, fragment = target.partition('#')
        filename, fragment = unquote(filename), unquote(fragment)
    if filename.startswith(('\\\\', '//')):
        raise LocalLinkError('unsupported network filesystem path')
    windows_drive = bool(re.match(r'^[A-Za-z]:', filename))
    if windows_drive and sys.platform != 'win32':
        raise LocalLinkError('Windows local path unavailable on this host')
    absolute = file_uri or windows_drive or Path(filename).is_absolute() or filename.startswith(('\\', '/'))
    path = (document.parent/filename).resolve() if filename else document.resolve()
    return path, fragment, absolute


def audit_document_links(root, docs, tracked=None):
    """两审计共用的本地链接检查；tracked=None时不核对Git索引归属。"""
    root = Path(root).resolve()
    tracked_names = {Path(name) for name in tracked} if tracked is not None else None
    broken, checked, cache = [], 0, {}
    nonportable_targets, external_skipped = [], 0
    untracked_targets, tracked_file_links = [], 0
    for name in docs:
        check_portability = tracked_names is None or Path(name) in tracked_names
        source = unfenced((root/name).read_text(encoding='utf-8-sig'))
        source = re.sub(r'`+[^`\n]*`+', '', source)
        for match in LINK.finditer(source):
            target = match[1].strip('<>')
            try:
                local = local_link_target(root/name, target)
            except ValueError as exc:
                checked += 1
                reason = str(exc) if isinstance(exc, LocalLinkError) else 'invalid local link'
                broken.append({'file': name, 'target': target, 'reason': reason})
                continue
            if local is None:
                external_skipped += 1
                continue
            path, fragment, absolute = local
            checked += 1
            portability_reason = ('absolute local path' if absolute else
                                  'backslash in local path' if '\\' in unquote(target.partition('#')[0]) else None)
            if check_portability and portability_reason:
                nonportable_targets.append({'file': name, 'target': target, 'reason': portability_reason})
            if not path.exists():
                broken.append({'file': name, 'target': target, 'reason': 'missing file'})
                continue
            if tracked_names is not None and Path(name) in tracked_names and path.is_file():
                tracked_file_links += 1
                try:
                    relative_target = path.relative_to(root)
                except ValueError:
                    reason = 'outside repository'
                else:
                    reason = None if relative_target in tracked_names else 'not in Git index'
                if reason:
                    untracked_targets.append({'file': name, 'target': target, 'reason': reason})
            if fragment and path.suffix.lower() in ('.md', '.html', '.htm'):
                if path not in cache:
                    cache[path] = anchors(path)
                if fragment not in cache[path]:
                    broken.append({'file': name, 'target': target, 'reason': 'missing anchor'})

    return {'local_links_checked': checked, 'broken_links_or_anchors': broken,
            'portability_scope': 'all_scanned_markdown' if tracked_names is None else 'tracked_markdown_only',
            'external_links_skipped': external_skipped,
            'tracked_file_links_checked': tracked_file_links,
            'untracked_link_targets': untracked_targets,
            'nonportable_link_targets': nonportable_targets}


def audit(root=ROOT, scan_credentials=True):
    root = Path(root).resolve()
    if not root.is_dir():
        raise AuditError('root_unavailable', '源码根目录不存在或不可访问；请检查--root。')
    tracked = git(root, 'ls-files', '-z').stdout.decode('utf-8').split('\0')[:-1]
    untracked = git(root, 'ls-files', '--others', '--exclude-standard', '-z').stdout.decode('utf-8').split('\0')[:-1]
    worktree = sorted(set(tracked + untracked))
    docs = [name for name in worktree if name.lower().endswith('.md')
            and (root/name).is_file()]
    links = audit_document_links(root, docs, tracked)

    ignore_errors = []
    for expect, paths in ((True, IGNORED_PATHS), (False, ALLOWED_PATHS)):
        for name in paths:
            result = git(root, 'check-ignore', '--no-index', '-q', name, allowed_codes=(0, 1))
            if (result.returncode == 0) != expect:
                ignore_errors.append({'path': name, 'expected_ignored': expect})
    tracked_ignored = git(root, 'ls-files', '-ci', '--exclude-standard', '-z').stdout.decode('utf-8').split('\0')[:-1]
    restricted = [name for name in worktree
                  if re.search(r'(^|/)(artifacts|data|node_modules|\.venv(?:-mineru)?|\.workbuddy|\.mimosa|tmp)/|^output/pdf/', name)
                  or Path(name).suffix.lower() in RESTRICTED_SUFFIXES
                  or Path(name).name.startswith('.env') and Path(name).name != '.env.example']

    credentials, credential_status = {}, 'skipped_by_option'
    env_path = root/'.env'
    if scan_credentials and not env_path.is_file():
        credential_status = 'skipped_no_local_env'
    elif scan_credentials:
        try:
            from dotenv import dotenv_values
        except ImportError:
            raise AuditError('dotenv_dependency_missing', '本地.env凭据核查需要python-dotenv；请安装项目依赖，或明确使用--skip-local-credentials。') from None
        for key, value in dotenv_values(env_path, interpolate=False).items():
            if (re.search(r'API_KEY|PASSWORD|TOKEN|SECRET', key, re.I)
                    and value and len(value) >= 8 and '${' not in value
                    and not re.fullmatch(r'(?:your[_-].*|.*placeholder.*|changeme|\*+)', value, re.I)):
                credentials[key] = value.encode('utf-8')
        credential_status = 'checked_local_literal_values'
    secret_matches = []
    for name in worktree:
        if credentials and (root/name).is_file():
            content = (root/name).read_bytes()
            fields = [key for key, value in credentials.items() if value in content]
            if fields:
                secret_matches.append({'path': name, 'credential_fields': fields})
    return {'scope': 'tracked_and_untracked_nonignored_worktree_files',
            'tracked_files': len(tracked), 'untracked_files': len(untracked),
            'worktree_files': len(worktree), 'markdown_files': len(docs),
            'tracked_markdown_files': sum(name in tracked for name in docs),
            **links,
            'ignore_probes': len(IGNORED_PATHS)+len(ALLOWED_PATHS),
            'ignore_errors': ignore_errors, 'tracked_ignored': tracked_ignored,
            'tracked_restricted_files': [name for name in restricted if name in tracked],
            'untracked_restricted_files': [name for name in restricted if name in untracked],
            'credential_scan_status': credential_status,
            'credential_values_checked': len(credentials),
            'local_credential_matches': secret_matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='含Git索引的源码仓库根目录')
    parser.add_argument('--output', type=Path, help='可选JSON报告路径，建议放在artifacts目录')
    parser.add_argument('--require-tracked-links', action='store_true',
                        help='发布门控：已跟踪Markdown须使用可移植的本地链接，文件目标须在Git索引中，否则退出1')
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument('--skip-local-credentials', action='store_true', help='明确跳过本地.env凭据值比对，报告标明未检查')
    credentials.add_argument('--require-local-credentials', action='store_true', help='CI门控：要求实际完成本地.env扫描且至少比对1个有效字面值，否则退出1')
    args = parser.parse_args()
    try:
        result = audit(args.root, not args.skip_local_credentials)
    except AuditError as exc:
        print(f'核查无法完成 [{exc.code}]：{exc}', file=sys.stderr)
        return 2
    except UnicodeError:
        print('核查无法完成 [file_encoding_error]：文件不是可读取的UTF-8文本。', file=sys.stderr)
        return 2
    except OSError:
        print('核查无法完成 [file_read_error]：文件读取失败；请检查文件访问权限。', file=sys.stderr)
        return 2
    except RuntimeError:
        print('核查无法完成 [audit_runtime_error]：审计执行失败。', file=sys.stderr)
        return 2
    result['credentials_required'] = args.require_local_credentials
    result['tracked_links_required'] = args.require_tracked_links
    result['tracked_link_requirement_met'] = (
        not args.require_tracked_links
        or not result['untracked_link_targets'] and not result['nonportable_link_targets']
    )
    result['credential_scan_requirement_met'] = (
        not args.require_local_credentials
        or result['credential_scan_status'] == 'checked_local_literal_values'
        and result['credential_values_checked'] > 0
    )
    report = json.dumps(result, ensure_ascii=False, indent=2)
    print(report)
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding='utf-8')
        except OSError:
            print('报告写入失败 [report_write_error]：请指定可写的输出路径。', file=sys.stderr)
            return 2
    return int(not result['credential_scan_requirement_met']
               or not result['tracked_link_requirement_met'] or any(result[key] for key in (
        'broken_links_or_anchors', 'ignore_errors', 'tracked_ignored',
        'tracked_restricted_files', 'untracked_restricted_files', 'local_credential_matches')))


if __name__ == '__main__':
    raise SystemExit(main())
