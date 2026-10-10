"""核查源码仓库的本地文档链接、Git忽略规则及本地凭据值混入。

在含Git索引的源码目录运行，无GPU、数据库或模型接口调用。
默认只读；仅指定--output时写入报告。凭据检查不输出凭据值。
"""
import argparse
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PATHS = (
    '.env', '.env.production', '.env.test', 'web_demo/.env.staging',
    'config.env', 'tmp-credentials.env', '.venv/a', 'data/docs.json',
    'artifacts/a.json', 'src/models/bge-m3/model.safetensors',
    'frontend/node_modules/a', 'frontend/dist/a', 'frontend/dist-slides/a',
    'logs/a.log', 'cache/a.json', '.workbuddy/a', 'docs/pages.json',
    '__pycache__/a.pyc',
)
ALLOWED_PATHS = (
    '.env.example', 'README.md', 'requirements.txt', 'pyproject.toml',
    'MANIFEST.in', 'uv.lock', 'tests/test_quality_pilot.py',
    'scripts/verify_faq_services.py', 'scripts/check_completion_gate.py',
    'docs/faq-guard.md', 'docs/quality_pilot_20261010.json',
    'docs/faq_guard_verification_20261010.json',
    'src/models/bge-m3/tokenizer.json', 'src/models/bge-m3/config.json',
    'src/models/bge-m3/sentencepiece.bpe.model', 'frontend/public/favicon.svg',
    'scripts/audit_repository.py',
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


def audit(root=ROOT, scan_credentials=True):
    root = Path(root).resolve()
    if not root.is_dir():
        raise AuditError('root_unavailable', '源码根目录不存在或不可访问；请检查--root。')
    tracked = git(root, 'ls-files', '-z').stdout.decode('utf-8').split('\0')[:-1]
    docs = [name for name in tracked if name.lower().endswith('.md')
            and (root/name).is_file()]
    broken, checked, cache = [], 0, {}
    for name in docs:
        source = unfenced((root/name).read_text(encoding='utf-8-sig'))
        source = re.sub(r'`+[^`\n]*`+', '', source)
        for match in LINK.finditer(source):
            target = match[1].strip('<>')
            if re.match(r'\w+:', target) or target.startswith('//'):
                continue
            filename, _, fragment = target.partition('#')
            filename, fragment = unquote(filename), unquote(fragment)
            path = ((root/name).parent/filename).resolve() if filename else root/name
            checked += 1
            if not path.exists():
                broken.append({'file': name, 'target': target, 'reason': 'missing file'})
            elif fragment and path.suffix.lower() in ('.md', '.html', '.htm'):
                if path not in cache:
                    cache[path] = anchors(path)
                if fragment not in cache[path]:
                    broken.append({'file': name, 'target': target, 'reason': 'missing anchor'})

    ignore_errors = []
    for expect, paths in ((True, IGNORED_PATHS), (False, ALLOWED_PATHS)):
        for name in paths:
            result = git(root, 'check-ignore', '--no-index', '-q', name, allowed_codes=(0, 1))
            if (result.returncode == 0) != expect:
                ignore_errors.append({'path': name, 'expected_ignored': expect})
    tracked_ignored = git(root, 'ls-files', '-ci', '--exclude-standard', '-z').stdout.decode('utf-8').split('\0')[:-1]
    restricted = [name for name in tracked
                  if re.search(r'(^|/)(artifacts|data|node_modules|\.venv|\.workbuddy|\.mimosa)/', name)
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
    for name in tracked:
        if credentials and (root/name).is_file():
            content = (root/name).read_bytes()
            fields = [key for key, value in credentials.items() if value in content]
            if fields:
                secret_matches.append({'path': name, 'credential_fields': fields})
    return {'tracked_files': len(tracked), 'tracked_markdown_files': len(docs),
            'local_links_checked': checked, 'broken_links_or_anchors': broken,
            'ignore_probes': len(IGNORED_PATHS)+len(ALLOWED_PATHS),
            'ignore_errors': ignore_errors, 'tracked_ignored': tracked_ignored,
            'tracked_restricted_files': restricted,
            'credential_scan_status': credential_status,
            'credential_values_checked': len(credentials),
            'local_credential_matches': secret_matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='含Git索引的源码仓库根目录')
    parser.add_argument('--output', type=Path, help='可选JSON报告路径，建议放在artifacts目录')
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
    return int(not result['credential_scan_requirement_met'] or any(result[key] for key in (
        'broken_links_or_anchors', 'ignore_errors', 'tracked_ignored',
        'tracked_restricted_files', 'local_credential_matches')))


if __name__ == '__main__':
    raise SystemExit(main())
