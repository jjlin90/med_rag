"""Read-only source, local documentation link and environment dependency audit."""
import ast
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def audit():
    files = [ROOT / 'main.py']
    for directory in ('src', 'scripts', 'web_demo', 'tests'):
        files.extend(p for p in (ROOT / directory).rglob('*.py')
                     if not {'.venv', 'venv', 'node_modules', '__pycache__', 'models'} & set(p.parts))
    errors = []
    for path in files:
        try:
            compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
        except (SyntaxError, UnicodeError) as exc:
            errors.append({'path': str(path.relative_to(ROOT)), 'error': str(exc)})
    broken = []
    docs = [ROOT / 'README.md', ROOT / 'GETTING_STARTED.md',
            ROOT / 'web_demo/README.md', ROOT / 'frontend/README.md',
            *(ROOT / 'docs').rglob('*.md')]
    for path in docs:
        source = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', source):
            target = unquote(target.split('#')[0].strip('<>'))
            if not target or re.match(r'\w+:', target) or target.startswith('//'):
                continue
            if not (path.parent / target).exists():
                broken.append({'path': str(path.relative_to(ROOT)), 'target': target})
    from packaging.requirements import Requirement
    deps = []
    for dist in metadata.distributions():
        for value in dist.requires or []:
            req = Requirement(value)
            if req.marker and not req.marker.evaluate({'extra': ''}):
                continue
            try:
                installed = metadata.version(req.name)
            except metadata.PackageNotFoundError:
                deps.append(f'{dist.metadata["Name"]}: missing {req.name}')
                continue
            if req.specifier and installed not in req.specifier:
                deps.append(f'{dist.metadata["Name"]}: requires {req}, installed {installed}')
    return {'python_files': len(files), 'syntax_errors': errors,
            'markdown_files': len(docs), 'broken_local_links': broken,
            'dependency_errors': sorted(set(deps))}


if __name__ == '__main__':
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if '--output' in sys.argv:
        path = Path(sys.argv[sys.argv.index('--output') + 1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    sys.exit(bool(result['syntax_errors'] or result['broken_local_links'] or result['dependency_errors']))
