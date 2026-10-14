"""Read-only source, local documentation link and environment dependency audit."""
import ast
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

if __package__:
    from .audit_repository import audit_document_links
else:
    from audit_repository import audit_document_links

ROOT = Path(__file__).resolve().parents[1]


def audit_torch_installation():
    """Check wheel records against the installed build without loading native DLLs."""
    distributions = [dist for dist in metadata.distributions()
                     if dist.metadata.get('Name', '').lower() == 'torch']
    result = {'metadata_versions': [dist.version for dist in distributions],
              'runtime_version': None, 'compiled_cuda': None, 'errors': []}
    errors = result['errors']
    try:
        from packaging.version import Version
    except ImportError as exc:
        errors.append(f'torch: build validation requires packaging ({exc}); '
                      'install the project dependencies')
        return result
    if not distributions:
        errors.append('torch: distribution is missing')
        return result
    if len(distributions) != 1:
        errors.append(f'torch: {len(distributions)} distributions found; expected one')
    for dist in distributions:
        records = {path.name: path for path in dist.files or []}
        for required in ('RECORD', 'WHEEL'):
            if (required not in records
                    or not Path(dist.locate_file(records[required])).is_file()):
                errors.append(f'torch {dist.version}: missing wheel {required}')

    version_path = Path(distributions[0].locate_file('torch/version.py'))
    try:
        values = {}
        for node in ast.parse(version_path.read_text(encoding='utf-8')).body:
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign) else [])
            for target in targets:
                if isinstance(target, ast.Name) and target.id in ('__version__', 'cuda'):
                    values[target.id] = ast.literal_eval(node.value)
        runtime = Version(values['__version__'])
        result['runtime_version'] = str(runtime)
        result['compiled_cuda'] = values.get('cuda')
        if not values.get('cuda'):
            errors.append('torch: CUDA build required for project model execution')
        for dist in distributions:
            declared = Version(dist.version)
            if (declared.base_version != runtime.base_version
                    or declared.local and declared.local != runtime.local):
                errors.append(f'torch: metadata {declared} differs from build {runtime}')
            if declared.local and declared.local.startswith('cu'):
                digits = declared.local[2:]
                if digits.isdigit():
                    expected = f'{digits[:-1]}.{digits[-1]}'
                    if values.get('cuda') != expected:
                        errors.append(f'torch {declared}: compiled CUDA is {values.get("cuda")}, '
                                      f'expected {expected}')
    except (OSError, SyntaxError, ValueError, KeyError) as exc:
        errors.append(f'torch: cannot verify build file: {exc}')
    return result


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
    docs = [ROOT / 'README.md', ROOT / 'GETTING_STARTED.md',
            ROOT / 'web_demo/README.md', ROOT / 'frontend/README.md',
            *(ROOT / 'docs').rglob('*.md')]
    links = audit_document_links(ROOT, [path.relative_to(ROOT).as_posix() for path in docs])
    broken = [{'path': item['file'], 'target': item['target'], 'reason': item['reason']}
              for item in links['broken_links_or_anchors']]
    deps = []
    try:
        from packaging.requirements import Requirement
    except ImportError as exc:
        deps.append(f'audit_static: dependency validation requires packaging ({exc}); '
                    'install the project dependencies')
    else:
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
    torch_build = audit_torch_installation()
    return {'python_files': len(files), 'syntax_errors': errors,
            'python_scope': 'main.py plus src/scripts/web_demo/tests, excluding model and runtime directories; includes untracked files',
            'markdown_files': len(docs), 'broken_local_links': broken,
            'local_links_checked': links['local_links_checked'],
            'external_links_skipped': links['external_links_skipped'],
            'nonportable_link_targets': links['nonportable_link_targets'],
            'markdown_scope': 'root and frontend/web_demo READMEs, GETTING_STARTED and docs/**/*.md; includes untracked files, excludes model READMEs',
            'link_check_scope': 'shared local path, file URI, anchor and portability checks; Git index membership is checked separately by audit_repository',
            'dependency_errors': sorted(set(deps)), 'torch_build': torch_build,
            'installation_errors': torch_build['errors']}


def main():
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if '--output' in sys.argv:
        path = Path(sys.argv[sys.argv.index('--output') + 1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return int(bool(result['syntax_errors'] or result['broken_local_links']
                    or result['nonportable_link_targets']
                    or result['dependency_errors'] or result['installation_errors']))


if __name__ == '__main__':
    raise SystemExit(main())
