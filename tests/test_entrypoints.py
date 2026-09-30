"""Check CLI loading without constructing models or connecting services."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class EntrypointTests(unittest.TestCase):
    def run_python(self, *args):
        return subprocess.run(
            [sys.executable, *args], cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
        )

    def test_module_help_has_no_double_import_warning(self):
        result = self.run_python('-m', 'src.online_service.main_api', '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--port', result.stdout)
        self.assertNotIn('RuntimeWarning', result.stderr)

    def test_lazy_api_exports_remain_compatible(self):
        result = self.run_python('-c', '''
import sys
import src.online_service as services
assert 'src.online_service.main_api' not in sys.modules
from src.online_service import RAGWebAPI, create_app
from src.online_service.main_api import RAGWebAPI as actual, create_app as factory
assert RAGWebAPI is actual and create_app is factory
try:
    services.missing_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export must raise AttributeError')
''')
        self.assertEqual(result.returncode, 0, result.stderr)
