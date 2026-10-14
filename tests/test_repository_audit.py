"""Audit new worktree files without staging them or scanning ignored artifacts."""
from contextlib import redirect_stdout
import io
import json
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import audit_repository
from scripts.audit_repository import audit


class RepositoryAuditTests(unittest.TestCase):
    def run_link_gate(self, root):
        output = io.StringIO()
        args = ['audit_repository.py','--root',str(root),
                '--skip-local-credentials','--require-tracked-links']
        with patch.object(audit_repository,'IGNORED_PATHS',()), \
                patch.object(audit_repository,'ALLOWED_PATHS',()), \
                patch('sys.argv',args), redirect_stdout(output):
            code = audit_repository.main()
        return code, json.loads(output.getvalue())

    def initialize(self, root):
        subprocess.run(['git','init',str(root)],check=True,capture_output=True)
        (root/'README.md').write_text('# Existing\n',encoding='utf-8')
        (root/'.gitignore').write_text('.env\n/artifacts/\n/output/pdf/\n/tmp/\n',encoding='utf-8')
        subprocess.run(['git','add','README.md','.gitignore'],cwd=root,check=True,capture_output=True)

    def test_untracked_source_and_document_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            (root/'new.md').write_text('[bad](missing.md)\n',encoding='utf-8')
            token = 'fixture-audit-token-72843'
            (root/'.env').write_text('TEST_API_KEY='+token,encoding='utf-8')
            (root/'new.py').write_text('TOKEN = '+repr(token),encoding='utf-8')
            report = audit(root)
            self.assertEqual(report['tracked_files'],2)
            self.assertEqual(report['untracked_files'],2)
            self.assertEqual(report['worktree_files'],4)
            self.assertEqual(report['tracked_markdown_files'],1)
            self.assertEqual(report['markdown_files'],2)
            self.assertEqual(report['broken_links_or_anchors'][0]['file'],'new.md')
            self.assertEqual(report['local_credential_matches'],[
                {'path':'new.py','credential_fields':['TEST_API_KEY']}])
            staged = subprocess.run(['git','diff','--cached','--name-only'],cwd=root,
                                    check=True,capture_output=True,text=True).stdout
            self.assertNotIn('new.md',staged)
            self.assertNotIn('new.py',staged)

    def test_ignored_private_files_stay_out_of_worktree_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            (root/'new.md').write_text('[ok](README.md#existing)\n',encoding='utf-8')
            for parent in ('artifacts','output/pdf','tmp'):
                path = root/parent/'private.md'
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('[private](missing.md)\n',encoding='utf-8')
            report = audit(root,scan_credentials=False)
            self.assertEqual(report['worktree_files'],3)
            self.assertEqual(report['markdown_files'],2)
            self.assertEqual(report['broken_links_or_anchors'],[])
            self.assertEqual(report['untracked_restricted_files'],[])

    def test_tracked_document_reports_existing_untracked_file_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            (root/'extra.md').write_text('# Extra\n',encoding='utf-8')
            (root/'README.md').write_text('[extra](extra.md#extra)\n',encoding='utf-8')
            report = audit(root,scan_credentials=False)
            self.assertEqual(report['broken_links_or_anchors'],[])
            self.assertEqual(report['tracked_file_links_checked'],1)
            self.assertEqual(report['untracked_link_targets'],[
                {'file':'README.md','target':'extra.md#extra',
                 'reason':'not in Git index'}])
            subprocess.run(['git','add','extra.md'],cwd=root,check=True,capture_output=True)
            self.assertEqual(audit(root,scan_credentials=False)['untracked_link_targets'],[])

    def test_untracked_document_can_link_to_untracked_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            (root/'draft.md').write_text('[extra](extra.md#extra)\n',encoding='utf-8')
            (root/'extra.md').write_text('# Extra\n',encoding='utf-8')
            report = audit(root,scan_credentials=False)
            self.assertEqual(report['local_links_checked'],1)
            self.assertEqual(report['broken_links_or_anchors'],[])
            self.assertEqual(report['tracked_file_links_checked'],0)
            self.assertEqual(report['untracked_link_targets'],[])

    def test_absolute_tracked_file_is_checked_but_not_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            path = root/'extra.md'
            path.write_text('# Extra\n',encoding='utf-8')
            subprocess.run(['git','add','extra.md'],cwd=root,check=True,capture_output=True)
            target = path.as_posix()+'#extra'
            (root/'README.md').write_text('[extra](<'+target+'>)\n',encoding='utf-8')
            code, report = self.run_link_gate(root)
            self.assertEqual(report['local_links_checked'],1)
            self.assertEqual(report['tracked_file_links_checked'],1)
            self.assertEqual(report['broken_links_or_anchors'],[])
            self.assertEqual(report['untracked_link_targets'],[])
            self.assertEqual(report['nonportable_link_targets'],[
                {'file':'README.md','target':target,'reason':'absolute local path'}])
            self.assertFalse(report['tracked_link_requirement_met'])
            self.assertEqual(code,1)

    def test_local_file_uris_check_encoded_paths_anchors_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            path = root/'extra file.md'
            path.write_text('# Some heading\n',encoding='utf-8')
            missing = root/'missing.md'
            for uri in (path.as_uri(),path.as_uri().replace('file:///','file://localhost/')):
                with self.subTest(uri=uri):
                    target = uri+'#some-heading'
                    (root/'README.md').write_text(
                        '[extra]('+target+')\n[missing]('+missing.as_uri()+')\n',encoding='utf-8')
                    code, report = self.run_link_gate(root)
                    self.assertEqual(report['local_links_checked'],2)
                    self.assertEqual(report['tracked_file_links_checked'],1)
                    self.assertEqual(report['untracked_link_targets'],[
                        {'file':'README.md','target':target,'reason':'not in Git index'}])
                    self.assertEqual(report['broken_links_or_anchors'],[
                        {'file':'README.md','target':missing.as_uri(),'reason':'missing file'}])
                    self.assertEqual(len(report['nonportable_link_targets']),2)
                    self.assertEqual(code,1)

    def test_external_urls_are_skipped_but_unsupported_file_uris_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            targets = ('https://example.invalid/a','http://example.invalid/b',
                       'mailto:example@example.invalid','//example.invalid/a',
                       'data:text/plain,example')
            (root/'README.md').write_text('# Existing\n[self](#existing)\n'+
                '\n'.join('[external]('+target+')' for target in targets),encoding='utf-8')
            code, report = self.run_link_gate(root)
            self.assertEqual(code,0)
            self.assertEqual(report['local_links_checked'],1)
            self.assertEqual(report['external_links_skipped'],len(targets))
            self.assertEqual(report['nonportable_link_targets'],[])
            unsupported = {
                'file://server/share/private.md':'unsupported file URI',
                'file:relative.md':'unsupported file URI',
                'file:////server/share/private.md':'unsupported network filesystem path',
            }
            for uri, reason in unsupported.items():
                with self.subTest(uri=uri):
                    (root/'README.md').write_text('[file]('+uri+')\n',encoding='utf-8')
                    code, report = self.run_link_gate(root)
                    self.assertEqual(code,1)
                    self.assertEqual(report['external_links_skipped'],0)
                    self.assertEqual(report['broken_links_or_anchors'],[
                        {'file':'README.md','target':uri,'reason':reason}])

    def test_tracked_link_gate_rejects_ignored_target_then_accepts_tracked_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            (root/'tmp').mkdir()
            (root/'tmp'/'private.md').write_text('# Private\n',encoding='utf-8')
            (root/'README.md').write_text('[private](tmp/private.md)\n',encoding='utf-8')
            arguments = ['audit_repository.py','--root',str(root),'--skip-local-credentials']

            def run_cli(strict):
                output = io.StringIO()
                args = arguments + (['--require-tracked-links'] if strict else [])
                # This fixture exercises delivery checks independently of project ignore probes.
                with patch.object(audit_repository,'IGNORED_PATHS',()), \
                        patch.object(audit_repository,'ALLOWED_PATHS',()), \
                        patch('sys.argv',args), redirect_stdout(output):
                    code = audit_repository.main()
                return code, json.loads(output.getvalue())

            code, report = run_cli(False)
            self.assertEqual(code,0)
            self.assertFalse(report['tracked_links_required'])
            self.assertTrue(report['untracked_link_targets'])
            code, report = run_cli(True)
            self.assertEqual(code,1)
            self.assertFalse(report['tracked_link_requirement_met'])
            (root/'README.md').write_text('[self](README.md#existing)\n# Existing\n',encoding='utf-8')
            code, report = run_cli(True)
            self.assertEqual(code,0)
            self.assertTrue(report['tracked_link_requirement_met'])
            self.assertEqual(report['untracked_link_targets'],[])
