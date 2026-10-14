"""Audit new worktree files without staging them or scanning ignored artifacts."""
import subprocess
import tempfile
from pathlib import Path
import unittest

from scripts.audit_repository import audit


class RepositoryAuditTests(unittest.TestCase):
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
