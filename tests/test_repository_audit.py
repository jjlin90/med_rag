"""Audit new worktree files without staging them or scanning ignored artifacts."""
from contextlib import redirect_stdout
import io
import json
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import audit_repository, audit_static
from scripts.audit_repository import audit


class RepositoryAuditTests(unittest.TestCase):
    def run_static_audit(self, root):
        (root/'main.py').write_text('pass\n',encoding='utf-8')
        (root/'GETTING_STARTED.md').write_text('# Getting started\n',encoding='utf-8')
        for name in ('web_demo','frontend'):
            directory = root/name
            directory.mkdir(exist_ok=True)
            (directory/'README.md').write_text('# Entry\n',encoding='utf-8')
        (root/'docs').mkdir(exist_ok=True)
        with patch.object(audit_static,'ROOT',root), \
                patch.object(audit_static.metadata,'distributions',return_value=[]), \
                patch.object(audit_static,'audit_torch_installation',return_value={'errors':[]}):
            return audit_static.audit()

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

    def test_static_and_repository_check_absolute_paths_and_file_uris_consistently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            missing = root/'missing.md'
            existing = root/'extra.md'
            existing.write_text('# Extra\n',encoding='utf-8')
            targets = (missing.as_posix(),missing.as_uri(),existing.as_uri()+'#absent')
            (root/'README.md').write_text('\n'.join('[target](<'+target+'>)' for target in targets),encoding='utf-8')
            static = self.run_static_audit(root)
            repo = audit(root,scan_credentials=False)
            self.assertTrue(any(item['target'] == missing.as_posix() for item in static['broken_local_links']))
            self.assertEqual(static['broken_local_links'],[
                {'path':item['file'],'target':item['target'],'reason':item['reason']}
                for item in repo['broken_links_or_anchors']])
            self.assertEqual(static['local_links_checked'],repo['local_links_checked'])
            self.assertEqual(static['nonportable_link_targets'],[
                {'path':item['file'],'target':item['target'],'reason':item['reason']}
                for item in repo['nonportable_link_targets']])

    def test_raw_and_encoded_backslash_paths_fail_delivery_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            path = root/'notes'/'sub'/'x.md'
            path.parent.mkdir(parents=True)
            path.write_text('# Note\n',encoding='utf-8')
            subprocess.run(['git','add','notes/sub/x.md'],cwd=root,check=True,capture_output=True)
            targets = ('notes\\sub\\x.md','notes%5Csub%5Cx.md')
            (root/'README.md').write_text('\n'.join('[note]('+target+')' for target in targets),encoding='utf-8')
            code, repo = self.run_link_gate(root)
            self.assertEqual(repo['nonportable_link_targets'],[
                {'file':'README.md','target':target,'reason':'backslash in local path'} for target in targets])
            self.assertEqual(code,1)
            static = self.run_static_audit(root)
            self.assertEqual(static['nonportable_link_targets'],[
                {'path':item['file'],'target':item['target'],'reason':item['reason']}
                for item in repo['nonportable_link_targets']])
            with patch.object(audit_static,'audit',return_value=static), \
                    patch('sys.argv',['audit_static.py']), redirect_stdout(io.StringIO()):
                self.assertEqual(audit_static.main(),1)
            (root/'README.md').write_text('[note](notes/sub/x.md#note)\n',encoding='utf-8')
            code, repo = self.run_link_gate(root)
            self.assertEqual(code,0)
            self.assertEqual(repo['nonportable_link_targets'],[])

    def test_static_and_repository_check_anchors_and_skip_code_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            (root/'README.md').write_text(
                '# Existing\n[ok](#existing)\n[bad](#missing)\n'
                '`[literal](inline-missing.md)`\n'
                '~~~markdown\n[example](fenced-missing.md)\n~~~\n'
                '[external](https://example.invalid/)\n',encoding='utf-8')
            static = self.run_static_audit(root)
            repo = audit(root,scan_credentials=False)
            self.assertEqual(static['broken_local_links'],[
                {'path':'README.md','target':'#missing','reason':'missing anchor'}])
            self.assertEqual(repo['broken_links_or_anchors'],[
                {'file':'README.md','target':'#missing','reason':'missing anchor'}])
            self.assertEqual(static['local_links_checked'],repo['local_links_checked'])
            self.assertEqual(static['external_links_skipped'],1)

    def test_untracked_draft_portability_scopes_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize(root)
            (root/'docs').mkdir()
            target = (root/'README.md').as_posix()+'#existing'
            draft = root/'docs'/'draft.md'
            draft.write_text('[readme](<'+target+'>)\n',encoding='utf-8')
            static = self.run_static_audit(root)
            code, repo = self.run_link_gate(root)
            self.assertEqual(static['portability_scope'],'all_scanned_markdown')
            self.assertEqual(repo['portability_scope'],'tracked_markdown_only')
            self.assertEqual(static['nonportable_link_targets'],[
                {'path':'docs/draft.md','target':target,'reason':'absolute local path'}])
            self.assertEqual(repo['nonportable_link_targets'],[])
            self.assertEqual(code,0)
            with patch.object(audit_static,'audit',return_value=static), \
                    patch('sys.argv',['audit_static.py']), redirect_stdout(io.StringIO()):
                self.assertEqual(audit_static.main(),1)
            subprocess.run(['git','add','docs/draft.md'],cwd=root,check=True,capture_output=True)
            code, repo = self.run_link_gate(root)
            self.assertEqual(code,1)
            self.assertEqual(repo['nonportable_link_targets'],[
                {'file':'docs/draft.md','target':target,'reason':'absolute local path'}])
            draft.write_text('[readme](../README.md#existing)\n',encoding='utf-8')
            code, repo = self.run_link_gate(root)
            self.assertEqual(code,0)
            self.assertEqual(repo['nonportable_link_targets'],[])
            self.assertEqual(self.run_static_audit(root)['nonportable_link_targets'],[])

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
