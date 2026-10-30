import subprocess
from unittest.mock import patch

import pytest


@pytest.mark.parametrize('failure', [
    FileNotFoundError('git'),
    subprocess.CalledProcessError(128, ['git', 'rev-parse', 'HEAD']),
])
def test_manifest_can_be_written_without_git(failure):
    from workflow.run_manifest import build_run_manifest
    with patch('workflow.run_manifest.subprocess.run', side_effect=failure):
        manifest = build_run_manifest(profile={'name': 'typed-evidence-v2', 'sha256': 'hash'})
    assert manifest['git'] == {'commit': None, 'dirty': None, 'available': False}
    assert manifest['workflow_profile'] == 'typed-evidence-v2'
