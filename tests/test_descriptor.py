import pytest


def test_final_descriptor_uses_official_seal_and_rejects_mutation():
 from tools.make_descriptor import make_descriptor
 from qfbench2_common.contracts.descriptor import SubmissionDescriptor
 from qfbench2_common.contracts.errors import ContractError
 models=[{'name':'test-house','version':'pinned-test-version','training_cutoff':'2025-01','access':'api','revision':'test-revision'}]
 result=make_descriptor('ghcr.io/test-org/under25@sha256:'+'a'*64,'team-'+'1'*32,models,'Apache-2.0')
 parsed=SubmissionDescriptor.from_mapping(result)
 assert parsed.track=='analysis' and parsed.phase=='final' and result['category']=='api'
 assert result['competition_id']=='agenthon2026-analysis-final' and result['image_access']=='public'
 result['license']='MIT'
 with pytest.raises(ContractError):SubmissionDescriptor.from_mapping(result)


@pytest.mark.parametrize('models',[[],[{'access':'local'}]])
def test_model_disclosure_cannot_claim_model_free_or_local(models):
 from tools.make_descriptor import make_descriptor
 with pytest.raises(ValueError):make_descriptor('ghcr.io/test-org/under25@sha256:'+'a'*64,'team-'+'1'*32,models,'MIT')
