"""Generate a final descriptor from the official fixture; never handles Team Keys."""
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
try:
    import qfbench2_common
except ImportError:
    sys.path.insert(0,str(ROOT/'vendor/official'))
from qfbench2_common.contracts.fixtures import load_fixture
from qfbench2_common.contracts.descriptor import SubmissionDescriptor,seal_descriptor_digest


def make_descriptor(image_ref,team_id,models,license_id):
    if not re.fullmatch(r'team-[0-9a-f]{32}',team_id):
        raise ValueError('Use the alias derived by the official submission alias/pack command')
    if not isinstance(models,list) or not models or any(m.get('access')!='api' for m in models):
        raise ValueError('Disclose the actual organizer House model as an api model')
    image,digest=image_ref.rsplit('@',1)
    registry,repository=image.split('/',1)
    body=load_fixture('c5/analysis_final.json')
    body.update(team_id=team_id,models=models,license=license_id,
        image={'registry':registry,'repository':repository,'digest':digest},
        image_access='public',competition_id='agenthon2026-analysis-final',phase='final',track='analysis',category='api')
    sealed=seal_descriptor_digest(body)
    SubmissionDescriptor.from_mapping(sealed)
    return sealed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True,help='Public registry/repository@sha256:... after push')
    p.add_argument('--team-id',required=True,help='Alias derived using the official toolkit')
    p.add_argument('--models',type=Path,required=True,help='JSON array describing the actual House model and cutoff')
    p.add_argument('--license',required=True,help='SPDX license you choose for your own code')
    p.add_argument('--out',type=Path,default=Path('submission.json'))
    a=p.parse_args();data=make_descriptor(a.image,a.team_id,json.loads(a.models.read_text()),a.license)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(str(a.out))


if __name__=='__main__':main()
