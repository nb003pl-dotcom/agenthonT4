"""Run vendored official deterministic checks; no synthetic total score."""
import sys
import hashlib
from dataclasses import asdict
from .common import ROOT, read


def evaluate(unit, answer, token_counter=None):
    sys.path.insert(0,str(ROOT/'vendor/official'))
    lock=read(ROOT/'vendor/official/VERSION_LOCK.json')
    for name,expected in lock['file_sha256'].items():
        if hashlib.sha256((ROOT/'vendor/official'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('official_code_hash_changed: '+name)
    if token_counter is None:token_counter=judge_token_counter()
    from qfbench2_track_analysis.scoring import _g1_schema
    from baselines.guardrails_example import citation_rail as rail
    task=read(unit/'task.json');corpus=rail.load_corpus(unit/'corpus')
    gate=_g1_schema({'_answer':answer})
    findings=[asdict(x) for x in rail.check_answer(answer,corpus,task['cutoff_date'],task=task)]
    claims=[asdict(x) for x in rail.check_claim_rules(answer,unit,token_counter=token_counter)]
    reasons=[asdict(x) for x in rail.check_submitted_reasons(answer,corpus,task['cutoff_date'])]
    return {'official_revision':read(ROOT/'data/official-revision.json'),'schema_gate':asdict(gate),
            'citation_findings':findings,'claim_findings':claims,'reason_findings':reasons,
            'judge_token_limit_checked':token_counter is not None,'nli':'not_run',
            'total_score':None,'reason':'Official naive and resolved gold not supplied to this run; no total score claimed.'}

from functools import lru_cache

@lru_cache(maxsize=1)
def judge_token_counter():
    from tokenizers import Tokenizer
    tokenizers=[]
    for item in read(ROOT/'data/judge-tokenizers.json'):
        path=ROOT/item['path']/'tokenizer.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['files']['tokenizer.json']:
            raise ValueError('judge_tokenizer_hash_changed')
        tok=Tokenizer.from_file(str(path));tok.no_truncation();tok.no_padding();tokenizers.append(tok)
    return lambda text:max(len(t.encode(text,add_special_tokens=False).ids) for t in tokenizers)
