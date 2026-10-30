"""Bounded, optional EPS interval postprocessing of an already validated answer."""
from copy import deepcopy
import time
from .quarter_item_disclosure import build,parse_source
from .item_stress_interval import expand
from .evaluation import evaluate
from .stage_deadline import wall_clock_budget

MAX_SECONDS=5.0
PUBLICATION_RESERVE_SECONDS=2.0


def _check_changes(before,after):
    a=deepcopy(before);b=deepcopy(after)
    old=a['entity_predictions'];new=b['entity_predictions']
    if len(old)!=len(new):raise ValueError('stress_entity_count_changed')
    for x,y in zip(old,new,strict=True):
        lo,hi=x['interval']['lo'],x['interval']['hi']
        if not y['interval']['lo']<=lo<=hi<=y['interval']['hi']:raise ValueError('stress_interval_narrowed')
        y['interval']['lo']=lo;y['interval']['hi']=hi
    if a!=b:raise ValueError('stress_noninterval_fields_changed')


def apply(inputs,answer,contract,deadline):
    start=time.monotonic();audit={'status':'not_applicable','model_requests':0,'mode':'bounded'}
    if contract['family'] not in ('eps_yoy_direction','eps_growth_regression'):
        return answer,{**audit,'seconds':time.monotonic()-start},None
    remaining=deadline-time.time()-PUBLICATION_RESERVE_SECONDS
    if remaining<=0:return answer,{**audit,'status':'skipped_budget','seconds':time.monotonic()-start},None
    audit['budget_seconds']=min(MAX_SECONDS,remaining)
    try:
        with wall_clock_budget(audit['budget_seconds']):
            report=build(inputs);audit['source_report']=report
            if not report['items']:
                return answer,{**audit,'status':'no_eligible_source','seconds':time.monotonic()-start},None
            candidate,scenarios=expand(inputs,deepcopy(answer),report,source_builder=build,source_parser=parse_source,effect_mode='bounded')
            _check_changes(answer,candidate)
            checks=evaluate(inputs.path,candidate)
            if not checks['schema_gate']['passed'] or any(checks[k] for k in ('citation_findings','claim_findings','reason_findings')):
                raise ValueError('stress_official_deterministic_checks_failed')
            audit.update(status='applied' if candidate!=answer else 'unchanged',scenarios=scenarios,
                changed_entities=[x['entity_id'] for x,y in zip(answer['entity_predictions'],candidate['entity_predictions'],strict=True) if x!=y])
        return candidate,{**audit,'seconds':time.monotonic()-start},checks
    except Exception as exc:
        return answer,{**audit,'status':'fallback','error_type':type(exc).__name__,'error':str(exc)[:300],'seconds':time.monotonic()-start},None
