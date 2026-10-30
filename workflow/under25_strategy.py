"""Opt-in online routing for the hard 25-request task strategy."""
from copy import deepcopy

from .common import ROOT
from .reason_content_gate import allocate_reason_batches


GENERIC_NEEDS=(
    ('target_history','historical target metric level','latest reported period before cutoff'),
    ('target_history','historical target metric change or growth','latest comparable period before cutoff'),
    ('guidance_expectation','management guidance or market expectation','latest publication before cutoff'),
    ('financial_driver','primary financial or quantitative driver','latest reported period before cutoff'),
    ('macro_market_driver','primary market or macro driver','latest observation before cutoff'),
    ('risk_counterevidence','material risk or counterevidence','latest available evidence before cutoff'),
)


def deterministic_v49_fallback(entities):
    return {'entity_queries':{eid:[list(row) for row in GENERIC_NEEDS] for eid in entities},
            '_deterministic_fallback':True}


def use_task_wide_query(config):
    return bool(config.get('under25_strategy') and not config.get('replay_from')
                and not config.get('query_plan_from'))


def query_system_prompt(config,contract):
    """Compose task-wide query instructions through family gates, never unit IDs."""
    from .roster_budget import QUERY_PROMPT,TASK_ONLY_QUERY_PROMPT
    from .reaction_evidence import enabled,query_prompt
    base=TASK_ONLY_QUERY_PROMPT if config.get('query_plan_context')=='task_only' else QUERY_PROMPT
    return query_prompt(base,False) if enabled(config,contract) else base


def prepare_task_only_query_payload(base):
    """Build a planner payload whose model-visible evidence comes only from the task."""
    from .roster_budget import query_plan_limits
    payload={key:deepcopy(base[key]) for key in ('task','target_contract','entities')}
    payload['limits']=query_plan_limits(len(payload['entities']),0)
    return payload,{'mode':'task_only','corpus_derived_inputs':0,'model_requests':0}


def prepare_query_payload(base,inputs,contract,index_path,table_inventory):
    """Attach zero-call corpus terminology evidence to the task-wide planner payload."""
    from .query_term_scan import build_query_evidence
    evidence=build_query_evidence(inputs,contract,index_path,table_inventory)
    from .roster_budget import query_plan_limits
    distinct_metrics={str(metric).casefold() for row in evidence['metric_time_inventory']
                      for metric in row.get('metric',[]) if str(metric).strip()}
    distinct_metrics.update(str(row.get('canonical','')).casefold()
                            for row in evidence['metric_vocabulary'] if str(row.get('canonical','')).strip())
    payload={**base,'metric_time_inventory':evidence['metric_time_inventory'],
             'metric_vocabulary':evidence['metric_vocabulary'],
             'table_header_inventory':(table_inventory or {}).get('tables',[]),
             'limits':query_plan_limits(len(inputs.entities),len(distinct_metrics))}
    return payload,evidence['audit']


def prediction_boundary_plan(model, payloads):
    """Allocate optional memo calls from actual upstream consumption."""
    batches=len(payloads);remaining=model.max_requests-model.requests
    if remaining<2*batches:
        return {'selected_batch_indexes':[],'decisions':[],
            'upstream_requests':model.requests,'prediction_attempt_reserve':min(remaining,2*batches),
            'reason_call_allowance':0,'projected_worst_case_requests':model.requests+max(0,remaining),
            'max_requests':model.max_requests,'within_limit':remaining>=batches,
            'full_repair_reserve':False,'hard_budget_risk':remaining<batches}
    return {**allocate_reason_batches(payloads,model.requests,model.max_requests),
        'full_repair_reserve':True,'hard_budget_risk':False}


def task_wide_query(call,entities,validator,*,reserve,requests,system=None,payload=None):
    """Spend at most one model request, then use safe deterministic templates."""
    from .roster_budget import QUERY_PROMPT,expand_templates
    before=requests()
    try:
        plan=call('query',system or QUERY_PROMPT,payload or {},validator,
            reserve=reserve,max_attempts=1)
        return plan,{'status':'model_plan','requests_before':before,'requests_after':requests()}
    except Exception as exc:
        plan=validator(deterministic_v49_fallback(entities))
        return plan,{'status':'deterministic_fallback','error':type(exc).__name__+': '+str(exc)[:500],
            'requests_before':before,'requests_after':requests()}


def enforce_reason_advisory(base_validator,payload,memo,audits):
    """Make an action conflict a validation failure, allowing Model's one repair."""
    from .predict_advisory_validator import advisory_issues,validate_or_retain
    def validate(data):
        rows=base_validator(data)
        enforce=not any(row.get('status')=='repair_required' for row in audits)
        try:
            rows,audit=validate_or_retain(payload,memo,rows,enforce=enforce)
        except ValueError:
            audits.append({'status':'repair_required','issues':advisory_issues(payload,memo,rows),
                'enforced':True})
            raise
        audits.append(audit)
        return rows
    return validate


def maybe_reason_payload(call, payload, *, reserve=0):
    """Try one strict structured memo; any failure retains direct prediction."""
    from scripts.replay_predict_reason_content import (
        memo_payload, prediction_with_structured_memo, validate_structured_memo)
    allowed=payload['allowed_fact_ids_by_entity']
    try:
        memo=call('reason_content',(ROOT/'prompts/predict-reason-content-v3.md').read_text(),
            memo_payload(payload),lambda d:validate_structured_memo(d,allowed),
            reserve=reserve,max_attempts=1)
        return prediction_with_structured_memo(payload,memo),{'status':'complete','memo':deepcopy(memo)}
    except Exception as exc:
        return payload,{'status':'direct_fallback','error':type(exc).__name__+': '+str(exc)[:500]}
