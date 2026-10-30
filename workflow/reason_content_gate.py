"""Task-independent routing for the optional structured reason-content stage."""
import math


def _number(value):
    return type(value) in (int,float) and math.isfinite(value)


def _entity_signal(ledger,*,target_type):
    if not isinstance(ledger,dict):ledger={}
    observations=ledger.get('observations',[])
    numeric=[row for row in observations if isinstance(row,dict) and _number(row.get('value'))]
    periods={row.get('period') for row in numeric if row.get('period') is not None}
    roles={row.get('evidence_role') for row in numeric if row.get('evidence_role')}
    priors=ledger.get('priors',[])
    supported_anchor=any(
        isinstance(row,dict) and row.get('kind')=='quantitative_anchor'
        and row.get('status')=='available' and type(row.get('observations')) is int
        and row['observations']>=2 and _number(row.get('point'))
        for row in priors)
    comparison_baseline=any(
        isinstance(row,dict) and row.get('kind')=='comparison_baseline' and _number(row.get('value'))
        for row in priors)
    comparison_transform=(target_type=='classification' and comparison_baseline and len(numeric)>=2 and
        bool(roles & {'target_history','historical_baseline','financial_driver','guidance_expectation'}))
    codes=[]
    if target_type=='classification' and supported_anchor and len(numeric)>=2:
        codes.append('supported_quantitative_anchor_with_updates')
    if comparison_transform:codes.append('comparison_baseline_with_numeric_history')
    return {'eligible':bool(codes),'reason_codes':codes,'numeric_observations':len(numeric),
        'distinct_periods':len(periods),'supported_anchor':supported_anchor,
        'comparison_baseline':comparison_baseline,'conflicts':len(ledger.get('conflicts',[])),
        'missing_inputs':len(ledger.get('missing_inputs',[]))}


def assess_reason_need(payload):
    entity_ids=payload.get('allowed_entity_ids',[]) if isinstance(payload,dict) else []
    ledgers=payload.get('evidence_ledger',{}) if isinstance(payload,dict) else {}
    contract=payload.get('target_contract',{}) if isinstance(payload,dict) else {}
    target_type=(contract.get('target') or {}).get('type')
    if target_type is None:target_type=((payload.get('task') or {}).get('target') or {}).get('type')
    signals={entity:_entity_signal(ledgers.get(entity,{}),target_type=target_type) for entity in entity_ids}
    eligible=sum(signal['eligible'] for signal in signals.values())
    required=math.ceil(len(entity_ids)/2) if entity_ids else 1
    trigger=bool(entity_ids) and eligible>=required
    score=(eligible/len(entity_ids) if entity_ids else 0.0)+sum(
        min(signal['numeric_observations'],12) for signal in signals.values())/1200
    return {'trigger':trigger,'eligible_entities':eligible,'required_entities':required,
        'entity_count':len(entity_ids),'score':score,'entity_signals':signals,
        'policy':'generic-evidence-anchor-ambiguity-v1'}


def allocate_reason_batches(payloads,upstream_requests,max_requests=25):
    if any(type(value) is not int or value<0 for value in (upstream_requests,max_requests)):
        raise ValueError('request counts must be nonnegative integers')
    assessments=[assess_reason_need(payload) for payload in payloads]
    prediction_reserve=2*len(payloads)
    if upstream_requests+prediction_reserve>max_requests:
        raise ValueError('prediction repair reserve exceeds request budget')
    slots=max_requests-upstream_requests-prediction_reserve
    ranked=sorted(((-row['score'],index) for index,row in enumerate(assessments) if row['trigger']))
    selected={index for _,index in ranked[:slots]}
    decisions=[]
    for index,row in enumerate(assessments):
        decisions.append({**row,'batch_index':index,'budget_selected':index in selected,
            'decision':'reason_then_predict' if index in selected else 'direct_predict',
            'budget_reason':('selected_with_reserved_predict_repairs' if index in selected else
                'evidence_gate_rejected' if not row['trigger'] else 'no_spare_reason_call')})
    projected=upstream_requests+prediction_reserve+len(selected)
    return {'selected_batch_indexes':sorted(selected),'decisions':decisions,
        'upstream_requests':upstream_requests,'prediction_attempt_reserve':prediction_reserve,
        'reason_call_allowance':slots,'projected_worst_case_requests':projected,
        'max_requests':max_requests,'within_limit':projected<=max_requests}
