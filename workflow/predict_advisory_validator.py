"""Task-independent checks between a structured advisory and final predictions."""
import math

VALIDATOR_MODES={'off','audit','repair-if-spare'}


def _number(value):return type(value) in (int,float) and math.isfinite(value)


def _close(a,b,relative_tolerance=.01,absolute_tolerance=1e-6):
    if not (_number(a) and _number(b)):return False
    return abs(a-b)<=max(absolute_tolerance,relative_tolerance*max(abs(a),abs(b)))


def _authoritative_baselines(payload):
    result={}
    for entity,ledger in (payload.get('evidence_ledger') or {}).items():
        priors=ledger.get('priors',[]) if isinstance(ledger,dict) else []
        for prior in priors:
            if (isinstance(prior,dict) and prior.get('status')=='available'
                and all(_number(prior.get(key)) for key in ('point','lo','hi'))):
                result[entity]={key:prior[key] for key in ('point','lo','hi')}
                break
    return result


def advisory_issues(payload,memo,rows,*,relative_tolerance=.01,absolute_tolerance=1e-6):
    authoritative=_authoritative_baselines(payload)
    advice={row.get('entity_id'):row for row in memo.get('reason_content',[]) if isinstance(row,dict)}
    predictions={row.get('entity_id'):row for row in rows if isinstance(row,dict)}
    issues=[]
    for entity,row in advice.items():
        baseline=row.get('baseline') or {};trusted=authoritative.get(entity)
        if trusted:
            for field in ('point','lo','hi'):
                if not _close(baseline.get(field),trusted[field],relative_tolerance,absolute_tolerance):
                    issues.append({'entity_id':entity,'code':'baseline_'+field+'_mismatch',
                        'advisory':baseline.get(field),'authoritative':trusted[field]})
        reference=trusted['point'] if trusted else baseline.get('point')
        point=(predictions.get(entity) or {}).get('point_forecast')
        if not (_number(reference) and _number(point)):continue
        tolerance=max(absolute_tolerance,relative_tolerance*max(abs(reference),abs(point)))
        action=row.get('point_action')
        if action=='increase' and point<=reference+tolerance:
            issues.append({'entity_id':entity,'code':'point_action_increase_not_applied',
                'baseline':reference,'point_forecast':point,'tolerance':tolerance})
        elif action=='decrease' and point>=reference-tolerance:
            issues.append({'entity_id':entity,'code':'point_action_decrease_not_applied',
                'baseline':reference,'point_forecast':point,'tolerance':tolerance})
        elif action=='retain' and abs(point-reference)>tolerance:
            issues.append({'entity_id':entity,'code':'point_action_retain_not_applied',
                'baseline':reference,'point_forecast':point,'tolerance':tolerance})
    return issues


def validate_or_retain(payload,memo,rows,*,enforce):
    issues=advisory_issues(payload,memo,rows)
    repairable=[item for item in issues if item['code'].startswith('point_action_')]
    audit={'status':('passed' if not issues else 'repair_required' if enforce and repairable
                     else 'retained_degraded_predict' if repairable else 'passed_with_advisory_warning'),
           'issues':issues,'enforced':bool(enforce)}
    if repairable and enforce:
        compact=[{k:item[k] for k in item if k in ('entity_id','code','baseline','point_forecast','authoritative')} for item in repairable]
        raise ValueError('advisory_prediction_conflict: '+str(compact)+
            '. Correct the full prediction batch using the authoritative quantitative baseline and structured point_action. '
            'If point_action is increase/decrease, move beyond tolerance; if retain, keep the authoritative baseline.')
    return rows,audit


def validation_policy(mode,requests,max_requests,later_batches=0):
    if mode not in VALIDATOR_MODES:raise ValueError('unknown generic predict validator mode')
    if mode=='off':return 'off'
    if mode=='audit':return 'audit'
    from .roster_budget import advisory_repair_available
    return 'enforce' if advisory_repair_available(requests,max_requests,later_batches) else 'retain'
