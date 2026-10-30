import pytest


def payload(point=0.0,lo=-10.0,hi=10.0,observations=0):
    return {'allowed_entity_ids':['A'],'evidence_ledger':{'A':{'priors':[{
        'kind':'quantitative_anchor','status':'available','point':point,'lo':lo,'hi':hi,
        'observations':observations,'method':'test anchor'}]}}}


def memo(action='increase',point=0.0,reliability='heuristic'):
    return {'reason_content':[{'entity_id':'A','baseline':{'point':point,'lo':-10.0,'hi':10.0,
        'reliability':reliability},'direction':'up','magnitude':'moderate',
        'numeric_support':'qualitative_only','point_action':action,'recommended_point':None,
        'counterevidence':'none','uncertainty_action':'retain','fact_ids':[]}]}


def rows(point):
    return [{'entity_id':'A','point_forecast':point,'interval':{'lo':-10.0,'hi':10.0}}]


def test_validator_detects_authoritative_baseline_and_action_conflicts():
    from workflow.predict_advisory_validator import advisory_issues
    issues=advisory_issues(payload(point=0.0),memo(action='increase',point=2.0),rows(0.0))
    assert {i['code'] for i in issues}=={'baseline_point_mismatch','point_action_increase_not_applied'}


def test_validator_uses_scale_aware_tolerance_and_accepts_directional_move():
    from workflow.predict_advisory_validator import advisory_issues
    assert advisory_issues(payload(point=100.0),memo(action='retain',point=100.0),rows(100.5))==[]
    assert advisory_issues(payload(point=0.0),memo(action='increase'),rows(5.0))==[]


def test_audit_retains_degraded_prediction_but_enforce_requests_repair():
    from workflow.predict_advisory_validator import validate_or_retain
    kept,audit=validate_or_retain(payload(),memo(),rows(0.0),enforce=False)
    assert kept==rows(0.0) and audit['status']=='retained_degraded_predict'
    with pytest.raises(ValueError,match='advisory_prediction_conflict'):
        validate_or_retain(payload(),memo(),rows(0.0),enforce=True)


def test_nonrepairable_memo_provenance_issue_is_warning_not_degraded_prediction():
    from workflow.predict_advisory_validator import validate_or_retain
    changed=memo(action='increase',point=2.0)
    kept,audit=validate_or_retain(payload(),changed,rows(5.0),enforce=True)
    assert kept==rows(5.0)
    assert audit['status']=='passed_with_advisory_warning'


def test_repair_allowance_reserves_later_generations_only():
    from workflow.roster_budget import advisory_repair_available
    assert advisory_repair_available(requests=20,max_requests=25,later_batches=4)
    assert not advisory_repair_available(requests=21,max_requests=25,later_batches=4)
    assert advisory_repair_available(requests=24,max_requests=25,later_batches=0)
    assert not advisory_repair_available(requests=25,max_requests=25,later_batches=0)


def test_validation_policy_degrades_when_repair_budget_is_not_spare():
    from workflow.predict_advisory_validator import validation_policy
    assert validation_policy('off',1,2)=='off'
    assert validation_policy('audit',1,2)=='audit'
    assert validation_policy('repair-if-spare',1,2)=='enforce'
    assert validation_policy('repair-if-spare',2,2)=='retain'
