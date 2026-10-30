from copy import deepcopy


def entity(*,anchor_status=None,anchor_observations=None,comparison=None,numeric=0,roles=None):
    priors=[]
    if anchor_status is not None:
        priors.append({'kind':'quantitative_anchor','status':anchor_status,'point':0.0,
            'lo':-1.0,'hi':1.0,'observations':anchor_observations})
    if comparison is not None:
        priors.append({'kind':'comparison_baseline','value':comparison})
    roles=roles or ['target_history']
    observations=[{'value':float(i+1),'period':f'P{i}','evidence_role':roles[i%len(roles)]}
                  for i in range(numeric)]
    return {'priors':priors,'observations':observations,'derived_values':[],
        'conflicts':[],'missing_inputs':[]}


def payload(ledgers,target_type='regression',**metadata):
    return {'task':{'task_id':metadata.get('task_id','arbitrary'),'family':metadata.get('family','arbitrary')},
        'target_contract':{'target':{'type':target_type}},
        'allowed_entity_ids':list(ledgers),'evidence_ledger':ledgers}


def test_gate_accepts_supported_anchor_or_comparison_transform_for_classification_without_family_rules():
    from workflow.reason_content_gate import assess_reason_need
    calibrated=payload({'A':entity(anchor_status='available',anchor_observations=9,numeric=3)},
        target_type='classification')
    transformed=payload({'A':entity(comparison=2.0,numeric=3,
        roles=['historical_baseline','financial_driver'])},target_type='classification')

    assert assess_reason_need(calibrated)['trigger'] is True
    assert assess_reason_need(transformed)['trigger'] is True
    changed=deepcopy(transformed)
    changed['task']={'task_id':'fomc-looking-name','family':'unseen_family'}
    assert assess_reason_need(changed)==assess_reason_need(transformed)


def test_gate_rejects_uncalibrated_comparison_only_regression_magnitude():
    from workflow.reason_content_gate import assess_reason_need
    result=assess_reason_need(payload({'A':entity(comparison=2.0,numeric=3,
        roles=['historical_baseline','financial_driver'])},target_type='regression'))
    assert result['trigger'] is False
    assert result['entity_signals']['A']['comparison_baseline'] is True


def test_gate_rejects_even_supported_anchor_for_continuous_regression():
    from workflow.reason_content_gate import assess_reason_need
    result=assess_reason_need(payload({'A':entity(
        anchor_status='available',anchor_observations=9,numeric=3)},target_type='regression'))
    assert result['trigger'] is False


def test_gate_rejects_zero_observation_prior_and_insufficient_numeric_evidence():
    from workflow.reason_content_gate import assess_reason_need
    zero_observation=payload({'A':entity(anchor_status='available',anchor_observations=0,numeric=4)})
    weak_comparison=payload({'A':entity(comparison=2.0,numeric=1)})

    assert assess_reason_need(zero_observation)['trigger'] is False
    assert assess_reason_need(weak_comparison)['trigger'] is False


def test_gate_requires_at_least_half_the_batch_to_have_verifiable_anchor_ambiguity():
    from workflow.reason_content_gate import assess_reason_need
    result=assess_reason_need(payload({
        'A':entity(comparison=2.0,numeric=3),
        'B':entity(numeric=3),
        'C':entity(numeric=3),
    },target_type='classification'))
    assert result['trigger'] is False
    assert result['eligible_entities']==1
    assert result['required_entities']==2


def test_allocator_reserves_two_predict_attempts_per_batch_before_reason_calls():
    from workflow.reason_content_gate import allocate_reason_batches
    eligible=[payload({'A':entity(comparison=2.0,numeric=3)},target_type='classification') for _ in range(3)]

    roomy=allocate_reason_batches(eligible,upstream_requests=16,max_requests=25)
    assert roomy['selected_batch_indexes']==[0,1,2]
    assert roomy['projected_worst_case_requests']==25

    tight=allocate_reason_batches(eligible,upstream_requests=18,max_requests=25)
    assert tight['selected_batch_indexes']==[0]
    assert tight['projected_worst_case_requests']==25
    assert tight['decisions'][1]['budget_selected'] is False


def test_allocator_never_spends_reason_budget_on_ineligible_batch():
    from workflow.reason_content_gate import allocate_reason_batches
    batches=[payload({'A':entity(anchor_status='available',anchor_observations=0,numeric=4)}),
             payload({'A':entity(comparison=2.0,numeric=3)},target_type='classification')]
    result=allocate_reason_batches(batches,upstream_requests=20,max_requests=25)
    assert result['selected_batch_indexes']==[1]
    assert result['projected_worst_case_requests']==25
