from types import SimpleNamespace
from pathlib import Path


def _eligible(entity_id):
    return {
        'task': {'target': {'type': 'classification'}},
        'target_contract': {'target': {'type': 'classification'}},
        'allowed_entity_ids': [entity_id],
        'allowed_fact_ids_by_entity': {entity_id: []},
        'evidence_ledger': {entity_id: {
            'priors': [{'kind': 'comparison_baseline', 'value': 1.0}],
            'observations': [
                {'value': 1.0, 'period': 'P1', 'evidence_role': 'target_history'},
                {'value': 2.0, 'period': 'P2', 'evidence_role': 'target_history'},
            ], 'conflicts': [], 'missing_inputs': []}},
    }


def test_under25_task_wide_query_is_opt_in_and_excludes_frozen_routes():
    from workflow.under25_strategy import use_task_wide_query
    assert use_task_wide_query({'under25_strategy': True}) is True
    assert use_task_wide_query({'under25_strategy': True, 'replay_from': 'x'}) is False
    assert use_task_wide_query({'under25_strategy': True, 'query_plan_from': 'x'}) is False
    assert use_task_wide_query({}) is False


def test_prediction_boundary_uses_actual_calls_and_reserves_two_calls_per_batch():
    from workflow.under25_strategy import prediction_boundary_plan
    model = SimpleNamespace(requests=18, max_requests=25)
    result = prediction_boundary_plan(model, [_eligible('A'), _eligible('B'), _eligible('C')])
    assert result['selected_batch_indexes'] == [0]
    assert result['upstream_requests'] == 18
    assert result['prediction_attempt_reserve'] == 6
    assert result['projected_worst_case_requests'] == 25


def test_prediction_boundary_skips_reason_for_regression():
    from workflow.under25_strategy import prediction_boundary_plan
    row = _eligible('A')
    row['task']['target']['type'] = 'regression'
    row['target_contract']['target']['type'] = 'regression'
    result = prediction_boundary_plan(SimpleNamespace(requests=20, max_requests=25), [row])
    assert result['selected_batch_indexes'] == []


def test_reason_failure_falls_back_to_original_payload():
    from workflow.under25_strategy import maybe_reason_payload
    calls = []
    def failed_call(*args, **kwargs):
        calls.append(args[0])
        raise ValueError('bad memo')
    payload = _eligible('A')
    final, audit = maybe_reason_payload(failed_call, payload)
    assert final is payload
    assert calls == ['reason_content']
    assert audit['status'] == 'direct_fallback'


def test_advisory_validator_enforces_conflict_then_accepts_unique_repair():
    import pytest
    from workflow.under25_strategy import enforce_reason_advisory
    payload = _eligible('A')
    payload['evidence_ledger']['A']['priors'].append({
        'kind': 'quantitative_anchor', 'status': 'available', 'point': 1.0,
        'lo': 0.0, 'hi': 2.0, 'observations': 2})
    memo = {'reason_content': [{'entity_id':'A','baseline':{'point':1.0,'lo':0.0,'hi':2.0,
        'reliability':'calibrated'},'direction':'up','magnitude':'small',
        'numeric_support':'derived_or_direct','point_action':'increase','recommended_point':2.0,
        'counterevidence':'none','uncertainty_action':'retain','fact_ids':[]}]}
    audits=[]
    validator=enforce_reason_advisory(lambda data:data['predictions'],payload,memo,audits)
    with pytest.raises(ValueError,match='advisory_prediction_conflict'):
        validator({'predictions':[{'entity_id':'A','point_forecast':1.0}]})
    accepted=validator({'predictions':[{'entity_id':'A','point_forecast':1.0}]})
    assert accepted[0]['point_forecast']==1.0
    assert [row['status'] for row in audits]==['repair_required','retained_degraded_predict']


def test_task_wide_query_failure_uses_deterministic_templates_after_one_attempt():
    from workflow.under25_strategy import task_wide_query
    from workflow.roster_budget import expand_templates
    from workflow.workflow import check_queries
    model=SimpleNamespace(requests=4,max_requests=25)
    calls=[]
    def failed_call(*args,**kwargs):
        calls.append(kwargs)
        model.requests+=1
        raise ValueError('invalid query schema')
    entities={'A':{'entity_id':'A','name':'Alpha'},'B':{'entity_id':'B','name':'Beta'}}
    validator=lambda d:check_queries(expand_templates(d,entities),list(entities),6,8)
    plan,audit=task_wide_query(failed_call,entities,validator,reserve=5,
        requests=lambda:model.requests)
    assert len(calls)==1 and calls[0]['max_attempts']==1
    assert len(plan['queries'])==12
    assert audit['status']=='deterministic_fallback'
    assert audit['requests_after']==5


def test_under25_query_payload_adds_local_scan_and_table_evidence_without_model_call(monkeypatch):
    from workflow.under25_strategy import prepare_query_payload
    calls=[]
    def fake_scan(inputs,contract,index_path,table_inventory):
        calls.append((inputs,contract,index_path,table_inventory))
        return {'metric_time_inventory':[{'metric':['revenue'],'time':['Q2 2024']}],
                'metric_vocabulary':[{'canonical':'Diluted EPS','types':['table_header']}],
                'audit':{'status':'complete','model_requests':0}}
    monkeypatch.setattr('workflow.query_term_scan.build_query_evidence',fake_scan)
    inputs=SimpleNamespace(task={'task_id':'u'},entities={'A':{}},docs={},relations={})
    base={'task':{'task_id':'u'},'target_contract':{},'entities':[{'entity_id':'A'}]}
    payload,audit=prepare_query_payload(base,inputs,{},Path('/tmp/index.gz'),{'tables':[]})
    assert payload['metric_time_inventory'][0]['metric']==['revenue']
    assert payload['metric_vocabulary'][0]['types']==['table_header']
    assert audit['model_requests']==0
    assert payload['limits']['min_queries_per_entity']==6
    assert payload['limits']['max_queries_per_entity']==8
    assert payload['limits']['min_subqueries']==6
    assert len(calls)==1


def test_task_only_query_payload_excludes_all_corpus_derived_context():
    from workflow.under25_strategy import prepare_task_only_query_payload
    base={'task':{'task_id':'synthetic'},'target_contract':{'family':'event_reaction'},
          'entities':[{'entity_id':'A','name':'Alpha'}],
          'corpus_catalog':[{'doc_id':'secret'}],
          'metric_time_inventory':[{'metric':['leak']}],
          'metric_vocabulary':[{'canonical':'leak'}],
          'table_header_inventory':[{'headers':['leak']}]}
    payload,audit=prepare_task_only_query_payload(base)
    assert set(payload)=={'task','target_contract','entities','limits'}
    assert payload['task']==base['task']
    assert payload['limits']['min_queries_per_entity']==6
    assert audit=={'mode':'task_only','corpus_derived_inputs':0,'model_requests':0}


def test_task_only_query_prompt_does_not_reference_scan_or_corpus_inputs():
    from workflow.under25_strategy import query_system_prompt
    prompt=query_system_prompt({'reaction_evidence':True,'query_plan_context':'task_only'},
                               {'family':'post_earnings_reaction'}).casefold()
    for forbidden in ('metric_time_inventory','metric_vocabulary','table_header_inventory',
                      'corpus_catalog','scanned terms','series_code'):
        assert forbidden not in prompt
    assert 'derive every retrieval need only from the task' in prompt
    assert 'expectation proxy' in prompt
    assert 'at least three metrics per entity' in prompt
    assert 'products, services, or segments' in prompt
    assert 'at most one target-history row' in prompt
    assert 'latest already disclosed consolidated revenue' in prompt
    assert 'consolidated operating income' in prompt
    assert 'volume and price or unit-economics' in prompt
    assert 'quarter being announced' in prompt
    assert 'do not spend a row restating' in prompt
    assert 'level and year-over-year change' in prompt
    assert 'previous earnings release' in prompt
    assert 'separate rows for volume and price' in prompt


def test_v49_fallback_emits_six_explicit_targets_per_entity_after_one_attempt():
    from workflow.under25_strategy import task_wide_query
    from workflow.roster_budget import expand_templates, query_plan_limits
    model=SimpleNamespace(requests=4,max_requests=25)
    entities={'A':{'entity_id':'A','name':'Alpha'},'B':{'entity_id':'B','name':'Beta'}}
    limits=query_plan_limits(2,distinct_metric_count=0)
    def failed_call(*args,**kwargs):
        model.requests+=1
        raise ValueError('invalid query schema')
    validator=lambda d:expand_templates(d,entities,limits=limits)
    plan,audit=task_wide_query(failed_call,entities,validator,reserve=5,
        requests=lambda:model.requests)
    assert len(plan['queries'])==12
    assert all(q['targets']==[{'metric':q['metric'],'time':q['period']}]
               for q in plan['queries'])
    assert audit['status']=='deterministic_fallback'


def test_under25_query_prompt_is_task_first_and_treats_scan_as_incomplete_context():
    from workflow.roster_budget import QUERY_PROMPT
    lowered=QUERY_PROMPT.casefold()
    assert 'task-first' in lowered
    assert 'measure, operation' in lowered
    assert 'metric_time_inventory' in QUERY_PROMPT
    assert 'table_header' in QUERY_PROMPT
    assert 'incomplete' in lowered


def test_task_wide_expander_preserves_metric_and_period_per_entity():
    from workflow.roster_budget import expand_templates
    entities={'A':{'name':'Alpha'}}
    result=expand_templates({'entity_queries':{'A':[
        ['target_history','GAAP diluted EPS growth','Q2 2024 versus Q2 2023'],
        ['guidance_expectation','revenue guidance','target quarter, published before cutoff'],
        ['risk_counterevidence','foreign exchange impact','latest reported quarter'],
        ['financial_driver','gross margin','latest reported quarter'],
        ['macro_market_driver','market demand','latest observation before cutoff'],
        ['definition_baseline','weighted average diluted shares','latest reported quarter'],
    ]}},entities)
    assert len(result['queries'])==6
    assert result['wire_format']=='v49_task_first_metric_time_per_entity'
    assert result['queries'][0]['metric']=='GAAP diluted EPS growth'
    assert result['queries'][0]['period']=='Q2 2024 versus Q2 2023'


def test_v49_expander_accepts_six_to_eight_needs_and_preserves_targets():
    from workflow.roster_budget import expand_templates, query_plan_limits
    entities={'A':{'name':'Alpha'}}
    rows=[
        ['target_history','GAAP diluted EPS growth','Q2 2024 versus Q2 2023'],
        ['target_history','revenue growth','Q2 2024 versus Q2 2023'],
        ['guidance_expectation','revenue guidance','Q3 2024 guidance published before cutoff'],
        ['financial_driver','gross margin','Q2 2024'],
        ['risk_counterevidence','foreign exchange impact','Q2 2024'],
        ['definition_baseline','weighted average diluted shares','Q2 2024'],
    ]
    limits=query_plan_limits(1,distinct_metric_count=4)
    result=expand_templates({'entity_queries':{'A':rows}},entities,limits=limits)
    assert len(result['queries'])==6
    assert result['wire_format']=='v49_task_first_metric_time_per_entity'
    assert result['queries'][0]['targets']==[{
        'metric':'GAAP diluted EPS growth','time':'Q2 2024 versus Q2 2023'}]


def test_v49_expander_rejects_fewer_than_six_queries_per_entity():
    import pytest
    from workflow.roster_budget import expand_templates, query_plan_limits
    entities={'A':{'name':'Alpha'}}
    rows=[['target_history',f'metric {i}','Q2 2024'] for i in range(5)]
    with pytest.raises(ValueError,match='min_queries_per_entity=6'):
        expand_templates({'entity_queries':{'A':rows}},entities,
                         limits=query_plan_limits(1,distinct_metric_count=4))


def test_v49_expander_rejects_operation_only_metrics():
    import pytest
    from workflow.roster_budget import expand_templates
    entities={'A':{'name':'Alpha'}}
    rows=[
        ['target_history','Year-over-year growth','2023'],
        ['guidance_expectation','revenue guidance','2024'],
        ['financial_driver','operating margin','2023'],
        ['risk_counterevidence','foreign exchange exposure','2023'],
        ['macro_market_driver','consumer demand','2023'],
        ['definition_baseline','cash and equivalents','2023'],
    ]
    with pytest.raises(ValueError,match='operation-only metric'):
        expand_templates({'entity_queries':{'A':rows}},entities)


def test_v49_expander_rejects_wholly_identical_multi_entity_need_signatures():
    import pytest
    from workflow.roster_budget import expand_templates
    entities={'A':{'name':'Alpha'},'B':{'name':'Beta'}}
    rows=[
        ['target_history','revenue year-over-year change','2023'],
        ['guidance_expectation','revenue guidance','2024'],
        ['financial_driver','operating margin','2023'],
        ['risk_counterevidence','foreign exchange exposure','2023'],
        ['macro_market_driver','consumer demand','2023'],
        ['definition_baseline','cash and equivalents','2023'],
    ]
    with pytest.raises(ValueError,match='entity-specific'):
        expand_templates({'entity_queries':{'A':rows,'B':rows}},entities)


def test_v49_limits_and_prompt_port_core_planner_invariants():
    from workflow.roster_budget import QUERY_PROMPT, query_plan_limits
    limits=query_plan_limits(3,distinct_metric_count=20)
    assert limits['min_queries_per_entity']==6
    assert limits['max_queries_per_entity']==8
    assert limits['max_subqueries']==24
    lowered=QUERY_PROMPT.casefold()
    assert 'operation audit' in lowered
    assert 'publication lag' in lowered
    assert 'remaining capacity' in lowered


def test_post_earnings_task_wide_prompt_prioritizes_predictive_inputs_without_hard_outcome_ban():
    from workflow.under25_strategy import query_system_prompt
    prompt=query_system_prompt({'reaction_evidence':True},
        {'family':'post_earnings_reaction'})
    lowered=prompt.casefold()
    assert 'historical derived change' in lowered
    assert 'expectation proxy' in lowered
    assert 'remaining capacity' in lowered
    assert 'never search for the pending' not in lowered
    for leaked in ('aapl','amzn','meta','2024-02-01','positive_reaction','negative_reaction'):
        assert leaked not in lowered
