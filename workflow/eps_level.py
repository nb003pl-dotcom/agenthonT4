"""Internal EPS-level interface; official percentage output is computed later."""
import copy
from .calculations import derive_eps_growth


def level_payload(payload, inputs, group):
    # The original task remains in input-contract.json. Preserve source facts,
    # permission-scoped IDs and gaps; remove the redundant percent objective.
    keep=('allowed_entity_ids','allowed_fact_ids_by_entity','facts','conflicts','gaps','evidence_ledger')
    result={k:copy.deepcopy(payload[k]) for k in keep if k in payload}
    result['forecast_request']={'variable':'quarterly GAAP diluted EPS','unit':'USD/share',
        'cutoff_date':inputs.task['cutoff_date'],'interval_level':inputs.task.get('interval_level',.9),
        'purpose':'Estimate target-quarter EPS levels; downstream code derives the official growth target.'}
    result['forecast_wire_format']='eps-level-v1'
    result['output_requirements']={'required_per_row':['entity_id','eps_forecast','eps_interval','fact_ids','reasoning'],
        'numeric_meaning':'future quarterly GAAP diluted EPS in USD/share; no model-generated percentage target',
        'entity_ids_exactly_once':list(group),'fact_ids_per_row':{'min':1,'max':3,'unique':True},
        'reasoning_max_characters':240}
    if 'evidence_ledger' in result:
        result['forecast_request']['ledger_instruction']=('Evidence ledger quantities retain their stated units and periods. '
            'An official-target percentage prior is not an EPS level. Use source-backed historical observations '
            'and explicit forecasting assumptions; output EPS levels only.')
    result['entities']=[{'entity_id':e,'name':inputs.entities[e].get('name'),
        'sector':inputs.entities[e].get('sector'),'target_quarter':inputs.entities[e].get('quarter_reported'),
        'historical_same_quarter':{'period':inputs.entities[e].get('prior_year_quarter'),
                                  'gaap_diluted_eps':inputs.entities[e].get('prior_year_q_eps')}} for e in group]
    return result


def restore_levels(data, contract):
    if not isinstance(data,dict) or not isinstance(data.get('predictions'),list):raise ValueError('predictions must be a list')
    rows=[]
    for raw in data['predictions']:
        if not isinstance(raw,dict):raise ValueError('EPS prediction must be an object')
        rows.append({k:raw[k] for k in ('entity_id','fact_ids','reasoning') if k in raw}|
                    {'underlying_eps':raw.get('eps_forecast'),'eps_interval':raw.get('eps_interval',{})})
    return derive_eps_growth({'predictions':rows},contract)
