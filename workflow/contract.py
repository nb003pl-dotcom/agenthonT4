"""Task-owned prediction definitions, independent of resolved outcomes."""
from datetime import date
import re

FAMILIES = {
    'eps_beat_consensus': ('USD_per_share', ['target_history','guidance_expectation','definition_baseline','risk_counterevidence']),
    'eps_yoy_direction': ('USD_per_share', ['target_history','guidance_expectation','financial_driver','risk_counterevidence']),
    'eps_growth_regression': ('percent', ['target_history','guidance_expectation','financial_driver','risk_counterevidence']),
    'auction_demand': ('ratio', ['target_history','macro_market_driver','definition_baseline']),
    'positioning_shift': ('pct_of_open_interest', ['target_history','macro_market_driver','definition_baseline']),
    'cpi_component_nowcast': ('percent', ['target_history','macro_market_driver','definition_baseline']),
    'rate_curve_cross_section': ('bps_change', ['target_history','macro_market_driver','definition_baseline']),
    'macro_revision_direction': (None, ['target_history','definition_baseline','risk_counterevidence']),
    'post_earnings_reaction': ('percent', ['target_history','guidance_expectation','financial_driver','risk_counterevidence']),
    'credit_event': ('probability', ['financial_driver','risk_counterevidence','definition_baseline']),
}

def clean_task(task):
    # The shipped rubric describes the retired 80% admission gate. It must not
    # teach the model that a source must prove a future outcome. Raw input is logged.
    # notes can carry an unfamiliar task's target definition, so preserve them.
    return {k:v for k,v in task.items() if k not in ('entities','faithfulness_rubric')}

def _specified(*candidates):
    """First explicit nonempty value with its original field path; zero is valid."""
    for value, source in candidates:
        if value is not None and value != '' and value != [] and value != {}:
            return value, source
    return None, 'not_specified'

def _target_definition(task, entity, default_unit, default_basis):
    target=task['target'];local=entity.get('target')
    local=local if isinstance(local,dict) else {}
    unit,unit_source=_specified((local.get('unit'),'entity.target.unit'),
        (entity.get('target_unit'),'entity.target_unit'),(target.get('unit'),'task.target.unit'),
        (target.get('units'),'task.target.units'),(task.get('target_unit'),'task.target_unit'),
        (entity.get('unit'),'entity.unit'),(entity.get('units'),'entity.units'),
        (default_unit,'family_default'))
    period,period_source=_specified((local.get('period'),'entity.target.period'),
        (entity.get('target_period'),'entity.target_period'),(target.get('period'),'task.target.period'),
        (target.get('target_period'),'task.target.target_period'),(task.get('target_period'),'task.target_period'),
        (entity.get('quarter_reported'),'entity.quarter_reported'),(entity.get('ref_month'),'entity.ref_month'),
        (entity.get('event_window'),'entity.event_window'),(entity.get('auction_date'),'entity.auction_date'))
    basis,basis_source=_specified((local.get('basis'),'entity.target.basis'),
        (entity.get('target_basis'),'entity.target_basis'),(target.get('basis'),'task.target.basis'),
        (task.get('target_basis'),'task.target_basis'),(entity.get('basis'),'entity.basis'),
        (default_basis,'family_default'))
    baseline,baseline_source=_specified((local.get('comparison_baseline'),'entity.target.comparison_baseline'),
        (entity.get('comparison_baseline'),'entity.comparison_baseline'),
        (target.get('comparison_baseline'),'task.target.comparison_baseline'),
        (entity.get('prior_year_q_eps'),'entity.prior_year_q_eps'),
        (entity.get('latest_precutoff_estimate'),'entity.latest_precutoff_estimate'),
        (entity.get('consensus_eps'),'entity.consensus_eps'))
    return {'target_unit':unit,'target_period':period,'basis':basis,'comparison_baseline':baseline,
        'comparison_period':entity.get('comparison_period',entity.get('prior_year_quarter',entity.get('latest_published_ref_month'))),
        'source':'explicit_task_fields_then_legacy_family_defaults',
        'field_sources':{'target_unit':unit_source,'target_period':period_source,
                         'basis':basis_source,'comparison_baseline':baseline_source}}

def build_contract(task):
    family=task.get('family',''); unit,roles=FAMILIES.get(family,(task['target'].get('unit'),['target_history','definition_baseline','risk_counterevidence']))
    declared_family=family
    default_basis='GAAP diluted EPS' if family in ('eps_yoy_direction','eps_growth_regression') else None
    entities={};incompatibilities=[]
    for e in task['entities']:
        definition=_target_definition(task,e,unit,default_basis)
        entities[e['entity_id']]=definition
        # The legacy specialized estimators are valid only in their documented
        # quantity. Do not re-label a percentage packet as EUR, fraction or bps.
        if family in FAMILIES and unit and canonical_unit(definition['target_unit'])!=canonical_unit(unit):
            incompatibilities.append({'entity_id':e['entity_id'],'field':'target_unit',
                'declared':definition['target_unit'],'family_default':unit})
        if default_basis and definition['basis'] and re.sub(r'\s+',' ',str(definition['basis']).strip()).casefold()!=default_basis.casefold():
            incompatibilities.append({'entity_id':e['entity_id'],'field':'basis',
                'declared':definition['basis'],'family_default':default_basis})
    if incompatibilities:
        family='generic'
        roles=['target_history','definition_baseline','risk_counterevidence']
    try:
        days=(date.fromisoformat(task['resolution_date'])-date.fromisoformat(task['cutoff_date'])).days
    except (KeyError,ValueError): days=None
    return {'version':2,'target':task['target'],'family':family,'declared_family':declared_family,
        'family_specialization':{'enabled':family in FAMILIES,'incompatibilities':incompatibilities},
        'cutoff_date':task['cutoff_date'],
        'resolution_date':task.get('resolution_date'),'horizon_calendar_days':days,
        'interval_level':task.get('interval_level',.9),'entities':entities,'required_roles':roles,
        'ranking_order':task.get('ranking_order','descending' if family=='positioning_shift' else None),
        'known_family':family in FAMILIES,
        'constraints':['Only frozen pre-cutoff evidence; guidance for future periods is permitted.',
            'A forecast is an uncertain future estimate, not a source fact.',
            'A claim must describe the cited pre-cutoff passage, not assert the realized future outcome.',
            'Unknown direction is not evidence that the flat/no_event class is most probable.',
            'Shared evidence may support multiple entities; do not infer entity permissions from text.']}

def canonical_unit(unit):
    s=re.sub(r'[^a-z0-9]+','_',str(unit).lower()).strip('_')
    groups={'percent':{'percent','pct','percentage','percent_change','mom_percent','pct_change','mom_pct_change_sa'},
        'percentage_points':{'percentage_points','percentage_point','pp','pct_points'},
        'bps_change':{'bps','bp','basis_points','bps_change','basis_points_change'},
        'USD_per_share':{'usd_per_share','usd_share','dollars_per_share','eps_usd'},
        'ratio':{'ratio','bid_to_cover','bid_to_cover_ratio','multiple','x'},
        'probability':{'probability','prob','0_1'},
        'pct_of_open_interest':{'pct_of_open_interest','percent_of_open_interest','percent_of_oi','pct_oi'}}
    return next((name for name,aliases in groups.items() if s in aliases),s)

def prediction_output_requirements(contract):
    family=contract['family'];kind=contract['target']['type']
    required=['entity_id','interval','fact_ids','reasoning']
    if kind in ('regression','ranking') or family in FAMILIES:required.append('point_forecast')
    if kind=='classification':required+=['label','class_probabilities']
    if kind=='ranking':required.append('ranking_order')
    if family=='eps_growth_regression':required+=['underlying_eps','eps_interval']
    meaning={'credit_event':'probability of credit_event in [0,1]',
        'eps_beat_consensus':'future quarterly reported EPS in USD/share, NEVER beat/miss confidence',
        'eps_yoy_direction':'future quarterly GAAP diluted EPS in USD/share, NEVER label confidence',
        'macro_revision_direction':'future revised LEVEL in task series units, NEVER label confidence',
        'post_earnings_reaction':'future abnormal return in percent, NEVER label confidence'}.get(family,'future numeric target in the explicit task unit')
    return {'required_per_row':required,'numeric_meaning':meaning,
        'interval_schema':{'lo':'finite numeric lower bound of SAME task variable','hi':'finite numeric upper bound of SAME task variable'},
        'labels':contract['target'].get('labels',[]),'ranking_direction':contract.get('ranking_order'),
        'ranking_direction_is_string_not_rank':True,'credit_exception':'class_probabilities can be derived from explicit credit_event probability; no numeric scale conversion'}

def apply_task_ranking_direction(data,contract):
    direction=contract.get('ranking_order')
    if contract['target']['type']!='ranking' or direction not in ('ascending','descending'):return data
    import copy
    result=copy.deepcopy(data)
    if not isinstance(result,dict) or not isinstance(result.get('predictions'),list):return result
    result['ranking_order']=direction
    for row in result['predictions']:
        row['model_ranking_order']=row.get('ranking_order')
        row['ranking_order']=direction
        row['ranking_serialization']='task-defined direction; global ranks computed by Python, point values unchanged'
    return result

def restore_missing_task_units(data,contract):
    import copy
    result=copy.deepcopy(data)
    if not isinstance(result,dict) or not isinstance(result.get('predictions'),list):return result
    for row in result['predictions']:
        if not isinstance(row,dict):raise ValueError('Each prediction row must be a JSON object')
        required=contract['entities'].get(row.get('entity_id'),{}).get('target_unit')
        if required and row.get('unit') is None:
            row['unit']=required
            row['unit_serialization']='missing redundant unit restored from explicit task contract, numeric values unchanged'
    return result

def validate_units(rows, contract):
    for row in rows:
        required=contract['entities'][row['entity_id']]['target_unit']
        if required and canonical_unit(row.get('unit'))!=canonical_unit(required):
            raise ValueError('target_unit_mismatch: '+row['entity_id']+' requires '+str(required))
    return rows

def apply_probabilities(rows, contract):
    """Classification maximizes class probability, not an arbitrary point's band."""
    if contract['target']['type']!='classification':return rows
    labels=contract['target']['labels']
    for row in rows:
        ps=row.get('class_probabilities')
        # A binary event probability already defines the whole distribution.
        # Restore redundant schema fields only; never reinterpret 75 as 0.75.
        if contract['family']=='credit_event' and ps is None and set(labels)=={'credit_event','no_event'}:
            p=row.get('point_forecast')
            if type(p) in (int,float) and 0<=p<=1:
                ps={'credit_event':p,'no_event':1-p}
                row['class_probabilities']=ps
                row['probability_serialization']='class probabilities derived from explicit event probability'
        if not isinstance(ps,dict) or set(ps)!=set(labels):
            raise ValueError('class_probabilities must cover exactly task labels '+str(labels)+
                '; fix label AND probability keys together for entity '+str(row.get('entity_id'))+
                '. Do not add other classes. Return all required rows and fields.')
        if any(type(x) not in (float,int) or not 0<=x<=1 for x in ps.values()) or abs(sum(ps.values())-1)>.02:
            raise ValueError('class_probabilities must be finite, nonnegative and sum to 1')
        row['label']=max(labels,key=lambda l:ps[l])
        if contract['family']=='credit_event':
            p=ps.get('credit_event')
            if p is not None:
                row['point_forecast']=p
                row['model_unit']=row.get('unit')
                row['unit']='probability'
            if not 0<=row['interval']['lo']<=row['interval']['hi']<=1:
                raise ValueError('credit probability interval must be within [0,1]')
            if p is not None and not row['interval']['lo']<=p<=row['interval']['hi']:
                raise ValueError('event probability outside interval')
    return rows
