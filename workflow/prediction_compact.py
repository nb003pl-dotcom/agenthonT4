"""Python owns redundant forecast units and local fact-reference serialization."""
import copy

def annotate_prediction(payload):
    result=copy.deepcopy(payload);mapping={};counters={}
    for fact in result['facts']:
        entity=fact['entity_id'];index=counters.get(entity,0);counters[entity]=index+1
        mapping.setdefault(entity,{})[index]=fact.pop('fact_id')
        fact['fact_index']=index
        for citation in fact.get('citations',[]):citation.pop('citation_id',None)
    result.pop('allowed_fact_ids_by_entity',None)
    result['allowed_fact_indices_by_entity']={e:list(m) for e,m in mapping.items()}
    contract=result['target_contract'];kind=contract['target']['type'];family=contract['family']
    meanings={'credit_event':'probability of credit_event, 0..1; not probability of the chosen label',
        'eps_beat_consensus':'future quarterly reported EPS in USD/share; compare with task consensus, NEVER label confidence',
        'eps_yoy_direction':'future quarterly GAAP diluted EPS in USD/share; NEVER label confidence',
        'macro_revision_direction':'future revised numeric LEVEL in the series units; NEVER label confidence',
        'post_earnings_reaction':'future abnormal return in percent; NEVER label confidence'}
    required=['entity_id','point_forecast','interval','fact_indices','reasoning']
    if kind=='classification':required+=['label','class_probabilities']
    if kind=='ranking':required+=['ranking_order']
    result['output_requirements']={'required_per_row':required,'point_forecast_must_be_finite_number':True,
        'numeric_meaning':meanings.get(family,'future target value in the explicit task unit'),
        'interval_meaning':'90% outcome interval in the SAME numeric target variable as point_forecast',
        'labels':contract['target'].get('labels',[]),'ranking_order_allowed':['ascending','descending'],
        'ranking_order_required_value':contract.get('ranking_order'),'no_batch_ranks':True}
    return result,mapping

def restore_prediction(data,mapping,contract):
    if not isinstance(data,dict) or not isinstance(data.get('predictions'),list):raise ValueError('predictions must be a list')
    result=copy.deepcopy(data)
    for row in result['predictions']:
        entity=row.get('entity_id')
        if entity not in mapping:raise ValueError('prediction references unrequested entity')
        indices=row.pop('fact_indices',None)
        if not isinstance(indices,list) or not 1<=len(indices)<=3 or any(type(i)is not int or i not in mapping[entity] for i in indices) or len(set(indices))!=len(indices):raise ValueError('fact_indices must be 1..3 allowed unique integers for this entity')
        row['fact_ids']=[mapping[entity][i] for i in indices]
        unit=contract['entities'][entity]['target_unit']
        if unit:
            row['model_unit']=row.get('unit')
            if row.get('unit') is not None:
                from .contract import canonical_unit
                if canonical_unit(row['unit'])!=canonical_unit(unit):raise ValueError('model supplied conflicting target unit')
            row['unit']=unit
            row['unit_serialization']='restored from explicit target contract; no numeric conversion'
    return result
