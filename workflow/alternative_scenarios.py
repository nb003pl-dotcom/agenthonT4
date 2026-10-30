"""Optional evidence-grounded alternative scenarios, not additional source facts."""
from .prediction import finite
from .facts import compact
from .contract import clean_task


def scenario_payload(inputs, contract, facts, anchors):
    return {'task':{**clean_task(inputs.task),'entities':list(inputs.entities.values())},
            'target_contract':contract,'facts':compact(facts),'quantitative_anchors':anchors,
            'allowed_entity_ids':list(inputs.entities),
            'allowed_fact_ids_by_entity':{e:[f['fact_id'] for f in facts if f['entity_id']==e]
                                          for e in inputs.entities}}


def validate_scenarios(data, entities, facts, contract):
    scenarios=data.get('scenarios') if isinstance(data,dict) else None
    if not isinstance(scenarios,list) or len(scenarios)!=3:
        raise ValueError('Exactly three alternative scenarios required')
    if any(not isinstance(s,dict) for s in scenarios):raise ValueError('Scenario must be an object')
    names=[s.get('name') for s in scenarios]
    if any(not isinstance(n,str) or not 0<len(n)<=80 for n in names) or len(set(names))!=3:
        raise ValueError('Three distinct short scenario names required')
    weights=[s.get('probability') for s in scenarios]
    if any(not finite(p) or not 0<p<1 for p in weights) or abs(sum(weights)-1)>.005:
        raise ValueError('Scenario probabilities must be positive and sum to one')
    fmap={f['fact_id']:f for f in facts};roster=set(entities)
    for scenario in scenarios:
        if not isinstance(scenario.get('mechanism'),str) or not 0<len(scenario['mechanism'])<=500:
            raise ValueError('A short conditional mechanism is required')
        outcomes=scenario.get('outcomes')
        if (not isinstance(outcomes,list) or any(not isinstance(r,dict) for r in outcomes)
            or len(outcomes)!=len(roster) or {r.get('entity_id') for r in outcomes}!=roster):
            raise ValueError('Each scenario must cover every entity exactly once')
        for row in outcomes:
            point=row.get('point_forecast');band=row.get('interval',{})
            if (not isinstance(band,dict) or not all(finite(x) for x in (point,band.get('lo'),band.get('hi')))
                or not band['lo']<=point<=band['hi']):
                raise ValueError('Each scenario needs numeric point and ordered conditional interval')
            if contract['family']=='credit_event' and not 0<=band['lo']<=band['hi']<=1:
                raise ValueError('Credit scenario must use event probability in [0,1]')
            ids=row.get('fact_ids')
            if (not isinstance(ids,list) or not 1<=len(ids)<=2 or len(set(ids))!=len(ids)
                or any(f not in fmap or fmap[f]['entity_id']!=row['entity_id'] for f in ids)):
                raise ValueError('Scenario fact_ids must be 1–2 facts for that entity')
    return {'scenarios':scenarios,'status':'model_hypotheses_not_source_facts',
            'instruction':'These are uncertain alternative mechanisms, not new observed facts or calibrated probabilities. Consider all three when forming the final target distribution; do not average interval endpoints and call that a 90% interval. Check signs and task starting levels. Final claims can cite only original facts, never these hypotheses.'}


def scoped_scenarios(packet, group):
    """Retain common regimes while excluding other entities' outcome/fact rows."""
    return {**packet,'scenarios':[{**s,'outcomes':[r for r in s['outcomes'] if r['entity_id'] in group]}
                                  for s in packet['scenarios']]}
