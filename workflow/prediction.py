import math
import json
from .common import digest


def finite(x):
    return type(x) in (int,float) and math.isfinite(x)


def validate_predictions(data, entities, target_type, labels, fact_map):
    rows=data.get('predictions') if isinstance(data,dict) else None
    if not isinstance(rows,list) or sorted(r.get('entity_id','') for r in rows)!=sorted(entities):
        raise ValueError('predictions must cover ONLY these entity_ids exactly once: '+str(list(entities)))
    if target_type=='classification':
        invalid_labels=[r['entity_id'] for r in rows if r.get('label') not in labels]
        if invalid_labels:
            # A label and its probability dictionary share the same vocabulary.
            # One repair must fix both; never silently drop a model's extra class.
            raise ValueError('invalid_label: fix label AND class_probabilities for ALL rows. '
                'label must be one of task.target.labels; class_probabilities must use exactly '
                'those keys, with finite values in [0,1] summing to 1. Allowed labels='+
                json.dumps(labels,ensure_ascii=False,separators=(',',':'))+
                '. Invalid label entities='+json.dumps(invalid_labels,ensure_ascii=False,separators=(',',':')))
    if target_type=='ranking':
        root_direction=data.get('ranking_order')
        directions=[r.get('ranking_order',root_direction) for r in rows]
        if (any(d not in ('ascending','descending') for d in directions)
            or len(set(directions))!=1
            or (root_direction is not None and root_direction!=directions[0])):
            raise ValueError('ranking_order must be the literal string ascending or descending, consistent across rows; never numeric ranks')
        rows=[dict(r,ranking_order=d) for r,d in zip(rows,directions)]
    invalid_reference_entities=[]
    for r in rows:
        if target_type in ('regression','ranking') and not finite(r.get('point_forecast')):
            raise ValueError('finite point_forecast required')
        interval=r.get('interval',{})
        if not finite(interval.get('lo')) or not finite(interval.get('hi')) or interval['lo']>interval['hi']:
            raise ValueError('invalid_interval for '+str(r.get('entity_id'))+': include interval={lo:finite number,hi:finite number} in the task numeric variable, lo<=hi; return the full required prediction schema')
        if finite(r.get('point_forecast')) and not interval['lo']<=r['point_forecast']<=interval['hi']:
            raise ValueError('point_outside_interval')
        ids=r.get('fact_ids',[])
        if (not isinstance(ids,list) or not ids or
            any(not isinstance(i,str) or i not in fact_map or
                fact_map[i]['entity_id']!=r['entity_id'] for i in ids)):
            invalid_reference_entities.append(r['entity_id'])
        if not isinstance(r.get('reasoning'),str):raise ValueError('reasoning required')
    if invalid_reference_entities:
        # One schema repair is available. Report the whole batch, not just the
        # first bad row; the original payload already supplies every allowed ID.
        raise ValueError('Invalid fact_ids for entities '+
            json.dumps(invalid_reference_entities,ensure_ascii=False,separators=(',',':'))+
            '. Fix ALL listed entities. fact_ids must be a nonempty list of strings from '
            'allowed_fact_ids_by_entity[entity_id] in the original input. '
            'Never borrow another entity\'s ID or invent one.')
    return rows


def assemble(inputs, rows, fact_map, *, contract=None, claim_cap=3):
    target=inputs.task['target']['type'];predictions=[]
    for r in rows:
        from .claim_risk import select_claims
        claims,_=select_claims(inputs,r['entity_id'],r['fact_ids'],fact_map,
                               contract=contract,cap=claim_cap)
        # Explicit per-entity claim cap matches evidence selection logs.  The
        # selector publishes exact source quotes only and rejects facts whose
        # structured numeric assertion is not supported by that citation.
        p={'entity_id':r['entity_id'],'interval':{**r['interval'],'level':inputs.task.get('interval_level',.9)},
           'claims':claims}
        if target=='classification':
            p['label']=r['label']
            if finite(r.get('point_forecast')):p['point_forecast']=r['point_forecast']
        else:p['point_forecast']=r['point_forecast']
        predictions.append(p)
    if target=='ranking':
        # Prompt explicitly requests higher-is-better target values; rank direction
        # must be derived from task semantics, never independent per-batch ranks.
        direction=inputs.task.get('ranking_order')
        if direction not in ('ascending','descending'):
            direction=rows[0].get('ranking_order')
        if direction not in ('ascending','descending') or any(r.get('ranking_order',direction)!=direction for r in rows):
            raise ValueError('ranking_order must be explicit and consistent')
        order=sorted(predictions,key=lambda x:(x['point_forecast'] if direction=='ascending' else -x['point_forecast'],x['entity_id']))
        for rank,p in enumerate(order,1):p['rank']=rank
    return {'task_id':inputs.task['task_id'],'target_type':target,'entity_predictions':predictions}
