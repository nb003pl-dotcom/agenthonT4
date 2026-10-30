"""Historical memo contracts, without experimental replay transports."""
from copy import deepcopy
from pathlib import Path
import json,re
from workflow.common import ROOT,read
from workflow.predict_input_wire import encode_input_wire

STRUCTURED_ROW_KEYS={'entity_id','baseline','direction','magnitude','numeric_support','point_action',
    'recommended_point','counterevidence','uncertainty_action','fact_ids'}
STRUCTURED_ENUMS={
    'direction':{'up','down','flat','mixed','unknown'},
    'magnitude':{'none','small','moderate','large','unknown'},
    'numeric_support':{'derived_or_direct','qualitative_only','none'},
    'point_action':{'retain','increase','decrease'},
    'uncertainty_action':{'retain','widen','narrow'},
}


def memo_payload(original):
    result=deepcopy(original)
    # These are instructions for the later output, not evidence or target data.
    result.pop('output_requirements',None)
    result.pop('forecast_update_contract',None)
    if 'memo_output_requirements' in result:raise ValueError('memo contract already supplied')
    result['memo_output_requirements']={'entity_ids_exactly_once':deepcopy(result['allowed_entity_ids']),
        'top_level_key':'reason_content','required_per_row':['entity_id','content','fact_ids'],
        'additional_row_fields':False,'content_max_characters':1000,'fact_ids_max':5,
        'fact_ids_rule':'Only allowed_fact_ids_by_entity[entity_id], never another entity even for a related component.'}
    return result


def validate_memo(data,allowed):
    rows=data.get('reason_content') if isinstance(data,dict) else None
    if not isinstance(rows,list) or set(data)!={'reason_content'}:raise ValueError('memo reason_content schema')
    if any(not isinstance(r,dict) for r in rows):raise ValueError('memo row schema')
    ids=[r.get('entity_id') for r in rows]
    if sorted(ids)!=sorted(allowed) or len(ids)!=len(set(ids)):raise ValueError('memo roster must match exactly')
    for row in rows:
        if set(row)!={'entity_id','content','fact_ids'}:raise ValueError('memo row keys')
        if not isinstance(row['content'],str) or not 0<len(row['content'].strip())<=1000:raise ValueError('memo content length')
        if not set(re.findall(r'\bf_[A-Za-z0-9_]+\b',row['content']))<=set(allowed[row['entity_id']]):
            raise ValueError('memo inline fact IDs must be owned')
        facts=row['fact_ids']
        if (not isinstance(facts,list) or any(not isinstance(f,str) for f in facts)
            or len(facts)>5 or len(set(facts))!=len(facts) or not set(facts)<=set(allowed[row['entity_id']])):
            raise ValueError('memo fact IDs must be unique and owned')
    return data


def validate_structured_memo(data,allowed):
    rows=data.get('reason_content') if isinstance(data,dict) else None
    if not isinstance(rows,list) or set(data)!={'reason_content'}:raise ValueError('memo reason_content schema')
    if any(not isinstance(row,dict) or set(row)!=STRUCTURED_ROW_KEYS for row in rows):
        raise ValueError('memo row keys')
    ids=[row.get('entity_id') for row in rows]
    if sorted(ids)!=sorted(allowed) or len(ids)!=len(set(ids)):raise ValueError('memo roster must match exactly')
    for row in rows:
        for key,values in STRUCTURED_ENUMS.items():
            if row[key] not in values:raise ValueError('memo '+key)
        baseline=row['baseline']
        if (not isinstance(baseline,dict) or set(baseline)!={'point','lo','hi','reliability'}
            or baseline['reliability'] not in {'calibrated','heuristic','unavailable'}):
            raise ValueError('memo baseline')
        numeric=[baseline[k] for k in ('point','lo','hi')]
        if baseline['reliability']=='unavailable':
            if any(value is not None for value in numeric):raise ValueError('memo unavailable baseline')
        elif any(type(value) not in (int,float) for value in numeric) or not baseline['lo']<=baseline['point']<=baseline['hi']:
            raise ValueError('memo baseline bounds')
        recommended=row['recommended_point']
        if row['numeric_support']!='derived_or_direct' and recommended is not None:
            raise ValueError('memo recommended_point requires numeric support')
        if row['numeric_support']=='derived_or_direct' and type(recommended) not in (int,float):
            raise ValueError('memo recommended_point required')
        if recommended is not None and baseline['point'] is not None:
            if row['point_action']=='increase' and recommended<=baseline['point']:raise ValueError('memo point_action increase')
            if row['point_action']=='decrease' and recommended>=baseline['point']:raise ValueError('memo point_action decrease')
            if row['point_action']=='retain' and recommended!=baseline['point']:raise ValueError('memo point_action retain')
        if not isinstance(row['counterevidence'],str) or not row['counterevidence'].strip():
            raise ValueError('memo counterevidence')
        facts=row['fact_ids']
        if (not isinstance(facts,list) or any(not isinstance(f,str) for f in facts) or len(facts)>5
            or len(set(facts))!=len(facts) or not set(facts)<=set(allowed[row['entity_id']])):
            raise ValueError('memo fact IDs must be unique and owned')
    return data


def prediction_with_memo(original,memo):
    if 'reason_content' in original:raise ValueError('original already contains memo')
    result=deepcopy(original)
    result['reason_content']={'status':'unverified_model_generated_advisory',
        'authority':'Original evidence and target contract remain authoritative. Verify this advisory against them, correct errors, and preserve the existing prediction schema. This memo supplies no new verified facts.',
        'entries':deepcopy(memo['reason_content'])}
    return result


def prediction_with_structured_memo(original,memo):
    if 'reason_content' in original:raise ValueError('original already contains memo')
    result=deepcopy(original)
    result['reason_content']={'status':'unverified_model_generated_advisory',
        'protocol':'generic-structured-v3',
        'authority':'Original evidence and target contract remain authoritative. Verify every memo field.',
        'entries':deepcopy(memo['reason_content'])}
    return result


def structured_predict_system(original):
    suffix=(ROOT/'prompts/predict-reason-application-v3.md').read_text()
    return original.rstrip()+'\n\n'+suffix.strip()+'\n'


def reason_wire(system,payload,mode):
    wire,audit=encode_input_wire(payload,mode)
    if audit is None:return system,wire,audit
    guide=(ROOT/'prompts/reason-content-input-wire-v1.md').read_text()
    return system.rstrip()+'\n\n'+guide.strip()+'\n',wire,audit


def within(record):
    n=(record.get('usage') or {}).get('completion_tokens')
    return type(n) is int and 0<n<4000


def adaptive_task_plan(source,batches,max_requests=25):
    from workflow.reason_content_gate import allocate_reason_batches
    records=[json.loads(line) for line in (Path(source)/'model-requests.jsonl').read_text().splitlines()]
    upstream=sum(record.get('stage')!='predict' for record in records)
    payloads=[]
    for batch in batches:
        request=read(batch['path'])['request']
        payloads.append(json.loads(request['messages'][1]['content']))
    return allocate_reason_batches(payloads,upstream,max_requests)


