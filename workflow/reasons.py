"""Build official submitted_reasons from supported fact ids, never model offsets."""
import json
from .common import ROOT

def build_reasons(data, inputs, fact_map, packets):
    rows=data.get('reasons') if isinstance(data,dict) else None
    if not isinstance(rows,list) or not 1<=len(rows)<=3:raise ValueError('Need 1..3 reasons')
    result=[]
    for i,row in enumerate(rows,1):
        ids=row.get('entity_ids')
        if not isinstance(ids,list) or not ids or len(set(ids))!=len(ids) or any(e not in inputs.entities for e in ids):
            raise ValueError('reason entity_ids must be a unique subset of the task roster')
        reason={'reason_id':f'r{i}','scope':{'entities':ids}}
        for field in ('premise','mechanism','answer_implication'):
            text=row.get(field)
            if not isinstance(text,str) or not text.strip():raise ValueError('reason missing '+field)
            reason[field]=text
        citations=[]
        for fid in row.get('fact_ids',[]):
            if fid not in fact_map:raise ValueError('reason unknown fact_id')
            if not all(any(inputs.allowed(e,c['doc_id']) for e in ids) for c in fact_map[fid]['citations']):
                raise ValueError('reason citation is outside official scope permissions')
            citations += [{k:c[k] for k in ('doc_id','span_start','span_end')} for c in fact_map[fid]['citations']]
        for eid in row.get('anchor_entity_ids',[]):
            if eid not in ids or packets.get(eid,{}).get('status')!='available':raise ValueError('reason unknown anchor')
            p=packets[eid]
            if p['source_end']-p['source_start']>8000:raise ValueError('reason anchor table too long; cite shorter fact spans instead')
            citations.append({'doc_id':p['doc_id'],'span_start':p['source_start'],'span_end':p['source_end']})
        if not citations:raise ValueError('reason needs at least one supported fact citation')
        seen=set();unique=[]
        for c in citations:
            key=(c['doc_id'],c['span_start'],c['span_end'])
            if key not in seen:unique.append(c);seen.add(key)
        reason['citations']=unique;result.append(reason)
    if set(e for reason in result for e in reason['scope']['entities'])!=set(inputs.entities):
        raise ValueError('Reasons must explain the full entity roster collectively')
    # Use the actual official rail for URI, duplicate, evidence and answer caps.
    import sys
    sys.path.insert(0,str(ROOT/'vendor/official'))
    from baselines.guardrails_example import citation_rail as rail
    corpus=rail.load_corpus(inputs.path/'corpus')
    findings=rail.check_submitted_reasons({'submitted_reasons':result},corpus,inputs.task['cutoff_date'])
    # Answer fields are checked again when published with the complete roster.
    if findings:raise ValueError('official reason checks: '+str(findings)[:400])
    return result

def reason_payload(inputs,rows,fact_map,packets,contract):
    selected=list(dict.fromkeys(fid for r in rows for fid in r['fact_ids']))
    facts=[{'fact_id':fid,**{k:fact_map[fid].get(k) for k in ('entity_id','statement','metric','value','unit','period','comparison_period','basis','evidence_role')},
        'quotes':[c['quote'] for c in fact_map[fid]['citations']]} for fid in selected]
    return {'task_prompt':inputs.task['prompt'],'target_contract':contract,
        'predictions':rows,'facts':facts,
        'quantitative_anchors':{e:p for e,p in packets.items() if p.get('status')=='available'},
        'limits':{'reasons':3,'total_reason_text_characters':4500,'all_entities_must_be_explained':True},
        'notice':'Group related entities into 1..3 distinct mechanisms. No extra forecasts; explain the exact submitted values and intervals. No outcome data exists here.'}
