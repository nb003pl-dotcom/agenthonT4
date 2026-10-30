"""Research-only issue-type conditioned auction forecast; no target outcomes."""
import math
import numpy as np
from .quantitative import center, distribution, norm, number, tenor_bucket, Quantitative


def point(past, target_type):
    values=[r['bid_to_cover'] for r in past]
    baseline=center(values)
    matching=[r['bid_to_cover'] for r in past if r['issue_type']==target_type]
    active=target_type in ('new','reopen') and len(matching)>=3 and len({r['issue_type'] for r in past})==2
    weight=len(matching)/(len(matching)+3) if active else 0.
    conditional=center(matching) if active else baseline
    return {'point':(1-weight)*baseline+weight*conditional,'baseline_point':baseline,
            'conditional_center':conditional,'matching_count':len(matching),'weight':weight,'active':active}


def forecast(history,before,target_type,level=.9):
    past=sorted((r for r in history if r['date']<before),key=lambda r:r['date'])
    if len({r['date'] for r in past})!=len(past):raise ValueError('duplicate_history_date')
    if any(r['issue_type'] not in ('new','reopen') or type(r['bid_to_cover']) not in (int,float) or not math.isfinite(r['bid_to_cover']) or r['bid_to_cover']<=0 for r in past):raise ValueError('invalid_history')
    if len(past)<6:return None
    prediction=point(past,target_type)
    errors=[];audit=[]
    for i in range(4,len(past)):
        pred=point(past[:i],past[i]['issue_type'])
        error=past[i]['bid_to_cover']-pred['point'];errors.append(error)
        audit.append({'date':past[i]['date'],'training_last_date':past[i-1]['date'],'error':error,**pred})
    d=distribution(prediction['point'],errors,level,floor=float(np.std([r['bid_to_cover'] for r in past],ddof=1))*.25)
    return {**d,**prediction,'training_count':len(past),'training_last_date':past[-1]['date'],
            'target_type':target_type,'target_date':before,'past_errors':audit,
            'method':'Within-entity conditional center shrunk to unconditional center with 3 fixed pseudo-observations; no tuned parameters.'}


def extract_histories(inputs,contract):
    q=Quantitative(inputs,contract);anchors=q.build();histories={};sources={}
    if contract['family']!='auction_demand':return histories,sources
    for eid,e in inputs.entities.items():
        a=anchors.get(eid,{})
        if a.get('status')!='available':continue
        did=a['doc_id']
        if not inputs.allowed(eid,did):raise ValueError('entity_permission_failure')
        table=next(t for t in q.tables[did] if t['start']==a['source_start'])
        keys=[norm(k) for k in table['header']]
        indexes=[keys.index(k) for k in ('term','newreopen','bidtocover')]
        term,issue,btc=indexes
        rows=[{'date':r[0],'issue_type':r[issue].strip().lower(),'bid_to_cover':number(r[btc])}
              for r in table['rows'] if r[0]<=contract['cutoff_date'] and tenor_bucket(r[term])==tenor_bucket(e['tenor'])]
        histories[eid]=sorted(rows,key=lambda r:r['date'])
        sources[eid]={k:a[k] for k in ('doc_id','source_start','source_end','doc_date')}
    return histories,sources
