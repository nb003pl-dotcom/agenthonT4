"""Locate and protect bounded exact quarterly sources in existing O4000 nodes."""
from itertools import combinations


def build_priorities(inputs):
    if inputs.task.get('family') not in ('eps_growth_regression','eps_yoy_direction'):
        return {}
    from .eps_source_frame import build as eps_build
    from .quarter_item_disclosure import build as item_build
    eps=eps_build(inputs);items=item_build(inputs);result={}
    for eid in inputs.entities:
        records=[r for r in eps['records'] if r['entity_id']==eid and r['duration_months']==3]
        if not records:continue
        latest=max(r['period_end'] for r in records)
        # A withheld newer conflicting period must not silently become an older anchor.
        if any(c['key'][0]==eid and c['key'][1]==3 and c['key'][2]>=latest for c in eps.get('conflicts',[])):continue
        recent=[r for r in records if r['period_end']==latest]
        # Different precision or conflicting values are not a unique anchor.
        if len({(r['period_precision'],r['value']) for r in recent})!=1:continue
        alternatives=[]
        for r in recent:
            for source in [r,*r.get('corroborating_sources',[])]:
                alternatives.append({'doc_id':source['doc_id'],'spans':[source['header'],source['row']]})
        result[eid]=[{'kind':'latest_quarterly_eps','period_end':latest,'alternatives':alternatives}]
        for item in items['items']:
            if item['entity_id']==eid and item['period_end'][:len(latest)]==latest:
                result[eid].append({'kind':'disclosed_quarterly_item','period_end':item['period_end'],
                    'alternatives':[{'doc_id':item['doc_id'],'spans':[item['source']]}]})
    return result


def _valid_span(text,span):
    a,b=span.get('start'),span.get('end')
    return (type(a) is int and type(b) is int and 0<=a<b<=len(text)
            and text[a:b]==span.get('text'))


def _options(inputs,eid,candidates,priority):
    options=[]
    for alternative in priority['alternatives']:
        did=alternative['doc_id'];doc=inputs.docs.get(did)
        if not doc or not inputs.allowed(eid,did) or doc['doc_date']>inputs.task['cutoff_date']:continue
        spans=alternative['spans']
        if not spans or not all(_valid_span(doc['text'],s) for s in spans):continue
        usable=[]
        for i,c in enumerate(candidates):
            if c['doc_id']!=did or c.get('entity_id',eid)!=eid:continue
            source=c.get('source_spans',[])
            if not source or not all(_valid_span(doc['text'],s) for s in source):continue
            covered={j for j,s in enumerate(spans) if any(t['start']<=s['start'] and s['end']<=t['end'] for t in source)}
            if covered:usable.append((i,covered))
        for size in (1,2):
            for combo in combinations(usable,size):
                if set.union(*(covered for _,covered in combo))==set(range(len(spans))):
                    options.append((tuple(i for i,_ in combo),alternative))
    return options


def prioritize(inputs,eid,candidates,selected,priorities):
    """Keep selection size and all source bytes; only change membership/order."""
    before=[c['chunk_id'] for c in selected];by_id={c['chunk_id']:c for c in candidates}
    if len(by_id)!=len(candidates) or len(set(before))!=len(before) or any(cid not in by_id for cid in before):
        raise ValueError('invalid_source_priority_candidates')
    protected=[];audit=[];cap=min(2,len(selected))
    for priority in priorities.get(eid,[]):
        options=_options(inputs,eid,candidates,priority)
        fitting=[(indices,proof) for indices,proof in options
                 if len(set(protected)|set(indices))<=cap]
        if not fitting:
            audit.append({'kind':priority['kind'],'period_end':priority['period_end'],
                'status':'protection_cap' if options else 'source_not_in_retained_candidates'});continue
        indices,proof=min(fitting,key=lambda p:(len(set(p[0])-set(protected)),len(p[0]),p[0]))
        for i in indices:
            if i not in protected:protected.append(i)
        audit.append({'kind':priority['kind'],'period_end':priority['period_end'],'status':'protected',
            'chunk_ids':[candidates[i]['chunk_id'] for i in indices],'source':proof})
    chosen=[candidates[i] for i in protected]
    chosen_ids={c['chunk_id'] for c in chosen}
    chosen.extend(c for c in selected if c['chunk_id'] not in chosen_ids)
    chosen=chosen[:len(selected)];after=[c['chunk_id'] for c in chosen]
    return chosen,{'before':before,'after':after,'changed':before!=after,
        'added':[cid for cid in after if cid not in before],
        'displaced':[cid for cid in before if cid not in after],
        'max_protected_chunks':cap,'priorities':audit,'no_new_retrieval_or_text':True}


def supplement(inputs,retrieval,candidates,priorities):
    """Add at most two exact-source nodes BEFORE relevance filtering.

    Existing candidates are preserved; final selected count remains a separate cap.
    No new text splitting, embedding, or fabricated chunk identity is performed.
    Replay readers without an index explicitly retain their original candidates.
    """
    if not hasattr(retrieval,'source_nodes'):
        return candidates,{'status':'index_lookup_unavailable','entities':{}}
    output={};audit={}
    for eid,rows in candidates.items():
        wanted={a['doc_id'] for p in priorities.get(eid,[]) for a in p['alternatives']}
        wanted={d for d in wanted if d in inputs.docs and inputs.allowed(eid,d)
                and inputs.docs[d]['doc_date']<=inputs.task['cutoff_date']}
        if not wanted:
            output[eid]=rows;continue
        all_rows=list(rows);seen={c['chunk_id'] for c in rows}
        for c in retrieval.source_nodes(eid,wanted):
            if c['chunk_id'] not in seen:all_rows.append(c);seen.add(c['chunk_id'])
        proposed,detail=prioritize(inputs,eid,all_rows,rows,priorities)
        additions=[c for c in proposed if c['chunk_id'] not in {r['chunk_id'] for r in rows}]
        # Explicit provenance; not a lexical/dense similarity score.
        additions=[{**c,'score':0.0,'query_ids':[],'pools':[],
                    'source_lookup':'exact_quarterly_source'} for c in additions]
        output[eid]=[*rows,*additions]
        audit[eid]={'added':[c['chunk_id'] for c in additions],
                    'candidate_count_before':len(rows),'candidate_count_after':len(output[eid]),
                    'lookup':detail,'requires_relevance_filter':True}
    return output,{'status':'complete','entities':audit,'max_additions_per_entity':2}


def prepare(inputs,retrieval,candidates,deadline):
    """Bound all parsing and index lookup; on failure retain original candidates."""
    import time
    from .stage_deadline import wall_clock_budget
    started=time.monotonic();audit={'enabled':True,'entities':{}}
    try:
        allowance=min(5.0,deadline-time.time()-2.0)
        if allowance<=0:raise TimeoutError('source_priority_no_time')
        with wall_clock_budget(allowance):
            priorities=build_priorities(inputs)
            augmented,lookup=supplement(inputs,retrieval,candidates,priorities)
        audit.update(status='parsed',lookup=lookup)
        return augmented,priorities,{**audit,'seconds':time.monotonic()-started}
    except (ValueError,RuntimeError,TimeoutError,KeyError) as exc:
        return candidates,{},dict(audit,status='fallback',error_type=type(exc).__name__,seconds=time.monotonic()-started)
