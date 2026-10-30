"""Budget-aware sentence presentation and reading, independent of ranking."""
from t4_splitter import count_tokens
from t4_evidence import digest


def present(engine,response):
    if engine.summary.get('retrieval_granularity')!='sentence':
        return response
    response={**response,'schema_version':'sentence-response-v1','retrieval_granularity':'sentence'}
    hits=[dict(h) for h in response['hits']]
    windows=[]
    for hit in hits:
        c=engine.by_id[hit['chunk_id']]
        if c['node_type']!='sentence':
            continue
        hit.update(child_id=c['child_id'],sentence_id=c['sentence_id'],sentence_text=c['text'],
                   is_sentence_fragment=c['is_sentence_fragment'],text_unit_type=c['text_unit_type'])
        # sentence_text is the exact excerpt. Keeping the legacy excerpt as well
        # would send the same sentence a third time (alongside the window).
        hit['excerpts']=[]
        windows.append({'doc_id':c['doc_id'],'region_id':c['region_id'],
                        'start':c['window_start'],'end':c['window_end'],
                        'matched_child_ids':[c['child_id']],'window_limited':c['window_limited']})
    merged=[]
    for w in sorted(windows,key=lambda w:(w['doc_id'],w['region_id'],w['start'],w['end'])):
        last=merged[-1] if merged else None
        if (last and last['doc_id']==w['doc_id'] and last['region_id']==w['region_id']
                and w['start']<=last['end']
                and count_tokens(engine.docs[w['doc_id']][last['start']:max(last['end'],w['end'])])<=512):
            last['end']=max(last['end'],w['end'])
            last['matched_child_ids'].extend(w['matched_child_ids'])
            last['window_limited'] |= w['window_limited']
        else:
            merged.append(dict(w))
    contexts=[]; links={}
    for w in merged:
        context_id='context-'+digest([w['doc_id'],w['start'],w['end']])[:24]
        for cid in w['matched_child_ids']:
            links[cid]=context_id
        contexts.append({**w,'context_id':context_id,'text':engine.docs[w['doc_id']][w['start']:w['end']],
                         'source_spans':[{'kind':'text','start':w['start'],'end':w['end']}]})
    parents={}
    for hit in hits:
        if hit.get('child_id'):
            hit['context_id']=links[hit['child_id']]
        for pid in hit['parent_chunk_ids']:
            p=engine.parents[pid]
            parents[pid]={k:p[k] for k in ('doc_id','section','source_spans','result_path')}
            parents[pid].update(parent_chunk_id=pid,token_count=count_tokens(p['agent_text']))
    return {**response,'hits':hits,'contexts':contexts,'parents':list(parents.values())}


def read_node(engine,node_id,view,budget):
    from hybrid_search import measured, integer
    integer(budget,'token_budget',256,100000)
    if view not in ('sentence','window','parent'):
        raise ValueError('Unknown node view')
    child=engine.by_id.get(node_id)
    if view=='parent':
        ids=[node_id] if node_id in engine.parents else child.get('parent_chunk_ids',[]) if child else []
        if not ids:
            raise ValueError('Unknown node or missing parent')
        nodes=[{**{k:engine.parents[pid][k] for k in
                   ('doc_id','section','source_spans','result_path','table_refs','start','end')},
                'node_id':pid,'text':engine.parents[pid]['agent_text']} for pid in dict.fromkeys(ids)]
        for n in nodes:
            n['table_refs']=engine._table_refs(n)
    else:
        if child is None or child['node_type']!='sentence':
            raise ValueError('sentence/window requires a sentence child_id')
        a,b=(child['start'],child['end']) if view=='sentence' else (child['window_start'],child['window_end'])
        nodes=[{'node_id':node_id,'doc_id':child['doc_id'],'text':engine.docs[child['doc_id']][a:b],
                'source_spans':[{'kind':'text','start':a,'end':b}],
                'parent_chunk_ids':child['parent_chunk_ids'],
                'is_sentence_fragment':child['is_sentence_fragment']}]
    out=measured({'task_id':engine.summary['task_id'],'schema_version':'sentence-response-v1',
                  'view':view,'nodes':nodes,'budget_limited':False})
    if out['response_tokens']>budget:
        return measured({'nodes':[],'budget_limited':True,'required_tokens':out['response_tokens']})
    return out
