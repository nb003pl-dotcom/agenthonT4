"""Bounded citation neighborhoods from exact, previously read source regions."""
from collections import Counter
from copy import deepcopy
from . import common  # Register the vendored tokenizer before importing it.
from t4_splitter import count_tokens


INSTRUCTION='''
read_source_contexts are bounded verbatim excerpts from previously read source
regions, not verified interpretations or new facts. They can end before later
rows and can begin after a relevant heading. Use each fact's source_context_ids
to check its metric identity, unit, period, basis and scope against nearby labels.
A number appearing in a quote does not establish the claimed row or column.
Prefer statements whose identified measure matches the actual source context;
do not treat component/subgroup measures as totals or one basis as another.
Ambiguous context is not confirmation. Choose less ambiguous complementary
existing facts where available. Do not rewrite statements, create facts, or
infer quantities or absence from omitted source regions. Keep the original
selection counts and output schema unchanged.
'''


def _merge(spans):
    result=[]
    for a,b in sorted(spans):
        if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(b,result[-1][1]))
        else:result.append((a,b))
    return result


def add_read_context(wf,payload,facts,*,max_tokens=24000,max_contexts=96):
    """No new source access: use only permitted, exact readback source intervals.

    Round-robin entity queues prevent an early entity exhausting the task budget.
    Offsets and metadata remain auditable; no semantic validity is asserted.
    """
    payload=deepcopy(payload);verified={};skipped=Counter()
    for entity,chunks in getattr(wf,'readback_selected_sources',{}).items():
        for chunk in chunks:
            doc=chunk['doc_id']
            if doc not in wf.inputs.docs or not wf.inputs.allowed(entity,doc):continue
            text=wf.inputs.docs[doc]['text']
            for span in chunk.get('source_spans',[]):
                a,b=span['start'],span['end']
                if type(a) is int and type(b) is int and 0<=a<b<=len(text) and text[a:b]==span.get('text'):
                    verified.setdefault((entity,doc),[]).append((a,b))
    verified={key:_merge(spans) for key,spans in verified.items()}
    rows={(e,f['fact_id']):f for e,v in payload['entities'].items() for f in v['facts']}
    queues={e:[] for e in payload['entities']}
    for fact in facts:
        entity= fact['entity_id']
        if (entity,fact['fact_id']) not in rows:continue
        queues[entity].extend((fact['fact_id'],c) for c in fact.get('citations',[]))
    contexts=[];known={};used=0
    for depth in range(max((len(q) for q in queues.values()),default=0)):
        for entity,queue in queues.items():
            if depth>=len(queue):continue
            fid,citation=queue[depth];doc=citation['doc_id']
            if doc not in wf.inputs.docs or not wf.inputs.allowed(entity,doc):
                skipped['permission_denied']+=1;continue
            text=wf.inputs.docs[doc]['text'];a=citation.get('span_start');b=citation.get('span_end')
            if type(a) is not int or type(b) is not int or not 0<=a<b<=len(text) or text[a:b]!=citation.get('quote'):
                skipped['citation_source_mismatch']+=1;continue
            region=next(((x,y) for x,y in verified.get((entity,doc),[]) if x<=a<b<=y),None)
            if region is None:
                skipped['citation_not_in_verified_read_source']+=1;continue
            start=max(region[0],a-1200);end=min(region[1],b+480);key=entity,doc,start,end
            if key not in known:
                if len(contexts)>=max_contexts:skipped['context_count_budget']+=1;continue
                tokens=count_tokens(text[start:end])
                if used+tokens>max_tokens:skipped['source_token_budget']+=1;continue
                cid=f'context{len(contexts):03d}';known[key]=cid;used+=tokens
                contexts.append({'context_id':cid,'entity_id':entity,'doc_id':doc,
                    'start':start,'end':end,'text':text[start:end],
                    'left_clipped_by_read_boundary':start>a-1200,'right_clipped_by_read_boundary':end<b+480})
            refs=rows[entity,fid].setdefault('source_context_ids',[])
            if known[key] not in refs:refs.append(known[key])
    payload['read_source_contexts']=contexts
    return payload,{'policy':'read-citation-neighborhood-v1','source_tokens':used,
        'max_source_tokens':max_tokens,'context_count':len(contexts),'max_contexts':max_contexts,
        'skipped':dict(skipped),'contexts':[{k:v for k,v in c.items() if k!='text'} for c in contexts],
        'semantic_status':'unverified','new_source_access':False}
