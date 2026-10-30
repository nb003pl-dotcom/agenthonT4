"""Bounded structural source context for opt-in joint extraction."""
from copy import deepcopy
from .common import write
from t4_splitter import count_tokens


def propose_context(retrieval,entity,seed_ids,*,max_added,token_ratio,parents=True,neighbors=True,excluded=()):
    """One-hop structural context, round-robin over frozen retrieval seed order."""
    seeds=list(dict.fromkeys(seed_ids));by_id=retrieval.by_id
    if any(not retrieval.inputs.allowed(entity,by_id[c]['doc_id']) for c in seeds):
        raise ValueError('seed permission mismatch')
    cost=lambda row:count_tokens(row['text'])
    seed_tokens=sum(cost(by_id[c]) for c in seeds)
    budget=int(seed_tokens*token_ratio);queues=[]
    for cid in seeds:
        seed=by_id[cid];doc=seed['doc_id'];kind=seed['node_type'];queue=[]
        same_doc=[r for r in retrieval.chunks if r['doc_id']==doc]
        if parents and kind=='table':
            parent_seqs=set(seed.get('parent_text_seqs',[]))
            queue.extend((r,'table_parent') for r in sorted(same_doc,key=lambda r:r['seq'])
                         if r['node_type']=='text' and r['seq'] in parent_seqs)
        if neighbors:
            peers=[r for r in same_doc if r['node_type']==kind]
            if kind=='table':
                refs={r['ref_id'] for r in seed.get('table_refs',[])}
                peers=[r for r in peers if refs&{x['ref_id'] for x in r.get('table_refs',[])}]
            peers.sort(key=lambda r:r['seq'])
            position=next((i for i,r in enumerate(peers) if r['chunk_id']==cid),None)
            if position is not None:
                for i in (position-1,position+1):
                    if 0<=i<len(peers):queue.append((peers[i],'same_kind_neighbor'))
        queues.append((cid,queue))
    seen=set(seeds)|set(excluded);added=[];used=0
    for depth in range(max((len(q) for _,q in queues),default=0)):
        for seed,queue in queues:
            if depth>=len(queue):continue
            row,reason=queue[depth];cid=row['chunk_id']
            if cid in seen:continue
            seen.add(cid)
            if not retrieval.inputs.allowed(entity,row['doc_id']):
                raise ValueError('context permission mismatch')
            tokens=cost(row)
            if len(added)>=max_added or used+tokens>budget:continue
            added.append({'chunk_id':cid,'seed_chunk_id':seed,'reason':reason,'source_tokens':tokens})
            used+=tokens
    return {'seed_ids':seeds,'seed_source_tokens':seed_tokens,'added_source_tokens':used,
            'added_token_budget':budget,'added':added}


def extend_selection(retrieval,kept,choices,audits,*,max_total=None,excluded=None):
    """Preserve selected seeds and add at most eight nodes / 50% source tokens.

    Supplemental extraction additionally retains its original total-source cap
    and excludes previously read sources unless an explicit reread was allowed.
    No additional semantic/query coverage is claimed by structural adjacency.
    """
    kept,choices,audits=deepcopy((kept,choices,audits))
    context={'policy':'bounded-source-context-v1','max_added_nodes_per_entity':8,
             'max_added_source_token_ratio':.5,'max_total_nodes_per_entity':max_total,'entities':{}}
    for entity,rows in kept.items():
        indices=choices[entity]['indices'];seeds=[rows[i]['chunk_id'] for i in indices]
        cap=8 if max_total is None else min(8,max(0,max_total-len(seeds)))
        proposal=propose_context(retrieval,entity,seeds,max_added=cap,token_ratio=.5,
            excluded=(excluded or {}).get(entity,()))
        context['entities'][entity]=proposal
        positions={r['chunk_id']:i for i,r in enumerate(rows)}
        qids=list(dict.fromkeys(q for r in rows for q in r.get('need_matches',{})))
        for addition in proposal['added']:
            cid=addition['chunk_id']
            if cid not in positions:
                positions[cid]=len(rows)
                rows.append({**retrieval.chunk(entity,cid),'query_ids':[],'pools':[],'score':0})
            index=positions[cid]
            rows[index]['need_matches']={q:'uncertain' for q in qids}
            rows[index]['source_context']={k:v for k,v in addition.items() if k!='chunk_id'}
            indices.append(index)
            audits[entity]['selected'].append({'chunk_id':cid,'reason':'source_context_neighbor',
                'need_matches':rows[index]['need_matches'],
                'seed_chunk_id':addition['seed_chunk_id'],'structure':addition['reason']})
        added_ids={r['chunk_id'] for r in proposal['added']}
        audits[entity]['omitted']=[cid for cid in audits[entity].get('omitted',[]) if cid not in added_ids]
        audits[entity]['source_context']=proposal
    return kept,choices,audits,context


def apply_context(wf,kept,choices,*,max_total=None,excluded=None):
    if not wf.config.get('joint_source_context'):return kept,choices
    kept,choices,wf.need_selection_audit,audit=extend_selection(wf.retrieval,kept,choices,
        wf.need_selection_audit,max_total=max_total,excluded=excluded)
    write(wf.output/'source-context.json',audit)
    return kept,choices

