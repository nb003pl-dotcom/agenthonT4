from collections import defaultdict
from pathlib import Path
from .common import write, check_time


class Retrieval:
    def __init__(self, inputs, folder, embedder):
        from preprocess_unit import preprocess
        from hybrid_search import SearchEngine
        self.inputs, self.folder, self.embedder = inputs, Path(folder), embedder
        assignments={d:{'doc_id':d,'entity_ids':r['entity_ids'],'shared':r['shared'],
                       'scope':('unassigned' if not r['entity_ids'] else 'entity' if len(r['entity_ids'])==1 else 'multi_entity'),
                       'source':'official_manifest','assignment_basis':['official_manifest'],
                       'assignment_confidence':1.0 if r['scope']!='unknown' else 0.0,'diagnostics':[]}
                     for d,r in inputs.relations.items()}
        self.summary = preprocess(inputs.path, self.folder, embedder=embedder, granularity='chunk',
                                  official_assignments=assignments)
        self.engine = SearchEngine(self.folder, inputs.path, embedder=embedder)
        self.embedder.role = 'query'
        # Official permissions are applied before ranking using doc filters. The
        # vendored heuristic association is not an authorization source.
        for c in self.engine.chunks:
            c['official_relation'] = inputs.relations[c['doc_id']]
        write(self.folder/'official-relations.json', inputs.relations)

    def search(self, queries, top=5, cap=12, *, duty_anchors=True):
        # Populate the exact-text SQLite cache in batches. SearchEngine still
        # checks model identity and vector validity, but subsequent per-pool
        # lookups no longer perform one remote HTTP encoding per query.
        dense_texts=list(dict.fromkeys(q['dense_query'].strip() for q in queries))
        if any(len(self.embedder.tokenizer.encode(q,add_special_tokens=True))>512 for q in dense_texts):
            raise ValueError('Query exceeds 512 BGE tokens; split the question')
        if dense_texts:self.embedder.encode(dense_texts)
        merged = {e:{} for e in self.inputs.entities}
        audit=[]
        for q in queries:
            check_time(self.embedder.deadline)
            for entity in q['entity_ids']:
                for pool, docs in self.inputs.pools(entity).items():
                    if not docs: continue  # An empty vendor filter means ALL docs.
                    result=self.engine._search_evidence({'task_id':self.inputs.task['task_id'],
                        'query':q['dense_query'], 'keywords':[q['lexical_query']],
                        'mode':'hybrid','top_k':top,'filters':{'doc_ids':docs}}, candidates_only=True)
                    audit.append({'entity_id':entity,'query_id':q['query_id'],'pool':pool,
                                  'hits':result['hits']})
                    for rank,h in enumerate(result['hits'],1):
                        cid=h['chunk_id']
                        item=merged[entity].setdefault(cid,{'chunk_id':cid,'score':0,'query_ids':[],'pools':[]})
                        item['score']+=1/(60+rank)
                        if q['query_id'] not in item['query_ids']:item['query_ids'].append(q['query_id'])
                        if pool not in item['pools']:item['pools'].append(pool)
        candidates={};omitted={}
        for entity, values in merged.items():
            rows=sorted(values.values(), key=lambda x:(-x['score'],x['chunk_id']))
            if cap is None:
                candidates[entity]=[{**r,**self.chunk(entity,r['chunk_id'])} for r in rows]
                omitted[entity]=[]
                continue
            # Round-robin query coverage before remaining global RRF candidates.
            picked=[]
            for q in queries:
                matches=[r for r in rows if q['query_id'] in r['query_ids'] and r not in picked]
                if entity in q['entity_ids'] and matches and len(picked)<cap:picked.append(matches[0])
            picked+=( [r for r in rows if r not in picked][:max(0,cap-len(picked))] )
            candidates[entity]=[{**r,**self.chunk(entity,r['chunk_id'])} for r in picked]
            omitted[entity]=[r for r in rows if r not in picked]
        return candidates, {'queries':audit,'omitted_candidates':omitted}

    def source_nodes(self,entity,doc_ids):
        """Read existing permitted nodes for exact-source lookup; no encoding."""
        return [self.chunk(entity,cid) for cid,c in self.engine.by_id.items()
                if c['doc_id'] in doc_ids and self.inputs.allowed(entity,c['doc_id'])]

    def chunk(self, entity, cid):
        c=self.engine.by_id[cid];doc=c['doc_id']
        if not self.inputs.allowed(entity,doc):raise ValueError('chunk_entity_not_allowed')
        # Send exactly source spans represented by this node. Do not use the
        # envelope across removed tables as if all of it had been retrieved.
        spans=[]
        for s in c['source_spans']:
            spans.append({'start':s['start'],'end':s['end'],'kind':s['kind'],
                          'text':self.inputs.docs[doc]['text'][s['start']:s['end']]})
        refs=self.engine._table_refs(c)
        return {'chunk_id':cid,'doc_id':doc,'entity_id':entity,'title':c.get('title',''),
                'doc_date':c['doc_date'],'section':c.get('section',''),
                'node_type':c['node_type'],'source_spans':spans,
                'text':'\n'.join(s['text'] for s in spans),
                'table_refs':refs,'official_relation':self.inputs.relations[doc]}

    def table(self, ref):
        return self.engine.read_table(self.inputs.task['task_id'],ref)
