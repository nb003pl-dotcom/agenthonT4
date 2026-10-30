"""Local, per-task retrieval over immutable preprocessed index snapshots."""
from collections import Counter, OrderedDict
import json
from pathlib import Path
import re

import bm25s
import numpy as np

from pipeline import CPUEmbedder
from read_table import read_table_ref
from search_index import (REVISION, TOKENIZER_VERSION, digest, financial_tokens,
                          read_json, verify_index, within)
from table_artifacts import sha256
from t4_splitter import count_tokens


def serialize(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False, default=str)


def measured(value):
    value['response_tokens'] = 0
    for _ in range(8):
        n = count_tokens(serialize(value))
        if n == value['response_tokens']:
            return value
        value['response_tokens'] = n
    raise ValueError('Unable to measure response tokens')


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


class SearchEngine:
    def __init__(self, root, unit=None, embedder=None, *, preview_batch_size=16, preview_threads=4):
        self.preview_batch_size=integer(preview_batch_size,'preview_batch_size',1,1024)
        self.preview_threads=integer(preview_threads,'preview_threads',1,256)
        self.root = Path(root).resolve()
        self.summary = verify_index(self.root)
        self.folder = within(self.root, read_json(self.root / 'current.json')['index_path'])
        self.manifest = read_json(self.folder / 'manifest.json')
        if self.manifest['index_version'] != self.summary['index_version']:
            raise ValueError('Index changed during loading; retry with a fresh engine')
        if self.manifest['configuration']['tokenizer'] != TOKENIZER_VERSION:
            raise ValueError('Tokenizer version mismatch; rebuild index')
        embedding = self.manifest['configuration']['embedding']
        if any(embedding.get(k) != v for k,v in {'revision':REVISION,'dimensions':1024,'pooling':'CLS','normalized':True,'dtype':'float32'}.items()):
            raise ValueError('Index embedding configuration is incompatible')
        self.chunks = [json.loads(s) for s in (self.folder / 'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        self.by_id = {c['chunk_id']: c for c in self.chunks}
        self.doc_chunk_counts = Counter(c['doc_id'] for c in self.chunks)
        self.matrix = np.load(self.folder / 'dense.npy', mmap_mode='r', allow_pickle=False)
        self.bm25 = bm25s.BM25.load(str(self.folder / 'bm25'), mmap=True)
        self.tables = read_json(self.folder / 'table_registry.json')
        self.parents = read_json(self.folder/'parents.json') if (self.folder/'parents.json').exists() else {}
        self.embedder = embedder
        self.query_cache = OrderedDict()
        self.entities = {}
        self.entity_order = []
        if unit is not None:
            path = Path(unit) / 'task.json'
            if sha256(path) != self.manifest['inputs']['files']['task.json']:
                raise ValueError('Task file differs from indexed task')
            task = read_json(path)
            self.entity_order = [e['entity_id'] for e in task.get('entities', [])]
            self.entities = {e['entity_id']: e for e in task.get('entities', [])}
        self.docs = read_json(self.folder/'sources.json') if 'sources.json' in self.manifest['files'] else {}
        for c in ([] if self.docs else self.chunks):
            old = self.docs.get(c['doc_id'], '')
            if c['start'] > len(old) or old[c['start']:min(len(old), c['end'])] != c['text'][:max(0, len(old)-c['start'])]:
                raise ValueError('Source coverage/overlap mismatch')
            self.docs[c['doc_id']] = old + c['text'][max(0, len(old)-c['start']):]

    def _task(self, task_id):
        if task_id != self.summary['task_id']:
            raise ValueError('task_id differs from loaded index')

    def _table_refs(self, chunk):
        # Preserve the portable reference contract; resolve runtime locations only
        # in responses, so moving the output directory needs no index rebuild.
        parent = within(self.root, chunk['result_path']).parent
        return [{**ref, 'absolute_path': str(within(parent, ref['path']))}
                for ref in chunk['table_refs']]

    def _location(self, chunk):
        section = chunk.get('section')
        return {'doc_type': chunk.get('doc_type', ''),
                'node_type':chunk.get('node_type','text'),
                'parent_chunk_ids':chunk.get('parent_chunk_ids',[c['chunk_id'] for c in self.chunks
                                    if c['doc_id']==chunk['doc_id'] and c.get('node_type','text')=='text'
                                    and c['seq'] in chunk.get('parent_text_seqs',[])]),
                'source_spans':chunk.get('source_spans',[{'kind':'text','start':chunk['start'],'end':chunk['end']}]),
                'source_offsets_are_envelope':chunk.get('source_offsets_are_envelope',False),
                'view_start':chunk.get('view_start'), 'view_end':chunk.get('view_end'),
                'table_part':chunk.get('table_part'), 'row_fragment':chunk.get('row_fragment',False),
                'section': section,
                'section_known': bool(section and section != '未恢复章节'),
                'seq': chunk['seq'],
                'doc_chunk_count': self.doc_chunk_counts[chunk['doc_id']],
                'result_path': chunk['result_path']}

    def _query_vector(self, query):
        if query in self.query_cache:
            self.query_cache.move_to_end(query)
            return self.query_cache[query], True
        if self.embedder is None:
            self.embedder = CPUEmbedder()
        if self.embedder.manifest['revision'] != REVISION:
            raise ValueError('Query model revision differs from document model')
        if len(self.embedder.tokenizer.encode(query, add_special_tokens=True)) > 512:
            raise ValueError('Query exceeds 512 BGE tokens; split the question')
        vector = np.asarray(self.embedder.encode([query])[0][0], dtype=np.float32)
        if vector.shape != (1024,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
            raise ValueError('Invalid query embedding')
        self.query_cache[query] = vector
        if len(self.query_cache) > 128:
            self.query_cache.popitem(last=False)
        return vector, False

    def search_evidence(self, request):
        return self._search_evidence(request)

    def _search_evidence(self, request, candidates_only=False):
        if not isinstance(request,dict):
            raise ValueError('Search request must be a JSON object')
        self._task(request.get('task_id'))
        allowed = {'task_id','request_id','query','entity_id','keywords','mode','top_k','candidate_k','filters','max_response_tokens','cursor'}
        if set(request) - allowed:
            raise ValueError('Unknown search fields: ' + ', '.join(sorted(set(request)-allowed)))
        query = request.get('query')
        if not isinstance(query, str) or not query.strip() or len(query) > 8000:
            raise ValueError('query must contain 1–8000 characters')
        mode = request.get('mode', 'hybrid')
        if mode not in ('hybrid','lexical','dense','exact'):
            raise ValueError('Unknown search mode')
        top = integer(request.get('top_k',20), 'top_k',1,100)
        budget = integer(request.get('max_response_tokens',8000),'max_response_tokens',256,100000)
        ks = request.get('candidate_k', {'bm25':50,'dense':50})
        if not isinstance(ks, dict) or set(ks) - {'bm25','dense'}:
            raise ValueError('Invalid candidate_k')
        kb, kd = [integer(ks.get(k,50),k,1,200) for k in ('bm25','dense')]
        keywords = request.get('keywords', [])
        if not isinstance(keywords,list) or len(keywords)>32 or any(not isinstance(s,str) or len(s)>256 for s in keywords):
            raise ValueError('Invalid keywords')
        entity_id = request.get('entity_id')
        if entity_id is not None and not isinstance(entity_id,str):
            raise ValueError('entity_id must be a string')
        if entity_id is not None and entity_id not in self.entities:
            raise ValueError('Unknown entity_id; supply the matching --unit to enable entity hints')
        entity = self.entities.get(entity_id, {})
        dense_query = ' '.join([query.strip(), *([entity_id, entity.get('name','')] if entity_id else [])])
        lexical_query = ' '.join([dense_query, *keywords])
        terms = list(dict.fromkeys(financial_tokens(lexical_query)))
        filters = request.get('filters', {})
        if not isinstance(filters,dict) or set(filters)-{'doc_ids','doc_types','node_types','has_table'}:
            raise ValueError('Invalid filters')
        for key in ('doc_ids','doc_types','node_types'):
            if not isinstance(filters.get(key,[]),list) or any(not isinstance(x,str) for x in filters.get(key,[])):
                raise ValueError('Invalid filter values')
        if any(k not in ('text','table','sentence') for k in filters.get('node_types',[])):
            raise ValueError('Unknown node_type')
        if self.parents and 'text' in filters.get('node_types',[]):
            raise ValueError('Sentence index stores text parents for reading only; search sentence/table or use the chunk index')
        if filters.get('has_table') is not None and type(filters['has_table']) is not bool:
            raise ValueError('has_table must be boolean or null')
        selected = [i for i,c in enumerate(self.chunks)
                    if (not filters.get('doc_ids') or c['doc_id'] in filters['doc_ids'])
                    and (not filters.get('doc_types') or c['doc_type'] in filters['doc_types'])
                    and (not filters.get('node_types') or c.get('node_type','text') in filters['node_types'])
                    and (filters.get('has_table') is None or bool(c['table_refs']) == filters['has_table'])]
        key = lambda i: (self.chunks[i]['doc_id'],self.chunks[i]['start'],self.chunks[i]['end'],self.chunks[i]['chunk_id'])
        signature = digest({'request': {k:v for k,v in request.items() if k!='cursor'}, 'version': self.summary['index_version']})
        cursor = request.get('cursor')
        offset = 0
        if cursor is not None:
            if not isinstance(cursor,dict) or cursor.get('signature') != signature:
                raise ValueError('Cursor does not match query, options or index version')
            offset = integer(cursor.get('offset'),'cursor.offset',0,10**9)
        ranks, raw = {'bm25':{},'dense':{}}, {'bm25':{},'dense':{}}
        exact = {}
        cache_hit = None
        if selected and mode in ('hybrid','lexical') and terms:
            scores = self.bm25.get_scores(terms)
            order = sorted((i for i in selected if scores[i]>0), key=lambda i:(-float(scores[i]),key(i)))[:kb]
            ranks['bm25'] = {i:r+1 for r,i in enumerate(order)}
            raw['bm25'] = {i:float(scores[i]) for i in order}
        if selected and mode in ('hybrid','dense'):
            vector, cache_hit = self._query_vector(dense_query)
            scores = self.matrix @ vector
            order = sorted(selected,key=lambda i:(-float(scores[i]),key(i)))[:kd]
            ranks['dense'] = {i:r+1 for r,i in enumerate(order)}
            raw['dense'] = {i:float(scores[i]) for i in order}
        if mode == 'exact':
            matches = {doc:[(m.start(),m.end()) for m in re.finditer(re.escape(query),text,re.I)] for doc,text in self.docs.items()}
            for i in selected:
                c = self.chunks[i]
                coverage = [s for s in c.get('source_spans',[{'kind':'text','start':c['start'],'end':c['end']}]) if s['kind']=='text']
                spans = [(a,b) for a,b in matches[c['doc_id']] if any(a<s['end'] and b>s['start'] for s in coverage)]
                if spans:
                    exact[i] = spans
            order = sorted(exact,key=lambda i:(-len(exact[i]),key(i)))
            fusion = {}
        else:
            union = set(ranks['bm25']) | set(ranks['dense'])
            fusion = {i:sum(1/(60+ranks[channel][i]) for channel in ranks if i in ranks[channel]) for i in union}
            order = sorted(union,key=lambda i:(-fusion[i],key(i)))
        if offset > len(order):
            raise ValueError('Cursor offset exceeds candidate pool')
        base = {'task_id':self.summary['task_id'], 'request_id':request.get('request_id'),
                'index_root':str(self.root),
                'location_contract':{'result_path_base':'index_root', 'table_path_base':'result_path parent',
                                     'offsets':'original document text, Python characters, [start,end)',
                                     'view_offsets':'reading_text; null for table nodes',
                                     'absolute_path_scope':'current execution filesystem'},
                'index_version':self.summary['index_version'], 'effective_mode':mode,
                'query':query,'dense_query':dense_query,'lexical_query':lexical_query,'effective_filters':filters,
                'query_embedding_cache_hit':cache_hit,
                'retrieval':{'bm25_candidates':len(ranks['bm25']),'dense_candidates':len(ranks['dense']),
                             'union_candidates':len(order),'rank_constant':60},
                'hits':[], 'returned':0,'offset':offset,'next_cursor':None,'budget_limited':False,
                'warnings':[], 'budget_scope':'complete compact JSON, cl100k_base; excludes tool wrapper'}
        def pack(hits, next_offset, limited=False):
            from sentence_retrieval import present
            return measured(present(self,{**base,'hits':hits,'returned':len(hits),'budget_limited':limited,
                             'next_cursor':{'signature':signature,'offset':next_offset} if next_offset<len(order) else None}))
        output = {'hits':[]} if candidates_only else pack([],offset)
        for pos in range(offset,min(len(order),offset+top)):
            i = order[pos]
            c = self.chunks[i]
            spans = exact.get(i, [])
            text = c['text']
            match = next((m for term in terms if (m:=re.search(re.escape(term),text,re.I))),None)
            original = self.docs[c['doc_id']][c['start']:c['end']]
            source_match = next((m for term in terms if (m:=re.search(re.escape(term),original,re.I))),None)
            center = max(0,spans[0][0]-c['start']) if spans else source_match.start() if source_match else 0
            a = max(0, center-80)
            b = min(len(original), a+700)
            while count_tokens(original[a:b])>160:
                b = a+(b-a)*3//4
            hit = {k:c[k] for k in ('chunk_id','doc_id','doc_date','title','evidence_id','start','end','agent_text_tokens')}
            hit.update(self._location(c))
            body_terms = set(financial_tokens(text))
            metadata_terms = set(financial_tokens(' '.join([c['doc_id'],c['title'],c.get('section',''),c.get('table_context',''),
                                  *[h for ref in c['table_refs'] for h in ref['header']]])))
            hit.update(source_tokens=count_tokens(original), indexed_text_tokens=c['tokens'], scores={'bm25':raw['bm25'].get(i),'dense_cosine':raw['dense'].get(i),'rrf':fusion.get(i)},
                       ranks={k:ranks[k].get(i) for k in ranks}, matched_by=['exact'] if spans else [k for k in ranks if i in ranks[k]],
                       matched_terms=[t for t in terms if t in body_terms],
                       matched_metadata_terms=[t for t in terms if t in metadata_terms],
                       excerpts=[{'text':original[a:b], 'start':c['start']+a,'end':c['start']+b,
                                  'is_full_chunk':a == 0 and b == len(original) and original==text,
                                  'view':'source',
                                  'method':'exact_match' if spans else 'lexical_window' if source_match else 'leading_context'}],
                       table_refs=self._table_refs(c))
            if spans:
                hit['exact_spans'] = [{'start':a,'end':b} for a,b in spans]
                coverage = [s for s in c.get('source_spans',[{'kind':'text','start':c['start'],'end':c['end']}]) if s['kind']=='text']
                hit['match_crosses_block'] = any(not any(s['start']<=a and b<=s['end'] for s in coverage) for a,b in spans)
            if candidates_only:
                output['hits'].append(hit)
                continue
            trial = pack([*output['hits'],hit],pos+1)
            if trial['response_tokens']>budget:
                if not output['hits']:
                    raise ValueError(f'Response budget too small; first complete hit requires {trial["response_tokens"]} tokens')
                output = pack(output['hits'],pos,True)
                break
            output = trial
        if candidates_only:
            return {**base, 'hits':output['hits'], 'returned':len(output['hits'])}
        if output['response_tokens']>budget:
            raise ValueError('Response budget too small for metadata')
        return output

    def search_evidence_batch(self, requests):
        if not isinstance(requests,list) or not 1<=len(requests)<=16:
            raise ValueError('Batch must contain 1–16 requests')
        return [self.search_evidence(request) for request in requests]

    def _validate_entity_search(self, request):
        if not isinstance(request,dict):
            raise ValueError('Entity search request must be a JSON object')
        allowed = {'task_id','request_id','query','entity_id','keywords','mode','top_k',
                   'candidate_k','filters','max_response_tokens'}
        if set(request)-allowed:
            raise ValueError('Unknown entity search fields: ' + ', '.join(sorted(set(request)-allowed)))
        self._task(request.get('task_id'))
        query = request.get('query')
        if not isinstance(query,str) or not query.strip() or len(query)>8000:
            raise ValueError('query must contain 1–8000 characters')
        entity_id = request.get('entity_id')
        if not isinstance(entity_id,str) or entity_id not in self.entities:
            raise ValueError('Unknown entity_id; supply the matching --unit for entity search')
        mode = request.get('mode','hybrid')
        if mode not in ('hybrid','lexical','dense','exact'):
            raise ValueError('Unknown search mode')
        top = integer(request.get('top_k',12),'top_k',1,100)
        budget = integer(request.get('max_response_tokens',8000),'max_response_tokens',256,100000)
        keywords = request.get('keywords',[])
        if not isinstance(keywords,list) or len(keywords)>32 or any(not isinstance(s,str) or len(s)>256 for s in keywords):
            raise ValueError('Invalid keywords')
        ks = request.get('candidate_k',{'bm25':50,'dense':50})
        if not isinstance(ks,dict) or set(ks)-{'bm25','dense'}:
            raise ValueError('Invalid candidate_k')
        for key in ('bm25','dense'):
            integer(ks.get(key,50),key,1,200)
        filters = request.get('filters',{})
        if not isinstance(filters,dict) or set(filters)-{'doc_ids','doc_types','node_types','has_table'}:
            raise ValueError('Invalid filters')
        for key in ('doc_ids','doc_types','node_types'):
            if not isinstance(filters.get(key,[]),list) or any(not isinstance(x,str) for x in filters.get(key,[])):
                raise ValueError('Invalid filter values')
        if any(k not in ('text','table','sentence') for k in filters.get('node_types',[])):
            raise ValueError('Unknown node_type')
        if self.parents and 'text' in filters.get('node_types',[]):
            raise ValueError('Sentence index stores text parents for reading only; search sentence/table or use the chunk index')
        if filters.get('has_table') is not None and type(filters['has_table']) is not bool:
            raise ValueError('has_table must be boolean or null')
        return entity_id, top, budget, filters

    def _pool_search(self, request, doc_ids, top, candidates_only=False):
        if not doc_ids:
            return {'hits':[], 'retrieval':{'bm25_candidates':0,'dense_candidates':0,
                                            'union_candidates':0,'rank_constant':60},
                    'query_embedding_cache_hit':None}
        filters = {**request.get('filters',{}),'doc_ids':sorted(doc_ids)}
        inner = {k:v for k,v in request.items() if k!='max_response_tokens'}
        inner.update(top_k=top,filters=filters)
        if not candidates_only:
            inner['max_response_tokens']=100000
        return self._search_evidence(inner, candidates_only=candidates_only)

    @staticmethod
    def _merge_pools(entity_hits, shared_hits, top):
        entity_target = (2*top+1)//3
        shared_target = top-entity_target
        chosen_entity = list(entity_hits[:entity_target])
        chosen_shared = list(shared_hits[:shared_target])
        remaining = top-len(chosen_entity)-len(chosen_shared)
        if remaining and len(chosen_entity)<entity_target:
            extra = shared_hits[len(chosen_shared):len(chosen_shared)+remaining]
            chosen_shared.extend(extra); remaining -= len(extra)
        if remaining and len(chosen_shared)<shared_target:
            extra = entity_hits[len(chosen_entity):len(chosen_entity)+remaining]
            chosen_entity.extend(extra); remaining -= len(extra)
        if remaining:
            extra = entity_hits[len(chosen_entity):len(chosen_entity)+remaining]
            chosen_entity.extend(extra); remaining -= len(extra)
        if remaining:
            chosen_shared.extend(shared_hits[len(chosen_shared):len(chosen_shared)+remaining])
        merged = []
        ie = ish = 0
        while ie<len(chosen_entity) or ish<len(chosen_shared):
            for source in ('entity','entity','shared'):
                if source=='entity' and ie<len(chosen_entity):
                    merged.append(chosen_entity[ie]); ie += 1
                elif source=='shared' and ish<len(chosen_shared):
                    merged.append(chosen_shared[ish]); ish += 1
                if len(merged)>=top:
                    return merged
            if ie>=len(chosen_entity) and ish<len(chosen_shared):
                merged.extend(chosen_shared[ish:top-len(merged)+ish]); break
            if ish>=len(chosen_shared) and ie<len(chosen_entity):
                merged.extend(chosen_entity[ie:top-len(merged)+ie]); break
        return merged[:top]

    def search_entity_evidence(self, request):
        return self._search_entity_evidence(request)

    def _search_entity_evidence(self, request, candidates_only=False):
        entity_id, top, budget, filters = self._validate_entity_search(request)
        allowed_docs = set(filters.get('doc_ids',[])) or None
        entity_docs = {c['doc_id'] for c in self.chunks if entity_id in c.get('entity_ids',[])}
        shared_docs = {c['doc_id'] for c in self.chunks if c.get('entity_scope')=='unassigned'}
        if allowed_docs is not None:
            entity_docs &= allowed_docs
            shared_docs &= allowed_docs
        entity_result = self._pool_search(request,entity_docs,top,candidates_only)
        shared_result = self._pool_search(request,shared_docs,top,candidates_only)
        entity_hits = [{**hit,'source_pool':'entity','pool_rank':rank}
                       for rank,hit in enumerate(entity_result['hits'],1)]
        shared_hits = [{**hit,'source_pool':'shared','pool_rank':rank}
                       for rank,hit in enumerate(shared_result['hits'],1)]
        merged = self._merge_pools(entity_hits,shared_hits,top)
        entity = self.entities[entity_id]
        # Match the exact string used by search_evidence, including a missing name.
        generated_query = ' '.join((request['query'].strip(),entity_id,entity.get('name','')))
        warnings = []
        if not entity_docs:
            warnings.append('entity pool is empty')
        if not shared_docs:
            warnings.append('shared pool is empty')
        base = {
            'task_id':self.summary['task_id'],'request_id':request.get('request_id'),
            'index_root':str(self.root),'index_version':self.summary['index_version'],
            'entity_id':entity_id,'entity_name':entity.get('name',''),
            'query':request['query'],'generated_query':generated_query,
            'dense_query':generated_query,
            'lexical_query':' '.join([generated_query,*request.get('keywords',[])]),
            'effective_mode':request.get('mode','hybrid'),'effective_filters':filters,
            'pool_policy':{
                'entity_definition':'entity_id in chunk.entity_ids',
                'shared_definition':'chunk.entity_scope == unassigned',
                'merge':'entity:shared approximately 2:1 with shortage fill',
            },
            'pool_retrieval':{
                'entity':{'documents':len(entity_docs),**entity_result['retrieval'],
                          'query_embedding_cache_hit':entity_result['query_embedding_cache_hit']},
                'shared':{'documents':len(shared_docs),**shared_result['retrieval'],
                          'query_embedding_cache_hit':shared_result['query_embedding_cache_hit']},
            },
            'hits':[],'entity_hits':[],'shared_hits':[],'returned':0,
            'budget_limited':False,'warnings':warnings,
            'budget_scope':'complete compact JSON, cl100k_base; excludes tool wrapper',
        }
        if candidates_only:
            return {**base, 'hits':merged, 'returned':len(merged)}
        def pack(hits,limited=False):
            from sentence_retrieval import present
            ranked = [{**hit,'combined_rank':rank} for rank,hit in enumerate(hits,1)]
            compact = lambda hit: {k:hit[k] for k in
                                   ('chunk_id','doc_id','source_pool','pool_rank','combined_rank')}
            return measured(present(self,{**base,'hits':ranked,
                             'entity_hits':[compact(h) for h in ranked if h['source_pool']=='entity'],
                             'shared_hits':[compact(h) for h in ranked if h['source_pool']=='shared'],
                             'returned':len(ranked),'budget_limited':limited}))
        output = pack([])
        for hit in merged:
            trial = pack([*output['hits'],hit])
            if trial['response_tokens']>budget:
                if not output['hits']:
                    raise ValueError(f'Response budget too small; first complete hit requires {trial["response_tokens"]} tokens')
                output = pack(output['hits'],True)
                break
            output = trial
        if output['response_tokens']>budget:
            raise ValueError('Response budget too small for entity search metadata')
        return output

    def search_task_entities(self, request):
        if not isinstance(request,dict):
            raise ValueError('Task entity search request must be a JSON object')
        allowed = {'task_id','request_id','query','entity_ids','keywords','mode','top_k',
                   'candidate_k','filters','max_response_tokens'}
        if set(request)-allowed:
            raise ValueError('Unknown task search fields: ' + ', '.join(sorted(set(request)-allowed)))
        self._task(request.get('task_id'))
        requested = request.get('entity_ids')
        if requested is None:
            selected = list(self.entity_order)
        else:
            if (not isinstance(requested,list) or any(not isinstance(x,str) for x in requested)):
                raise ValueError('entity_ids must be a list of strings')
            if len(requested)!=len(set(requested)):
                raise ValueError('entity_ids contains duplicate values')
            unknown = set(requested)-set(self.entities)
            if unknown:
                raise ValueError('Unknown entity_id: ' + ', '.join(sorted(unknown)))
            selected = [entity_id for entity_id in self.entity_order if entity_id in set(requested)]
        if not selected:
            raise ValueError('Task has no selected entities; supply the matching --unit')
        base_request = {k:v for k,v in request.items() if k!='entity_ids'}
        results = [self.search_entity_evidence({**base_request,'entity_id':entity_id})
                   for entity_id in selected]
        return measured({'task_id':self.summary['task_id'],'request_id':request.get('request_id'),
                         'query':request.get('query'),'entity_count':len(results),'results':results,
                         'entity_response_tokens':sum(r['response_tokens'] for r in results),
                         'budget_scope':'max_response_tokens applies independently to each entity result'})

    def read_evidence(self, task_id, chunk_ids, view='agent', token_budget=12000, before=0, after=0):
        self._task(task_id)
        if view not in ('agent','source') or not isinstance(chunk_ids,list) or not 1<=len(chunk_ids)<=100 or any(not isinstance(c,str) for c in chunk_ids):
            raise ValueError('Invalid evidence request')
        integer(token_budget,'token_budget',256,100000)
        integer(before,'before',0,1); integer(after,'after',0,1)
        rows = {}
        for cid in chunk_ids:
            if cid not in self.by_id:
                raise ValueError('Unknown chunk_id')
            c = self.by_id[cid]
            for n in self.chunks:
                if n['chunk_id']==cid or (n['doc_id']==c['doc_id'] and c.get('node_type','text')=='text'
                    and n.get('node_type','text')=='text' and c['seq']-before<=n['seq']<=c['seq']+after):
                    rows[n['chunk_id']] = n
        output = []
        for c in sorted(rows.values(),key=lambda c:(c['doc_id'],c['seq'])):
            entry = {k:c[k] for k in ('chunk_id','doc_id','evidence_id','start','end','table_refs','result_path')}
            entry.update(self._location(c), table_refs=self._table_refs(c),
                         title=c['title'], doc_date=c['doc_date'])
            entry.update(view=view, text=c['agent_text'] if view=='agent' else self.docs[c['doc_id']][c['start']:c['end']], requested=c['chunk_id'] in chunk_ids)
            output.append(entry)
        result = measured({'task_id':task_id,'index_root':str(self.root),'index_version':self.summary['index_version'],'chunks':output,'budget_limited':False})
        if result['response_tokens']>token_budget:
            error = measured({'task_id':task_id,'chunks':[],'budget_limited':True,'required_tokens':result['response_tokens'],
                             'requested_chunk_ids':chunk_ids,'hint':'Reduce neighbours or increase token_budget; no text was truncated'})
            if error['response_tokens']>token_budget:
                raise ValueError('Evidence budget too small even for the request identifiers')
            return error
        return result

    def search_entity_previews(self, request):
        from chunk_preview import search_previews
        return search_previews(self, request)

    def read_entity_chunks(self, request):
        from chunk_preview import read_chunks
        return read_chunks(self, request)

    def read_table(self, task_id, ref_id):
        self._task(task_id)
        if ref_id not in self.tables:
            raise ValueError('Unknown table ref_id')
        item = self.tables[ref_id]
        return read_table_ref(item['ref'], within(self.root,item['result_path']))

    def read_node(self, task_id, node_id, view='window', token_budget=12000):
        self._task(task_id)
        from sentence_retrieval import read_node
        return read_node(self,node_id,view,token_budget)
