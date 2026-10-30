"""On-demand sentence previews over the existing O4000 retrieval index."""
from collections import OrderedDict
from time import perf_counter

import numpy as np

from hybrid_search import integer, measured, serialize
from pipeline import CPUEmbedder
from search_index import REVISION, digest
from sentence_nodes import fragments, parser, regions, sentence_configuration
from t4_splitter import count_tokens

SCHEMA = 'chunk-preview-v1'
TABLE_FIELDS = ('ref_id','absolute_path','sheet','range','header_row','header',
                'row_count','column_count','source','partial_in_chunk')


class PreviewRequestError(ValueError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


class PayloadCache:
    """Bound serialized text/vector payload; Python container overhead is extra."""
    def __init__(self, limit=256*1024*1024):
        self.limit, self.size, self.items = limit, 0, OrderedDict()

    def get(self, key):
        if key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key][0]

    def put(self, key, value, size):
        if key in self.items:
            self.size -= self.items.pop(key)[1]
        if size > self.limit:
            return
        while self.items and self.size+size > self.limit:
            _, (_, old_size) = self.items.popitem(last=False)
            self.size -= old_size
        self.items[key] = value, size
        self.size += size


def document_sentences(engine, doc_id):
    """Recover source sentence boundaries once, without embedding the document.

    Looking at contiguous document spans avoids cutting a sentence at an O4000
    edge. Only sentences overlapping retrieved parents will later be encoded.
    """
    text = engine.docs[doc_id]
    parents = sorted((c for c in engine.chunks if c['doc_id']==doc_id and c.get('node_type','text')=='text'),
                     key=lambda c:c['seq'])
    rows = []
    for a,b,kind in regions(text,parents):
        raw = text[a:b]
        splits = [raw] if kind!='sentence' else parser().sentence_splitter(raw)
        cursor = a
        for piece in splits:
            piece = piece.strip()
            if not piece:
                continue
            start = text.find(piece,cursor,b)
            if start<cursor or text[cursor:start].strip():
                raise PreviewRequestError('Sentence alignment failed',[{'doc_id':doc_id,'start':cursor}])
            end = start+len(piece)
            parts = list(fragments(text,start,end))
            for left,right in parts:
                ids = [p['chunk_id'] for p in parents if any(
                    s['kind']=='text' and left<s['end'] and right>s['start'] for s in p['source_spans'])]
                rows.append({'text':text[left:right],
                             'source_spans':[{'kind':'text','start':left,'end':right}],
                             'is_sentence_fragment':len(parts)>1,'context_chunk_ids':ids})
            cursor = end
        if text[cursor:b].strip():
            raise PreviewRequestError('Sentence coverage incomplete',[{'doc_id':doc_id,'start':cursor}])
    return rows


def _cache(engine):
    if not hasattr(engine,'preview_cache'):
        engine.preview_cache = PayloadCache()
        engine.preview_configuration = digest(sentence_configuration())
    return engine.preview_cache


def _refs(engine, chunk):
    return [{k:r[k] for k in TABLE_FIELDS if k in r} for r in engine._table_refs(chunk)]


def _require_chunk_index(engine):
    if engine.summary.get('retrieval_granularity')=='sentence' or engine.parents:
        raise ValueError('Use the O4000 chunk index for chunk previews and entity reads')


def _selected_entities(engine, requested):
    if requested is None:
        requested = engine.entity_order
    if not isinstance(requested,list) or not requested or any(not isinstance(e,str) for e in requested):
        raise ValueError('entity_ids must be a nonempty list; supply the matching --unit')
    if len(requested)!=len(set(requested)) or any(e not in engine.entities for e in requested):
        raise ValueError('Unknown or duplicate entity_ids')
    return requested


def _prepare_sentences(engine, candidates, stats):
    cache = _cache(engine)
    by_chunk, vectors, missing = {}, {}, {}
    started = perf_counter()
    doc_rows = {}
    for candidate in candidates:
        chunk = engine.by_id[candidate['chunk_id']]
        if chunk.get('node_type','text')!='text':
            continue
        doc_id = chunk['doc_id']
        if doc_id not in doc_rows:
            # The engine holds an immutable verified index snapshot. This prefix
            # also binds vectors to both source content and parser/model version.
            prefix = (engine.summary['index_version'],doc_id,digest(engine.docs[doc_id]),
                      engine.preview_configuration,REVISION)
            key = ('boundaries',prefix)
            rows = cache.get(key)
            if rows is None:
                rows = document_sentences(engine,doc_id)
                cache.put(key,rows,len(serialize(rows).encode('utf-8')))
            doc_rows[doc_id] = prefix,rows
        prefix,rows = doc_rows[doc_id]
        selected = []
        for row in rows:
            if chunk['chunk_id'] not in row['context_chunk_ids']:
                continue
            span = row['source_spans'][0]
            key = ('vector',prefix,span['start'],span['end'])
            selected.append((row,key))
            if key in vectors or key in missing:
                continue
            vector = cache.get(key)
            if vector is None:
                missing[key] = row['text']
            else:
                vectors[key] = vector
                stats['cached_sentence_vectors'] += 1
        if not selected and any(s['kind']=='text' and engine.docs[doc_id][s['start']:s['end']].strip()
                                for s in chunk['source_spans']):
            raise PreviewRequestError('No source-exact preview for nonempty chunk',
                                      [{'doc_id':doc_id,'chunk_id':chunk['chunk_id']}])
        by_chunk[chunk['chunk_id']] = selected
    stats['sentence_split_seconds'] += perf_counter()-started
    if missing:
        started = perf_counter()
        if engine.embedder is None:
            engine.embedder = CPUEmbedder(batch_size=engine.preview_batch_size,threads=engine.preview_threads)
            stats['model_load_seconds'] += perf_counter()-started
        if engine.embedder.manifest['revision'] != REVISION:
            raise ValueError('Sentence model revision differs from index')
        started = perf_counter()
        keys = list(missing)
        encoded = np.asarray(engine.embedder.encode([missing[k] for k in keys])[0],dtype=np.float32)
        if (encoded.shape!=(len(keys),1024) or not np.isfinite(encoded).all()
                or not np.allclose(np.linalg.norm(encoded,axis=1),1,atol=1e-4)):
            raise ValueError('Invalid sentence embedding')
        for key,vector in zip(keys,encoded):
            vector = vector.copy()
            vectors[key] = vector
            cache.put(key,vector,vector.nbytes+len(serialize(key).encode('utf-8')))
        stats['encoded_sentences'] += len(keys)
        stats['sentence_encoding_seconds'] += perf_counter()-started
    return by_chunk,vectors


def search_previews(engine, request):
    started = perf_counter()
    _require_chunk_index(engine)
    if not isinstance(request,dict):
        raise ValueError('Preview request must be an object')
    allowed = {'task_id','request_id','query','entity_ids','mode','top_k','candidate_k',
               'keywords','filters','max_response_tokens','debug'}
    if set(request)-allowed:
        raise ValueError('Unknown preview fields: '+', '.join(sorted(set(request)-allowed)))
    engine._task(request.get('task_id'))
    selected = _selected_entities(engine,request.get('entity_ids'))
    debug = request.get('debug',False)
    if type(debug) is not bool:
        raise ValueError('debug must be boolean')
    inner = {k:v for k,v in request.items() if k not in ('entity_ids','debug')}
    requests = [{**inner,'entity_id':eid} for eid in selected]
    for item in requests:
        engine._validate_entity_search(item)
    budget = request.get('max_response_tokens',8000)
    stats = {k:0 for k in ('model_load_seconds','retrieval_seconds','sentence_split_seconds',
                           'sentence_encoding_seconds','selection_seconds','encoded_sentences',
                           'cached_sentence_vectors')}
    # Reuse the existing query encoder, but load the default model with batches
    # suitable for sentence workloads. Pure lexical empty results load no model.
    if engine.embedder is None and request.get('mode','hybrid') in ('dense','hybrid'):
        t=perf_counter(); engine.embedder=CPUEmbedder(batch_size=engine.preview_batch_size,threads=engine.preview_threads)
        stats['model_load_seconds']=perf_counter()-t
    t=perf_counter()
    groups = [engine._search_entity_evidence(item,candidates_only=True) for item in requests]
    stats['retrieval_seconds']=perf_counter()-t
    unique = {h['chunk_id']:h for g in groups for h in g['hits']}
    sentences,vectors = _prepare_sentences(engine,unique.values(),stats)
    t=perf_counter()
    results=[]
    for group in groups:
        vector = None
        if any(sentences.get(h['chunk_id']) for h in group['hits']):
            vector,_ = engine._query_vector(group['dense_query'])
        hits=[]
        for rank,hit in enumerate(group['hits'],1):
            chunk = engine.by_id[hit['chunk_id']]
            entry = {k:chunk.get(k) for k in ('chunk_id','doc_id','title','section')}
            entry.update(rank=rank,node_type=chunk.get('node_type','text'),sentence=None,
                         table_refs=_refs(engine,chunk))
            options=sentences.get(chunk['chunk_id'],[])
            if options:
                scores=np.asarray([vectors[key] for _,key in options]) @ vector
                best=int(np.argmax(scores))
                entry['sentence']=options[best][0]
            elif chunk.get('node_type','text')=='table':
                # A source-exact, token-bounded table excerpt; full data is Excel.
                text=chunk['text']; end=min(len(text),700)
                while count_tokens(text[:end])>160:
                    end=end*3//4
                entry.update(table_excerpt=text[:end],table_excerpt_truncated=end<len(text))
            else:
                entry['preview_status']='table_only'
            hits.append(entry)
        base={'entity_id':group['entity_id'],'entity_name':group['entity_name'],
              'hits':hits,'budget_limited':False,'omitted_chunk_ids':[]}
        output=measured(base)
        while output['response_tokens']>budget and output['hits']:
            omitted=output['hits'].pop()['chunk_id']
            output['omitted_chunk_ids'].insert(0,omitted)
            output['budget_limited']=True
            output=measured(output)
        if output['response_tokens']>budget:
            raise ValueError('Preview budget too small for identifiers')
        results.append(output)
    stats['selection_seconds']=perf_counter()-t
    response={'schema_version':SCHEMA,'task_id':request['task_id'],'results':results,
              'first_stage_mode':request.get('mode','hybrid'),'sentence_selection_mode':'dense',
              'budget_scope':'max_response_tokens per entity; top-level response_tokens includes all entities'}
    if debug:
        response['debug']={**stats,'total_seconds':perf_counter()-started,
                           'cache_payload_bytes':engine.preview_cache.size,
                           'candidates':[{'entity_id':g['entity_id'],'pool_retrieval':g['pool_retrieval'],
                                          'hits':[{k:h.get(k) for k in ('chunk_id','source_pool','pool_rank')} for h in g['hits']]}
                                         for g in groups]}
    return measured(response)


def read_chunks(engine, request):
    _require_chunk_index(engine)
    if not isinstance(request,dict) or set(request)-{'task_id','selections','view','max_response_tokens'}:
        raise ValueError('Invalid entity read request fields')
    engine._task(request.get('task_id'))
    selections=request.get('selections')
    if not isinstance(selections,list) or not 1<=len(selections)<=100:
        raise ValueError('selections must contain 1–100 entity groups')
    view=request.get('view','agent')
    if view not in ('agent','source'):
        raise ValueError('view must be agent or source')
    budget=integer(request.get('max_response_tokens',12000),'max_response_tokens',256,100000)
    selected=OrderedDict(); errors=[]
    for selection in selections:
        if not isinstance(selection,dict) or set(selection)!={'entity_id','chunk_ids'}:
            raise ValueError('Each selection requires entity_id and chunk_ids')
        eid,ids=selection['entity_id'],selection['chunk_ids']
        if not isinstance(eid,str) or eid not in engine.entities:
            raise ValueError('Unknown entity_id; supply the matching --unit')
        if not isinstance(ids,list) or not 1<=len(ids)<=100 or any(not isinstance(cid,str) for cid in ids):
            raise ValueError('chunk_ids must contain 1–100 strings')
        selected.setdefault(eid,[])
        for cid in ids:
            c=engine.by_id.get(cid)
            if c is None or (eid not in c.get('entity_ids',[]) and c.get('entity_scope')!='unassigned'):
                errors.append({'entity_id':eid,'chunk_id':cid,
                               'reason':'unknown_chunk' if c is None else 'entity_mismatch'})
            elif cid not in selected[eid]:
                selected[eid].append(cid)
    if errors:
        raise PreviewRequestError('Invalid entity/chunk selections',errors)
    loaded={}; results=[]
    for eid,ids in selected.items():
        chunks=[]
        for cid in ids:
            if cid not in loaded:
                c=engine.by_id[cid]
                entry={k:c.get(k) for k in ('chunk_id','doc_id','title','section')}
                entry.update(node_type=c.get('node_type','text'),view=view,
                             text=c['agent_text'] if view=='agent' else engine.docs[c['doc_id']][c['start']:c['end']],
                             source_spans=c['source_spans'] if view=='agent' else
                                 [{'kind':'text','start':c['start'],'end':c['end']}],
                             table_refs=_refs(engine,c))
                loaded[cid]=entry
            chunks.append(loaded[cid])
        results.append({'entity_id':eid,'entity_name':engine.entities[eid].get('name',''),'chunks':chunks})
    response=measured({'schema_version':'entity-chunks-v1','task_id':request['task_id'],
                       'results':results,'budget_limited':False})
    if response['response_tokens']>budget:
        response=measured({'schema_version':'entity-chunks-v1','task_id':request['task_id'],
                           'results':[{**g,'chunks':[]} for g in results],'budget_limited':True,
                           'required_tokens':response['response_tokens'],
                           'selections':[{'entity_id':eid,'chunk_ids':ids} for eid,ids in selected.items()]})
        if response['response_tokens']>budget:
            raise ValueError('Read budget too small for request identifiers')
    return response
