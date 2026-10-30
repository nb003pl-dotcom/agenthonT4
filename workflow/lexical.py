"""Corpus-only BM25 retrieval with the exact O4000/table-first source mapping.

No neural checkpoint or fabricated embedding metadata. For the official profile
unless express approval for auxiliary neural models is separately obtained.
"""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import bm25s
from .common import digest, write, check_time
from .selection import duties

def index_chunk_size(config):
    value=config.get('index_chunk_size',4000)
    if type(value) is not int or not 1000<=value<=8000:
        raise ValueError('index chunk size must be an integer from 1000 to 8000')
    return value

class LexicalRetrieval:
    def __init__(self,inputs,folder,deadline,chunk_size=4000):
        from table_extraction import extract_tables_with_diagnostics
        from table_artifacts import publish_result
        from pipeline import attach_tables
        from table_first import build_nodes
        from t4_splitter import configuration
        from t4_evidence import chunk_identity, digest as index_digest
        from search_index import financial_tokens
        self.inputs=inputs;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.deadline=deadline;self.by_id={};self.tables={};self.chunks=[];start=time.monotonic()
        for did,doc in inputs.docs.items():
            check_time(deadline)
            text=doc['text'];path=self.folder/'documents'/digest(did)/'result.json'
            tables,diagnostics=extract_tables_with_diagnostics(text)
            seed={'status':0,'meta':{'doc_id':did,'source_tables':len(tables)},
                'result':[attach_tables({'text':text,'start':0,'end':len(text),'evidence_id':f'{did}::0::{len(text)}'},tables)]}
            exported=publish_result(seed,path)
            refs=exported['result'][0]['table_refs']
            for ref in refs:self.tables[ref['ref_id']]={'ref':ref,'result_path':path}
            _,_,nodes=build_nodes(did,text,refs,chunk_size=chunk_size)
            for n in nodes:
                spans=[{**s,'text':text[s['start']:s['end']]} for s in n['source_spans']]
                row={**n,'source_spans':spans,'text':'\n'.join(s['text'] for s in spans),
                    'doc_id':did,'title':doc.get('title',''),'doc_date':doc['doc_date'],
                    'official_relation':inputs.relations[did],'result_path':str(path)}
                cid=chunk_identity(inputs.task['task_id'],did,index_digest(text),configuration(chunk_size),n['start'],n['end'])
                if n.get('algorithm_version')=='table-first-v2':
                    cid=index_digest({'base':cid,'kind':n['node_type'],'seq':n['seq'],'view_start':n['view_start'],'text':n['text'],'version':'table-first-v2'})
                row['chunk_id']=cid
                row['evidence_duties']=duties(row)
                self.chunks.append(row);self.by_id[row['chunk_id']]=row
            write(path,{'status':0,'result':nodes,'source_text':text,'meta':{
                'doc_id':did,'complete_document':True,'total_chunks':len(nodes),'model':None,
                'retrieval':'BM25-only','neural_weights_used':False,'table_extraction':diagnostics},
                'artifacts':exported['artifacts']})
        if not self.chunks:raise ValueError('No complete corpus nodes')
        texts=[' '.join(financial_tokens(r['title']+' '+r.get('embedding_text',r['text']))) for r in self.chunks]
        tokens=bm25s.tokenize(texts,stopwords='en',show_progress=False)
        self.bm25=bm25s.BM25(method='lucene',k1=1.5,b=.75)
        self.bm25.index(tokens,show_progress=False)
        self.bm25.save(str(self.folder/'bm25'))
        self.summary={'task_id':inputs.task['task_id'],'documents':len(inputs.docs),
            'chunks':len(self.chunks),'text_chunks':sum(r['node_type']=='text' for r in self.chunks),
            'table_chunks':sum(r['node_type']=='table' for r in self.chunks),'chunk_size':chunk_size,
            'method':'BM25 + deterministic content-duty anchors','neural_weights_used':False,
            'elapsed_seconds':time.monotonic()-start}
        write(self.folder/'lexical-index.json',{'summary':self.summary,'inputs':inputs.identity,
            'chunk_ids':[r['chunk_id'] for r in self.chunks],'relations':inputs.relations})

    @classmethod
    def from_existing(cls, inputs, folder, deadline, chunk_size=4000):
        """Load a verified lexical index and its source nodes without rebuilding it."""
        folder=Path(folder).resolve()
        metadata_path=folder/'lexical-index.json'
        if not metadata_path.is_file():raise ValueError('reuse_index_missing_lexical_metadata')
        from .common import read
        from t4_splitter import configuration
        from t4_evidence import chunk_identity, digest as index_digest
        metadata=read(metadata_path)
        if metadata.get('inputs')!=inputs.identity or metadata.get('relations')!=inputs.relations:
            raise ValueError('reuse_index_input_hashes_or_permissions_differ')
        if metadata.get('summary',{}).get('task_id')!=inputs.task['task_id']:
            raise ValueError('reuse_index_task_mismatch')
        if metadata.get('summary',{}).get('chunk_size',4000)!=chunk_size:
            raise ValueError('reuse_index_chunk_size_mismatch')
        self=cls.__new__(cls)
        self.inputs=inputs;self.folder=folder;self.deadline=deadline
        self.by_id={};self.tables={};self.chunks=[]
        for did,doc in inputs.docs.items():
            check_time(deadline)
            result_path=folder/'documents'/digest(did)/'result.json'
            if not result_path.is_file():raise ValueError('reuse_index_missing_document_nodes: '+did)
            result=read(result_path)
            if result.get('meta',{}).get('doc_id')!=did or result.get('source_text')!=doc['text']:
                raise ValueError('reuse_index_document_content_mismatch: '+did)
            for n in result.get('result',[]):
                spans=[{**s,'text':doc['text'][s['start']:s['end']]} for s in n['source_spans']]
                row={**n,'source_spans':spans,'text':'\n'.join(s['text'] for s in spans),
                    'doc_id':did,'title':doc.get('title',''),'doc_date':doc['doc_date'],
                    'official_relation':inputs.relations[did],'result_path':str(result_path)}
                cid=chunk_identity(inputs.task['task_id'],did,index_digest(doc['text']),configuration(chunk_size),n['start'],n['end'])
                if n.get('algorithm_version')=='table-first-v2':
                    cid=index_digest({'base':cid,'kind':n['node_type'],'seq':n['seq'],'view_start':n['view_start'],'text':n['text'],'version':'table-first-v2'})
                row['chunk_id']=cid
                row['evidence_duties']=duties(row)
                self.chunks.append(row);self.by_id[row['chunk_id']]=row
                for ref in row.get('table_refs',[]):
                    self.tables[ref['ref_id']]={'ref':ref,'result_path':result_path}
        if not self.chunks:raise ValueError('reuse_index_has_no_chunks')
        if [r['chunk_id'] for r in self.chunks]!=metadata.get('chunk_ids'):
            raise ValueError('reuse_index_chunk_manifest_mismatch')
        bm25_folder=folder/'bm25'
        if not bm25_folder.is_dir():raise ValueError('reuse_index_missing_bm25_data')
        self.bm25=bm25s.BM25.load(str(bm25_folder),show_progress=False)
        self.summary={**metadata['summary'],'reused':True,'elapsed_seconds':0}
        return self

    def scores(self,q):
        from search_index import financial_tokens
        toks=bm25s.tokenize([' '.join(financial_tokens(q))],stopwords='en',return_ids=False,show_progress=False)
        return self.bm25.get_scores(toks[0])

    def search(self,queries,top=5,cap=12,*,duty_anchors=True):
        candidates={};audit=[]
        for entity in self.inputs.entities:
            merged={};pools=self.inputs.pools(entity)
            for q in queries:
                if entity not in q['entity_ids']:continue
                check_time(self.deadline)
                scores=self.scores(q['lexical_query']+' '+q['dense_query'])
                for pool,docs in pools.items():
                    allowed=[i for i,c in enumerate(self.chunks) if c['doc_id'] in docs]
                    chosen=sorted(allowed,key=lambda i:(-float(scores[i]),self.chunks[i]['chunk_id']))[:top]
                    hits=[]
                    for rank,i in enumerate(chosen,1):
                        if scores[i]<=0:continue
                        c=self.chunks[i];cid=c['chunk_id']
                        row=merged.setdefault(cid,{'chunk_id':cid,'score':0,'query_ids':[],'pools':[]})
                        row['score']+=1/(60+rank)
                        if q['query_id'] not in row['query_ids']:row['query_ids'].append(q['query_id'])
                        if pool not in row['pools']:row['pools'].append(pool)
                        hits.append({'chunk_id':cid,'score':float(scores[i]),'doc_id':c['doc_id']})
                    audit.append({'entity_id':entity,'query_id':q['query_id'],'pool':pool,'hits':hits})
            # Quantified forward guidance is sparse and can be outranked by many
            # generic earnings rows. Add the strongest corpus-allowed anchor.
            eligible=[c for c in self.chunks if duty_anchors and self.inputs.allowed(entity,c['doc_id']) and c['evidence_duties'].get('guidance_expectation',0)>=5]
            for c in sorted(eligible,key=lambda c:(-c['evidence_duties']['guidance_expectation'],c['chunk_id']))[:2]:
                merged.setdefault(c['chunk_id'],{'chunk_id':c['chunk_id'],'score':.001,'query_ids':[],
                    'pools':['shared' if self.inputs.relations[c['doc_id']]['shared'] else 'entity'],'anchor':'quantified_guidance'})
            rows=sorted(merged.values(),key=lambda r:(-r['score'],r['chunk_id']))
            if cap is None:
                candidates[entity]=[{**r,**self.chunk(entity,r['chunk_id'])} for r in rows]
                continue
            chosen=[]
            for q in queries:
                matches=[r for r in rows if q['query_id'] in r['query_ids'] and r not in chosen]
                if entity in q['entity_ids'] and matches and len(chosen)<cap:chosen.append(matches[0])
            for r in rows:
                if r.get('anchor') and r not in chosen and len(chosen)<cap:chosen.append(r)
            chosen += [r for r in rows if r not in chosen][:max(0,cap-len(chosen))]
            candidates[entity]=[{**r,**self.chunk(entity,r['chunk_id'])} for r in chosen]
        return candidates,{'queries':audit,'method':'BM25 + corpus-allowed quantified-guidance anchors' if duty_anchors else 'BM25 per-subquery', 'candidate_cap':cap}

    def source_nodes(self,entity,doc_ids):
        """Read existing permitted nodes for exact-source lookup; no encoding."""
        return [self.chunk(entity,cid) for cid,c in self.by_id.items()
                if c['doc_id'] in doc_ids and self.inputs.allowed(entity,c['doc_id'])]

    def chunk(self,entity,cid):
        row=self.by_id[cid]
        if not self.inputs.allowed(entity,row['doc_id']):raise ValueError('chunk_entity_not_allowed')
        refs=[{**r,'absolute_path':str((Path(row['result_path']).parent/r['path']).resolve())} for r in row['table_refs']]
        keys=('chunk_id','doc_id','title','doc_date','section','node_type','source_spans','text','official_relation','evidence_duties')
        return {k:row.get(k) for k in keys}|{'entity_id':entity,'table_refs':refs}

    def table(self,rid):
        from read_table import read_table_ref
        info=self.tables[rid]
        return read_table_ref(info['ref'],info['result_path'])
