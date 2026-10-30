"""Fast local query-planning evidence from the prebuilt term automaton."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .query_term_index import load_index, token_key


GENERIC={"analysis","balance","business","company","financial","financial statements",
         "information","note","other","report","risk","securities","year"}
TIME_RE=re.compile(
    r"\b(?:Q[1-4]\s*(?:FY)?\s*20\d{2}|(?:FY\s*)?20\d{2}\s*Q[1-4]|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}|"
    r"20\d{2}-\d{2}(?:-\d{2})?|20\d{2})\b",re.I)


def _authorized(inputs,doc_id):
    relation=inputs.relations.get(doc_id,{})
    return bool(relation.get('shared')) or bool(set(relation.get('entity_ids',[])).intersection(inputs.entities))


def _entities(inputs,doc_id):
    relation=inputs.relations.get(doc_id,{})
    return list(inputs.entities) if relation.get('shared') else [x for x in relation.get('entity_ids',[]) if x in inputs.entities]


def _context(text,start,end,words=4):
    # Bound work per match: do not re-tokenize a multi-megabyte filing for every hit.
    left=max(0,start-100);right=min(len(text),end+100)
    return " ".join(text[left:right].split())[:220]


def _dedupe_prefixes(values):
    keys=[" ".join(token_key(value)) for value in values]
    return [value for i,(value,key) in enumerate(zip(values,keys))
            if key and not any(j!=i and other.startswith(key+" ") for j,other in enumerate(keys))]


def build_query_evidence(inputs,contract,index_path,table_inventory=None,*,max_inventory=100,
                         max_vocabulary=40,max_chars=14000):
    try:index=load_index(Path(index_path))
    except (OSError,ValueError) as exc:
        return {'metric_time_inventory':[],'metric_vocabulary':[],
                'audit':{'status':'index_unavailable','error':type(exc).__name__+': '+str(exc)[:300],
                         'model_requests':0,'table_headers_selected':0}}
    inventory=[];aggregates={};scanned=0
    for doc_id,doc in inputs.docs.items():
        if not _authorized(inputs,doc_id):continue
        text=doc.get('text','');hits=[hit for hit in index.scan(text)
            if hit['type']!='financial_term' or len(token_key(hit['surface']))>1];scanned+=1
        metrics=_dedupe_prefixes(list(dict.fromkeys(hit['canonical'] for hit in hits)))
        times=list(dict.fromkeys(match.group(0) for match in TIME_RE.finditer(text)))
        if metrics:
            inventory.append({'doc_id':doc_id,'entity_ids':_entities(inputs,doc_id),
                              'metric':metrics[:30],'time':times[:20],
                              'doc_date':str(doc.get('doc_date',''))[:10]})
        for hit in hits:
            key=(hit['canonical'].casefold(),hit['type'])
            row=aggregates.setdefault(key,{'canonical':hit['canonical'],'types':{hit['type']},
                'aliases':Counter(),'entity_ids':set(),'contexts':Counter(),'count':0})
            row['count']+=1;row['aliases'][hit['surface']]+=1
            row['entity_ids'].update(_entities(inputs,doc_id))
            row['contexts'][_context(text,hit['start'],hit['end'])]+=1
    header_count=0
    observed_metric_keys={key for key,kind in aggregates if kind in {'series_code','metric','financial_term'}}
    for table in (table_inventory or {}).get('tables',[]):
        if not table.get('shared') and not set(table.get('entity_ids',[])).intersection(inputs.entities):continue
        for header in table.get('header',[]):
            canonical=" ".join(str(header).split())[:160]
            if not canonical:continue
            normalized=canonical.casefold()
            if any(normalized==metric or normalized.startswith(metric+' ') for metric in observed_metric_keys):continue
            key=(normalized,'table_header')
            row=aggregates.setdefault(key,{'canonical':canonical,'types':{'table_header'},'aliases':Counter(),
                'entity_ids':set(),'contexts':Counter(),'count':0})
            row['count']+=1;row['entity_ids'].update(table.get('entity_ids',[]) or inputs.entities)
            if table.get('table_context'):row['contexts'][str(table['table_context'])[:220]]+=1
            header_count+=1
    task_text=' '.join([str(inputs.task.get('prompt','')),json.dumps(inputs.task.get('target',{}),ensure_ascii=False),
                        json.dumps(contract,ensure_ascii=False)])
    task_tokens=set(token_key(task_text));observed_tokens=set(token_key(' '.join(x for row in inventory for x in row['metric'])))
    ranked=[]
    priority={'series_code':5,'metric':4,'table_header':3,'financial_term':1}
    for (normalized,kind),row in aggregates.items():
        if normalized in GENERIC or len(normalized)<4:continue
        words=set(token_key(normalized))
        score=priority.get(kind,0)+4*len(words&task_tokens)+1.5*len(words&observed_tokens)+min(row['count'],10)*.15
        ranked.append((-score,normalized,kind,row))
    ranked.sort()
    vocabulary=[];used=0
    for _score,_normalized,kind,row in ranked:
        item={'canonical':row['canonical'],'types':sorted(row['types']),
              'aliases':[x for x,_ in row['aliases'].most_common(2) if x.casefold()!=row['canonical'].casefold()]}
        if row['entity_ids']:item['entity_ids']=sorted(row['entity_ids'])
        if row['contexts']:item['term_windows']=[x for x,_ in row['contexts'].most_common(1)]
        cost=len(json.dumps(item,ensure_ascii=False,separators=(',',':')))
        if vocabulary and (len(vocabulary)>=max_vocabulary or used+cost>max_chars):break
        vocabulary.append(item);used+=cost
    return {'metric_time_inventory':inventory[:max_inventory],'metric_vocabulary':vocabulary,
            'audit':{'status':'complete','model_requests':0,'documents_scanned':scanned,
                     'inventory_records':min(len(inventory),max_inventory),'vocabulary_records':len(vocabulary),
                     'vocabulary_characters':used,'table_headers_selected':header_count,
                     'index_format':index.metadata.get('format'),'index_surface_count':index.metadata.get('surface_count')}}
