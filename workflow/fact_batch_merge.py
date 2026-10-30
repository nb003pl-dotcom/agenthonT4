"""Lossless frozen-source packing; new extraction may still change selected facts."""
from copy import deepcopy
from collections import Counter
import re
from .citation_catalog import annotate_payload
from .model import input_tokens


def restore(system,payload,inputs,source):
    raw=deepcopy(payload);eid=raw['entity']['entity_id']
    for c in raw['chunks']:
        if not inputs.allowed(eid,c['doc_id']):raise ValueError('source_entity_permission')
        d=inputs.docs[c['doc_id']]
        if not d.get('doc_date') or d['doc_date']>inputs.task['cutoff_date']:raise ValueError('source_date')
        for span in c['source_spans']:
            a,b=span['start'],span['end']
            if type(a) is not int or type(b) is not int or not 0<=a<b<=len(d['text']):raise ValueError('source_offsets')
            span['text']=d['text'][a:b]
    encoded,catalog=annotate_payload(raw)
    if encoded!=payload:raise ValueError('frozen_catalog_mismatch')
    return {'system':system,'payload':encoded,'raw':raw,'catalog':catalog,'sources':[str(source)]}


def citations(batch):
    return Counter((c['doc_id'],c['chunk_id'],c['_catalog_span']['start'],c['_catalog_span']['end'],c['quote']) for c in batch['catalog'].values())


def merge(a,b,max_tokens=23400):
    if a['system']!=b['system'] or a['raw']['entity']!=b['raw']['entity']:return None
    x,y=a['raw'],b['raw']
    excluded={'chunks','table_reads','limits','instruction'}
    if {k:v for k,v in x.items() if k not in excluded}!={k:v for k,v in y.items() if k not in excluded}:return None
    if {k:v for k,v in x['limits'].items() if k!='max_facts'}!={k:v for k,v in y['limits'].items() if k!='max_facts'}:return None
    n,m=x['limits']['max_facts'],y['limits']['max_facts']
    if type(n) is not int or type(m) is not int or min(n,m)<1 or n+m>12:return None
    pattern=r'Compact output: retain the \d+ most decisive distinct facts'
    if len(re.findall(pattern,x['instruction']))!=1 or len(re.findall(pattern,y['instruction']))!=1:return None
    if re.sub(pattern,'CAPACITY',x['instruction'])!=re.sub(pattern,'CAPACITY',y['instruction']):return None
    tables=deepcopy(x['table_reads'])
    for k,v in y['table_reads'].items():
        if k in tables and tables[k]!=v:return None
        tables[k]=deepcopy(v)
    raw=deepcopy(x);raw['chunks']+=deepcopy(y['chunks']);raw['table_reads']=tables
    raw['limits']['max_facts']=n+m
    raw['instruction']=re.sub(pattern,f'Compact output: retain the {n+m} most decisive distinct facts',raw['instruction'])
    system=a['system'].replace('Prefer three distinct important facts to redundant numbers.',
                                'Prefer up to limits.max_facts distinct important facts to redundant numbers.')
    payload,catalog=annotate_payload(raw)
    if input_tokens(system,payload)>max_tokens:return None
    result={'system':system,'payload':payload,'raw':raw,'catalog':catalog,'sources':a['sources']+b['sources']}
    if citations(result)!=citations(a)+citations(b):raise ValueError('citation_multiset_changed')
    return result


def pack(batches,max_tokens=23400):
    result=[]
    for batch in batches:
        # Normalize only for comparing a previously merged system to a new batch.
        merged=None
        if result:
            previous=deepcopy(result[-1]);current=deepcopy(batch)
            if len(previous['sources'])>1:
                current['system']=current['system'].replace('Prefer three distinct important facts to redundant numbers.',
                    'Prefer up to limits.max_facts distinct important facts to redundant numbers.')
            merged=merge(previous,current,max_tokens)
        if merged is not None:result[-1]=merged
        else:result.append(deepcopy(batch))
    if sum((citations(x) for x in result),Counter())!=sum((citations(x) for x in batches),Counter()):raise ValueError('packing_lost_source')
    return result
