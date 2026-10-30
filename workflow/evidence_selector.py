"""Optional House-model evidence selection from fixed candidate previews."""
import re
from .selection import select

def preview(row,queries,limit=900):
    # A long flattened financial table can be a single "sentence". Previously
    # its beginning exhausted all 900 characters and hid every later match.
    # Allocate explicit budgets to exact excerpts, never new retrieval nodes.
    text=row.get('text','')
    if len(text)<=limit:return text
    terms=set(re.findall(r'[a-z]{3,}', ' '.join(q['lexical_query']+' '+q['dense_query'] for q in queries).lower()))
    important=re.compile(r'one[ -]time|non[ -]?recurring|unrealized|fair.value|restructur|(?:gains?|loss(?:es)?).{0,60}(?:sales?|disposals?|investments?|securities|equity|shares)|tax.benefit',re.I)
    positions=[]
    for term in sorted(terms):
        matches=list(re.finditer(r'\b'+re.escape(term)+r'\b',text,re.I))
        # Sample across the whole chunk, rather than only its earliest table.
        for j in sorted({round(k*(len(matches)-1)/7) for k in range(8)}) if matches else []:
            positions.append(matches[j].start())
    positions.extend(m.start() for m in list(important.finditer(text))[:24])
    candidates={}
    for pos in positions:
        a=max(0,pos-90);b=min(len(text),a+230);a=max(0,b-230)
        s=text[a:b];tokens=set(re.findall(r'[a-z]{3,}',s.lower()))
        candidates[(a,b)]=(tokens&terms,bool(important.search(s)),bool(re.search(r'\d',s)))
    if not candidates:return text[:limit]
    head=min(150,max(1,limit//5));pieces=[text[:head]];chosen=[(0,head)];covered=set();used=head
    while candidates and len(pieces)<4:
        eligible={span:info for span,info in candidates.items() if all(max(span[0],a)>=min(span[1],b) for a,b in chosen)}
        if not eligible:break
        span=max(eligible,key=lambda s:(len(eligible[s][0]-covered)+.5*len(eligible[s][0])+3*eligible[s][1]+.5*eligible[s][2],-s[0]))
        a,b=span;remaining=limit-used-1
        if remaining<=0:break
        b=min(b,a+remaining);pieces.append(text[a:b]);used+=b-a+1;chosen.append((a,b));covered|=eligible[span][0];candidates.pop(span)
    return '\n'.join(pieces)

def selection_payload(inputs,contract,candidates,queries,group,cap):
    entries={}
    for e in group:
        qs=[q for q in queries if e in q['entity_ids']]
        entries[e]=[{'candidate_index':idx,'doc_id':r['doc_id'],'doc_date':r['doc_date'],
                    'title':r.get('title',''),'section':r.get('section',''),'node_type':r.get('node_type'),
                    'duties':r.get('evidence_duties',{}),'table_headers':[t.get('header',[]) for t in r.get('table_refs',[])],
                    'preview':preview(r,qs)} for idx,r in enumerate(candidates[e])]
    return {'task':inputs.task['prompt'],'target_contract':{**contract,'entities':{e:contract['entities'][e] for e in group}},
            'allowed_entity_ids':group,'max_selected':cap,'candidates':entries,
            'notice':'Previews are incomplete data, not source facts. Selected full chunks are read next. Do not judge absence from a preview.'}

def validate_selection(data,candidates,group,cap):
    rows=data.get('selections') if isinstance(data,dict) else None
    if not isinstance(rows,list) or sorted(r.get('entity_id','') for r in rows)!=sorted(group):raise ValueError('Selection must cover only '+str(group))
    result={}
    for row in rows:
        e=row['entity_id'];indices=row.get('candidate_indices')
        if not isinstance(indices,list) or not indices or len(indices)>cap or len(set(indices))!=len(indices) or any(type(i)is not int or not 0<=i<len(candidates[e]) for i in indices):raise ValueError('Invalid candidate_indices for '+e)
        result[e]={'indices':indices,'rationale':str(row.get('rationale',''))[:300]}
    return result


def selection_groups(inputs,contract,candidates,queries,ids,cap,system,input_limit=23400):
    """Shrink entity batches, never candidate lists, when preview packing is large."""
    from .model import input_tokens
    start=0
    while start<len(ids):
        size=min(6,len(ids)-start)
        while True:
            group=ids[start:start+size];payload=selection_payload(inputs,contract,candidates,queries,group,cap)
            if input_tokens(system,payload)<=input_limit:break
            if size==1:raise ValueError('Single entity selection metadata exceeds context budget')
            size=max(1,size//2)
        yield group,payload
        start+=size
