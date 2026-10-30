"""Convert explicit source items to ordinary cited facts in real retrieved chunks."""
import re
from .facts import validate_facts
from .reported_item_sensitivity import parse_sentence


def build(inputs,report,candidates,existing):
    facts=list(existing);added=[];audit=[];mapping={}
    for item in report['items']:
        eid,did=item['entity_id'],item['doc_id'];doc=inputs.docs.get(did)
        if doc is None or not inputs.allowed(eid,did):raise ValueError('item_source_not_allowed')
        if doc['doc_date']!=item['doc_date'] or doc['doc_date']>inputs.task['cutoff_date']:raise ValueError('item_invalid_date')
        for key in ('source','context'):
            s=item[key];a,b=s['start'],s['end']
            if type(a) is not int or type(b) is not int or not 0<=a<b<=len(doc['text']) or doc['text'][a:b]!=s['text']:raise ValueError('item_source_span_changed')
        parsed,why=parse_sentence(item['source']['text'])
        if why or any(item.get(k)!=v for k,v in parsed.items()):raise ValueError('item_source_value_changed')
        if item['period_end']>doc['doc_date']:raise ValueError('item_period_after_document')
        source=item['source'];context=item['context']
        if not context['start']<=source['start']<source['end']<=context['end']:raise ValueError('item_context_does_not_contain_source')
        duplicate=next((f for f in facts if f['entity_id']==eid and f.get('metric')=='recognized_'+item['direction'] and f.get('value')==item['displayed_amount'] and f.get('unit')=='reported_dollars_'+item['displayed_scale'] and f.get('period')=='three months ended '+item['period_end'] and any(c['doc_id']==did and c['span_start']<=source['start'] and c['span_end']>=source['end'] for c in f['citations'])),None)
        if duplicate:
            mapping[item['item_id']]=duplicate['fact_id'];audit.append({'item_id':item['item_id'],'reason':'source_already_cited','fact_id':duplicate['fact_id']});continue
        matches=[c for c in candidates.get(eid,[]) if c['doc_id']==did and any(s['start']<=source['start']<source['end']<=s['end'] for s in c['source_spans'])]
        if not matches:audit.append({'item_id':item['item_id'],'reason':'complete_sentence_not_in_retrieved_chunks'});continue
        chunk=matches[0]
        quote=context if any(s['start']<=context['start']<context['end']<=s['end'] for s in chunk['source_spans']) else source
        if quote is context and quote['start']>0:
            boundary=re.search(r'(?<=[.!?])\s+(?=[A-Z])',quote['text'])
            if boundary and quote['start']+boundary.end()<=source['start']:
                a=quote['start']+boundary.end();quote={'start':a,'end':quote['end'],'text':doc['text'][a:quote['end']]}
            else:quote=source
        raw={'entity_id':eid,'source_entity_id':eid,'statement':source['text'],'evidence_role':'financial_driver',
             'metric':'recognized_'+item['direction'],'value':item['displayed_amount'],'unit':'reported_dollars_'+item['displayed_scale'],
             'period':'three months ended '+item['period_end'],'comparison_period':None,'basis':'reported','reporting_scope':'unknown','earnings_character':'unknown','information_role':'historical_result',
             'citations':[{'doc_id':did,'chunk_id':chunk['chunk_id'],'quote':quote['text'],'_catalog_span':{'start':quote['start'],'end':quote['end']}}]}
        checked=validate_facts({'facts':[raw]},inputs,eid,[chunk],trust_catalog=True)
        if checked['rejected']:
            audit.append({'item_id':item['item_id'],'reason':'fact_validation_rejected','details':checked['rejected']});continue
        f=checked['facts'][0];facts.append(f);added.append(f);mapping[item['item_id']]=f['fact_id']
        audit.append({'item_id':item['item_id'],'reason':'added','fact_id':f['fact_id'],'chunk_id':chunk['chunk_id']})
    return {'facts':facts,'added':added,'item_fact_ids':mapping,'audit':audit}
