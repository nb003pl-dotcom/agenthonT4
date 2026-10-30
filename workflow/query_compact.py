"""Compact wire schema; Python builds the expanded retrieval contract."""
def expand_queries(data,entities,expected_count=3):
    rows=data.get('entity_queries') if isinstance(data,dict) else None
    if not isinstance(rows,list) or sorted(r.get('entity_id','') for r in rows)!=sorted(entities):raise ValueError('entity_queries must cover only '+str(entities))
    result=[]
    allowed={'target_history','guidance_expectation','financial_driver','macro_market_driver','risk_counterevidence','definition_baseline'}
    for row in rows:
        qs=row.get('queries')
        if not isinstance(qs,list) or len(qs)!=expected_count:raise ValueError(f'Exactly {expected_count} compact queries per entity')
        for q in qs:
            kind=q.get('type');text=q.get('text')
            if kind not in allowed or not isinstance(text,str) or not 0<len(text)<=140:raise ValueError('Compact query needs a valid type and <=140-char text')
            result.append({'query_id':f'q{len(result)+1:03d}','entity_ids':[row['entity_id']],
                'query_type':kind,'dense_query':text,'lexical_query':text,'purpose':kind.replace('_',' ')})
    return {'queries':result,'wire_format':'compact_entity_queries'}
