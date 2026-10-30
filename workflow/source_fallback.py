"""Explicit degraded evidence: exact source excerpts, never invented facts.

Used only for entities without a usable structured fact after the time budget.
The forecast model must interpret these original excerpts; fields remain unknown.
"""
import re
from .citation_catalog import annotate_payload
from .facts import validate_facts
from t4_splitter import count_tokens


def source_facts(inputs, entity, chunks, queries, cap=3):
    # All quotes retain original doc/chunk coordinates. Do not concatenate
    # disjoint passages, substitute a preview or create a new embedding node.
    _,catalog=annotate_payload({'chunks':chunks})
    source_bounds={}
    for chunk in chunks:
        source_bounds.setdefault((chunk['doc_id'],chunk['chunk_id']),[]).extend(chunk.get('source_spans',[]))
    terms=set(re.findall(r'[a-z]{3,}',' '.join(q['lexical_query'] for q in queries
                  if entity in q['entity_ids']).lower()))
    candidates=[]
    entries=list(catalog.values())
    for idx,entry in enumerate(entries):
        quote=entry['quote'];span=dict(entry['_catalog_span'])
        # Up to three adjacent source pieces provide metric/period context.
        for other in entries[idx+1:idx+3]:
            if other['chunk_id']!=entry['chunk_id'] or other['_catalog_span']['start']!=span['end']:break
            # Adjacent catalog pieces may belong to different authorized spans.
            # Keep every merged citation inside one original source span, as
            # required by validate_facts; adjacency alone is not authorization.
            if other['doc_id']!=entry['doc_id'] or not any(
                bound['start']<=span['start']<other['_catalog_span']['end']<=bound['end']
                for bound in source_bounds.get((entry['doc_id'],entry['chunk_id']),[])
            ):break
            proposed=quote+other['quote']
            if len(proposed)>1200 or count_tokens(proposed)>300:break
            quote=proposed;span['end']=other['_catalog_span']['end']
        if not quote.strip() or count_tokens(quote)>300:continue
        words=set(re.findall(r'[a-z]{3,}',quote.lower()))
        score=len(words&terms)+int(bool(re.search(r'\d',quote)))
        if re.search(r'forward.looking statements|safe harbor|table of contents',quote,re.I):score-=4
        candidates.append((score,idx,{**entry,'quote':quote,'_catalog_span':span}))
    candidates.sort(key=lambda x:(-x[0],x[1]))
    chosen=[];seen=set()
    for score,idx,c in candidates:
        # First cover distinct selected chunks, rather than near-duplicate rows.
        if c['chunk_id'] in seen:continue
        if not inputs.allowed(entity,c['doc_id']):continue
        seen.add(c['chunk_id']);chosen.append(c)
        if len(chosen)>=cap:break
    data={'facts':[{'entity_id':entity,'source_entity_id':entity,
        'statement':c['quote'],'metric':'unstructured_source_excerpt','value':None,
        'reporting_scope':'unknown','comparison_kind':'unknown','earnings_character':'unknown',
        'evidence_role':'unstructured_source_context','citations':[c]} for c in chosen]}
    result=validate_facts(data,inputs,entity,chunks,trust_catalog=True)
    for f in result['facts']:
        f['extraction_method']='verbatim_source_fallback'
        f['validation']['notice']='Exact excerpt only; metric, period, columns and scope have NOT been interpreted.'
    if not result['facts']:raise ValueError('no_valid_source_fallback_for_'+entity)
    return result
