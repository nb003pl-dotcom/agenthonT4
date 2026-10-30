"""Prediction-only view: keep source identity and quotes, remove interpretations."""
from copy import deepcopy

INSTRUCTION = '''
Some supplied fact records may contain only entity/fact IDs and citations. These are unstructured verbatim excerpts, NOT missing values to fill from memory. Determine metric, period, consolidated/segment scope and column meaning only where supported by the original quotes and statement_contexts. A quote may be clipped or lack its header: acknowledge uncertainty rather than invent an interpretation. Select the same allowed fact IDs for relevant original evidence; keep the required prediction schema unchanged.
'''


def excerpt_view(facts):
    output=[];seen=set()
    for fact in facts:
        fid=fact.get('fact_id');eid=fact.get('entity_id');citations=fact.get('citations')
        if not isinstance(fid,str) or not fid or fid in seen:raise ValueError('invalid_or_duplicate_fact_id')
        if not isinstance(eid,str) or not eid:raise ValueError('missing_fact_entity')
        if not isinstance(citations,list) or not citations:raise ValueError('missing_fact_citations')
        quotes=[]
        for c in citations:
            if not isinstance(c,dict) or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('citation_id','quote','doc_date')):
                raise ValueError('invalid_source_citation')
            quotes.append({k:deepcopy(c[k]) for k in ('citation_id','quote','doc_date')})
        output.append({'fact_id':fid,'entity_id':eid,'citations':quotes});seen.add(fid)
    return output
