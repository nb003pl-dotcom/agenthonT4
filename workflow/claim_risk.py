"""Deterministic, task-agnostic claim eligibility and selection.

The gate never paraphrases evidence and never calls a model.  It distinguishes
source-location validity from the narrower question of whether a structured
fact is safe enough to expose as a scored claim.
"""
import math
import re


ROLE_PRIORITY = {
    "target_history": 6,
    "guidance_expectation": 6,
    "risk_counterevidence": 5,
    "financial_driver": 4,
    "macro_market_driver": 3,
    "definition_baseline": 2,
}


def _numbers(text):
    """Return signed literal numbers while preserving accounting parentheses."""
    results=[]
    pattern=r'(?<![\w.])(?P<open>\()?\s*(?P<sign>[+-])?\s*\$?\s*(?P<num>(?:\d[\d,]*\.\d+|\.\d+|\d[\d,]*))\s*(?P<close>\))?'
    for match in re.finditer(pattern,str(text)):
        raw=match.group('num').replace(',','')
        try:value=float(raw)
        except ValueError:continue
        if match.group('sign')=='-' or (match.group('open') and match.group('close')):value=-value
        results.append(value)
    return results


def _contains_number(values, wanted):
    return any(math.isclose(value,wanted,rel_tol=1e-8,abs_tol=1e-8) for value in values)


def _visible_characters(text):
    return len(re.sub(r'\s+','',str(text)))


def assess_claim(inputs, entity_id, fact, citation, contract=None):
    reasons=[];quote=citation.get('quote') if isinstance(citation,dict) else None
    if fact.get('entity_id')!=entity_id:reasons.append('fact_wrong_entity')
    doc_id=citation.get('doc_id') if isinstance(citation,dict) else None
    if not doc_id or doc_id not in inputs.docs or not inputs.allowed(entity_id,doc_id):
        reasons.append('citation_wrong_entity')
    elif not isinstance(quote,str) or not quote.strip():reasons.append('empty_quote')
    else:
        start,end=citation.get('span_start'),citation.get('span_end')
        text=inputs.docs[doc_id]['text']
        if (type(start) is not int or type(end) is not int or not 0<=start<end<=len(text)
            or text[start:end]!=quote):reasons.append('citation_span_mismatch')
        if _visible_characters(quote)<20:reasons.append('quote_not_self_contained')
        if len(quote)>1600:reasons.append('quote_too_long')
    statement=fact.get('statement')
    if not isinstance(statement,str) or not statement.strip():reasons.append('missing_statement')
    elif isinstance(quote,str):
        source_numbers=_numbers(quote);statement_numbers=_numbers(statement)
        missing=[value for value in statement_numbers if not _contains_number(source_numbers,value)]
        if missing:
            opposite=[value for value in missing if _contains_number(source_numbers,-value) and value!=0]
            reasons.append('numeric_sign_mismatch' if opposite else 'statement_numbers_not_supported')
        value=fact.get('value')
        if type(value) in (int,float) and math.isfinite(value):
            if not _contains_number(source_numbers,float(value)):
                loss_sign=(value<0 and _contains_number(source_numbers,abs(float(value)))
                           and bool(re.search(r'\b(?:net |operating |gross )?loss\b',statement,re.I))
                           and bool(re.search(r'\b(?:net |operating |gross )?loss\b',quote,re.I)))
                if loss_sign:reason=None
                else:reason='numeric_sign_mismatch' if _contains_number(source_numbers,-float(value)) and value!=0 else 'fact_value_not_supported'
                if reason and reason not in reasons:reasons.append(reason)
    validation=fact.get('validation',{})
    if validation.get('source_located') is False:reasons.append('source_not_located')
    if validation.get('permission_valid') is False:reasons.append('permission_not_validated')
    if validation.get('date_valid') is False:reasons.append('date_not_validated')
    if fact.get('extraction_method')=='verbatim_source_fallback':reasons.append('verbatim_fallback_not_structured')
    role=fact.get('evidence_role')
    relevance=ROLE_PRIORITY.get(role,1)
    if contract:
        target=contract.get('target',{})
        if fact.get('metric') and fact.get('metric')==target.get('name'):relevance+=3
        entity_contract=contract.get('entities',{}).get(entity_id,{})
        if fact.get('period') and fact.get('period') in (entity_contract.get('target_period'),entity_contract.get('comparison_period')):
            relevance+=2
    return {'eligible':not reasons,'reasons':reasons,'risk':len(reasons),'relevance':relevance,
            'evidence_role':role,'fact_id':fact.get('fact_id'),'citation_id':citation.get('citation_id')}


def select_claims(inputs, entity_id, fact_ids, fact_map, contract=None, cap=3):
    candidates=[];audit=[];seen=set()
    for fact_order,fid in enumerate(fact_ids):
        fact=fact_map.get(fid)
        if not fact:
            audit.append({'fact_id':fid,'eligible':False,'reasons':['unknown_fact_id'],'decision':'rejected'})
            continue
        for citation_order,citation in enumerate(fact.get('citations',[])):
            assessment=assess_claim(inputs,entity_id,fact,citation,contract)
            key=(citation.get('doc_id'),citation.get('span_start'),citation.get('span_end'))
            row={**assessment,'fact_order':fact_order,'citation_order':citation_order,'citation_key':key}
            if key in seen:
                row.update(eligible=False,reasons=assessment['reasons']+['duplicate_citation'],decision='rejected')
                audit.append(row);continue
            seen.add(key);row['decision']='candidate' if assessment['eligible'] else 'rejected';audit.append(row)
            if assessment['eligible']:
                candidates.append((fact,citation,row))
    candidates.sort(key=lambda item:(-item[2]['relevance'],item[2]['fact_order'],item[2]['citation_order']))
    selected=[];roles=set()
    # First prefer distinct evidence roles, then fill remaining safe slots.
    for diverse in (True,False):
        for fact,citation,row in candidates:
            if len(selected)>=cap:break
            if row['decision']=='selected':continue
            role=fact.get('evidence_role')
            if diverse and role in roles:continue
            row['decision']='selected';roles.add(role);selected.append({
                'doc_id':citation['doc_id'],'span_start':citation['span_start'],'span_end':citation['span_end'],
                'claim':citation['quote']})
        if len(selected)>=cap:break
    # The published schema requires grounded claims.  If conservative semantic
    # checks cannot certify any alternative, retain the prior exact-source
    # behavior instead of turning a valid historical answer into a schema
    # failure.  The audit makes this compatibility fallback explicit.
    if not selected:
        fallback_seen=set()
        for fid in fact_ids:
            fact=fact_map.get(fid,{})
            if fact.get('entity_id')!=entity_id:continue
            for citation in fact.get('citations',[]):
                doc_id=citation.get('doc_id');start=citation.get('span_start');end=citation.get('span_end')
                key=(doc_id,start,end)
                if key in fallback_seen or doc_id not in inputs.docs or not inputs.allowed(entity_id,doc_id):continue
                text=inputs.docs[doc_id]['text'];quote=citation.get('quote')
                if (type(start) is not int or type(end) is not int or not isinstance(quote,str)
                    or not 0<=start<end<=len(text) or text[start:end]!=quote):continue
                fallback_seen.add(key);selected.append({'doc_id':doc_id,'span_start':start,'span_end':end,'claim':quote})
                audit.append({'fact_id':fid,'citation_key':key,'eligible':False,
                    'reasons':['compatibility_fallback_no_eligible_claim'],'decision':'fallback'})
                if len(selected)>=cap:break
            if len(selected)>=cap:break
    return selected,audit
