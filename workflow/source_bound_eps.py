"""Rebind explicitly cited quarterly EPS rows; never infer forecast outcomes."""
import re
import math
from .facts import validate_facts

EPS=re.compile(r'\bdiluted\s+(?:EPS|earnings\s+per\s+(?:common\s+)?share)',re.I)
EXCLUDED=re.compile(r'continuing|discontinued|adjusted|non[- ]GAAP|\bcore\b|\bbasic\b|\bsegment\b|\bexpects?\b|\bguidance\b',re.I)
PERIOD=re.compile(r'^(?:three|3) months ended\s+(\d{4}-\d{2}(?:-\d{2})?)$',re.I)


def _row_key(r):
    return r['doc_id'],r['header']['start'],r['header']['end'],r['row']['start'],r['row']['end']


def _citations(inputs,eid,r,chunks):
    did=r['doc_id'];doc=inputs.docs[did]
    if not inputs.allowed(eid,did) or doc['doc_date']!=r['doc_date'] or doc['doc_date']>inputs.task['cutoff_date']:
        return None
    if r['period_end']>doc['doc_date'][:len(r['period_end'])]:return None
    citations=[]
    for name in ('header','row'):
        q=r[name];a,b=q['start'],q['end']
        if type(a) is not int or type(b) is not int or not 0<=a<b<=len(doc['text']) or doc['text'][a:b]!=q['text']:
            return None
        matches=[c for c in chunks if c['doc_id']==did and c.get('entity_id',eid)==eid
                 and any(s['start']<=a<b<=s['end'] and doc['text'][s['start']:s['end']]==s['text'] for s in c['source_spans'])]
        if not matches:return None
        citations.append({'chunk_id':matches[0]['chunk_id'],'doc_id':did,'quote':q['text'],'_catalog_span':{'start':a,'end':b}})
    return citations


def bind(inputs,report,eid,chunks,facts):
    """report is produced by the trusted local parser, never accepted from a model."""
    output=[];audit=[]
    records=[r for r in report['records'] if r['entity_id']==eid and r['duration_months']==3]
    for fact in facts:
        detail={'old_fact_id':fact['fact_id'],'status':'unchanged'}
        period=PERIOD.fullmatch(str(fact.get('period') or '').strip())
        statement=fact.get('statement','');role=fact.get('information_role')
        if (fact['entity_id']!=eid or not period or not EPS.search(statement)
            or EXCLUDED.search(statement) or fact.get('reporting_scope')=='segment'
            or role in ('management_guidance','market_expectation')):
            output.append(fact);audit.append({**detail,'reason':'outside_narrow_scope'});continue
        claimed=period.group(1)
        matches=[r for r in records if (r['period_end']==claimed if r['period_precision']=='day' else claimed[:7]==r['period_end'])
                 and any(c['doc_id']==r['doc_id'] and max(c['span_start'],r['row']['start'])<min(c['span_end'],r['row']['end']) for c in fact['citations'])]
        conflicted=any(c['key'][0]==eid and c['key'][1]==3 and c['key'][2][:7]==claimed[:7] for c in report.get('conflicts',[]))
        if conflicted or not matches or len({(r['value'],r['period_end']) for r in matches})!=1:
            output.append(fact);audit.append({**detail,'reason':'no_unique_source_row'});continue
        available=[(r,_citations(inputs,eid,r,chunks)) for r in matches]
        available=[(r,c) for r,c in available if c]
        if not available:
            output.append(fact);audit.append({**detail,'reason':'source_not_fully_read_or_invalid'});continue
        r,citations=available[0]
        # Only replace an explicit numeric mismatch, not already-correct facts.
        tail=statement[EPS.search(statement).end():]
        amount=re.match(r'\s+(?:was|were|of)\s+\$?\s*(\()?([+-]?(?:\d[\d,]*\.\d+|\d+|\.\d+))(\))?(?![\w.%])',tail,re.I)
        if amount is None or bool(amount[1])!=bool(amount[3]):
            output.append(fact);audit.append({**detail,'reason':'explicit_current_amount_not_identified'});continue
        claimed_value=float(amount[2].replace(',',''))*(-1 if amount[1] else 1)
        if math.isclose(claimed_value,r['value'],abs_tol=1e-8,rel_tol=1e-8):
            output.append(fact);audit.append({**detail,'reason':'current_amount_already_agrees'});continue
        peers=[p for p in records if _row_key(p)==_row_key(r) and p['period_end'][5:7]==r['period_end'][5:7]
               and int(p['period_end'][:4])==int(r['period_end'][:4])-1]
        comparison=peers[0] if peers and len({p['value'] for p in peers})==1 else None
        statement=f"Reported diluted EPS (net income) was {r['value']:g} in report-currency units per share for the three months ended {r['period_end']}."
        raw={'entity_id':eid,'source_entity_id':eid,'statement':statement,'metric':'diluted_eps_net_income',
             'value':r['value'],'unit':'report_currency_per_share','period':'three months ended '+r['period_end'],
             'basis':'reported','reporting_scope':'unknown','information_role':'historical_result',
             'earnings_character':'reported','evidence_role':'historical_baseline','citations':citations}
        if comparison:
            raw['statement']+=f" The comparative three-month period ended {comparison['period_end']} was {comparison['value']:g}."
            raw.update(comparison_value=comparison['value'],comparison_period='three months ended '+comparison['period_end'],comparison_kind='yoy')
        checked=validate_facts({'facts':[raw]},inputs,eid,chunks,trust_catalog=True)
        if checked['rejected']:
            output.append(fact);audit.append({**detail,'reason':'canonical_citation_rejected','errors':[x['error'] for x in checked['rejected']]});continue
        new=checked['facts'][0];output.append(new)
        audit.append({**detail,'status':'canonicalized','old_fact':fact,'new_fact':new,
                      'source_row':r,'comparison_row':comparison,'scope_warning':r.get('scope_basis')})
    return {'facts':output,'audit':audit}
