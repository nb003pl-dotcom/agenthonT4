"""Explicit single-quarter footnotes and disclosed diluted per-share effects."""
import re,math,calendar
from datetime import date
from .reported_item_sensitivity import build as original_build,parse_sentence as original_parse,sentences,PERIOD,EXCLUDE
from .eps_source_frame import build as eps_build,MONTH,MONTHS,normalized
from .common import digest

FOOT=re.compile(r'\bIncluded\s+(?:the\s+|a\s+)?(?:net\s+)?(?P<direction>gain|loss|charge)(?:\s+(?:related to|on|from|associated with)\s+[^.;$]{1,100}?)?\s+of\s+\$\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<scale>thousand|million|billion)\s+for\s+the\s+three(?:\s+and\s+(six|nine))?\s+months\s+ended\s+('+MONTH+r')\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b',re.I)
Q=re.compile(r'\b(first|second|third|fourth)\s+quarter\s+of\s+((?:19|20)\d{2})\b',re.I)
EPS=re.compile(r'\b(increased|decreased|reduced)\s+diluted\s+earnings\s+per\s+(?:common\s+)?share(?:\s+for\s+the\s+(?:first|second|third|fourth)\s+quarter\s+of\s+(?:19|20)\d{2})?\s+by\s+\$\s*(\d+(?:\.\d+)?|\.\d+)\b',re.I)
BAD=re.compile(r'\b(?:not|no|never|hypothetical|example|illustrative|pro forma|fiscal)\b',re.I)


def parse_money(text):
    old,reason=original_parse(text)
    if old:return old,None
    t=normalized(text);matches=list(FOOT.finditer(t))
    if not matches:return None,reason
    if len(matches)!=1 or len(re.findall(r'\$',t))!=1 or len(set(re.findall(r'\b(?:19|20)\d{2}\b',t)))!=1:return None,'ambiguous_footnote_amount_or_period'
    if BAD.search(t) or EXCLUDE.search(t):return None,'excluded_footnote_scope'
    m=matches[0]
    try:end=date(int(m[7]),MONTHS[m[5].lower().rstrip('.')],int(m[6])).isoformat()
    except ValueError:return None,'invalid_footnote_date'
    amount=float(m['amount'].replace(',',''));value=amount*{'thousand':1e3,'million':1e6,'billion':1e9}[m['scale'].lower()]
    if not math.isfinite(value) or value<=0:return None,'invalid_amount'
    return {'period_end':end,'duration_months':3,'period_precision':'day','direction':m['direction'].lower(),'displayed_amount':amount,'displayed_scale':m['scale'].lower(),'amount_report_currency':value,'source_type':'quarterly_footnote_tax_basis_unverified'},None


def parse_eps(text):
    t=normalized(text);effects=list(EPS.finditer(t));qs=list(Q.finditer(t))
    if not effects:return None,'no_direct_effect'
    if len(effects)!=1 or len(qs)!=1:return None,'ambiguous_direct_effect_or_quarter'
    if BAD.search(t) or re.search(r'\b(?:continuing operations|non[- ]GAAP|operating|basic|year[- ](?:to[- ])?year|compared|versus|six months|nine months)\b',t,re.I) or re.search(r'\b(?:expect(?:ed|s)?|will|would|could|may)\b',t,re.I):return None,'excluded_direct_effect_scope'
    q=qs[0];month={'first':3,'second':6,'third':9,'fourth':12}[q[1].lower()];year=int(q[2]);end=date(year,month,calendar.monthrange(year,month)[1]).isoformat()
    effect=float(effects[0][2]);effect*=1 if effects[0][1].lower()=='increased' else -1
    if not math.isfinite(effect) or effect==0:return None,'invalid_direct_effect'
    return {'source_type':'disclosed_eps_effect','period_end':end,'duration_months':3,'period_precision':'day','eps_effect':effect},None


def parse_source(text):
    r,why=parse_eps(text)
    return (r,why) if r else parse_money(text)


def build(inputs):
    result=original_build(inputs,parser=parse_money)
    if result['status']=='not_applicable':return result
    eps=eps_build(inputs)
    for did,doc in inputs.docs.items():
        owners=[e for e in inputs.entities if inputs.allowed(e,did)]
        if len(owners)!=1 or doc['doc_date']>inputs.task['cutoff_date']:continue
        eid=owners[0]
        for a,b,s in sentences(doc['text']):
            x,why=parse_eps(s)
            if not x:
                if why!='no_direct_effect':result['audit'].append({'entity_id':eid,'doc_id':did,'start':a,'end':b,'reason':why})
                continue
            key={'entity_id':eid,'doc_id':did,'duration_months':3,'period_end':x['period_end'],'period_precision':'day'}
            es=[e for e in eps['records'] if all(e[k]==v for k,v in key.items())]
            if len(es)!=1 or x['period_end']>doc['doc_date']:
                result['audit'].append({**key,'start':a,'end':b,'reason':'direct_effect_no_matching_quarterly_eps_header'});continue
            result['items'].append({**key,**x,'doc_date':doc['doc_date'],'source':{'start':a,'end':b,'text':s},'context':{'start':max(0,a-600),'end':b,'text':doc['text'][max(0,a-600):b]},'input_sources':{'eps':es[0]},'not_additive_or_a_forecast':True,'economic_recurrence':'unverified','item_id':'direct_'+digest([did,a,b])[:16]})
    result['status']='parsed' if result['items'] else 'no_supported_items'
    return result
