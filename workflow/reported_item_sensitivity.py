"""Narrow explicit quarterly items; conditional arithmetic, never forecasts."""
import re,math
from datetime import date
from .eps_source_frame import build as eps_build,MONTH,MONTHS,normalized
from .financial_input_frame import build as financial_build
from .common import digest

PERIOD=re.compile(r'\bthree months ended\s+('+MONTH+r')\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b',re.I)
ITEM=re.compile(r'\b(?P<direction>gain|loss|charge)\s+of\s+\$\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<scale>million|billion|thousand)\b',re.I)
ACCOUNT=re.compile(r'\brecorded (?:to|in)\s+(?:other income(?:\s*\(expense\))?(?:,?\s+net)?|(?:other )?operating expenses?|cost of (?:sales|revenue))\b',re.I)
EXCLUDE=re.compile(r'\b(?:after[- ]tax|net of tax|other comprehensive income|AOCI|expect(?:ed|s)?|anticipat(?:e|ed|es)|estimate(?:d|s)?|will|would|could|may)\b',re.I)
INSTRUCTION='''
reported_item_sensitivities contains historical source-bound conditional arithmetic, NOT forecasts or proof an item will not recur. Each original sentence explicitly reports one quarterly gain/loss/charge; use the context to assess what the item means. The historical EPS with only this item hypothetically removed is NOT adjusted GAAP, NOT additive across items, and NOT automatically next-quarter EPS. Two tax scenarios are shown: zero tax and the same-period historical overall effective rate. Neither establishes the item's actual tax rate; actual tax can lie outside these two scenarios. Shares and tax rate are held fixed only for sensitivity calculation. Reconcile original facts with the dated source sentence: a whole income-statement line is not the disclosed component amount. Source permissions/positions and arithmetic have been checked, not economic recurrence or complete accounting scope. Retain business drivers and the original task comparison baseline. Do not invent extra adjustments. Keep the original output schema and cite only existing allowed fact IDs; calculation IDs are not publishable citation IDs.
'''


def sentences(text):
    start=0
    for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])',text):
        yield start,m.start(),text[start:m.start()];start=m.end()
    if start<len(text):yield start,len(text),text[start:]


def parse_sentence(text):
    t=normalized(text);items=list(ITEM.finditer(t));periods=list(PERIOD.finditer(t))
    if not items:return None,'no_item'
    if len(items)!=1:return None,'multiple_items'
    if len(periods)!=1 or len(set(re.findall(r'\b(?:19|20)\d{2}\b',t)))!=1:return None,'ambiguous_period'
    if EXCLUDE.search(t):return None,'excluded_tax_scope_or_forward_statement'
    if re.search(r'\b(?:not|no|never|hypothetical|example|illustrative)\b',t,re.I):return None,'negated_or_illustrative_statement'
    if not ACCOUNT.search(t):return None,'missing_income_statement_recognition'
    if not re.search(r'\b(?:recognized|recorded|incurred)\b',t,re.I):return None,'missing_recognition'
    p=periods[0]
    try:end=date(int(p[3]),MONTHS[p[1].lower().rstrip('.')],int(p[2])).isoformat()
    except ValueError:return None,'invalid_period'
    m=items[0];amount=float(m['amount'].replace(',',''));scale={'thousand':1e3,'million':1e6,'billion':1e9}[m['scale'].lower()]
    value=amount*scale
    if not math.isfinite(value) or value<=0:return None,'invalid_amount'
    return {'period_end':end,'duration_months':3,'period_precision':'day','direction':m['direction'].lower(),'displayed_amount':amount,'displayed_scale':m['scale'].lower(),'amount_report_currency':value},None


def build(inputs, *, parser=None):
    parser=parser or parse_sentence
    financial=financial_build(inputs);eps=eps_build(inputs)
    if financial['status']=='not_applicable':return {'status':'not_applicable','items':[],'audit':[]}
    result=[];audit=[]
    for did,doc in inputs.docs.items():
        owners=[e for e in inputs.entities if inputs.allowed(e,did)]
        if len(owners)!=1 or doc['doc_date']>inputs.task['cutoff_date']:continue
        eid=owners[0]
        for a,b,s in sentences(doc['text']):
            item,why=parser(s)
            if why=='no_item':continue
            entry={'entity_id':eid,'doc_id':did,'start':a,'end':b,'reason':why}
            if not item:audit.append(entry);continue
            if item['period_end']>doc['doc_date'] or item['period_end']>inputs.task['cutoff_date']:
                audit.append({**entry,'reason':'future_period'});continue
            key={'entity_id':eid,'doc_id':did,'duration_months':3,'period_end':item['period_end'],'period_precision':'day'}
            same=lambda r:all(r.get(k)==v for k,v in key.items())
            es=[r for r in eps['records'] if same(r)]
            sh=[r for r in financial['records'] if same(r) and r['metric']=='diluted_shares']
            tx=[r for r in financial['tax_checks'] if same(r) and r['status']=='derived' and r['usable_for_ordinary_sensitivity'] and r['scope']!='continuing_operations']
            if len(es)!=1 or len(sh)!=1 or len(tx)!=1 or sh[0]['displayed_value'] is None:
                audit.append({**entry,'reason':'missing_or_ambiguous_period_inputs'});continue
            den=sh[0]['displayed_value']*{'thousands':1e3,'millions':1e6}[sh[0]['scale']]
            if not math.isfinite(den) or den<=0:raise ValueError('invalid_shares')
            sign=1 if item['direction']=='gain' else -1;scenarios=[]
            for name,tax in [('zero_tax_assumption',0),('historical_overall_tax_assumption',tx[0]['effective_tax_rate'])]:
                effect=sign*item['amount_report_currency']*(1-tax)/den;without=es[0]['value']-effect
                if not math.isfinite(effect) or not math.isfinite(without):raise ValueError('nonfinite_sensitivity')
                scenarios.append({'assumption':name,'tax_rate':tax,'eps_effect':effect,'historical_eps_without_only_this_item':without})
            # Include bounded preceding context for economic interpretation, not period/amount binding.
            c=max(0,a-600);source={'start':a,'end':b,'text':s};context={'start':c,'end':b,'text':doc['text'][c:b]}
            result.append({**key,**item,'doc_date':doc['doc_date'],'source':source,'context':context,'input_sources':{'eps':es[0],'shares':sh[0],'tax_check':tx[0]},'scenarios':scenarios,'not_additive_or_a_forecast':True,'economic_recurrence':'unverified','item_id':'item_'+digest([did,a,b])[:16]})
    return {'status':'parsed' if result else 'no_supported_items','items':result,'audit':audit}


def packet(report,group):
    return [r for r in report['items'] if r['entity_id'] in group]
