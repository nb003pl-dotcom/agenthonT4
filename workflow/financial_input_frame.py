"""Source-bound historical shares/tax rows; not an EPS forecast or item tax rate."""
import re,math
from collections import defaultdict
from .eps_source_frame import header,normalized,NEXT_STATEMENT
from .statement_context import build as contexts,TITLES

NUM=r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+'
CELL=re.compile(r'\s*\$?\s*(?P<v>\(\s*\$?\s*(?:'+NUM+r')\s*\)|-?(?:'+NUM+r')|[—–])(?=\s|$|\))')
LABELS={
 'pretax_income':re.compile(r'\b(?:income|earnings)(?:\s*\(loss\)|/\(loss\))?(?:\s+from\s+continuing\s+operations)?\s+before\s+(?:provision\s+for\s+)?income\s+tax(?:es|\s+expense)(?:\s+and\s+noncontrolling\s+interests)?\b|\bpre[- ]tax earnings\b',re.I),
 'tax_line':re.compile(r'\b(?:income tax expense|provision(?:\s*\(credit\))? for (?:income )?taxes|applicable income taxes|income taxes)\b',re.I),
 'posttax_income':re.compile(r'\b(?:net (?:income|earnings)(?:\s*\(loss\)|/\(loss\))?(?: before (?:attribution to )?noncontrolling interests)?|income from continuing operations)(?=\s|$)',re.I),
 'aftertax_equity':re.compile(r'\bequity in earnings/\(losses\) of unconsolidated entities\b',re.I),
 'diluted_shares':re.compile(r'\b(?:(?:total|adjusted)\s+)?(?:(?:weighted[- ]average|average)\s+diluted(?:\s+common)?\s+shares(?:\s+issued\s+and)?(?:\s+outstanding)?|diluted\s+(?:weighted[- ]average|average)(?:\s+common)?\s+shares(?:\s+outstanding)?(?:\s*\(denominator\))?|weighted[- ]average\s+common\s+shares\s+outstanding\s*(?:applicable to diluted EPS|-\s*diluted|and common stock equivalents,\s*diluted))(?:\s*\(in\s+(?:millions|thousands)\))?',re.I),
}
SHARES_HEADING=re.compile(r'\b(?:shares used in calculation of earnings per share|(?:weighted[- ]average|average)(?: common)? shares)\s*:?\s*',re.I)
SCALE=re.compile(r'\b(?:in|of)\s+(millions|thousands)\b',re.I)


def cells(text,start,n):
    values=[];cursor=start
    for _ in range(n):
        m=CELL.match(text,cursor)
        if not m:return None
        token=m['v'];v=None if token in ('—','–') else float(re.sub(r'[\s,$()]','',token))
        if v is not None:
            if not math.isfinite(v):return None
            if '(' in token:v=-abs(v)
        values.append(v);cursor=m.end()
    if CELL.match(text,cursor):return None
    return values,cursor


def scale_basis(prefix,label,metric):
    local=SCALE.search(label)
    if local and metric=='diluted_shares':return local[1].lower(),'row',max(0,local.start()-25),local.end()
    # A dollars-only header does not state share scale.
    for m in SCALE.finditer(prefix):
        before=prefix[max(0,m.start()-30):m.start()]
        if metric=='diluted_shares' and re.search(r'dollars\s*$',before,re.I):continue
        if metric=='diluted_shares' and re.match(r'\s+of\s+dollars',prefix[m.end():],re.I):continue
        return m[1].lower(),'statement_header',max(0,m.start()-25),m.end()
    return None,None,None,None


def parse_window(w):
    text=w['text'];t=normalized(text);h=header(t)
    if not h:return [],{'reason':'unsupported_header'}
    if not any(p.match(t) for p in TITLES.values()):return [],{'reason':'unsupported_heading'}
    stop=NEXT_STATEMENT.search(t,h['end']);limit=stop.start() if stop else len(t);body=t[:limit]
    income=bool(TITLES['income_statement'].match(t));found=[]
    eps_start=re.search(r'\b(?:earnings(?:\s*\(loss\))? per (?:common )?share|per common share information|basic EPS)\b',body[h['end']:],re.I)
    money_limit=h['end']+eps_start.start() if eps_start else limit
    for metric,pattern in LABELS.items():
        if metric!='diluted_shares' and not income:continue
        for m in pattern.finditer(body,h['end'],limit if metric=='diluted_shares' else money_limit):
            # Never treat the 'income taxes' inside a pretax label as the tax row.
            if metric=='tax_line' and re.search(r'\b(?:before(?:\s+provision\s+for)?|benefit for)\s*$',body[max(0,m.start()-35):m.start()],re.I):continue
            v=cells(body,m.end(),len(h['columns']))
            if v:found.append((metric,m.start(),v[1],v[0],m[0],m.start()))
    for m in re.finditer(r'\bDiluted\b',body[h['end']:],re.I):
        a=h['end']+m.start();b=h['end']+m.end();hs=list(SHARES_HEADING.finditer(body,max(h['end'],a-180),a))
        if not hs:continue
        head=hs[-1]
        if re.search(r'earnings(?:\s*\(loss\))?\s+per\s+(?:common\s+)?share',body[head.end():a],re.I):continue
        v=cells(body,b,len(h['columns']))
        if v:found.append(('diluted_shares',a,v[1],v[0],body[head.start():head.end()],head.start()))
    records=[];skipped=[]
    for metric,a,b,values,label,label_start in found:
        nearest=header(text[:a],latest=True)
        if metric=='diluted_shares' and any(v is not None and v<=0 for v in values):
            skipped.append({'metric':metric,'reason':'nonpositive_shares','start':w['start']+a});continue
        if not nearest or nearest['columns']!=h['columns']:
            skipped.append({'metric':metric,'reason':'different_header','start':w['start']+a});continue
        scale,basis,ua,ub=scale_basis(t[:min(a,h['end']+150)],label,metric)
        if not scale:
            skipped.append({'metric':metric,'reason':'unsupported_unit','start':w['start']+a});continue
        if basis=='row':ua+=label_start;ub+=label_start
        for i,(col,value) in enumerate(zip(h['columns'],values)):
            records.append({**col,'metric':metric,'displayed_value':value,'scale':scale,'unit':('shares_' if metric=='diluted_shares' else 'report_currency_')+scale,
                'unit_basis':basis,'unit_source':{'start':w['start']+ua,'end':w['start']+ub,'text':text[ua:ub]},'column_index':i,'entity_id':w['entity_id'],'doc_id':w['doc_id'],'doc_date':w['doc_date'],
                'scope_basis':'exclusive corpus association; source issuer not independently resolved',
                'header':{'start':w['start']+nearest['start'],'end':w['start']+nearest['end'],'text':text[nearest['start']:nearest['end']]},
                'row':{'start':w['start']+a,'end':w['start']+b,'text':text[a:b]},
                'window_start':w['start'],'scope':'continuing_operations' if 'continuing operations' in label.lower() else 'before_noncontrolling_interests' if re.search(r'before (?:attribution to )?noncontrolling interests',label,re.I) else 'reported_statement'})
    return records,{'reason':'parsed' if records else 'no_supported_rows','skipped':skipped}


def tax_check(w,rows):
    """Tax display sign is resolved only with complete local equation rows."""
    results=[];bycol=defaultdict(list)
    for r in rows:
        if r['metric']!='diluted_shares':bycol[(r['duration_months'],r['period_end'],r['period_precision'],r['column_index'])].append(r)
    for key,rs in bycol.items():
        by=defaultdict(list)
        for r in rs:by[r['metric']].append(r)
        if len(by['pretax_income'])!=1:continue
        pre=by['pretax_income'][0]
        taxes=[r for r in by['tax_line'] if r['row']['start']>pre['row']['end']]
        if not taxes:continue
        tax=min(taxes,key=lambda r:r['row']['start'])
        posts=[r for r in by['posttax_income'] if r['row']['start']>tax['row']['end']]
        if not posts:continue
        post=min(posts,key=lambda r:r['row']['start'])
        if pre['scope']=='continuing_operations' and post['scope']!='continuing_operations':continue
        extras=[r for r in by['aftertax_equity'] if tax['row']['end']<=r['row']['start']<post['row']['start']]
        base=w['start'];between=w['text'][tax['row']['end']-base:post['row']['start']-base]
        for r in extras:between=between.replace(r['row']['text'],'',1)
        used=[pre,tax,post]+extras
        record={'entity_id':w['entity_id'],'doc_id':w['doc_id'],'doc_date':w['doc_date'],**{k:pre[k] for k in ('duration_months','period_end','period_precision','scope')},
                'sources':used,'status':'uncomputed','not_item_specific_tax_rate':True}
        if between.strip():
            record['reason']='unknown_intermediate_rows';results.append(record);continue
        if any(r['displayed_value'] is None for r in used) or len({r['scale'] for r in used})!=1:
            record['reason']='missing_value_or_scale_mismatch';results.append(record);continue
        p,t,n=[r['displayed_value'] for r in (pre,tax,post)];e=sum(r['displayed_value'] for r in extras)
        # Displayed amounts rounded to whole millions/thousands; allow half a displayed unit per participating row.
        tolerance=len(used)*.5
        signs=[s for s in (1,-1) if abs(p-s*t+e-n)<=tolerance]
        if len(signs)==2 and t==0:signs=[1]
        if len(signs)!=1:
            record['reason']='tax_sign_ambiguous_or_identity_mismatch';results.append(record);continue
        expense=signs[0]*t;rate=expense/p if p else None
        if rate is not None and not math.isfinite(rate):
            record['reason']='arithmetic_overflow';results.append(record);continue
        record.update(status='derived',display_sign=signs[0],tax_expense=expense,effective_tax_rate=rate,
           formula='(pretax_income - tax_expense + disclosed_aftertax_equity) = posttax_income',
           residual=p-expense+e-n,rounding_tolerance=tolerance,usable_for_ordinary_sensitivity=bool(p>0 and rate is not None and 0<=rate<=1),
           warning='Historical overall effective rate, not a disclosed marginal tax rate for any item. Future use is an assumption.')
        results.append(record)
    return results


def build(inputs):
    cs,audit=contexts(inputs)
    if audit==[{'reason':'unsupported_family'}]:return {'status':'not_applicable','records':[],'tax_checks':[],'windows':[],'conflicts':[]}
    records=[];checks=[];windows=[]
    for eid,ws in cs.items():
        for w in ws:
            rs,detail=parse_window(w);accepted=[]
            for r in rs:
                if not inputs.allowed(eid,r['doc_id']) or r['doc_date']>inputs.task['cutoff_date']:raise ValueError('source_not_allowed')
                for name in ('header','row','unit_source'):
                    s=r[name]
                    if inputs.docs[r['doc_id']]['text'][s['start']:s['end']]!=s['text']:raise ValueError('source_span_mismatch')
                if r['period_end']>r['doc_date'][:len(r['period_end'])]:raise ValueError('period_after_document')
                accepted.append(r)
            records.extend(accepted);checks.extend(tax_check(w,accepted));windows.append({'entity_id':eid,'doc_id':w['doc_id'],'start':w['start'],**detail})
    groups=defaultdict(list)
    for r in records:groups[(r['entity_id'],r['doc_id'],r['metric'],r['scope'],r['duration_months'],r['period_end'],r['period_precision'])].append(r)
    accepted=[];conflicts=[]
    for key,rs in groups.items():
        if len({(r['displayed_value'],r['scale']) for r in rs})>1:conflicts.append({'key':key,'sources':rs})
        else:accepted.append({**rs[0],'corroborating_sources':rs[1:]})
    # A conflicting tax input must not remain as an apparently usable derived rate.
    bad={(r['entity_id'],r['doc_id'],r['duration_months'],r['period_end'],r['period_precision']) for c in conflicts for r in c['sources'] if r['metric']!='diluted_shares'}
    for c in checks:
        if (c['entity_id'],c['doc_id'],c['duration_months'],c['period_end'],c['period_precision']) in bad:
            c.update(status='uncomputed',reason='conflicting_source_values',usable_for_ordinary_sensitivity=False)
            c.pop('effective_tax_rate',None)
    return {'status':'parsed' if accepted else 'no_supported_records','records':accepted,'tax_checks':checks,'windows':windows,'conflicts':conflicts,
            'warning':'Historical source rows only. No automatic EPS adjustment or future forecast.'}
