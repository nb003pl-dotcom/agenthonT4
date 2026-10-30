"""Conservative, source-bound diluted EPS table reader. No model or outcomes."""
import re
import math
from datetime import date
from .statement_context import build as statement_contexts, TITLES

MONTHS={name:i for i,name in enumerate(('january','february','march','april','may','june','july','august','september','october','november','december'),1)}
MONTHS.update({k[:3]:v for k,v in list(MONTHS.items())})
MONTH=r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?'
MD=re.compile(r'\b('+MONTH+r')(?:\s+(\d{1,2})(?!\d))?',re.I)
FULL_DATE=re.compile(r'\b('+MONTH+r')\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b',re.I)
PERIOD=re.compile(r'\b(?:(three|six|nine)\s+months?|quarter)\s+ended\b',re.I)
YEAR_BLOCK=re.compile(r'\b(?:19|20)\d{2}(?:\s+(?:19|20)\d{2}){1,3}\b')
EPS_LABEL=re.compile(r'\bdiluted\s+earnings(?:\s*\(loss\))?(?:\s+per\s+(?:common\s+)?share)?\b|\bearnings\s*(?:\(loss\)\s*)?per\s+(?:common\s+)?share(?:\s+of\s+common\s+stock)?\s*[-—–]\s*(?:diluted|assuming dilution)\b',re.I)
BARE=re.compile(r'\bdiluted\b',re.I)
CELL=re.compile(r'\s*\$?\s*(?P<value>\(\s*\$?\s*(?:\d+\.\d+|\.\d+)\s*\)|-?(?:\d+\.\d+|\.\d+))')
NEXT_STATEMENT=re.compile(r'\b(?:condensed\s+)?consolidated\s+(?:statements?\s+of|(?:comprehensive\s+)?income\s+statement)',re.I)


def normalized(text):return text.translate(str.maketrans({'\xa0':' ','\u200b':' ','−':'-'}))

def months(marker):return {'three':3,'six':6,'nine':9}.get((marker.group(1) or '').lower(),3)

def column(year,month,day,duration):
    y=int(year);m=MONTHS[month.lower().rstrip('.')]
    end=f'{y:04d}-{m:02d}';precision='month'
    if day:
        end=date(y,m,int(day)).isoformat();precision='day'
    return {'duration_months':duration,'period_end':end,'period_precision':precision}


def header(text, *, latest=False):
    t=normalized(text);prefix=t if latest else t[:900];options=[]
    # End dates written in each column, e.g. April 1, 2023 March 26, 2022.
    dates=list(FULL_DATE.finditer(prefix))
    for a,b in zip(dates,dates[1:]):
        if prefix[a.end():b.start()].strip():continue
        ps=[m for m in PERIOD.finditer(prefix[:a.start()])]
        if not ps or a.start()-ps[-1].end()>200:continue
        if int(a[3])!=int(b[3])+1 or months(ps[-1])!=3:continue
        try:cols=[column(d[3],d[1],d[2],3) for d in (a,b)]
        except ValueError:continue
        options.append({'start':ps[-1].start(),'end':b.end(),'columns':cols})
    # Shared end-month/day labels followed by a pure row of years.
    for block in YEAR_BLOCK.finditer(prefix):
        years=re.findall(r'\d{4}',block[0]);n=len(years)
        if n not in (2,4) or int(years[0])!=int(years[1])+1 or years!=years[:2]*(n//2):continue
        ps=list(PERIOD.finditer(prefix[:block.start()]))
        if len(ps)<n//2:continue
        selected=ps[-n//2:]
        if months(selected[0])!=3 or (n==4 and months(selected[1]) not in (6,9)):continue
        if block.start()-selected[0].end()>350:continue
        segments=[prefix[m.end():(selected[i+1].start() if i+1<len(selected) else block.start())] for i,m in enumerate(selected)]
        try:
            cols=[]
            for i,(marker,segment) in enumerate(zip(selected,segments)):
                md=list(MD.finditer(segment))
                if len(md)==1:md=md*2
                if len(md)!=2:raise ValueError('ambiguous_month_columns')
                # Do not read prose annual comparisons as a table header.
                if re.search(r'\b(?:19|20)\d{2}\b',segment):raise ValueError('prose_years_in_header')
                cols.extend(column(years[i*2+j],x[1],x[2],months(marker)) for j,x in enumerate(md))
        except (ValueError,KeyError):continue
        options.append({'start':selected[0].start(),'end':block.end(),'columns':cols})
    if not options:return None
    return (max if latest else min)(options,key=lambda h:(h['end'],h['start']))


def cells(text,start,count):
    cursor=start
    foot=re.match(r'\s*\([a-z0-9]\)',text[cursor:],re.I)
    if foot:cursor+=foot.end()
    values=[];end=None
    for _ in range(count):
        m=CELL.match(text,cursor)
        if not m:return None
        token=m['value'];num=re.sub(r'[\s$()]','',token)
        value=float(num)
        if not math.isfinite(value):return None
        if '(' in token:value=-abs(value)
        values.append(value);cursor=m.end();end=cursor
    if CELL.match(text,cursor):return None  # Extra numeric column: refuse.
    return values,end


def parse_window(window):
    original=window['text'];t=normalized(original)
    if not any(p.match(t) for p in TITLES.values()):return [],{'reason':'unsupported_statement_heading'}
    h=header(original)
    if not h:return [],{'reason':'unsupported_or_ambiguous_header'}
    boundary=NEXT_STATEMENT.search(t,h['end']);limit=boundary.start() if boundary else len(t)
    body=t[:limit];matches=[];n=len(h['columns'])
    for m in EPS_LABEL.finditer(body,h['end']):
        if re.search(r'(?:adjusted|non[- ]GAAP|core)\s*$',body[max(0,m.start()-40):m.start()],re.I):continue
        if re.match(r'\s*from\s+(?:continuing|discontinued)\s+operations',body[m.end():],re.I):continue
        parsed=cells(body,m.end(),n)
        if parsed:matches.append((m.start(),parsed[1],parsed[0],'explicit_diluted_eps'))
        # Continuing and discontinued operations require the net-income total.
        if not parsed and re.search(r'diluted\s+earnings.*per.*share',m[0],re.I):
            stop=min(limit,m.end()+1000)
            other_basis=re.search(r'\bBasic\b',body[m.end():stop],re.I)
            if other_basis:stop=m.end()+other_basis.start()
            for total in re.finditer(r'\bNet income\b',body[m.end():stop],re.I):
                a=m.end()+total.start();b=m.end()+total.end();v=cells(body,b,n)
                if v:matches.append((m.start(),v[1],v[0],'diluted_section_net_income'));break
    for m in BARE.finditer(body,h['end']):
        v=cells(body,m.end(),n)
        if not v:continue
        before=body[max(h['end'],m.start()-250):m.start()]
        eps_heads=list(re.finditer(r'earnings\s*(?:[/( ]loss[) ]*)?per\s+(?:common\s+)?share',before,re.I))
        if not eps_heads:continue
        scope_text=before[max(0,eps_heads[-1].start()-60):]
        before=before[eps_heads[-1].start():]
        if re.search(r'(?:continuing|discontinued)\s+operations|non[- ]GAAP',scope_text,re.I):continue
        if re.search(r'(?:adjusted|non[- ]GAAP|core)\s*$',before,re.I):continue
        if re.search(r'average.*shares|weighted.*shares|shares used|denominator',before,re.I):continue
        matches.append((m.start(),v[1],v[0],'diluted_row_in_eps_section'))
    for m in re.finditer(r'\bAssuming dilution\b',body[h['end']:],re.I):
        a=h['end']+m.start();b=h['end']+m.end()
        before=body[max(h['end'],a-150):a]
        if not re.search(r'earnings.*per share',before,re.I):continue
        segment=body[b:min(limit,b+600)];basic=re.search(r'\bBasic\b',segment,re.I)
        if basic:segment=segment[:basic.start()]
        total=re.search(r'\bTotal\b',segment,re.I)
        if total:
            v=cells(body,b+total.end(),n)
            if v:matches.append((a,v[1],v[0],'diluted_section_total'))
    # A non-GAAP/adjusted-EPS section is outside this narrow reported-EPS reader.
    matches=[r for r in matches if not re.search(r'non[- ]GAAP|\badjusted\s+(?:diluted\s+)?earnings|\bcore\s+(?:diluted\s+)?earnings',body[h['end']:r[1]],re.I)]
    matched_headers={a:header(original[:a],latest=True) for a,_,_,_ in matches}
    skipped_period_rows=sum(not matched_headers[a] or matched_headers[a]['columns']!=h['columns'] for a,_,_,_ in matches)
    matches=[r for r in matches if matched_headers[r[0]] and matched_headers[r[0]]['columns']==h['columns']]
    distinct={tuple(v) for _,_,v,_ in matches}
    if len(distinct)!=1:return [],{'reason':'no_supported_diluted_row' if not distinct else 'conflicting_diluted_rows','candidates':len(matches)}
    a,b,values,method=min(matches,key=lambda x:(x[1]-x[0],x[0]))
    h=matched_headers[a]
    result=[]
    for i,(c,value) in enumerate(zip(h['columns'],values)):
        result.append({**c,'column_index':i,'value':value,'metric':'diluted_eps_net_income','unit':'report_currency_per_share',
            'row_method':method,'entity_id':window['entity_id'],'doc_id':window['doc_id'],'doc_date':window['doc_date'],
            'scope_basis':'exclusive corpus association and statement/EPS heading; issuer not independently resolved',
            'header':{'start':window['start']+h['start'],'end':window['start']+h['end'],'text':original[h['start']:h['end']]},
            'row':{'start':window['start']+a,'end':window['start']+b,'text':original[a:b]}})
    return result,{'reason':'parsed','columns':n,'row_method':method,'skipped_different_period_rows':skipped_period_rows}


def build(inputs):
    contexts,context_audit=statement_contexts(inputs);records=[];audits=[];rejected=[]
    if context_audit==[{'reason':'unsupported_family'}]:
        return {'status':'not_applicable','records':[],'conflicts':[],'windows':[],'rejected_records':[],'entities_without_records':[],
                'warning':'This parser only applies to the supported EPS task families.'}
    for eid,windows in contexts.items():
        for w in windows:
            rows,audit=parse_window(w)
            for r in rows:
                for kind in ('header','row'):
                    s=r[kind]
                    if inputs.docs[r['doc_id']]['text'][s['start']:s['end']]!=s['text']:raise ValueError('frame_source_mismatch')
                if r['period_end']>r['doc_date'][:len(r['period_end'])]:
                    rejected.append({**r,'reason':'period_after_document_date'})
                else:records.append(r)
            audits.append({'entity_id':eid,'doc_id':w['doc_id'],'window_start':w['start'],**audit})
    groups={}
    for r in records:
        key=(r['entity_id'],r['duration_months'],r['period_end'],r['period_precision'],r['metric'])
        groups.setdefault(key,[]).append(r)
    accepted=[];conflicts=[]
    for key,rows in groups.items():
        if len({r['value'] for r in rows})!=1:conflicts.append({'key':key,'sources':rows})
        else:accepted.append({**rows[0],'corroborating_sources':rows[1:]})
    return {'status':'parsed' if accepted else 'no_supported_records','records':accepted,'conflicts':conflicts,'windows':audits,'rejected_records':rejected,
        'entities_without_records':[e for e in inputs.entities if not any(r['entity_id']==e for r in accepted)],
        'warning':'Historical source parsing only. Not a future EPS estimate; scope/issuer association is not independent semantic verification.'}
