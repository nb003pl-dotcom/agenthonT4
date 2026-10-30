"""Content-based evidence duties; query labels are hints, never completeness proof."""
import re

NUMBER=re.compile(r'(?<![A-Za-z])[-+]?\d[\d,.]*')
GUIDANCE=re.compile(r'(?is)(?:guidance|outlook|\bexpect(?:ed|s)?\b|\banticipat\w*|\bproject(?:ed|s)?\b|\bforecast\w*)[^.\n]{0,280}\d')
FORWARD=re.compile(r'(?i)\bguidance\b|\boutlook\b|\bexpect(?:ed|s)?\b|\banticipat\w*|\bproject(?:ed|s)?\b|\bforecast\w*')
QUANTIFIED=re.compile(r'\$\s*\d|\d[\d.,]*\s*(?:%|percent|billion|million|basis points|per share)',re.I)
HISTORY=re.compile(r'(?i)earnings per share|diluted earnings|net income|net sales|revenue|bid.to.cover|noncomm_net|net_%OI|as_of_\d|DGS\d|reference_month|CPI|open_interest|cash flows?')
RISK=re.compile(r'(?i)going concern|default|liquidity|bankruptcy|covenant|impairment|restructur|credit losses|uncertain|risk|decreas|declin')
DEFINITION=re.compile(r'(?i)methodology|definition|divided by|computed as|futures only|seasonally adjusted|units:|revision schedule|GAAP|diluted|90%|basis points|SAAR')
DRIVER=re.compile(r'(?i)operating (?:income|margin)|interest (?:income|rate)|inflation|employment|demand|margin|share repurchase|tax rate|provision|policy|federal funds|energy|gasoline')

def duties(row):
    text=row.get('text') or '\n'.join(s.get('text','') for s in row.get('source_spans',[]))
    roles={}
    strengths=[]
    for m in FORWARD.finditer(text):
        sentence=re.split(r'\.(?=\s+[A-Z]|$)|\n',text[m.start():m.start()+450],maxsplit=1)[0]
        quantitative=QUANTIFIED.findall(sentence)
        if quantitative:
            strengths.append(2+min(5,len(quantitative))+2*bool(re.search(r'(?i)\bbetween\b|\brange\b',sentence)))
    if strengths:roles['guidance_expectation']=max(strengths)
    if NUMBER.search(text) and HISTORY.search(text):roles['target_history']=2.0
    if DEFINITION.search(text):roles['definition_baseline']=1.0
    if RISK.search(text):roles['risk_counterevidence']=1.0
    if DRIVER.search(text):roles['financial_driver']=1.0;roles['macro_market_driver']=1.0
    return roles

def annotate(candidates, contract):
    result={}
    for entity,rows in candidates.items():
        result[entity]=[{**r,'evidence_duties':duties(r)} for r in rows]
        # Protect the strongest available slot winner, rather than all documents
        # mentioning a metric. This remains conservative, not semantic truth.
        for role in contract['required_roles']:
            eligible=[r for r in result[entity] if role in r['evidence_duties']]
            if eligible:
                best=max(eligible,key=lambda r:(r['evidence_duties'][role],r.get('score',0)))
                best.setdefault('protected_roles',[]).append(role)
    return result

def select(rows, contract, cap=5):
    chosen=[];required=contract['required_roles']; audit=[]
    for role in required:
        eligible=[r for r in rows if role in r.get('evidence_duties',duties(r))]
        if not eligible or len(chosen)>=cap:continue
        best=max(eligible,key=lambda r:(r.get('evidence_duties',duties(r)).get(role,0),r.get('relevance_status')=='relevant',r.get('score',0)))
        if best not in chosen:chosen.append(best)
        audit.append({'chunk_id':best['chunk_id'],'reason':'duty_coverage','duty':role})
    for row in sorted(rows,key=lambda r:(r.get('relevance_status')!='relevant',-r.get('score',0))):
        if row not in chosen and len(chosen)<cap:
            # Avoid spending remaining slots on duplicate overlapping passages.
            spans=row.get('source_spans',[])
            duplicate=any(other['doc_id']==row['doc_id'] and any(
                max(0,min(s['end'],t['end'])-max(s['start'],t['start']))/max(1,s['end']-s['start'])>.85
                for s in spans for t in other.get('source_spans',[])) for other in chosen)
            if duplicate:continue
            chosen.append(row);audit.append({'chunk_id':row['chunk_id'],'reason':'score_and_diversity'})
    return chosen,{'selected':audit,'omitted':[r['chunk_id'] for r in rows if r not in chosen],
        'required_roles':required,'missing_roles':[r for r in required if r not in {k for c in chosen for k in c.get('evidence_duties',duties(c))}],
        'notice':'Lexical duty detection selects source for reading; fact semantics must still be validated.'}
