import math
import re
from .common import digest
from t4_splitter import count_tokens
from .contract import clean_task


def locate(text, quote, spans):
    if not isinstance(quote,str) or not quote.strip():raise ValueError('empty_quote')
    matches=set()
    for s in spans:
        a=s['start']
        while True:
            i=text.find(quote,a,s['end'])
            if i<0:break
            matches.add((i,i+len(quote)));a=i+1
    if len(matches)>1:raise ValueError('quote_missing_or_ambiguous')
    if not matches:
        # Only whitespace may differ. Search inside each already-read span and
        # map the match back to original characters; no fuzzy words or numbers.
        parts=re.split(r'(\s+)',quote)
        pattern=''.join(r'\s+' if part.isspace() else re.escape(part) for part in parts)
        for span in spans:
            segment=text[span['start']:span['end']]
            for match in re.finditer('(?=('+pattern+'))',segment):
                a,b=match.span(1)
                matches.add((span['start']+a,span['start']+b))
    if len(matches)!=1:raise ValueError('quote_missing_or_ambiguous')
    return next(iter(matches))


def validate_facts(data, inputs, entity, chunks, trust_catalog=False, source_guard=False, *, max_facts=12):
    if not isinstance(data,dict) or not isinstance(data.get('facts'),list):raise ValueError('facts must be a list')
    if type(max_facts) is not int or not 1<=max_facts<=12:raise ValueError('max_facts must be 1..12')
    if len(data['facts'])>max_facts:raise ValueError(f'at most {max_facts} facts per batch')
    by_id={}
    for c in chunks:
        if c['chunk_id'] not in by_id: by_id[c['chunk_id']]={**c,'source_spans':list(c['source_spans'])}
        else: by_id[c['chunk_id']]['source_spans'].extend(c['source_spans'])
    good=[];rejected=[]
    for raw in data['facts']:
        try:
            if raw.get('_citation_restore_error'):raise ValueError(raw['_citation_restore_error'])
            if raw.get('entity_id')!=entity:raise ValueError('fact_wrong_entity')
            statement=raw.get('statement')
            if not isinstance(statement,str) or not statement.strip():raise ValueError('missing_statement')
            citations=[]
            if not raw.get('citations'):raise ValueError('fact_without_citations')
            for c in raw['citations']:
                chunk=by_id.get(c.get('chunk_id'))
                if chunk is None or chunk['doc_id']!=c.get('doc_id'):raise ValueError('citation_not_in_read_chunks')
                if not inputs.allowed(entity,c['doc_id']):raise ValueError('citation_wrong_entity')
                quote=c['quote']
                if len(quote)>1600 or count_tokens(quote)>350:raise ValueError('quote_too_long_for_claim')
                if trust_catalog and '_catalog_span' in c:
                    s=c['_catalog_span'];a,b=s['start'],s['end']
                    if not any(t['start']<=a<b<=t['end'] for t in chunk['source_spans']) or inputs.docs[c['doc_id']]['text'][a:b]!=quote:raise ValueError('Invalid catalog source span')
                else:a,b=locate(inputs.docs[c['doc_id']]['text'],quote,chunk['source_spans'])
                source_quote=inputs.docs[c['doc_id']]['text'][a:b]
                from .evaluation import judge_token_counter
                if judge_token_counter()(source_quote)>400:raise ValueError('quote_exceeds_official_judge_limit')
                if len(source_quote)>1600 or count_tokens(source_quote)>350:raise ValueError('quote_too_long_for_claim')
                citation={'doc_id':c['doc_id'],'chunk_id':c['chunk_id'],'quote':source_quote,
                          'alignment':'exact' if source_quote==quote else 'whitespace_only',
                          'model_quote':quote,
                          'span_start':a,'span_end':b,'doc_date':inputs.docs[c['doc_id']]['doc_date'],
                          'table_refs':chunk['table_refs']}
                citation['citation_id']='c_'+digest(citation)[:16];citations.append(citation)
            fields=('statement','source_entity_id','evidence_role','metric','value','unit','period','comparison_period','basis')
            fact={k:raw.get(k) for k in fields}
            from .financial_scope import metadata
            fact.update(metadata(raw))
            if fact.get('value') is not None and (type(fact['value']) not in (int,float) or not math.isfinite(fact['value'])):raise ValueError('nonfinite_or_nonnumeric_fact')
            fact.update(entity_id=entity,citations=citations,validation={'source_located':True,'permission_valid':True,
                       'date_valid':True,'semantic_status':'unverified'})
            if trust_catalog and raw.get('_citation_completion'):
                fact['validation']['citation_completion']=raw['_citation_completion']
            if source_guard:
                from .fact_source_guard import numeric_source_issue
                issue=numeric_source_issue(fact)
                if issue:raise ValueError(issue)
                fact['validation']['numeric_source_guard']='passed_narrow_check_not_semantic_validation'
            missing_range=unsupported_literal_range(fact)
            if missing_range:raise ValueError(missing_range)
            missing_eps=unsupported_literal_eps_amounts(fact)
            if missing_eps:raise ValueError(missing_eps)
            from .financial_scope import arithmetic_issue
            issue=arithmetic_issue(fact)
            if issue:raise ValueError(issue+'; verify periods/amounts and remove the contradictory comparison')
            fact['fact_id']='f_'+digest([entity,fact])[:16]
            good.append(fact)
        except (ValueError,KeyError,TypeError) as exc:
            rejected.append({'fact':raw,'error':str(exc)})
    numeric_warnings=[w for f in good for checker in (numeric_direction_warning,money_scale_warning) if (w:=checker(f))]
    return {'facts':good,'rejected':rejected,'conflicts':list(data.get('conflicts',[]))+numeric_warnings,
            'unresolved_questions':data.get('unresolved_questions',[])}


def fact_payload(task, entity, chunks, output, table_reader=None, contract=None, legacy=False):
    tables={}
    for chunk in chunks:
        for ref in chunk['table_refs']:
            rid=ref.get('ref_id')
            if rid:
                info={'ref':ref,'read_artifact':str(output/'tables-read'/f'{digest(rid)}.json')}
                if table_reader and rid not in tables:
                    contents=table_reader(rid)
                    # The complete workbook remains in the read artifact and source
                    # spans. Explicit previews never masquerade as a full table.
                    original=contents.get('original',{})
                    table_rows=original.get('raw_rows') or original.get('rows') or contents.get('rows',[])
                    row_spans=original.get('row_spans',[])
                    # Large workbooks: expose only rows represented by this read
                    # window, with an explicit row map. Never expose duplicated
                    # full original+typed+raw tables in every source window.
                    indices=list(range(len(table_rows)))
                    if row_spans and len(row_spans)==len(table_rows):
                        def overlaps(rs):
                            a,b=(rs.get('start'),rs.get('end')) if isinstance(rs,dict) else rs[:2]
                            return any(max(a,s['start'])<min(b,s['end']) for s in chunk['source_spans'])
                        indices=[i for i,rs in enumerate(row_spans) if overlaps(rs)]
                    info['contents']={'header':original.get('raw_header') or contents.get('header'),
                        'rows':[table_rows[i] for i in indices], 'row_indices':indices,
                        'total_rows':len(table_rows),'complete':len(indices)==len(table_rows),
                        'source':contents.get('source'), 'numeric_representation':'original cells; percentage signs preserved'}
                tables.setdefault(rid,info)
    if legacy:
        return {'task':{k:v for k,v in task.items() if k!='entities'},'entity':entity,'chunks':chunks,'table_reads':tables,
            'limits':{'max_facts':5,'quote_max_characters':700},
            'instruction':'Table source text is present in source_spans. Use those exact quotes. Return facts and conflicts, not predictions.'}
    return {'task':clean_task(task), 'target_contract':contract,
            'entity':entity,'chunks':chunks,'table_reads':tables,
            'limits':{'max_facts':7,'quote_max_characters':1000},
            'instruction':'Use source_spans for exact citations. Table contents supply headers/units/columns. Extract target-period guidance BEFORE redundant historical facts. Absence means not found in the PROVIDED windows, never absent from the entire corpus.'}


def extraction_batches(chunks, max_tokens=16000, *, fits=None):
    # Source windows are prompt packing, not new retrieval/embedding chunks.
    if max_tokens<=1500:raise ValueError('facts token budget too small')
    fits=fits or (lambda batch:True)
    units=[]
    for chunk in chunks:
        metadata={k:v for k,v in chunk.items() if k not in ('text','source_spans')}
        for span in chunk['source_spans']:
            text=span['text'];a=0
            while a<len(text):
                def window(end):
                    return {**metadata,'source_spans':[{**span,'start':span['start']+a,
                            'end':span['start']+end,'text':text[a:end]}]}
                lo=a+1;hi=len(text);end=None
                while lo<=hi:
                    mid=(lo+hi)//2
                    if count_tokens(text[a:mid])<=max_tokens-1500 and fits([window(mid)]):end=mid;lo=mid+1
                    else:hi=mid-1
                if end is None:raise ValueError('facts metadata exceeds input budget even with one source character')
                units.append(window(end))
                a=end
    batch=[];size=0
    for unit in units:
        n=count_tokens(unit['source_spans'][0]['text'])+300
        if batch and (size+n>max_tokens or not fits(batch+[unit])):
            yield batch;batch=[];size=0
        batch.append(unit);size+=n
    if batch:yield batch


def compact(facts):
    return [{k:v for k,v in f.items() if k!='validation'} | {
        'citations':[{'citation_id':c['citation_id'],'quote':c['quote'],'doc_date':c['doc_date']} for c in f['citations']]
        } for f in facts]


def handle_rejected_window(result, had_facts, attempt, entity, batch):
    """Repair once, then keep rejection audit instead of losing earlier facts.

    This does not create a usable fact. The caller must still require supported
    facts for every entity before forecasting.
    """
    if had_facts and not result['facts']:
        if attempt==1:
            # Model.call keeps only 500 characters. Put the actionable reasons
            # first; full fact/citation serialization used to hide every error.
            prefix='all facts invalid: '
            suffix='; Fix listed facts using exact supporting quote_ids. Do not invent values.'
            errors=[' '.join(str(r.get('error','invalid_fact')).split()).split(';',1)[0]
                    for r in result['rejected']] or ['invalid_fact']
            width=max(1,(500-len(prefix)-len(suffix)-2*(len(errors)-1))//len(errors))
            issues=[(str(i+1)+':'+error)[:width] for i,error in enumerate(errors)]
            raise ValueError(prefix+'; '.join(issues)+suffix)
        result['unresolved_questions'].append({'entity_id':entity,
            'reason':'all_facts_rejected_after_one_repair','batch':batch})
    return result


def unsupported_literal_range(fact):
    """Narrow literal-bound check; does not claim to verify entailment.

    Skip ambiguous date/count and monetary-scale conversions. Check combined
    quotations so separate range/header Q entries remain valid.
    """
    statement=fact.get('statement','')
    if re.search(r'\b(?:million|billion|thousand)\b',statement,re.I):return None
    numeric=r'([+-]?(?:\d+\.\d+|\d+))'
    patterns=[r'\bbetween\s+\$?'+numeric+r'\s*(?:%|percent)?\s+and\s+\$?'+numeric,
              r'\b(?:range|ranging)\s+(?:of\s+|from\s+)?\$?'+numeric+r'\s*(?:%|percent)?\s*(?:to|–|—|-)\s*\$?'+numeric]
    quotes=' '.join(c['quote'] for c in fact.get('citations',[]))
    values=[float(n.replace(',','')) for n in re.findall(r'(?<![\w.])[+-]?(?:\d[\d,]*\.\d+|\.\d+|\d[\d,]*)(?![\w.])',quotes)]
    for whole,numerator,denominator in re.findall(r'\b(\d+)-(\d+)/(\d+)\b',quotes):
        if int(denominator):values.append(int(whole)+int(numerator)/int(denominator))
    for pattern in patterns:
        for match in re.finditer(pattern,statement,re.I):
            a,b=map(float,match.groups())
            if 1900<=a<=2100 or 1900<=b<=2100:continue
            if '.' not in match[0] and '%' not in statement and 'percent' not in statement.lower():continue
            missing=[x for x in (a,b) if not any(math.isclose(x,v,rel_tol=1e-8,abs_tol=1e-8) for v in values)]
            if missing:return 'literal_range_bounds_not_in_citations: '+str(missing)+'; select the original range plus its metric/period context, or omit this fact'
    return None


def unsupported_literal_eps_amounts(fact):
    """Require literal reported EPS dollar magnitudes in their actual quotes.

    Does not resolve columns/periods, signs or calculated EPS. These remain
    semantic checks. It only catches plainly invented quoted dollar amounts.
    """
    statement=fact.get('statement','')
    if not re.search(r'\b(?:EPS|earnings per (?:common )?share)\b',statement,re.I):return None
    if re.search(r'\b(?:calculated|computed|derived|approximately|estimated|million|billion|thousand)\b',statement,re.I):return None
    # A mixed EPS fact may also mention net earnings "$2.89B", whose
    # source table is in millions (2891). Such scale conversions are outside
    # this literal EPS check; do not mistake the abbreviated billions for EPS.
    tokens=re.findall(r'\$\s*([+-]?(?:\d[\d,]*\.\d+|\.\d+|\d[\d,]*))(?:\s*(billion|million|thousand|bn|mn|[BMK])\b)?',statement,re.I)
    amounts=[abs(float(n.replace(',',''))) for n,scale in tokens if not scale]
    quotes=' '.join(c['quote'] for c in fact.get('citations',[]))
    values=[abs(float(n.replace(',',''))) for n in re.findall(r'(?<![\w.])[+-]?(?:\d[\d,]*\.\d+|\.\d+|\d[\d,]*)(?![\w.])',quotes)]
    missing=[x for x in amounts if not any(math.isclose(x,v,rel_tol=1e-8,abs_tol=1e-8) for v in values)]
    if missing:return 'literal_eps_amount_not_in_citations: '+str(missing)+'; select actual EPS row with year/quarter headers, or omit this fact'
    return None


def numeric_direction_warning(fact):
    """Flag a plainly contradictory up/down statement, without guessing columns."""
    value=fact.get('value');text=fact.get('statement','')
    if type(value) not in (float,int):return None
    match=re.search(r'\b(up|down|higher|lower|increased|decreased)\b[^;]{0,35}?\b(?:from|versus|compared with)\s+\$?([+-]?(?:\d[\d,]*\.?\d*|\.\d+))',text,re.I)
    if not match:return None
    comparison=float(match[2].replace(',',''))
    quotes=' '.join(c['quote'] for c in fact.get('citations',[]))
    # Both amounts must occur literally in the source; don't impose a trend
    # inferred from a different metric, date, or invented comparison amount.
    numbers={float(n.replace(',','')) for n in re.findall(r'(?<![A-Za-z\d])[+-]?(?:\d[\d,]*\.\d+|\.\d+|\d[\d,]*)(?![\d])',quotes)}
    if value not in numbers or comparison not in numbers:return None
    direction=match[1].lower();up=direction in ('up','higher','increased')
    if (up and value<comparison) or (not up and value>comparison):
        return f"Arithmetic warning for {fact['entity_id']} fact {fact['fact_id']}: statement says {direction}, but {value} versus {comparison} has the opposite direction. Verify the comparison belongs to this metric and period before using that directional word."
    return None


def money_scale_warning(fact):
    """Detect a narrow statement/field scaling disagreement, not source semantics."""
    value=fact.get('value');unit=str(fact.get('unit','')).lower().strip()
    if type(value) not in (int,float) or not value or unit not in ('usd','usd thousands','usd millions','usd billions'):return None
    multiplier={'usd':1,'usd thousands':1000,'usd millions':1000000,'usd billions':1000000000}[unit]
    amounts=[]
    for m in re.finditer(r'\$\s*([\d,]+(?:\.\d+)?)\s*(thousand|million|billion)?\b',fact.get('statement',''),re.I):
        n=float(m[1].replace(',',''))*{None:1,'thousand':1000,'million':1000000,'billion':1000000000}[m[2].lower() if m[2] else None]
        amounts.append(n)
    represented=value*multiplier
    if amounts and not any(math.isclose(represented,n,rel_tol=.001) for n in amounts) and any(any(math.isclose(abs(represented/n),scale,rel_tol=.001) for scale in (1000,1/1000,1000000,1/1000000,1000000000,1/1000000000)) for n in amounts if n):
        return f"Unit-scale warning for {fact['entity_id']} fact {fact['fact_id']}: value={value} {unit} does not match the dollar amount in its statement. Verify original table units and period before using the numeric field. No source correction was made."
    return None
