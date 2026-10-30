"""Model selects quote IDs; Python owns exact quote text and raw source spans."""
import copy

def annotate_payload(payload, max_quote_chars=320):
    if type(max_quote_chars) is not int or not 100 <= max_quote_chars <= 1600:
        raise ValueError('citation window must be 100..1600 characters')
    if max_quote_chars != 320:
        from t4_splitter import count_tokens
        from .evaluation import judge_token_counter
        judge_count = judge_token_counter()
        def fits(text):
            return count_tokens(text) <= 350 and judge_count(text) <= 400
    result=copy.deepcopy(payload);catalog={};serial=0
    for chunk in result['chunks']:
        for span in chunk['source_spans']:
            text=span['text'];pieces=[];a=0
            while a<len(text):
                b=min(len(text),a+max_quote_chars)
                if max_quote_chars != 320 and not fits(text[a:b]):
                    low,high=a+1,b
                    while low<=high:
                        middle=(low+high)//2
                        if fits(text[a:middle]):b=middle;low=middle+1
                        else:high=middle-1
                    if not fits(text[a:b]):raise ValueError('source character exceeds citation token limit')
                if b<len(text):
                    cuts=[text.rfind('\n',a+100,b),text.rfind(' ',a+100,b)]
                    cut=max(cuts)
                    if cut>=a+100:b=cut+1
                quote=text[a:b]
                # Token counts are not perfectly monotone under arbitrary BPE
                # boundaries. Recheck the final whitespace cut conservatively.
                if max_quote_chars != 320:
                    while quote and not fits(quote):
                        b-=1;quote=text[a:b]
                    if not quote:raise ValueError('empty bounded citation window')
                if quote.strip():
                    serial+=1;qid=f'Q{serial:04d}'
                    catalog[qid]={'chunk_id':chunk['chunk_id'],'doc_id':chunk['doc_id'],'quote':quote,
                                  '_catalog_window':{'start':span['start'],'end':span['end']},
                                  '_catalog_span':{'start':span['start']+a,'end':span['start']+b}}
                    pieces.append('[[ '+qid+' ]]'+quote)
                else:pieces.append(quote)
                a=b
            span['text']='\n'.join(pieces)
    result['citation_contract']='Use only quote_ids like Q0001 from [[ Q0001 ]] labels. Python restores exact text and offsets. Labels are not source facts. Do not generate quotes, doc IDs, chunk IDs or offsets. Prefer 1–2 IDs, allow up to 4 for complete table/period context. Include a separate column-header ID for table facts if necessary.'
    return result,catalog

def restore_quotes(data,catalog,strict=True):
    result=copy.deepcopy(data)
    for fact in result.get('facts',[]):
        try:
            citations=[];seen=set()
            for c in fact.get('citations',[]):
                ids=c.get('quote_ids')
                if not isinstance(ids,list) or not 1<=len(ids)<=4:raise ValueError('Compact citation requires 1..4 quote_ids')
                for qid in ids:
                    if qid not in catalog:raise ValueError('Unknown quote_id '+str(qid))
                    if qid not in seen:citations.append(dict(catalog[qid]));seen.add(qid)
            if not 1<=len(citations)<=4:raise ValueError('Use at most 4 distinct quote IDs per fact')
            fact['citations']=citations
        except (ValueError,TypeError,KeyError) as exc:
            if strict:raise
            fact['_citation_restore_error']=str(exc)
    return result
