"""Conservative source-preserving boundary repair for flattened English filings.

No document-specific offsets/names or hand-labelled evaluation cases are used.
Tables are protected text regions, not reconstructed cells. Headings are candidates.
"""
import bisect
import re

PAGE = re.compile(r'\b\d{1,3}\s+Table of Contents\s+')
ABBREV = re.compile(r'(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Corp|Inc|Ltd|Co|L\.P|L\.L\.C|U\.S|U\.K|No|vs|St)|\b[A-Z]|(?:[A-Z][A-Za-z]{0,2}\.){1,}[A-Za-z]{1,3})\.$',re.I)
ANAPHORA = re.compile(r'(?i)(?:this|these|those|such|it|they|its|their|the (?:increase|decrease|payment|amount|latter|former))\b')


def structure(text):
    n=len(text)
    pages=list(PAGE.finditer(text))
    headers=[]
    for p in pages:
        end=p.end()
        extra=re.match(r"(?:[A-Z][A-Z &.,’'\-]{1,70}(?:INC|CORP|LTD)\.\s+)?(?:NOTES TO THE CONSOLIDATED FINANCIAL STATEMENTS\s+DECEMBER\s+\d+,\s+\d{4}\s+)?",text[end:])
        if extra:end+=extra.end()
        headers.append((p.start(),end))
    safe={0,n}
    for m in re.finditer(r'[.!?][\)\]”\"\']*\s+',text):
        punct=m.start()
        if re.match(r'[a-z]|[’\']s\b|[”\"]',text[m.end():]):continue
        if text[punct]=='.' and ABBREV.search(text[max(0,punct-20):punct+1]):continue
        # Avoid splitting after section/footnote labels.
        if re.search(r'\b(?:Note|Item)\s+\d+[A-Z]?\.$',text[max(0,punct-25):punct+1],re.I):continue
        if any(a<=m.end()<b for a,b in headers):continue
        safe.add(m.end())
    sections=[]
    expected=1
    for m in re.finditer(r'\bNote\s+(\d+)\.\s+',text):
        num=int(m[1]);prefix=text[max(0,m.start()-45):m.start()]
        if num!=expected or re.search(r'(?:see|in|to|under|of|with)\s*$',prefix,re.I):continue
        if not re.match(r'[A-Z]',text[m.end():]):continue
        sections.append({'start':m.start(),'label':f'财务报表附注 / Note {num}','confidence':'规则识别，待人工确认'})
        safe.add(m.start());expected+=1
    # SEC Item labels after the table of contents; accept nonbreaking spaces.
    body_start=pages[0].start() if pages else 0
    for m in re.finditer(r'\bItem\s+(\d+[A-Z]?)\.\s+(?=[A-Z])',text):
        prefix=text[max(0,m.start()-35):m.start()]
        if m.start()<=body_start or re.search(r'(?:see|in|under|“|"|—|–)\s*$',prefix,re.I):continue
        sections.append({'start':m.start(),'label':f'Item {m[1]}','confidence':'编号标题候选，非官方目录'})
        safe.add(m.start())
    for m in re.finditer(r'SUPPLEMENTARY INFORMATION CONSOLIDATING FINANCIAL STATEMENTS',text):
        sections.append({'start':m.start(),'label':'补充合并财务信息','confidence':'大写标题候选'})
        safe.add(m.start())
    sections.sort(key=lambda x:x['start'])
    sentence_safe=sorted(safe)
    table_regions=[]
    for i,p in enumerate(pages):
        a=p.end();b=pages[i+1].start() if i+1<len(pages) else n
        body=text[a:b];words=re.findall(r'\S+',body);count=len(re.findall(r'(?<![A-Za-z])\d[\d,.]*',body));ratio=count/max(1,len(words))
        units=bool(re.search(r'\bAmounts in (?:millions|thousands)\b',body,re.I))
        title=bool(re.search(r'\bCONSOLIDAT(?:ED|ING) (?:BALANCE SHEET|STATEMENT)',body[:400]))
        if (count>=25 and ratio>=.16) or (units and count>=15) or title:
            # Include page marker to avoid leaving a detached header before the table.
            a=p.start()
            # Expand page boundaries outwards when a sentence continues across a page.
            before=text[:a].rstrip()
            if before and not re.search(r'[.!?\d%)\]]$',before):
                a=max(x for x in sentence_safe if x<a)
            tail=text[:b].rstrip()
            if tail and not re.search(r'[.!?\d%)\]]$',tail):
                b=next((x for x in sentence_safe if x>b),n)
            # Keep an immediately preceding introduction ending in a colon.
            if text[:a].rstrip().endswith(':'):
                prev=max(x for x in safe if x<a)
                if a-prev<1200:a=prev
            # Footnotes continuing on the next page belong to this text region.
            if i+1<len(pages):
                next_start=headers[i+1][1]
                if re.match(r'\(\d+\)\s',text[next_start:]):
                    following=sorted(x for x in safe if x>next_start)
                    if following:
                        b=max(b,following[0])
                        while re.match(r'\(\d+\)\s',text[b:]):
                            nxt=next((x for x in following if x>b),b)
                            if nxt==b:break
                            b=nxt
            table_regions.append({'start':a,'end':b,'kind':'表格候选整区保护','reason':f'数字项{count}，数字项/空白词{ratio:.3f}；金额单位={units}；财务表题={title}'})
    merged=[]
    for t in table_regions:
        if merged and t['start']<merged[-1]['end']:
            merged[-1]['end']=max(merged[-1]['end'],t['end']);merged[-1]['reason']+='；合并交叠区域'
        else:merged.append(t.copy())
    for t in merged:safe.update([t['start'],t['end']])
    safe=sorted(x for x in safe if not any(t['start']<x<t['end'] for t in merged))
    return safe,merged,sections,headers


def optimize(text, baseline_chunks, tokenizer, *, min_core=1800, small_tail=800, merge_ceiling=6000, label='O4000'):
    safe,tables,sections,headers=structure(text);n=len(text)
    cuts=[0];adjustments=[]
    # Repair existing pipeline's draft ends; discard cuts that would create tiny cores.
    for old in [c['end'] for c in baseline_chunks[:-1]]:
        idx=bisect.bisect_left(safe,old)
        candidates=safe[max(0,idx-6):idx+7]
        candidates=[x for x in candidates if x>cuts[-1] and len(tokenizer(text[cuts[-1]:x]))>=min_core]
        if not candidates:continue
        nearby_sections=[h['start'] for h in sections if abs(h['start']-old)<=1500 and h['start'] in candidates]
        end=min(nearby_sections or candidates,key=lambda x:abs(x-old))
        if end>=n:continue
        cuts.append(end);adjustments.append({'old_end':old,'new_end':end,'delta_chars':end-old})
    cuts.append(n)
    if len(cuts)>2 and len(tokenizer(text[cuts[-2]:n]))<small_tail and len(tokenizer(text[cuts[-3]:n]))<=merge_ceiling:cuts.pop(-2)
    output=[]
    for i,(core_a,b) in enumerate(zip(cuts,cuts[1:]),1):
        a=core_a
        if i>1:
            idx=bisect.bisect_left(safe,core_a)
            candidates=[x for x in safe[max(0,idx-20):idx] if x>=cuts[i-2] and len(tokenizer(text[x:core_a]))<=200]
            if candidates:a=min(candidates)
            # Move backwards by up to 500 tokens to include the antecedent.
            while ANAPHORA.match(text[a:]):
                j=bisect.bisect_left(safe,a)
                if j==0:break
                prev=safe[j-1]
                if prev<cuts[i-2] or len(tokenizer(text[prev:core_a]))>500:break
                a=prev
        active=[h for h in sections if h['start']<=core_a]
        path=active[-1]['label'] if active else '文档前部 / 未恢复章节'
        crossed=[h['label'] for h in sections if a<h['start']<b]
        spans=[t for t in tables if a<=t['start'] and t['end']<=b]
        tokens=len(tokenizer(text[a:b]))
        output.append({'config':label,'seq':i,'start':a,'end':b,'text':text[a:b],'tokens':tokens,'chars':b-a,'overlap_chars':max(0,output[-1]['end']-a) if output else 0,'core_start':core_a,'section':path,'sections_crossed':crossed,'table_regions':len(spans),'notes':('超过合并警戒；需继续处理' if tokens>merge_ceiling else ''),'type':'text_with_protected_regions','mapping':'exact_source_slice'})
    return {'chunks':output,'tables':tables,'sections':sections,'headers':headers,'adjustments':adjustments,'safe_positions':safe}
