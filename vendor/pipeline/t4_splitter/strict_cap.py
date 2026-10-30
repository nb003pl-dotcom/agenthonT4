"""Strict token ceiling adapter. Splits oversized source slices without inventing tables."""
import bisect,math,re
from .protected_split import ABBREV,ANAPHORA

def raw_edges(text):
    edges={0,len(text)}
    for m in re.finditer(r'[.!?][\)\]”\"\']*\s+|\n+',text):
        if m[0].startswith('\n'):
            edges.add(m.end());continue
        left=text[max(0,m.start()-25):m.start()+1];right=text[m.end():]
        if ABBREV.search(left) or re.match(r'[a-z]|[”\"]|[’\']s\b',right):continue
        if re.search(r'\b(?:Note|Item)\s+\d+[A-Z]?\.$',left,re.I):continue
        edges.add(m.end())
    return sorted(edges)

def cap(text,chunks,tokenizer,limit=4000):
    edges=raw_edges(text);out=[]
    def tok(a,b):return len(tokenizer(text[a:b]))
    def fit(a,b,budget):
        lo=a;hi=b
        while lo<hi:
            mid=(lo+hi+1)//2
            if tok(a,mid)<=budget:lo=mid
            else:hi=mid-1
        while lo>a and tok(a,lo)>budget:lo-=1
        return lo
    for original in chunks:
        a,b=original['start'],original['end'];total=tok(a,b)
        if total<=limit:
            out.append(original.copy()|{'soft_seq':original['seq'],'cap_reason':'原优化块未超限'});continue
        pieces=math.ceil(total/(limit-200));core=a
        for part in range(pieces):
            remain=pieces-part
            if remain==1:end=b
            else:
                budget=min(limit-200,math.ceil(tok(core,b)/remain))
                ideal=fit(core,b,budget)
                j=bisect.bisect_right(edges,ideal)
                options=[p for p in edges[max(0,j-6):j+6] if core<p<b and tok(core,p)<=limit-200 and tok(core,p)>=800]
                end=min(options,key=lambda p:abs(p-ideal)) if options else ideal
                if not options:
                    # Prefer whitespace near the token ceiling, never splitting a number if avoidable.
                    spaces=[m.end()+core for m in re.finditer(r'\s+',text[core:ideal])]
                    if spaces and ideal-spaces[-1]<200:end=spaces[-1]
            start=core
            if part:
                j=bisect.bisect_left(edges,core)
                options=[p for p in edges[max(0,j-10):j] if p>=a and tok(p,core)<=200 and tok(p,end)<=limit]
                if options:start=min(options)
            # A long final sentence may exceed the balanced allocation. Continue with explicit capped slices.
            while tok(start,end)>limit:
                cut=fit(start,end,limit-200)
                j=bisect.bisect_right(edges,cut)
                candidates=[p for p in edges[max(0,j-8):j] if p>start and tok(start,p)>=800]
                stop=max(candidates) if candidates else cut
                out.append(original.copy()|{'start':start,'end':stop,'core_start':start,'soft_seq':original['seq'],'cap_reason':'硬上限兜底；需核对句子/表格关系'})
                start=stop
            out.append(original.copy()|{'start':start,'end':end,'core_start':core,'soft_seq':original['seq'],'cap_reason':'超限原块均衡拆分；需结合保护区范围取证'})
            core=end
    for i,c in enumerate(out):
        c.update(seq=i+1,text=text[c['start']:c['end']],tokens=tok(c['start'],c['end']),chars=c['end']-c['start'],overlap_chars=max(0,out[i-1]['end']-c['start']) if i else 0)
        assert c['tokens']<=limit
    return out
