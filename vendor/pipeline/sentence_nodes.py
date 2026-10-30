"""Source-exact sentence leaves; O4000 parents remain reading-only nodes."""
from copy import deepcopy
from functools import lru_cache
import importlib.metadata
from pathlib import Path
import re

from pipeline import HERE
from t4_splitter import count_tokens
from t4_evidence import digest

VERSION = 'sentence-window-v1'
LIMIT = 512


def sentence_configuration():
    return {'version':VERSION,'window_size':1,'child_limit':LIMIT,'window_limit':LIMIT,
            'parser':'SentenceWindowNodeParser',
            'llama_index':importlib.metadata.version('llama-index-core'),
            'nltk':importlib.metadata.version('nltk'),
            'source_sha256':digest(Path(__file__).read_text(encoding='utf-8'))}


@lru_cache(maxsize=1)
def parser():
    # O4000 configures vendored NLTK resources; loading the local trained English
    # tokenizer avoids downloads and respects abbreviations/decimal numbers.
    count_tokens('initialize offline resources')
    from nltk.tokenize.punkt import PunktSentenceTokenizer, load_punkt_params
    from nltk.data import FileSystemPathPointer
    import t4_splitter
    from llama_index.core.node_parser import SentenceWindowNodeParser
    resource=Path(t4_splitter.__file__).parent/'resources/nltk_data/tokenizers/punkt_tab/english'
    import nltk
    resource_root=str(resource.parents[2])
    if resource_root not in nltk.data.path:
        nltk.data.path.insert(0,resource_root)
    tokenizer = PunktSentenceTokenizer(load_punkt_params(FileSystemPathPointer(str(resource))))
    def sentences(text):
        out=[]
        for a,b in tokenizer.span_tokenize(text):
            piece=text[a:b]
            # Punkt English alone does not split Chinese full stops.
            out.extend(s for s in re.split(r'(?<=[。！？])',piece) if s.strip())
        return out
    return SentenceWindowNodeParser.from_defaults(window_size=1,sentence_splitter=sentences)


def regions(text, parents):
    spans=sorted((s['start'],s['end']) for p in parents for s in p['source_spans'] if s['kind']=='text')
    merged=[]
    for a,b in spans:
        if merged and a<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],b)
        else:
            merged.append([a,b])
    # Headings and list items are standalone units; tables are already excluded.
    pattern=re.compile(r'(?m)^[ \t]*(?:#{1,6}\s+[^\n]+|(?:[-*+] |\d+[.)] )[^\n]+)(?:\n|$)')
    section_cuts=set()
    previous=None
    for p in parents:
        spans=[s for s in p['source_spans'] if s['kind']=='text']
        if not spans:
            continue
        section=p.get('section')
        if previous and section and section!='未恢复章节' and previous!=section:
            section_cuts.add(spans[0]['start'])
        previous=section
    bounded=[]
    for a,b in merged:
        cuts=[a,*sorted(c for c in section_cuts if a<c<b),b]
        bounded.extend(zip(cuts,cuts[1:]))
    for a,b in bounded:
        cursor=a
        for match in pattern.finditer(text,a,b):
            if cursor<match.start():
                yield cursor,match.start(),'sentence'
            kind='heading' if match.group().lstrip().startswith('#') else 'list'
            yield match.start(),match.end(),kind
            cursor=match.end()
        if cursor<b:
            yield cursor,b,'sentence'


def fragments(text,a,b):
    while a<b:
        if count_tokens(text[a:b])<=LIMIT:
            yield a,b
            return
        lo,hi=a,b
        while lo+1<hi:
            mid=(lo+hi)//2
            if count_tokens(text[a:mid])<=LIMIT:
                lo=mid
            else:
                hi=mid
        if lo<=a:
            raise ValueError('Cannot fit one source character')
        boundary=max(text.rfind(mark,a,lo) for mark in (';', '；','\n'))
        end=boundary+1 if boundary>a+(lo-a)//2 else lo
        yield a,end
        a=end


def build_sentence_nodes(doc_id,text,parents):
    from llama_index.core import Document
    parents=deepcopy(parents)
    config=sentence_configuration()
    source=digest(text)
    for p in parents:
        p['chunk_id']='parent-'+digest([doc_id,source,p['seq'],p['text'],p['source_spans']])[:24]
        p['child_ids']=[]
    nodes=[]
    for region_index,(ra,rb,kind) in enumerate(regions(text,parents)):
        raw=text[ra:rb]
        if not raw.strip():
            continue
        splits=([raw] if kind!='sentence' else
                [n.text for n in parser().get_nodes_from_documents([Document(text=raw)])])
        located=[]; cursor=ra
        for split in splits:
            split=split.strip()
            if not split:
                continue
            a=text.find(split,cursor,rb)
            if a<cursor or text[cursor:a].strip():
                raise ValueError(f'{doc_id}: sentence alignment failed at {cursor}')
            b=a+len(split)
            located.append((a,b)); cursor=b
        if text[cursor:rb].strip():
            raise ValueError(f'{doc_id}: sentence coverage incomplete at {cursor}')
        region_nodes=[]
        for i,(a,b) in enumerate(located):
            sid='sentence-'+digest([doc_id,source,a,b,config])[:24]
            pieces=list(fragments(text,a,b))
            for fa,fb in pieces:
                wa,wb=(located[max(0,i-1)][0],located[min(len(located)-1,i+1)][1])
                limited=len(pieces)>1
                if limited:
                    wa,wb=fa,fb
                else:
                    if count_tokens(text[wa:wb])>LIMIT:
                        wb=b; limited=True
                    if count_tokens(text[wa:wb])>LIMIT:
                        wa=a
                linked=[p for p in parents if any(s['kind']=='text' and fa<s['end'] and fb>s['start'] for s in p['source_spans'])]
                if not linked:
                    raise ValueError(f'{doc_id}: sentence has no parent')
                cid='child-'+digest([sid,fa,fb])[:24]
                node={'doc_id':doc_id,'node_type':'sentence','chunk_id':cid,'child_id':cid,
                      'sentence_id':sid,'text':text[fa:fb],'agent_text':text[fa:fb],
                      'embedding_text':text[fa:fb],'start':fa,'end':fb,
                      'source_spans':[{'kind':'text','start':fa,'end':fb}],
                      'source_offsets_are_envelope':False,'view_start':None,'view_end':None,
                      'parent_chunk_ids':[p['chunk_id'] for p in linked],
                      'section':linked[0]['section'],'table_refs':[],
                      'tokens':count_tokens(text[fa:fb]),'seq':len(nodes)+1,
                      'evidence_id':f'{doc_id}::{fa}::{fb}','algorithm_version':VERSION,
                      'is_sentence_fragment':len(pieces)>1,'text_unit_type':kind,
                      'region_id':region_index,'window_start':wa,'window_end':wb,
                      'window_text':text[wa:wb],'window_limited':limited,
                      'content_type':'text'}
                for p in linked:
                    p['child_ids'].append(cid)
                region_nodes.append(node); nodes.append(node)
        for i,n in enumerate(region_nodes):
            n['prev_child_id']=region_nodes[i-1]['child_id'] if i else None
            n['next_child_id']=region_nodes[i+1]['child_id'] if i+1<len(region_nodes) else None
    return nodes,parents
