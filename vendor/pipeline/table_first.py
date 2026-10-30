"""Table-first preprocessing with separate source and reading coordinates."""
import base64
import contextlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zlib

from pipeline import attach_tables
from table_artifacts import publish_result, sha256
from table_extraction import extract_tables_with_diagnostics
from t4_splitter import count_tokens, split_document

VERSION = 'table-first-v2'


def clip_tokens(text, limit):
    if count_tokens(text) <= limit:
        return text
    lo, hi = 0, len(text)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if count_tokens(text[:mid] + '…') <= limit:
            lo = mid
        else:
            hi = mid
    return text[:lo] + '…'


def reading_document(text, refs):
    parts, mapping, cursor, offset = [], [], 0, 0
    def append(value, a, b, kind, ref_id=None):
        nonlocal offset
        if not value:
            return
        item = {'view_start': offset, 'view_end': offset + len(value),
                'start': a, 'end': b, 'kind': kind}
        if ref_id:
            item['ref_id'] = ref_id
        mapping.append(item)
        parts.append(value)
        offset += len(value)
    for ref in sorted(refs, key=lambda r:r['source']['start']):
        a, b = ref['source']['start'], ref['source']['end']
        if not cursor <= a < b <= len(text):
            raise ValueError('Overlapping or invalid table spans')
        append(text[cursor:a], cursor, a, 'text')
        header = clip_tokens(' | '.join(ref['header']), 256)
        value = (f'\n[[TABLE_REF:{ref["ref_id"]}]]\n'
                 f'Table columns: {header}\nRows: {ref["row_count"]}; columns: {ref["column_count"]}.\n')
        append(value, a, b, 'table_ref', ref['ref_id'])
        cursor = b
    append(text[cursor:], cursor, len(text), 'text')
    return ''.join(parts), mapping


def map_spans(mapping, a, b):
    """Map view intervals; replacement summaries map to the complete source table."""
    out = []
    for item in mapping:
        lo, hi = max(a, item['view_start']), min(b, item['view_end'])
        if lo >= hi:
            continue
        start, end = item['start'], item['end']
        if item['kind'] == 'text':
            start += lo - item['view_start']
            end = start + hi - lo
        out.append({**item, 'view_start':lo-a, 'view_end':hi-a, 'start':start, 'end':end})
    return out


def capped_end(text, start, end, limit):
    if count_tokens(text[start:end]) <= limit:
        return end
    lo, hi = start, end
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if count_tokens(text[start:mid]) <= limit:
            lo = mid
        else:
            hi = mid
    if lo == start:
        raise ValueError('Token budget cannot hold one character')
    newline = text.rfind('\n', start, lo)
    return newline + 1 if newline > start and newline + 1 - start >= (lo-start)//2 else lo


def split_reading(doc_id, text, mapping, chunk_size=4000):
    # Use O4000's natural endpoints, with no duplicated overlap. Move any
    # endpoint inside a generated table summary to its start/end atomically.
    with contextlib.redirect_stdout(sys.stderr):
        drafts = split_document(doc_id, text, chunk_size=chunk_size)
    ends = sorted({c['end'] for c in drafts})
    refs = [m for m in mapping if m['kind'] == 'table_ref']
    start, result = 0, []
    while start < len(text):
        end = next(e for e in ends if e > start)
        end = capped_end(text, start, end, chunk_size)
        for region in refs:
            if region['view_start'] < end < region['view_end']:
                end = region['view_start'] if region['view_start'] > start else region['view_end']
                break
        if end <= start or count_tokens(text[start:end]) > chunk_size:
            raise ValueError('Atomic reference exceeds chunk budget')
        draft = next(c for c in drafts if c['start'] <= start < c['end'])
        result.append({'view_start':start, 'view_end':end, 'text':text[start:end],
                       'section':draft['section'], 'source_spans':map_spans(mapping,start,end)})
        start = end
    return result


def build_nodes(doc_id, text, refs, chunk_size=4000):
    refs = sorted(refs,key=lambda r:r['source']['start'])
    reading, mapping = reading_document(text, refs)
    by_ref = {r['ref_id']:r for r in refs}
    nodes = []
    for i, node in enumerate(split_reading(doc_id, reading, mapping,chunk_size), 1):
        linked = [by_ref[s['ref_id']] for s in node['source_spans'] if s['kind']=='table_ref']
        nodes.append({**node, 'node_type':'text', 'seq':i, 'table_refs':linked,
                      'agent_text':node['text'], 'embedding_text':node['text']})
    for ref_index, ref in enumerate(refs):
        a, b = ref['source']['start'], ref['source']['end']
        parents = [n['seq'] for n in nodes if n['node_type']=='text' and any(r['ref_id']==ref['ref_id'] for r in n['table_refs'])]
        section = next((n['section'] for n in nodes if n['node_type']=='text' and n['seq'] in parents), '未恢复章节')
        previous_end = refs[ref_index-1]['source']['end'] if ref_index else 0
        next_start = refs[ref_index+1]['source']['start'] if ref_index+1<len(refs) else len(text)
        context = (clip_tokens(text[max(previous_end,a-600):a],128) + '\n' +
                   clip_tokens(text[b:min(next_start,b+600)],128))
        prefix = ('Table context: ' + context + '\nTable columns: ' +
                  clip_tokens(' | '.join(ref['header']),256) + '\n')
        budget = chunk_size - count_tokens(prefix) - 8
        cursor, part = a, 0
        while cursor < b:
            end = capped_end(text, cursor, b, budget)
            part += 1
            body = text[cursor:end]
            nodes.append({'node_type':'table', 'seq':len(nodes)+1, 'table_part':part,
                          'parent_text_seqs':parents, 'section':section,
                          'table_context':context,
                          'text':body, 'embedding_text':prefix+body,
                          'agent_text':f'[[TABLE_REF:{ref["ref_id"]}]]',
                          'table_refs':[dict(ref, partial_in_chunk=cursor>a or end<b,
                                             overlap_start=cursor, overlap_end=end)],
                          'view_start':None, 'view_end':None,
                          'source_spans':[{'kind':'text','start':cursor,'end':end,'view_start':0,'view_end':len(body)}],
                          'row_fragment':(cursor>a and text[cursor-1]!='\n') or (end<b and text[end-1]!='\n')})
            cursor = end
    for node in nodes:
        start = min(s['start'] for s in node['source_spans'])
        end = max(s['end'] for s in node['source_spans'])
        node.update(doc_id=doc_id, start=start, end=end, evidence_id=f'{doc_id}::{start}::{end}',
                    tokens=count_tokens(node['text']), content_type=node['node_type'],
                    algorithm_version=VERSION, source_offsets_are_envelope=True)
        if count_tokens(node['embedding_text']) > chunk_size:
            raise ValueError('Embedding text exceeds configured token budget')
    return reading, mapping, nodes


def process_table_first(doc_id, text, embedder, output, previews=None, granularity='chunk'):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.table-first-', dir=output.parent) as tmp:
        staging = Path(tmp)
        with contextlib.redirect_stdout(sys.stderr):
            tables, report = extract_tables_with_diagnostics(text)
        seed = {'status':0,'meta':{'doc_id':doc_id,'source_tables':len(tables)},
                'result':[attach_tables({'text':text,'start':0,'end':len(text),
                                         'evidence_id':f'{doc_id}::0::{len(text)}'},tables)]}
        exported = publish_result(seed, staging/output.name, previews)
        refs = exported['result'][0]['table_refs']
        reading, mapping, nodes = build_nodes(doc_id,text,refs)
        parents=[]
        if granularity=='sentence':
            from sentence_nodes import build_sentence_nodes
            children,parents=build_sentence_nodes(doc_id,text,[n for n in nodes if n['node_type']=='text'])
            tables=[n for n in nodes if n['node_type']=='table']
            for table in tables:
                table['parent_chunk_ids']=[p['chunk_id'] for p in parents if p['seq'] in table['parent_text_seqs']]
            nodes=children+tables
            for seq,node in enumerate(nodes,1):
                node['seq']=seq
        vectors, lengths = embedder.encode([n['embedding_text'] for n in nodes])
        if len(vectors)!=len(nodes) or len(lengths)!=len(nodes):
            raise ValueError('Embedding result count differs from node count')
        for node, vector, length in zip(nodes,vectors,lengths):
            node.update(embedding=base64.b64encode(zlib.compress(vector.astype('<f4').tobytes())).decode('ascii'),
                        embedding_dim=1024, embeding_dim=1024, embedding_input_tokens=length,
                        embedding_source='embedding_text', embedding_encoding='float32-le+zlib+base64')
        result = {'status':0,'result':nodes,'source_text':text,'reading_text':reading,'source_map':mapping,
                  'artifacts':exported['artifacts'], 'meta':{**exported['meta'],
                    'processing_version':VERSION,'table_extraction':report,
                    'total_chunks':len(nodes),'returned_chunks':len(nodes),'complete_document':True,
                    'text_chunks':sum(n['node_type']=='text' for n in nodes),
                    'table_chunks':sum(n['node_type']=='table' for n in nodes),
                    'model':'BAAI/bge-m3','model_revision':embedder.manifest['revision'],
                    'normalized':True,'precision':'float32','device':embedder.manifest.get('device','cpu'),
                    'embedding_source':'embedding_text', 'overlap_tokens':0,
                    'agent_text_usage':'Reading text with atomic table references; tables indexed separately'}}
        if granularity=='sentence':
            from sentence_nodes import VERSION as SENTENCE_VERSION, sentence_configuration
            result['parents']=parents
            result['meta'].update(processing_version=SENTENCE_VERSION,
                                  sentence_configuration=sentence_configuration(),
                                  sentence_chunks=len(children),parent_chunks=len(parents))
        target = staging/output.name
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n', encoding='utf-8')
        # Publish referenced immutable bundles before the final JSON, never expose
        # the unembedded export seed at the requested result path.
        for artifact in result['artifacts']:
            bundle = Path(artifact['path']).parent
            source, dest = staging/bundle, output.parent/bundle
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if any(not (dest/p.name).is_file() or sha256(p)!=sha256(dest/p.name)
                       for p in source.iterdir() if p.is_file()):
                    raise ValueError('Existing table bundle differs')
            else:
                shutil.move(str(source),str(dest))
        target.replace(output)
    return result
