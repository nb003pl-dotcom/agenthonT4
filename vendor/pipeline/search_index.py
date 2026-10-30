"""Build and verify local BM25 + dense indexes from complete pipeline results."""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import tempfile
import unicodedata
import zlib

import bm25s
import numpy as np

from entity_assignment import SCOPES, assign_documents, assignment_configuration
from pipeline import HERE  # initializes the existing splitter import path
from read_table import read_table_ref
from table_artifacts import sha256
from t4_splitter import configuration, count_tokens
from t4_evidence import chunk_identity, digest

REVISION = 'remote-unknown'
INDEX_VERSION = 't4-preprocess-index-v3'
TOKENIZER_VERSION = 'financial-tokens-v1'
BM25_PARAMETERS = {'method': 'lucene', 'k1': 1.5, 'b': 0.75}
TOKEN_PATTERN = re.compile(r'[\u3400-\u9fff]+|[a-z][a-z0-9]*(?:[_.&/-][a-z0-9]+)*|[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[-/]\d+)*(?:-[a-z]+)?%?', re.I)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.json', delete=False) as f:
        temporary = Path(f.name)
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Path escapes root: {relative}')
    return path


def financial_tokens(text):
    tokens = []
    for token in TOKEN_PATTERN.findall(unicodedata.normalize('NFKC', text).lower()):
        if re.fullmatch(r'[\u3400-\u9fff]+', token):
            import jieba
            tokens.extend(w for w in jieba.cut(token, HMM=False) if w.strip())
            continue
        forms = [token]
        if ',' in token:
            forms.append(token.replace(',', ''))
        if re.search('[a-z]', token):
            forms.extend(x for x in re.split(r'[_./&-]', token) if x)
        tokens.extend(dict.fromkeys(forms))
    return tokens


def decode_vector(chunk):
    if chunk.get('embedding_source') not in ('text', 'embedding_text') or chunk.get('embedding_encoding') != 'float32-le+zlib+base64':
        raise ValueError('Embedding source/encoding mismatch')
    if chunk.get('embedding_dim') != 1024:
        raise ValueError('Embedding dimension mismatch')
    try:
        raw = zlib.decompress(base64.b64decode(chunk['embedding'], validate=True))
        vector = np.frombuffer(raw, dtype='<f4')
    except Exception as exc:
        raise ValueError('Invalid embedding encoding') from exc
    if vector.shape != (1024,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
        raise ValueError('Embedding must be finite, normalized and 1024-dimensional')
    return vector


def validate_result(doc, result, path):
    meta = result['meta']
    if (result.get('status') != 0 or meta.get('doc_id') != doc['doc_id']
            or meta.get('complete_document') is not True or meta.get('model') != 'BAAI/bge-m3'
            or meta.get('model_revision') != REVISION or meta.get('normalized') is not True
            or meta.get('precision') != 'float32' or meta.get('table_reference_version') != 'table-ref-v1'):
        raise ValueError('Incomplete or incompatible document result')
    chunks = result['result']
    if not chunks or meta['total_chunks'] != len(chunks) or meta['returned_chunks'] != len(chunks):
        raise ValueError('Chunk count mismatch')
    if meta.get('processing_version') in ('table-first-v2','sentence-window-v1'):
        from table_first import build_nodes
        if result.get('source_text') != doc['text']:
            raise ValueError('Original source text changed')
        refs = {r['ref_id']:r for c in result.get('parents',chunks) if c['node_type']=='text' for r in c['table_refs']}
        refs = sorted(refs.values(),key=lambda r:r['source']['start'])
        for ref in refs:
            table = read_table_ref(ref,path)
            a,b = ref['source']['start'],ref['source']['end']
            if ref['source']['doc_id']!=doc['doc_id'] or table['original']['text']!=doc['text'][a:b]:
                raise ValueError('Table source mismatch')
        reading,mapping,expected = build_nodes(doc['doc_id'],doc['text'],refs)
        if meta.get('processing_version')=='sentence-window-v1':
            from sentence_nodes import build_sentence_nodes, sentence_configuration
            children,parents=build_sentence_nodes(doc['doc_id'],doc['text'],[n for n in expected if n['node_type']=='text'])
            tables=[n for n in expected if n['node_type']=='table']
            for table in tables:
                table['parent_chunk_ids']=[p['chunk_id'] for p in parents if p['seq'] in table['parent_text_seqs']]
            expected=children+tables
            for seq,node in enumerate(expected,1):
                node['seq']=seq
            if (result.get('parents')!=parents or meta.get('sentence_configuration')!=sentence_configuration()
                    or meta.get('sentence_chunks')!=len(children) or meta.get('parent_chunks')!=len(parents)):
                raise ValueError('Sentence or parent structure mismatch')
        if (result.get('reading_text')!=reading or result.get('source_map')!=mapping
                or len(expected)!=len(chunks) or len(refs)!=meta['source_tables']
                or meta['text_chunks']!=sum(c['node_type']=='text' for c in chunks)
                or meta['table_chunks']!=sum(c['node_type']=='table' for c in chunks)):
            raise ValueError('Reading text, table or node coverage mismatch')
        for c,e in zip(chunks,expected):
            if any(c.get(k)!=v for k,v in e.items()) or c.get('embedding_source')!='embedding_text':
                raise ValueError('Node source mapping or content mismatch')
            decode_vector(c)
        return
    covered = 0
    checked = set()
    for seq, c in enumerate(chunks, 1):
        a, b = c['start'], c['end']
        if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(doc['text'])
                or a > covered or b <= covered or c['seq'] != seq
                or c.get('doc_id') != doc['doc_id'] or c['text'] != doc['text'][a:b]
                or c['evidence_id'] != f'{doc["doc_id"]}::{a}::{b}'
                or c['tokens'] != count_tokens(c['text']) or c['tokens'] > 4000):
            raise ValueError('Chunk source/coverage mismatch')
        covered = b
        decode_vector(c)
        if not isinstance(c.get('agent_text'), str) or not isinstance(c.get('table_refs'), list):
            raise ValueError('Missing agent text/table references')
        for ref in c['table_refs']:
            if ref['source']['doc_id'] != doc['doc_id']:
                raise ValueError('Table belongs to another document')
            if ref['ref_id'] not in checked:
                table = read_table_ref(ref, path)
                ta, tb = ref['source']['start'], ref['source']['end']
                if table['original']['text'] != doc['text'][ta:tb]:
                    raise ValueError('Table source mismatch')
                checked.add(ref['ref_id'])
    if covered != len(doc['text']) or len(checked) != meta['source_tables']:
        raise ValueError('Incomplete source or table coverage')


def index_configuration():
    return {'version': INDEX_VERSION, 'tokenizer': TOKENIZER_VERSION,
            'bm25': BM25_PARAMETERS, 'packages': {p: importlib.metadata.version(p) for p in ('bm25s', 'scipy', 'numpy', 'jieba')},
            'source_sha256': sha256(Path(__file__)), 'splitter': configuration(),
            'entity_assignment': {**assignment_configuration(),
                                  'source_sha256': sha256(Path(__file__).with_name('entity_assignment.py'))},
            'embedding': {'model': 'BAAI/bge-m3', 'revision': REVISION, 'dimensions': 1024,
                          'pooling': 'CLS', 'normalized': True, 'dtype': 'float32'}}


def build_search_index(root, task, docs, records, input_identity, official_assignments=None):
    root = Path(root).resolve()
    if len(records) != len(docs) or {r['doc_id'] for r in records} != set(docs):
        raise ValueError('Processed document set differs from the complete unit')
    config = index_configuration()
    assignments = official_assignments if official_assignments is not None else assign_documents(task, docs)
    roster = sorted(e['entity_id'] for e in task.get('entities', []))
    identity = {'task_id': task['task_id'], 'inputs': input_identity, 'configuration': config,
                'documents': records, 'entity_assignment_roster': roster,
                'official_assignments': official_assignments}
    version = digest(identity)
    destination = root / 'search' / version
    if destination.exists():
        return verify_index(root, destination), True
    chunks, vectors, corpus_tokens, registry, sources = [], [], [], {}, {}
    parent_registry={}
    granularities=set()
    for record in records:
        doc = docs[record['doc_id']]
        result_path = within(root, record['result_path'])
        if sha256(result_path) != record['result_sha256']:
            raise ValueError('Result checksum mismatch')
        result = read_json(result_path)
        validate_result(doc, result, result_path)
        sources[doc['doc_id']] = doc['text']
        assignment = assignments[doc['doc_id']]
        granularities.add('sentence' if result['meta'].get('processing_version')=='sentence-window-v1' else 'chunk')
        def enrich(node):
            return {**node,'task_id':task['task_id'],'doc_date':doc['doc_date'],
                    'title':doc.get('title',''),'doc_type':doc.get('doc_type',''),
                    'result_path':record['result_path'],'entity_ids':assignment['entity_ids'],
                    'entity_scope':assignment['scope']}
        for parent in result.get('parents',[]):
            parent_registry[parent['chunk_id']]=enrich(parent)
        for c in result['result']:
            cid = chunk_identity(task['task_id'], doc['doc_id'], digest(doc['text']), config['splitter'], c['start'], c['end'])
            if c.get('algorithm_version') == 'table-first-v2':
                cid = digest({'base':cid, 'kind':c['node_type'], 'seq':c['seq'],
                              'view_start':c['view_start'], 'text':c['text'], 'version':'table-first-v2'})
            if c.get('node_type')=='sentence':
                cid=c['child_id']
            chunk = {k: v for k, v in c.items() if k not in ('embedding',)}
            chunk.update(chunk_id=cid, task_id=task['task_id'], doc_date=doc['doc_date'],
                         title=doc.get('title', ''), doc_type=doc.get('doc_type', ''),
                         source_sha256=digest(doc['text']), result_path=record['result_path'],
                         agent_text_tokens=count_tokens(c['agent_text']),
                         entity_ids=assignment['entity_ids'], entity_scope=assignment['scope'],
                         entity_assignment_basis=assignment['assignment_basis'],
                         entity_assignment_confidence=assignment['assignment_confidence'],
                         shared=assignment.get('shared', False),
                         association_source=assignment.get('source','legacy_inference'))
            chunks.append(chunk)
            vectors.append(decode_vector(c))
            headers = []
            for ref in c['table_refs']:
                headers.extend(ref['header'])
                item = registry.setdefault(ref['ref_id'], {'doc_id': doc['doc_id'], 'result_path': record['result_path'],
                                                         'ref': ref, 'chunk_ids': []})
                if item['doc_id'] != doc['doc_id']:
                    raise ValueError('Duplicate table reference across documents')
                item['chunk_ids'].append(cid)
            section = c.get('section', '')
            if section == '未恢复章节':
                section = ''
            lexical = '\n'.join([doc['doc_id'], doc.get('title', ''), section,
                                 ' '.join(dict.fromkeys(headers)), c.get('table_context',''), c['text']])
            tokens = financial_tokens(lexical)
            if not tokens:
                raise ValueError('Empty BM25 document tokens')
            corpus_tokens.append(tokens)
    if not chunks or len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('No chunks or duplicate chunk identity')
    if len(granularities)!=1:
        raise ValueError('Mixed retrieval granularities in one index')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.staging-', dir=destination.parent) as tmp:
        folder = Path(tmp)
        (folder / 'chunks.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in chunks), encoding='utf-8')
        write_json(folder / 'dense_rows.json', [c['chunk_id'] for c in chunks])
        np.save(folder / 'dense.npy', np.stack(vectors).astype('<f4'), allow_pickle=False)
        write_json(folder / 'entity_assignments.json', assignments)
        write_json(folder / 'table_registry.json', registry)
        write_json(folder / 'sources.json', sources)
        if parent_registry:
            write_json(folder/'parents.json',parent_registry)
        retriever = bm25s.BM25(**BM25_PARAMETERS)
        retriever.index(corpus_tokens, show_progress=False)
        retriever.save(str(folder / 'bm25'))
        files = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        document_scopes = {scope:sum(a['scope']==scope for a in assignments.values()) for scope in SCOPES}
        chunk_scopes = {scope:sum(c['entity_scope']==scope for c in chunks) for scope in SCOPES}
        write_json(folder / 'manifest.json', {'index_version': version, **identity, 'files': files,
                   'counts': {'documents': len(records), 'chunks': len(chunks), 'tables': len(registry),
                              'text_chunks':sum(c.get('node_type','text')=='text' for c in chunks),
                              'table_chunks':sum(c.get('node_type')=='table' for c in chunks),
                              'sentence_chunks':sum(c.get('node_type')=='sentence' for c in chunks),
                              'parent_chunks':len(parent_registry),
                              'retrieval_granularity':next(iter(granularities)),
                              'dense_shape': [len(chunks), 1024], 'vocabulary': len(retriever.vocab_dict),
                              'entity_scope_documents':document_scopes,
                              'entity_scope_chunks':chunk_scopes}})
        summary = verify_index(root, folder)
        folder.rename(destination)
    return summary, False


def verify_index(root, folder=None):
    root = Path(root).resolve()
    if folder is None:
        pointer = read_json(root / 'current.json')
        folder = within(root, pointer['index_path'])
        if sha256(folder / 'manifest.json') != pointer['manifest_sha256']:
            raise ValueError('Index manifest checksum mismatch')
    folder = Path(folder)
    manifest = read_json(folder / 'manifest.json')
    identity_fields = ('task_id','inputs','configuration','documents','entity_assignment_roster')
    if 'official_assignments' in manifest:
        identity_fields += ('official_assignments',)
    missing = [key for key in identity_fields if key not in manifest]
    if missing:
        raise ValueError('Index lacks entity assignment metadata; rebuild the search index from verified document results')
    if 'entity_assignments.json' not in manifest.get('files',{}):
        raise ValueError('Index lacks entity assignments; rebuild the search index from verified document results')
    identity = {k:manifest[k] for k in identity_fields}
    if digest(identity) != manifest['index_version']:
        raise ValueError('Index identity mismatch')
    for name, checksum in manifest['files'].items():
        if sha256(within(folder, name)) != checksum:
            raise ValueError(f'Index checksum mismatch: {name}')
    chunks = [json.loads(line) for line in (folder / 'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = read_json(folder / 'dense_rows.json')
    vectors = np.load(folder / 'dense.npy', mmap_mode='r', allow_pickle=False)
    if rows != [c['chunk_id'] for c in chunks] or len(set(rows)) != len(rows):
        raise ValueError('Dense row mapping mismatch')
    if vectors.shape != (len(rows), 1024) or vectors.dtype != np.dtype('<f4') or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Invalid stored dense matrix')
    registry = read_json(folder / 'table_registry.json')
    if manifest['counts'].get('retrieval_granularity')=='sentence':
        parents=read_json(folder/'parents.json')
        by_id={c['chunk_id']:c for c in chunks}
        sources=read_json(folder/'sources.json')
        for c in chunks:
            for pid in c.get('parent_chunk_ids',[]):
                if pid not in parents or parents[pid]['doc_id']!=c['doc_id'] or parents[pid]['entity_ids']!=c['entity_ids']:
                    raise ValueError('Invalid parent link')
            if c['node_type']=='sentence':
                if not c['parent_chunk_ids'] or c['text']!=sources[c['doc_id']][c['start']:c['end']]:
                    raise ValueError('Sentence source mismatch')
                if any(c['chunk_id'] not in parents[pid]['child_ids'] for pid in c['parent_chunk_ids']):
                    raise ValueError('Missing reverse child link')
        for pid,p in parents.items():
            if any(cid not in by_id or pid not in by_id[cid].get('parent_chunk_ids',[]) for cid in p['child_ids']):
                raise ValueError('Invalid child link')
    assignments = read_json(folder / 'entity_assignments.json')
    counts = manifest['counts']
    if (counts['chunks'] != len(chunks) or counts['tables'] != len(registry)
            or counts['documents'] != len(manifest['documents']) or counts['dense_shape'] != list(vectors.shape)
            or any(c['task_id'] != manifest['task_id'] for c in chunks)):
        raise ValueError('Index summary or task identity mismatch')
    doc_ids = {record['doc_id'] for record in manifest['documents']}
    roster = manifest['entity_assignment_roster']
    if roster != sorted(set(roster)) or set(assignments) != doc_ids:
        raise ValueError('Entity assignment roster or document set mismatch')
    for doc_id, assignment in assignments.items():
        entity_ids = assignment.get('entity_ids')
        scope = assignment.get('scope')
        expected_scope = ('unassigned' if not entity_ids else
                          'entity' if len(entity_ids) == 1 else 'multi_entity')
        if (assignment.get('doc_id') != doc_id or entity_ids != sorted(set(entity_ids or []))
                or not set(entity_ids or []).issubset(roster) or scope not in SCOPES
                or scope != expected_scope or not isinstance(assignment.get('assignment_basis'), list)
                or not isinstance(assignment.get('diagnostics'), list)
                or type(assignment.get('assignment_confidence')) not in (int, float)):
            raise ValueError('Invalid entity assignment')
    for chunk in chunks:
        assignment = assignments[chunk['doc_id']]
        expected = {'entity_ids':assignment['entity_ids'], 'entity_scope':assignment['scope'],
                    'entity_assignment_basis':assignment['assignment_basis'],
                    'entity_assignment_confidence':assignment['assignment_confidence']}
        if any(chunk.get(key) != value for key,value in expected.items()):
            raise ValueError('Chunk entity assignment mismatch')
    expected_document_scopes = {scope:sum(a['scope']==scope for a in assignments.values()) for scope in SCOPES}
    expected_chunk_scopes = {scope:sum(c['entity_scope']==scope for c in chunks) for scope in SCOPES}
    if (counts.get('entity_scope_documents') != expected_document_scopes
            or counts.get('entity_scope_chunks') != expected_chunk_scopes):
        raise ValueError('Entity assignment counts mismatch')
    for record in manifest['documents']:
        if sha256(within(root, record['result_path'])) != record['result_sha256']:
            raise ValueError('Processed result changed')
    for item in registry.values():
        read_table_ref(item['ref'], within(root, item['result_path']))
    retriever = bm25s.BM25.load(str(folder / 'bm25'), mmap=True)
    if int(retriever.scores['num_docs']) != len(rows):
        raise ValueError('BM25 row count mismatch')
    return {'task_id': manifest['task_id'], 'index_version': manifest['index_version'],
            **manifest['counts'], 'verified': True}
