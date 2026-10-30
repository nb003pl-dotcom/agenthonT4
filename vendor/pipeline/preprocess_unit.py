"""Preprocess a complete T4 unit; publish BM25 and dense indexes together."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import time
import tomllib

from pipeline import CPUEmbedder, DEFAULT_MODEL, HERE, process_document
from table_artifacts import publish_result, sha256
from search_index import (REVISION, atomic_json, build_search_index, digest, read_json,
                          validate_result, verify_index, within, write_json)
from t4_splitter import configuration
from table_first import process_table_first, VERSION


def load_unit(unit):
    unit = Path(unit).resolve()
    task = read_json(unit / 'task.json')
    if not isinstance(task.get('task_id'), str) or not task['task_id']:
        raise ValueError('Missing task_id')
    cutoff = date.fromisoformat(task['cutoff_date'])
    tomllib.loads((unit / 'card.toml').read_text(encoding='utf-8'))
    manifest = read_json(unit / 'manifest.json')
    if not isinstance(manifest.get('unit_id'), str):
        raise ValueError('Missing manifest unit_id')
    declared, docs = {}, {}
    for entry in manifest['files']:
        name = entry['path']
        if name in declared:
            raise ValueError('Duplicate manifest path')
        path = within(unit, name)
        if sha256(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ValueError(f'Manifest checksum/size mismatch: {name}')
        declared[name] = entry['sha256']
        if entry.get('role') == 'corpus':
            doc = read_json(path)
            if not isinstance(doc.get('doc_id'), str) or not doc['doc_id'] or doc['doc_id'] in docs:
                raise ValueError('Missing or duplicate doc_id')
            if not isinstance(doc.get('text'), str) or not doc['text'].strip():
                raise ValueError('Missing corpus text')
            if doc.get('task_id', task['task_id']) != task['task_id'] or date.fromisoformat(doc['doc_date'][:10]) > cutoff:
                raise ValueError('Corpus task/date mismatch')
            docs[doc['doc_id']] = doc
    # Some official units omit card.toml from the root manifest. It must still
    # exist and parse (above); record its local hash without claiming an official checksum.
    unlisted_controls = [n for n in ('task.json', 'card.toml') if n not in declared]
    for name in unlisted_controls:
        declared[name] = sha256(unit / name)
    nested_name = task.get('corpus_manifest', 'corpus/manifest.json')
    if nested_name != 'manifest.json' and nested_name not in declared:
        raise ValueError('Corpus manifest not covered by root manifest')
    nested = read_json(within(unit, nested_name))
    root_corpus = {e['path']: e['sha256'] for e in manifest['files'] if e.get('role') == 'corpus'}
    nested_corpus = {e['path']: e['sha256'] for e in nested['files'] if e.get('role') == 'corpus'}
    if nested_corpus != root_corpus or nested.get('unit_id') != manifest['unit_id']:
        raise ValueError('Corpus manifests disagree')
    actual = {p.relative_to(unit).as_posix() for p in (unit / 'corpus').rglob('*.json')}
    if actual != {n for n in declared if n.startswith('corpus/') and n.endswith('.json')} or not docs:
        raise ValueError('Unlisted/missing corpus files or empty corpus')
    identity = {'unit_id': manifest['unit_id'], 'files': {**declared, 'manifest.json': sha256(unit / 'manifest.json')},
                'locally_hashed_unlisted_controls': unlisted_controls}
    return task, dict(sorted(docs.items())), identity


def processing_config(granularity='chunk'):
    config = {'splitter': configuration(), 'processing_version': VERSION, 'model_revision': REVISION,
            'code': {name: sha256(HERE / name) for name in
                     ('pipeline.py', 'table_first.py', 'table_extraction.py', 'table_artifacts.py', 'portable_xlsx.py', 'read_table.py')}}
    if granularity=='sentence':
        from sentence_nodes import sentence_configuration
        config['sentence']=sentence_configuration()
    return config


def preprocess(unit, output, model_dir=DEFAULT_MODEL, batch_size=1, threads=4, index_only=False, embedder=None, granularity='chunk', official_assignments=None):
    if granularity not in ('chunk','sentence'):
        raise ValueError('Unknown granularity')
    started = time.monotonic()
    unit, output = Path(unit).resolve(), Path(output).resolve()
    if output.is_relative_to(unit):
        raise ValueError('Output must be outside the immutable input unit')
    task, docs, inputs = load_unit(unit)
    output.mkdir(parents=True, exist_ok=True)
    owner_path = output / 'owner.json'
    owner = {'unit_id': inputs['unit_id'], 'task_id': task['task_id']}
    if owner_path.exists() and read_json(owner_path) != owner:
        raise ValueError('Output directory belongs to another task')
    atomic_json(owner_path, owner)
    config = processing_config(granularity)
    if embedder is not None and hasattr(embedder, 'binding'):
        config['remote_embedding_binding'] = embedder.binding
    reused = 0
    records = []
    for i, (doc_id, doc) in enumerate(docs.items(), 1):
        identity = {'doc': doc, 'processing': config}
        key = digest(identity)
        directory = output / 'documents' / key
        path = directory / 'result.json'
        cache = directory / 'cache.json'
        print(f'[{i}/{len(docs)}] {doc_id}', file=sys.stderr, flush=True)
        if cache.exists():
            stamp = read_json(cache)
            if stamp['identity'] != key or stamp['result_sha256'] != sha256(path):
                raise ValueError(f'Corrupt document cache: {directory}; use a new --out directory')
            validate_result(doc, read_json(path), path)
            reused += 1
            print('  reuse verified result (no model call)', file=sys.stderr, flush=True)
        else:
            if directory.exists():
                raise ValueError(f'Incomplete existing cache: {directory}')
            if index_only:
                raise ValueError(f'Missing verified result for {doc_id}; run without --index-only first')
            if embedder is None:
                print('Loading local BGE-M3 once...', file=sys.stderr, flush=True)
                embedder = CPUEmbedder(model_dir, batch_size, threads)
            directory.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.staging-', dir=directory.parent) as tmp:
                folder = Path(tmp)
                process_table_first(doc_id, doc['text'], embedder, folder / 'result.json', granularity=granularity)
                validate_result(doc, read_json(folder / 'result.json'), folder / 'result.json')
                write_json(folder / 'cache.json', {'identity': key, 'result_sha256': sha256(folder / 'result.json')})
                folder.rename(directory)
        records.append({'doc_id': doc_id, 'result_path': path.relative_to(output).as_posix(), 'result_sha256': sha256(path)})
    if load_unit(unit)[2] != inputs:
        raise ValueError('Input files changed during preprocessing; index not published')
    summary, hit = build_search_index(output, task, docs, records, inputs, official_assignments=official_assignments)
    folder = output / 'search' / summary['index_version']
    atomic_json(output / 'current.json', {'task_id': task['task_id'], 'index_version': summary['index_version'],
                'index_path': folder.relative_to(output).as_posix(), 'manifest_sha256': sha256(folder / 'manifest.json')})
    summary.update(reused_documents=reused, processed_documents=len(docs) - reused, index_cache_hit=hit,
                   elapsed_seconds=round(time.monotonic() - started, 3), output=str(output))
    atomic_json(output / 'preprocessing-report.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', type=Path, help='Directory with task/card/manifest/corpus')
    parser.add_argument('--out', type=Path, required=True, help='Dedicated output directory for one unit')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--index-only', action='store_true', help='Require existing verified document caches; never load a model')
    parser.add_argument('--granularity', choices=['chunk','sentence'], default='chunk')
    parser.add_argument('--check', action='store_true', help='Only verify the already published index and table references')
    args = parser.parse_args()
    if args.check:
        report = verify_index(args.out)
    else:
        if not args.unit:
            parser.error('--unit is required unless --check is used')
        report = preprocess(args.unit, args.out, args.model_dir, args.batch_size, args.threads, args.index_only, granularity=args.granularity)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
