"""O4000 -> complete tables -> local CPU BGE-M3 -> XLSX + referenced JSON."""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE
sys.path.insert(0, str(SCRIPTS))
DEFAULT_MODEL = HERE / 'models/bge-m3'


def extract_tables(text: str) -> list[dict]:
    """Compatibility entry point: standard and conservative separatorless tables."""
    from table_extraction import extract_tables_with_diagnostics
    return extract_tables_with_diagnostics(text)[0]


def attach_tables(chunk: dict, tables: list[dict]) -> dict:
    a, b = chunk['start'], chunk['end']
    attached = []
    outside = []
    cursor = a
    for table in tables:
        lo, hi = max(a, table['start']), min(b, table['end'])
        if lo >= hi:
            continue
        outside.append(chunk['text'][cursor - a:lo - a])
        cursor = hi
        attached.append({**table, 'partial_in_chunk': not (a <= table['start'] and table['end'] <= b),
                         'overlap_start': lo, 'overlap_end': hi})
    outside.append(chunk['text'][cursor - a:])
    content_type = 'text' if not attached else ('mixed' if ''.join(outside).strip() else 'table')
    return {**chunk, 'item_id': chunk['evidence_id'], 'tables': attached, 'content_type': content_type}


def verify_model(model_dir: Path) -> dict:
    manifest = json.loads((model_dir / 'MODEL_MANIFEST.json').read_text())
    if manifest['repo'] != 'BAAI/bge-m3' or manifest['revision'] != '5617a9f61b028005a4858fdac845db406aefb181':
        raise ValueError('Unexpected model identity; use download_model.py')
    for name, info in manifest['files'].items():
        path = model_dir / name
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha.update(data)
        if path.stat().st_size != info['bytes'] or sha.hexdigest() != info['sha256']:
            raise ValueError(f'Model file checksum mismatch: {name}')
    return manifest


class CPUEmbedder:
    """Official dense representation: normalized last_hidden_state[:, 0]."""
    def __init__(self, model_dir=DEFAULT_MODEL, batch_size=1, threads=4):
        if batch_size < 1 or threads < 1:
            raise ValueError('batch_size and threads must be positive')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.manifest = verify_model(Path(model_dir))
        torch.set_num_threads(threads)
        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True,
                                               use_safetensors=False, weights_only=True).float().cpu().eval()

    def encode(self, texts: list[str]) -> tuple[list, list[int]]:
        vectors, lengths = [], []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset:offset + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=False, return_tensors='pt')
            counts = encoded['attention_mask'].sum(dim=1).tolist()
            if max(counts) > 8192:
                raise ValueError(f'BGE-M3 input exceeds 8192 tokens at batch {offset}: {counts}; no truncation performed')
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state[:, 0]
                dense = self.torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy()
            if dense.shape != (len(batch), 1024) or not self.torch.isfinite(self.torch.from_numpy(dense)).all():
                raise ValueError('Invalid BGE-M3 embedding shape or non-finite value')
            vectors.extend(dense)
            lengths.extend(counts)
            print(f'Embedded {min(offset + len(batch), len(texts))}/{len(texts)} chunks', file=sys.stderr, flush=True)
        return vectors, lengths


def process_document(doc_id: str, text: str, embedder: CPUEmbedder, limit_chunks=None) -> dict:
    from t4_splitter import split_document
    if limit_chunks is not None and limit_chunks < 1:
        raise ValueError('limit_chunks must be positive')
    started = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        chunks = split_document(doc_id, text)
        from table_extraction import extract_tables_with_diagnostics
        tables, table_report = extract_tables_with_diagnostics(text)
    total = len(chunks)
    selected = chunks if limit_chunks is None else chunks[:limit_chunks]
    result = [attach_tables(chunk, tables) for chunk in selected]
    vectors, lengths = embedder.encode([c['text'] for c in result])
    if len(vectors) != len(result) or len(lengths) != len(result):
        raise ValueError('Embedding result count differs from chunk count')
    for chunk, vector, length in zip(result, vectors, lengths):
        chunk.update(embedding=base64.b64encode(zlib.compress(vector.astype('<f4').tobytes())).decode('ascii'),
                     embeding_dim=1024, embedding_dim=1024, embedding_input_tokens=length,
                     embedding_source='text', embedding_encoding='float32-le+zlib+base64')
    return {'status': 0, 'result': result, 'meta': {
        'doc_id': doc_id, 'total_chunks': total, 'returned_chunks': len(result),
        'complete_document': len(result) == total, 'source_tables': len(tables),
        'table_extraction': table_report,
        'model': 'BAAI/bge-m3', 'model_revision': embedder.manifest['revision'],
        'device': 'cpu', 'precision': 'float32', 'normalized': True,
        'embedding_source': 'final chunk.text only; table attachments are not concatenated',
        'elapsed_processing_seconds': round(time.monotonic() - started, 3),
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='T4 JSON or UTF-8 Markdown/text')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--doc-id')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--table-previews', type=Path, help='Optional Excel preview directory for validation')
    parser.add_argument('--limit-chunks', type=int, help='Smoke test only: embed the first N chunks; output explicitly marked incomplete')
    args = parser.parse_args()
    if args.limit_chunks is not None:
        parser.error('--limit-chunks is unsupported by table-first preprocessing; complete source coverage is required')
    if args.input.resolve() == args.out.resolve():
        parser.error('Output must not overwrite input')
    started = time.monotonic()
    raw = args.input.read_text(encoding='utf-8-sig')
    document = json.loads(raw) if args.input.suffix.lower() == '.json' else {'text': raw}
    doc_id = args.doc_id or document.get('doc_id') or document.get('id') or args.input.stem
    print('Verifying and loading local BGE-M3 on CPU...', file=sys.stderr, flush=True)
    embedder = CPUEmbedder(args.model_dir, args.batch_size, args.threads)
    from table_first import process_table_first
    data = process_table_first(doc_id, document['text'], embedder, args.out, args.table_previews)
    print(json.dumps({'output': str(args.out.resolve()), **data['meta']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
