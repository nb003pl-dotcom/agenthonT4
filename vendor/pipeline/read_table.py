"""Agent-side local XLSX reader. No model calls, no formula evaluation."""
import argparse
from datetime import date, datetime
import hashlib
import json
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries


def _hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _resolve(root, relative, checksum):
    if Path(relative).is_absolute():
        raise ValueError('Table path must be relative to result directory')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Table path escapes result directory')
    if _hash(path) != checksum:
        raise ValueError(f'Table artifact checksum mismatch: {relative}')
    return path


def _matrix(workbook, table):
    sheet = workbook[table['sheet']]
    if sheet.max_row is None or sheet.max_column is None:
        sheet.calculate_dimension(force=True)
    left, top, right, bottom = range_boundaries(table['range'])
    if left != 1 or top != 1 or right != table['column_count'] or bottom != table['row_count'] + 1:
        raise ValueError('Table range and dimensions disagree')
    if sheet.max_row != bottom or sheet.max_column != right:
        raise ValueError('Workbook contains an incomplete or unexpected table range')
    values = []
    for row in sheet.iter_rows(min_row=top, max_row=bottom, min_col=left, max_col=right):
        if any(c.data_type == 'f' for c in row):
            raise ValueError('Unexpected executable formula in source table')
        values.append([c.value for c in row])
    return values


def verify_workbook(path, spec):
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        if workbook.sheetnames != [t['sheet'] for t in spec['tables']]:
            raise ValueError('Workbook sheets differ from extracted tables')
        for table in spec['tables']:
            values = _matrix(workbook, table)
            for row, expected in zip(values, table['cells']):
                for value, cell in zip(row, expected):
                    target = cell['value']
                    if cell['kind'] == 'date':
                        value = value.date().isoformat() if isinstance(value, datetime) else value.isoformat() if isinstance(value, date) else value
                    # Empty Excel strings round-trip as None; blank remains distinct from zero.
                    if target == '' and value is None:
                        value = ''
                    if value != target:
                        raise ValueError(f'Excel cell round-trip mismatch: {table["sheet"]}: {value!r} != {target!r}')
    finally:
        workbook.close()


def read_table_ref(ref, result_path):
    """Return {header, rows, source, original}; accepts only this contract's local refs."""
    if ref.get('type') != 'table_ref' or ref.get('version') != 1:
        raise ValueError('Unsupported table reference')
    root = Path(result_path).resolve().parent
    path = _resolve(root, ref['path'], ref['sha256'])
    metadata_path = _resolve(root, ref['metadata_path'], ref['metadata_sha256'])
    spec = json.loads(metadata_path.read_text(encoding='utf-8'))
    if spec.get('version') != 'table-ref-v1':
        raise ValueError('Unsupported table metadata version')
    table = next((t for t in spec['tables'] if t['ref_id'] == ref['ref_id']), None)
    if table is None or any(ref[k] != table[k] for k in ('table_id', 'sheet', 'range', 'row_count', 'column_count')):
        raise ValueError('Table reference disagrees with metadata')
    if ref['source'] != {'doc_id': spec['doc_id'], 'start': table['start'], 'end': table['end']}:
        raise ValueError('Table source disagrees with metadata')
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        values = _matrix(workbook, table)
        return {'header': values[0], 'rows': values[1:], 'source': ref['source'],
                'original': {k: v for k, v in table.items() if k not in ('cells',)}}
    finally:
        workbook.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read a complete table through a validated chunk reference')
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--ref-id', help='Default: first referenced table')
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding='utf-8'))
    refs = [r for c in data['result'] for r in c.get('table_refs', [])]
    ref = next((r for r in refs if args.ref_id is None or r['ref_id'] == args.ref_id), None)
    if ref is None:
        parser.error('No matching table reference')
    table = read_table_ref(ref, args.result)
    print(json.dumps({k: v for k, v in table.items() if k != 'original'}, ensure_ascii=False, indent=2, default=str))
