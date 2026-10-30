"""Publish complete tables once; source chunks and embedding bytes stay unchanged."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
from decimal import Decimal
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
VERSION = 'table-ref-v1'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def column_name(number):
    out = ''
    while number:
        number, rem = divmod(number - 1, 26)
        out = chr(65 + rem) + out
    return out


class PlainCell(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag == 'br':
            self.parts.append('\n')


def plain(value):
    parser = PlainCell()
    parser.feed(str(value))
    return ''.join(parser.parts)


def cell(value, header='', is_header=False):
    """Conservative cell types; original cells remain in tables.json."""
    value = plain(value)
    if len(value.encode('utf-16-le')) // 2 > 32767 or re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', value):
        raise ValueError('Excel cell exceeds limits or contains unsupported control characters')
    result = {'value': value, 'kind': 'text', 'format': '@'}
    if is_header or re.search(r'(^|[_\s])(id|code|ticker|cusip|isin)($|[_\s])|代码|编码', header, re.I):
        return result
    token = value.strip()
    if not token:
        return {'value': None, 'kind': 'blank', 'format': 'General'}
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', token):
        try:
            parsed = date.fromisoformat(token)
            if parsed >= date(1900, 3, 1):
                return {'value': token, 'kind': 'date', 'format': 'yyyy-mm-dd'}
        except ValueError:
            pass
    number = re.fullmatch(r'([+-]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%)?', token)
    if number:
        digits = number[2].replace(',', '')
        precision = len((digits + (number[3] or '')).replace('.', '').lstrip('0'))
        if (len(digits) > 1 and digits.startswith('0')) or precision > 15:
            return result
        val = Decimal(token.replace(',', '').rstrip('%'))
        decimals = len(number[3] or '') - 1 if number[3] else 0
        fmt = ('#,##0' if ',' in token else '0') + ('.' + '0' * decimals if decimals else '')
        if number[4]:
            val /= 100
            fmt += '%'
        if number[1] == '+':
            fmt = '+' + fmt + ';-' + fmt + ';' + fmt
        return {'value': int(val) if val == val.to_integral_value() else float(val), 'kind': 'number', 'format': fmt}
    return result


def make_spec(data):
    unique = {}
    for chunk in data['result']:
        for table in chunk.get('tables', []):
            source = {k: v for k, v in table.items() if k not in ('partial_in_chunk', 'overlap_start', 'overlap_end')}
            key = table['table_id']
            if key in unique and unique[key] != source:
                raise ValueError(f'Conflicting copies of table {key}')
            unique[key] = source
    spec = {'version': VERSION, 'doc_id': data['meta']['doc_id'], 'tables': []}
    for index, table in enumerate(sorted(unique.values(), key=lambda t: t['start']), 1):
        width = len(table['header'])
        rows = table['rows']
        if not 1 <= width <= 16384 or len(rows) + 1 > 1048576 or any(len(r) != width for r in rows):
            raise ValueError('Invalid or oversized table dimensions')
        key = json.dumps([spec['doc_id'], table['table_id'], table['start'], table['end'], table['text']], ensure_ascii=False)
        ref_id = 'tbl_' + hashlib.sha256(key.encode()).hexdigest()[:24]
        last = column_name(width)
        spec['tables'].append({
            **table, 'ref_id': ref_id, 'sheet': f'table_{index:03d}',
            'range': f'A1:{last}{len(rows) + 1}', 'last_column': last,
            'row_count': len(rows), 'column_count': width,
            'cells': [[cell(v, is_header=True) for v in table['header']],
                      *[[cell(v, plain(table['header'][j])) for j, v in enumerate(row)] for row in rows]],
        })
    return spec


from portable_xlsx import export_xlsx


def publish_result(data, output: Path, previews=None):
    """Commit JSON only after XLSX export and full cell verification succeed."""
    started = time.monotonic()
    output = Path(output).resolve()
    if data.get('meta', {}).get('table_reference_version'):
        raise ValueError('Input is already a reference result; use the original inline result')
    data = deepcopy(data)
    spec = make_spec(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    base = output.parent / (output.stem + '.tables')
    refs = {}
    artifacts = []
    if spec['tables']:
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.staging-', dir=base) as tmp:
            staging = Path(tmp)
            workbook = staging / 'tables.xlsx'
            export_xlsx(spec, workbook, previews)
            from read_table import verify_workbook
            verify_workbook(workbook, spec)
            metadata = staging / 'tables.json'
            metadata.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
            workbook_hash, metadata_hash = sha256(workbook), sha256(metadata)
            bundle_id = hashlib.sha256((workbook_hash + metadata_hash).encode()).hexdigest()[:24]
            destination = base / bundle_id
            if destination.exists():
                if sha256(destination / 'tables.xlsx') != workbook_hash or sha256(destination / 'tables.json') != metadata_hash:
                    raise ValueError('Artifact directory already exists with different contents')
            else:
                # Same-filesystem atomic publication; existing bundles are never overwritten.
                staging.rename(destination)
            relative = destination.relative_to(output.parent)
            for table in spec['tables']:
                refs[table['table_id']] = {
                    'type': 'table_ref', 'version': 1,
                    'ref_id': table['ref_id'], 'table_id': table['table_id'],
                    'path': (relative / 'tables.xlsx').as_posix(), 'sha256': workbook_hash,
                    'metadata_path': (relative / 'tables.json').as_posix(), 'metadata_sha256': metadata_hash,
                    'sheet': table['sheet'], 'range': table['range'], 'header_row': 1,
                    'header': [plain(v) for v in table['header']],
                    'row_count': table['row_count'], 'column_count': table['column_count'],
                    'source': {'doc_id': spec['doc_id'], 'start': table['start'], 'end': table['end']},
                }
            artifacts = [{'type': 'xlsx', 'path': (relative / 'tables.xlsx').as_posix(), 'sha256': workbook_hash,
                          'table_count': len(refs)}]
    for chunk in data['result']:
        attached = sorted(chunk.pop('tables', []), key=lambda t: t['start'])
        chunk['table_refs'] = []
        parts, cursor = [], chunk['start']
        for table in attached:
            lo, hi = max(chunk['start'], table['start']), min(chunk['end'], table['end'])
            if not cursor <= lo < hi or chunk['text'][lo-chunk['start']:hi-chunk['start']] != table['text'][lo-table['start']:hi-table['start']]:
                raise ValueError('Table overlap does not match original chunk text')
            ref = {**refs[table['table_id']], 'partial_in_chunk': not (chunk['start'] <= table['start'] and table['end'] <= chunk['end']),
                   'overlap_start': lo, 'overlap_end': hi}
            chunk['table_refs'].append(ref)
            parts.extend([chunk['text'][cursor-chunk['start']:lo-chunk['start']], f'\n[[TABLE_REF:{ref["ref_id"]}]]\n'])
            cursor = hi
        parts.append(chunk['text'][cursor-chunk['start']:])
        chunk['agent_text'] = ''.join(parts)
    data['artifacts'] = artifacts
    data['meta'].update(table_reference_version=VERSION, exported_tables=len(refs),
                        elapsed_table_export_seconds=round(time.monotonic() - started, 3),
                        table_export_scope='complete tables referenced by returned chunks',
                        agent_text_usage='Use agent_text plus table_refs in agent prompts; original text remains embedding input')
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(output)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export tables from an existing inline pipeline result without rerunning embeddings')
    parser.add_argument('--input-result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--previews', type=Path)
    args = parser.parse_args()
    if args.input_result.resolve() == args.out.resolve():
        parser.error('Do not overwrite the input result')
    result = publish_result(json.loads(args.input_result.read_text()), args.out, args.previews)
    print(json.dumps({'output': str(args.out.resolve()), 'tables': result['meta']['exported_tables'], 'artifacts': result['artifacts']}, ensure_ascii=False, indent=2))
