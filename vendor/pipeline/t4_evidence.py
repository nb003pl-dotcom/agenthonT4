"""Per-unit, source-verified whole-chunk evidence retrieval. No model calls."""
import hashlib
import json
import re
from datetime import date
from pathlib import Path

FORMAT = 1
DEFAULT_BUDGET = 4000


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inputs(unit_dir):
    unit = Path(unit_dir)
    task = read_json(unit / 'task.json')
    cutoff = date.fromisoformat(task['cutoff_date'])
    if not task.get('task_id'):
        raise ValueError('task_id 缺失')
    docs = {}
    files = {}
    for path in sorted((unit / 'corpus').glob('*.json')):
        doc = read_json(path)
        if path.name == 'manifest.json' and 'text' not in doc and 'doc_id' not in doc:
            continue
        if not isinstance(doc.get('doc_id'), str) or not doc['doc_id']:
            raise ValueError(f'{path.name}: doc_id 缺失')
        if not isinstance(doc.get('text'), str) or not doc['text'].strip():
            raise ValueError(f"{doc['doc_id']}: 正文缺失、为空或只有空白，不能构建证据索引")
        if doc['doc_id'] in docs:
            raise ValueError('重复 doc_id: ' + doc['doc_id'])
        if doc.get('task_id', task['task_id']) != task['task_id']:
            raise ValueError('文档不属于当前题目: ' + doc['doc_id'])
        if not isinstance(doc.get('doc_date'), str):
            raise ValueError('文档日期缺失: ' + doc['doc_id'])
        if date.fromisoformat(doc['doc_date'][:10]) > cutoff:
            raise ValueError('文档日期晚于截止日: ' + doc['doc_id'])
        docs[doc['doc_id']] = doc
        files[path.name] = digest(doc)
    if not docs:
        raise ValueError('本题没有可用正文')
    return task, docs, {'task_sha256': digest(task), 'corpus_files': files}


def index_path(cache_dir):
    if not cache_dir:
        raise ValueError('请通过 --cache 指定本题证据索引目录，先由宿主执行 index')
    return Path(cache_dir) / 'index.json'


def chunk_identity(task_id, doc_id, source_sha, config, start, end):
    return 'chunk-' + digest([task_id, doc_id, source_sha, config, start, end])[:24]


def check_chunks(task, docs, config, chunks):
    from t4_splitter import count_tokens
    grouped = {doc_id: [] for doc_id in docs}
    source_hashes = {doc_id: digest(doc['text']) for doc_id, doc in docs.items()}
    seen = set()
    for chunk in chunks:
        doc = docs[chunk['doc_id']]
        a, b = chunk['start'], chunk['end']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(doc['text']):
            raise ValueError('缓存块位置无效')
        source_sha = source_hashes[doc['doc_id']]
        expected = chunk_identity(task['task_id'], doc['doc_id'], source_sha, config, a, b)
        if chunk['chunk_id'] != expected or expected in seen:
            raise ValueError('缓存块身份无效或重复')
        seen.add(expected)
        if (chunk['task_id'] != task['task_id'] or chunk['source_sha256'] != source_sha
                or chunk['doc_date'] != doc['doc_date']
                or chunk['algorithm_version'] != config['algorithm_version']):
            raise ValueError('缓存块归属或版本不匹配')
        if chunk['text'] != doc['text'][a:b] or chunk['evidence_id'] != f"{doc['doc_id']}::{a}::{b}":
            raise ValueError('缓存正文或引用与原文不一致')
        actual = count_tokens(chunk['text'])
        if actual != chunk['tokens'] or actual > 4000:
            raise ValueError('缓存块 token 计数无效或超限')
        grouped[doc['doc_id']].append(chunk)
    for doc_id, group in grouped.items():
        covered = 0
        for seq, chunk in enumerate(group, 1):
            if chunk['seq'] != seq or chunk['start'] > covered:
                raise ValueError('缓存编号不连续或原文覆盖存在缺口')
            covered = max(covered, chunk['end'])
        if covered != len(docs[doc_id]['text']):
            raise ValueError('缓存没有覆盖全部原文')


def load_index(unit_dir, cache_dir):
    from t4_splitter import configuration
    task, docs, source = inputs(unit_dir)
    path = index_path(cache_dir)
    if not path.is_file():
        raise ValueError('证据索引不存在，请先执行 index')
    data = read_json(path)
    config = configuration()
    if (data.get('format') != FORMAT or data.get('task_id') != task['task_id']
            or data.get('source') != source or data.get('configuration') != config):
        raise ValueError('证据索引已过期或属于其他题目，请重新执行 index')
    if data.get('chunks_sha256') != digest(data['chunks']):
        raise ValueError('证据索引内容摘要不匹配，请重新执行 index')
    check_chunks(task, docs, config, data['chunks'])
    return data, task, docs


def build_unit_index(unit_dir, cache_dir):
    from t4_splitter import configuration, split_document
    task, docs, source = inputs(unit_dir)
    path = index_path(cache_dir)
    config = configuration()
    if path.exists():
        try:
            data, _, _ = load_index(unit_dir, cache_dir)
            return summary(data, path, True)
        except (ValueError, KeyError, TypeError):
            pass  # Only an explicit index command rebuilds stale caches.
    chunks = []
    for doc_id, doc in docs.items():
        source_sha = digest(doc['text'])
        for chunk in split_document(doc_id, doc['text']):
            a, b = chunk['start'], chunk['end']
            chunks.append(dict(chunk, task_id=task['task_id'], doc_id=doc_id,
                               doc_date=doc['doc_date'], source_sha256=source_sha,
                               algorithm_version=config['algorithm_version'],
                               chunk_id=chunk_identity(task['task_id'], doc_id, source_sha, config, a, b),
                               evidence_id=f'{doc_id}::{a}::{b}'))
    check_chunks(task, docs, config, chunks)
    data = {'format': FORMAT, 'task_id': task['task_id'], 'cutoff_date': task['cutoff_date'],
            'configuration': config, 'source': source, 'chunks': chunks,
            'chunks_sha256': digest(chunks)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return summary(data, path, False)


def summary(data, path, cache_hit):
    return {'task_id': data['task_id'], 'algorithm_version': data['configuration']['algorithm_version'],
            'documents': len({c['doc_id'] for c in data['chunks']}), 'chunks': len(data['chunks']),
            'max_chunk_tokens': max(c['tokens'] for c in data['chunks']),
            'index': str(path), 'cache_hit': cache_hit}


def result_chunk(chunk):
    # Keep text once in the tool result; quote remains compatible with existing evidence packs.
    return {k: v for k, v in chunk.items() if k != 'text'} | {
        'span_start': chunk['start'], 'span_end': chunk['end'], 'quote': chunk['text']}


def transport_result(result):
    """Lossless JSON display: avoid long-line clipping risks in tool readers."""
    def convert(value):
        if isinstance(value, list):
            return [convert(x) for x in value]
        if not isinstance(value, dict):
            return value
        out = {k: convert(v) for k, v in value.items() if k != 'quote'}
        if 'quote' in value:
            quote = value['quote']
            out['quote_parts'] = [quote[i:i+400] for i in range(0, len(quote), 400)]
        return out
    displayed = convert(result)
    displayed['quote_encoding'] = 'quote_parts 按顺序无分隔拼接即完整原文；未删改正文或重新切块'
    displayed['end_of_evidence'] = '本JSON结束；同时检查工具截断/省略提示，不能仅凭本标记认定完整'
    return displayed


def select_chunks(candidates, *, limit, token_budget, offset=0):
    if type(limit) is not int or limit < 1 or type(token_budget) is not int or token_budget < 1:
        raise ValueError('limit 和 token_budget 必须为正整数')
    if type(offset) is not int or not 0 <= offset <= len(candidates):
        raise ValueError('offset 越界')
    selected = []
    used = 0
    cursor = offset
    while cursor < len(candidates) and len(selected) < limit:
        chunk = candidates[cursor]
        if used + chunk['tokens'] > token_budget:
            break
        selected.append(result_chunk(chunk))
        used += chunk['tokens']
        cursor += 1
    return {'chunks': selected, 'total_matches': len(candidates), 'returned': len(selected),
            'body_tokens': used, 'body_token_budget': token_budget, 'offset': offset,
            'next_offset': cursor if cursor < len(candidates) else None,
            'minimum_next_chunk_tokens': candidates[cursor]['tokens'] if cursor < len(candidates) else None,
            'budget_limited': cursor < len(candidates) and used + candidates[cursor]['tokens'] > token_budget,
            'budget_scope': 'cl100k_base 正文；不含JSON/工具包装、提示词、历史或模型输出',
            'selection_complete': cursor == len(candidates)}


def search_chunks(unit_dir, query, cache_dir, doc_id=None, limit=12, token_budget=DEFAULT_BUDGET, offset=0):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query 不能为空')
    data, task, docs = load_index(unit_dir, cache_dir)
    if doc_id is not None and doc_id not in docs:
        raise ValueError('文档不存在: ' + doc_id)
    pattern = re.compile(re.escape(query), re.I)
    # Search the actual full source first; even a phrase spanning a hard block
    # boundary must remain discoverable, with both intersecting blocks available.
    source_hits = {key: list(pattern.finditer(doc['text'])) for key, doc in docs.items()
                   if doc_id is None or key == doc_id}
    candidates = []
    for chunk in data['chunks']:
        if doc_id is not None and chunk['doc_id'] != doc_id:
            continue
        hits = [hit for hit in source_hits[chunk['doc_id']]
                if hit.start() < chunk['end'] and hit.end() > chunk['start']]
        if hits:
            candidates.append(dict(chunk, match_count=len(hits),
                                   first_match_start=hits[0].start(),
                                   match_crosses_block=any(h.start()<chunk['start'] or h.end()>chunk['end'] for h in hits)))
    candidates.sort(key=lambda c: (-c['match_count'], c['doc_id'], c['start'], c['end']))
    return dict(select_chunks(candidates, limit=limit, token_budget=token_budget, offset=offset),
                task_id=task['task_id'], query=query, ranking='字面命中数降序，其次doc_id与原文位置')


def read_chunk(unit_dir, chunk_id, cache_dir, before=0, after=0, token_budget=DEFAULT_BUDGET):
    if type(before) is not int or type(after) is not int or min(before, after) < 0:
        raise ValueError('before/after 必须为非负整数')
    data, task, docs = load_index(unit_dir, cache_dir)
    selected = next((c for c in data['chunks'] if c['chunk_id'] == chunk_id), None)
    if selected is None:
        raise ValueError('chunk_id 不存在或不属于本题')
    same_doc = [c for c in data['chunks'] if c['doc_id'] == selected['doc_id']]
    pos = selected['seq'] - 1
    candidates = same_doc[max(0, pos-before):min(len(same_doc), pos+after+1)]
    # Neighbour reads are all-or-nothing so an omitted centre is never presented as success.
    required = sum(c['tokens'] for c in candidates)
    if type(token_budget) is not int or token_budget < 1:
        raise ValueError('token_budget 必须为正整数')
    if required > token_budget:
        return {'chunks': [], 'returned': 0, 'task_id': task['task_id'], 'budget_limited': True,
                'required_body_tokens': required, 'body_token_budget': token_budget,
                'selection_complete': False, 'center_chunk_id': chunk_id,
                'prev_chunk_id': same_doc[pos-1]['chunk_id'] if pos else None,
                'next_chunk_id': same_doc[pos+1]['chunk_id'] if pos+1 < len(same_doc) else None,
                'hint': '分别按相邻ID读取、减少相邻块数量或提高正文预算；未截断任何块'}
    result = select_chunks(candidates, limit=len(candidates), token_budget=token_budget)
    # Do not concatenate overlapping text. Each returned citation remains an exact source slice.
    result.update(task_id=task['task_id'], center_chunk_id=chunk_id,
                  source_union={'doc_id': selected['doc_id'], 'start': candidates[0]['start'],
                                'end': max(c['end'] for c in candidates)},
                  prev_chunk_id=same_doc[pos-1]['chunk_id'] if pos else None,
                  next_chunk_id=same_doc[pos+1]['chunk_id'] if pos+1 < len(same_doc) else None)
    return result


def quote_span(unit_dir, chunk_id, quote, cache_dir):
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError('引用文字不能为空')
    data, task, docs = load_index(unit_dir, cache_dir)
    chunk = next((c for c in data['chunks'] if c['chunk_id'] == chunk_id), None)
    if chunk is None:
        raise ValueError('chunk_id 不存在或不属于本题')
    a = chunk['text'].find(quote)
    if a < 0:
        raise ValueError('文字不是块内连续原文')
    if chunk['text'].find(quote, a+1) >= 0:
        raise ValueError('块内引用文字不唯一，请补充上下文')
    start, end = chunk['start'] + a, chunk['start'] + a + len(quote)
    if docs[chunk['doc_id']]['text'][start:end] != quote:
        raise ValueError('引用回读不一致')
    return {'task_id': task['task_id'], 'doc_id': chunk['doc_id'], 'doc_date': chunk['doc_date'],
            'chunk_id': chunk_id, 'evidence_id': f"{chunk['doc_id']}::{start}::{end}",
            'span_start': start, 'span_end': end, 'quote': quote}


def initial_pack(unit_dir, cache_dir, token_budget=DEFAULT_BUDGET):
    data, task, docs = load_index(unit_dir, cache_dir)
    eps = 'eps' in task['family'] or task['family'] == 'post_earnings_reaction'
    terms = (r'diluted.{0,35}(?:earnings|income|share)|earnings per.{0,15}share|net interest income|outlook|guidance|provision for credit|net income'
             if eps else r'going concern|liquidity|default|bankruptcy|maturit|cash flows|debt covenant|restructur|cash and cash equivalents')
    pattern = re.compile(terms, re.I)
    grouped = {}
    for chunk in data['chunks']:
        count = len(pattern.findall(chunk['text']))
        grouped.setdefault(chunk['doc_id'], []).append(dict(chunk, match_count=count))
    # Round-robin across documents; within each document use hit count then source order.
    # Unmatched chunks only seed small corpora; long corpora remain available via search.
    for group in grouped.values():
        group.sort(key=lambda c: (-c['match_count'], c['start']))
    ranked = []
    while any(grouped.values()):
        for doc_id in sorted(grouped):
            if grouped[doc_id]:
                ranked.append(grouped[doc_id].pop(0))
    pack = select_chunks(ranked, limit=len(ranked), token_budget=token_budget)
    pack.update(task_id=task['task_id'], algorithm_version=data['configuration']['algorithm_version'],
                documents=[{'doc_id': k, 'doc_date': d['doc_date'], 'chars': len(d['text'])} for k, d in docs.items()],
                selection_method='每文档按指标关键词命中数排序，再按doc_id轮询；预算不足时停止，后续按需检索')
    return pack
