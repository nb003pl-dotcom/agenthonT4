"""Complete one already-read adjacent quote; never search new documents for numbers."""
import copy
import sys
from .common import ROOT
from .evaluation import judge_token_counter
from t4_splitter import count_tokens


def numeric_status(statement, citations, names=()):
    sys.path.insert(0, str(ROOT / 'vendor/official'))
    from qfbench2_track_analysis.numeric import claim_number_status
    return claim_number_status(statement, [c['quote'] for c in citations], rule='every', names=names)


def complete_adjacent_citation(fact, catalog, names=()):
    """Minimum added source characters among bounded contiguous candidates.

    This proves literal numeric presence only. It never certifies period,
    column, metric, sign or causal entailment. A derived quantity is left alone.
    """
    current = fact.get('citations', [])
    statement = fact.get('statement', '')
    if not isinstance(statement, str) or not statement.strip():
        return fact, {'changed': False, 'reason': 'invalid_statement_left_for_fact_validator',
                      'semantic_status': 'unverified'}
    before = numeric_status(statement, current, names)
    audit = {'status_before': before, 'status_after': before, 'changed': False,
             'semantic_status': 'unverified'}
    if before != 'unanchored':
        return fact, audit
    solutions = []
    for index, selected in enumerate(current):
        span = selected.get('_catalog_span')
        if not span:
            continue
        ordered = sorted([c for c in catalog.values()
                          if c['doc_id'] == selected['doc_id'] and c['chunk_id'] == selected['chunk_id']],
                         key=lambda c: c['_catalog_span']['start'])
        positions = [j for j, c in enumerate(ordered) if c['_catalog_span'] == span and c['quote'] == selected['quote']]
        if not positions:
            continue
        j = positions[0]
        window = ordered[j].get('_catalog_window')
        for left in range(max(0, j - 2), j + 1):
            for right in range(j + 1, min(len(ordered), j + 3) + 1):
                pieces = ordered[left:right]
                if len(pieces) < 2 or len(pieces) > 4:
                    continue
                if window and any(c.get('_catalog_window') != window for c in pieces):
                    continue
                if any(a['_catalog_span']['end'] != b['_catalog_span']['start'] for a, b in zip(pieces, pieces[1:])):
                    continue
                quote = ''.join(c['quote'] for c in pieces)
                if len(quote) > 1600 or count_tokens(quote) > 350 or judge_token_counter()(quote) > 400:
                    continue
                expanded = {**selected, 'quote': quote,
                            '_catalog_span': {'start': pieces[0]['_catalog_span']['start'], 'end': pieces[-1]['_catalog_span']['end']}}
                proposed = current[:index] + [expanded] + current[index + 1:]
                if numeric_status(statement, proposed, names) == 'anchored':
                    solutions.append((len(quote) - len(selected['quote']), index, expanded))
    if not solutions:
        return fact, {**audit, 'reason': 'no_bounded_already_read_completion'}
    extra, index, expanded = min(solutions, key=lambda r: (r[0], r[1], r[2]['_catalog_span']['start']))
    result = copy.deepcopy(fact)
    result['citations'][index] = expanded
    return result, {**audit, 'changed': True, 'status_after': 'anchored', 'added_characters': extra,
                    'doc_id': expanded['doc_id'], 'chunk_id': expanded['chunk_id'],
                    'old_span': current[index]['_catalog_span'], 'new_span': expanded['_catalog_span'],
                    'reason': 'contiguous_already_read_source_within_quote_limits'}
