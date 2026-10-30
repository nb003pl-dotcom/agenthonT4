"""Preserve verbatim qualifications from already-read, single-owner sources.

No accounting adjustment, recurrence classification, or new retrieval. These
are source passages for the predictor to interpret, not verified adjustments.
"""
import re

from .facts import validate_facts
from t4_splitter import count_tokens

TRIGGER = re.compile(
    r"\b(?:results|net income|earnings)\b.{0,110}?\b(?:included|includes|reflects|reflected)\b"
    r"|\bexcluding\s+(?:(?:the|these|certain|significant|notable|special)\s+){0,3}(?:items|effects|impact)\b",
    re.I,
)
AMOUNT = re.compile(r"\$\s*\d|\b(?:million|billion|per share)\b", re.I)
# Decimal points do not terminate a sentence. SEC flattened text may contain
# page numbers after a full stop; retain the complete sentence, not the page.
END = re.compile(r"[.!?](?=\s|$)|[\n•]")
ABBREVIATION = re.compile(r"(?:\b(?:[A-Za-z]\.){2,}|\b(?:Inc|Corp|Co|Ltd)\.)$", re.I)
TABLE_HEADER = re.compile(r"\bdollars\s+in\s+(?:millions|billions|thousands)\b|\bActual\s+Actual\s+Outlook\b", re.I)


def sentences(text, start, end):
    # Expand only to determine whether a full source sentence was actually read.
    previous = 0
    for match in END.finditer(text):
        if match.group() == '.' and ABBREVIATION.search(text[max(0, match.end()-20):match.end()]):
            continue
        stop = match.end() if match.group() in '.!?' else match.start()
        a, b = previous, stop
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b-1].isspace():
            b -= 1
        if a >= end:
            break
        if start <= a < b <= end:
            yield a, b, text[a:b]
        previous = match.end()
    if previous < len(text):
        a = previous
        while a < len(text) and text[a].isspace():
            a += 1
        if start <= a < len(text) <= end:
            yield a, len(text), text[a:]


def supplement(inputs, facts, read_chunks, max_per_entity=2):
    if type(max_per_entity) is not int or max_per_entity < 1:
        raise ValueError('invalid_qualifier_limit')
    if inputs.task.get('family') not in ('eps_growth_regression', 'eps_yoy_direction', 'eps_beat_consensus'):
        return [], [{'reason': 'unsupported_family', 'family': inputs.task.get('family')}]
    added, audit = [], []
    for eid in inputs.entities:
        candidates = {}
        for chunk in read_chunks.get(eid, []):
            did = chunk['doc_id']; relation = inputs.relations.get(did, {})
            if not inputs.allowed(eid, did) or relation.get('shared') or relation.get('entity_ids') != [eid]:
                audit.append({'entity_id': eid, 'doc_id': did, 'reason': 'not_exclusive_source'})
                continue
            doc = inputs.docs[did]
            if doc['doc_date'] > inputs.task['cutoff_date']:
                raise ValueError('qualifier_source_after_cutoff')
            text = doc['text']
            for span in chunk['source_spans']:
                start, end = span['start'], span['end']
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                    raise ValueError('qualifier_invalid_source_span')
                for a, b, quote in sentences(text, start, end):
                    if not TRIGGER.search(quote) or not AMOUNT.search(quote):
                        continue
                    key = (did, a, b)
                    if key not in candidates:
                        candidates[key] = (chunk, quote, doc['doc_date'])
        ordered = sorted(candidates, key=lambda k: (k[0], k[1], k[2]))
        ordered.sort(key=lambda k: candidates[k][2], reverse=True)
        accepted = 0
        for did, a, b in ordered:
            chunk, quote, _ = candidates[(did, a, b)]
            item = {'entity_id': eid, 'doc_id': did, 'start': a, 'end': b, 'quote': quote}
            if TABLE_HEADER.search(quote):
                audit.append({**item, 'reason': 'flattened_table_not_prose'}); continue
            if any(c['doc_id'] == did and c['span_start'] <= a and c['span_end'] >= b
                   for f in facts if f['entity_id'] == eid for c in f['citations']):
                audit.append({**item, 'reason': 'already_covered'}); continue
            if len(quote) > 1600 or count_tokens(quote) > 350:
                audit.append({**item, 'reason': 'complete_sentence_too_long'}); continue
            raw = {'entity_id': eid, 'source_entity_id': eid, 'statement': quote,
                'evidence_role': 'financial_driver', 'metric': 'reported_results_qualification',
                'basis': 'verbatim source passage; recurrence and forecast effect not inferred',
                'citations': [{'doc_id': did, 'chunk_id': chunk['chunk_id'], 'quote': quote,
                    '_catalog_span': {'start': a, 'end': b}}]}
            result = validate_facts({'facts': [raw]}, inputs, eid, [chunk], trust_catalog=True)
            if not result['facts']:
                audit.append({**item, 'reason': 'validation_rejected', 'details': result['rejected']}); continue
            if accepted >= max_per_entity:
                audit.append({**item, 'reason': 'explicit_entity_limit'}); continue
            fact = result['facts'][0]
            added.append(fact); accepted += 1
            audit.append({**item, 'reason': 'added', 'fact_id': fact['fact_id']})
    return added, audit
