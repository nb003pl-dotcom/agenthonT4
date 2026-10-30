"""Domain-free subquery relevance and deterministic budgeted source selection."""
from collections import defaultdict, deque
import math
import json
import re
import calendar
from copy import deepcopy
import time

import httpx

from .common import write
from .model import input_tokens
from t4_splitter import count_tokens

STATUSES = ('direct', 'partial', 'uncertain', 'irrelevant')
QUALITY = {'direct': 1.0, 'partial': .5, 'uncertain': 0.0, 'irrelevant': 0.0}
DOMAIN_FIELDS = {'evidence_duties', 'protected_roles', 'anchor'}


def evidence_policy(config):
    policy = config.get('evidence_policy') or ('subquery' if config.get('workflow_profile') == 'typed-evidence-v2' else 'legacy')
    if policy not in ('subquery', 'legacy'): raise ValueError('unknown evidence policy')
    if policy == 'subquery' and (config.get('replay_from') or config.get('small_corpus_mode', 'off') != 'off' or config.get('policy') == 'legacy'):
        raise ValueError('subquery evidence requires fresh retrieval, coverage policy and small_corpus_mode=off')
    return policy


def retrieval_top(config):
    value=config.get('subquery_retrieval_top',5)
    if type(value) is not int or not 1<=value<=20:
        raise ValueError('subquery retrieval top must be an integer from 1 to 20')
    return value


def entity_needs(queries, entity):
    """Keep planner constraints verbatim; query types never impose selection quotas."""
    fields = ('query_id', 'information_need', 'purpose', 'targets', 'metric', 'period',
              'basis', 'unit', 'scope', 'constraints', 'priority', 'dense_query', 'lexical_query')
    return [{k: q[k] for k in fields if k in q} for q in queries if entity in q['entity_ids']]


def selection_budget(config):
    """The extraction-window budget never implicitly caps all selected sources."""
    budget = config.get('selection_source_token_budget')
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ValueError('invalid total selection budget')
    return budget


def neutral_source_batch(batch):
    """For controlled experiments, remove only selection-specific annotations."""
    fields = DOMAIN_FIELDS | {'need_matches', 'relevance_status', 'source_lookup'}
    return [{k: v for k, v in c.items() if k not in fields} for c in batch]


def add_need_context(payload, needs, batch):
    """Research needs are hints; preserve the task-driven extraction limits."""
    return {**payload, 'information_needs': needs,
        'source_need_matches': {c['chunk_id']: c.get('need_matches', {}) for c in batch},
        'instruction': payload.get('instruction', '') +
        ' Research needs are search directions, not an exhaustive fact checklist. '
        'Retain material task-useful context beyond those needs, including conflicting information. '
        'Preserve the task metric, scope, period and basis explicitly; a related measure is not interchangeable. '
        'Need matches are unverified relevance hints. Report unmet needs without claiming corpus absence.'}


def _need_groups(needs):
    groups = {}
    for q in needs:
        descriptor = {k: q[k] for k in ('information_need', 'purpose', 'metric', 'period', 'basis', 'unit', 'scope', 'constraints') if k in q}
        if q.get('targets'):
            canonical = {json.dumps(t, sort_keys=True, ensure_ascii=False) for t in q['targets']}
            descriptor['targets'] = sorted(canonical)
        else:
            descriptor.update({k: q[k] for k in ('dense_query', 'lexical_query') if k in q})
        key = json.dumps(descriptor, sort_keys=True, ensure_ascii=False)
        p = q.get('priority', 1)
        weight = 1/max(1, p) if type(p) in (int, float) and math.isfinite(p) and p > 0 else 1
        entry = groups.setdefault(key, {'ids': [], 'weight': 0})
        entry['ids'].append(q['query_id']); entry['weight'] = max(entry['weight'], weight)
    return list(groups.values())


def _payload(windows, needs, task):
    entities = dict.fromkeys(w['entity_id'] for w in windows)
    return {'task': task, 'needs': {e: needs[e] for e in entities}, 'windows': windows}


def judgment_packets(candidates, queries, system, input_limit=23400, *, task=None, max_pairs=160, max_windows=None):
    """Pack exact source text, splitting only for context; no preview or truncation."""
    if max_windows is not None and (type(max_windows) is not int or max_windows < 1):
        raise ValueError('max_windows must be a positive integer')
    needs = {e: entity_needs(queries, e) for e in candidates}
    pending = []
    def fits(windows):
        pairs = sum(len(needs[w['entity_id']]) for w in windows)
        return pairs <= max_pairs and input_tokens(system, _payload(windows, needs, task)) <= input_limit
    def windows_for(entity, rows):
        if not needs[entity]: raise ValueError('entity_without_subquery_needs: '+entity)
        for idx, row in enumerate(rows):
            text = row.get('text') or '\n'.join(s['text'] for s in row['source_spans'])
            start = 0; seq = 0
            while start < len(text):
                base = {'entity_id': entity, 'candidate_index': idx, 'window_index': seq,
                        'chunk_id': row['chunk_id'], 'doc_id': row['doc_id'],
                        'doc_date': row['doc_date'], 'title': row.get('title', ''),
                        'node_type': row.get('node_type'), 'section': row.get('section', ''),
                        'table_headers': [t.get('header', []) for t in row.get('table_refs', [])],
                        'retrieved_by': row.get('query_ids', []),
                        'allowed_query_ids':[q['query_id'] for q in needs[entity]]}
                def window(end): return {**base, 'start': start, 'end': end, 'text': text[start:end]}
                end = len(text)
                if not fits([window(end)]):
                    lo, hi = start+1, end; end = None
                    while lo <= hi:
                        mid = (lo+hi)//2
                        if fits([window(mid)]): end = mid; lo = mid+1
                        else: hi = mid-1
                    if end is None: raise ValueError('need_metadata_exceeds_judgment_budget')
                w = window(end)
                yield w
                if end == len(text): break
                # Context overlap is source text, not a new retrieval node.
                start = max(start+1, end-min(200, (end-start)//4)); seq += 1
    def packets_for(entity, rows):
        pending = []
        for w in windows_for(entity, rows):
            if pending and ((max_windows is not None and len(pending)>=max_windows) or not fits(pending+[w])):
                yield _payload(pending, needs, task); pending = []
            pending.append(w)
        if pending: yield _payload(pending, needs, task)
    # Keep entity scopes separate; rotate whole packets for budget fairness.
    streams = deque(iter(packets_for(e, rows)) for e, rows in candidates.items())
    while streams:
        stream = streams.popleft()
        try: packet = next(stream)
        except StopIteration: continue
        streams.append(stream)
        yield packet


def validate_judgments(data, payload):
    rows = data.get('judgments') if isinstance(data, dict) else None
    if not isinstance(rows, list): raise ValueError('judgments must be a list')
    expected = {(w['entity_id'], w['candidate_index'], w['window_index']) for w in payload['windows']}
    seen = set()
    for row in rows:
        if not isinstance(row, dict): raise ValueError('invalid judgment row')
        entity = row.get('entity_id'); idx = row.get('candidate_index'); win = row.get('window_index')
        if not isinstance(entity, str) or type(idx) is not int or type(win) is not int:
            raise ValueError('invalid judgment identity')
        key = (entity, idx, win)
        if key not in expected or key in seen: raise ValueError('unexpected or duplicate judgment')
        seen.add(key)
        qids = [q['query_id'] for q in payload['needs'][entity]]
        assigned = []
        for status in STATUSES:
            ids = row.get(status)
            if not isinstance(ids, list) or any(not isinstance(q, str) for q in ids):
                raise ValueError('invalid judgment partition')
            assigned.extend(ids)
        if len(set(assigned)) != len(assigned) or sorted(assigned) != sorted(qids):
            raise ValueError('judgment must partition all entity needs')
    if seen != expected: raise ValueError('judgments must cover all source windows')
    return rows


def _date_mentions(text):
    dates=set()
    for match in re.finditer(r'(?<!\d)(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?(?!\d)',text):
        y,m,d=match.groups();dates.add((int(y),int(m),int(d) if d else None))
    for month in range(1,13):
        name=r'(?:'+calendar.month_name[month]+r'|'+calendar.month_abbr[month]+r')'
        for match in re.finditer(r'\b'+name+r'\s+(\d{1,2}),?\s+(\d{4})\b',text,re.I):
            d,y=match.groups();dates.add((int(y),month,int(d)))
        for match in re.finditer(r'\b(\d{1,2})\s+'+name+r',?\s+(\d{4})\b',text,re.I):
            d,y=match.groups();dates.add((int(y),month,int(d)))
        for match in re.finditer(r'\b'+name+r'\s+(\d{4})\b',text,re.I):
            dates.add((int(match[1]),month,None))
    return dates


def _quarters(text):
    result=set()
    for match in re.finditer(r'\b(\d{4})\s*[-/]?\s*[Qq]([1-4])\b',text):result.add((int(match[1]),int(match[2])))
    for match in re.finditer(r'\b[Qq]([1-4])\s*[-/,]?\s*(\d{4})\b',text):result.add((int(match[2]),int(match[1])))
    for i,word in enumerate(('first','second','third','fourth'),1):
        for match in re.finditer(r'\b'+word+r'\s+quarter(?:\s+of)?\s+(\d{4})\b',text,re.I):result.add((int(match[1]),i))
    return result


def _witness_period(period, text):
    """Check coherent literal dates/quarters; never mix fields from different dates."""
    period=str(period);wanted=_date_mentions(period);found=_date_mentions(text)
    if wanted:
        return all(any(y==sy and m==sm and (d is None or d==sd) for sy,sm,sd in found) for y,m,d in wanted)
    wanted_quarters=_quarters(period)
    if wanted_quarters:
        found_quarters=_quarters(text)|{(y,(m-1)//3+1) for y,m,d in found if 1<=m<=12}
        return wanted_quarters<=found_quarters
    years=re.findall(r'(?<!\d)\d{4}(?!\d)',period)
    return all(re.search(r'(?<!\d)'+year+r'(?!\d)',text) for year in years)


def grounded_judgments(data, payload):
    """A label alone cannot certify coverage or justify irreversible exclusion."""
    rows=deepcopy(validate_judgments(data,payload))
    windows={(w['entity_id'],w['candidate_index'],w['window_index']):w for w in payload['windows']}
    for row in rows:
        window=windows[(row['entity_id'],row['candidate_index'],row['window_index'])]
        needs={q['query_id']:q for q in payload['needs'][row['entity_id']]}
        evidence=row.get('evidence',[]);support=row.get('support',{});issues=[]
        if not isinstance(evidence,list):evidence=[]
        if not isinstance(support,dict):support={}
        original=re.sub(r'\s+',' ',window['text']).strip()
        for qid in list(row['direct']):
            indices=support.get(qid,[])
            quotes=[evidence[i] for i in indices] if isinstance(indices,list) and indices and len(indices)<=3 and all(type(i) is int and 0<=i<len(evidence) for i in indices) else []
            valid=bool(quotes) and all(isinstance(q,str) and 0<len(q)<=350 and bool(re.sub(r'\s+',' ',q).strip()) and re.sub(r'\s+',' ',q).strip() in original for q in quotes)
            joined=' '.join(quotes) if valid else ''
            targets=needs[qid].get('targets') or [{'metric':needs[qid].get('metric'),'time':needs[qid].get('period')}]
            for target in targets:
                if not valid:break
                period=target.get('time') or target.get('period')
                if period and not _witness_period(period,joined):valid=False
                terms=set(re.findall(r'[a-z][a-z0-9_]*',str(target.get('metric') or '').lower()))-{'the','and','for','per','of','in','a','an'}
                if terms and not terms<=set(re.findall(r'[a-z][a-z0-9_]*',joined.lower())):valid=False
            if not valid:
                row['direct'].remove(qid);row['partial'].append(qid)
                issues.append({'query_id':qid,'reason':'direct_label_without_matching_source_witness'})
        # Semantic negatives remain soft until independently substantiated.
        # A single fallible classifier must not erase potentially useful sources.
        if row['irrelevant']:
            issues.extend({'query_id':q,'reason':'unverified_negative_retained_for_backfill'} for q in row['irrelevant'])
            row['uncertain'].extend(row['irrelevant']);row['irrelevant']=[]
        row['grounding_issues']=issues
    return rows


def aggregate_judgments(candidates, queries, judgments):
    grouped = defaultdict(list)
    for row in judgments: grouped[(row['entity_id'], row['candidate_index'])].append(row)
    kept = {}; audit = {}
    for entity, rows in candidates.items():
        qids = [q['query_id'] for q in entity_needs(queries, entity)]
        kept[entity] = []; audit[entity] = []
        for idx, row in enumerate(rows):
            windows = grouped[(entity, idx)]; matches = {}
            for qid in qids:
                labels = [next((s for s in STATUSES if qid in w.get(s, [])), 'uncertain') for w in windows]
                matches[qid] = next((s for s in STATUSES if s in labels), 'uncertain')
            retain = not matches or any(s != 'irrelevant' for s in matches.values())
            audit[entity].append({'chunk_id': row['chunk_id'], 'need_matches': matches,
                                  'retained': retain, 'windows': len(windows)})
            if retain:
                clean = {k: v for k, v in row.items() if k not in DOMAIN_FIELDS}
                kept[entity].append({**clean, 'need_matches': matches})
    return kept, audit


def _duplicate(row, chosen):
    """Deduplicate represented source coverage, never documents or metrics alone."""
    spans = row.get('source_spans', [])
    total = sum(s['end']-s['start'] for s in spans)
    if not total: return False
    coverage = 0
    for span in spans:
        intervals = sorted((max(span['start'], t['start']), min(span['end'], t['end']))
            for c in chosen if c['doc_id'] == row['doc_id'] for t in c.get('source_spans', [])
            if max(span['start'], t['start']) < min(span['end'], t['end']))
        end = span['start']
        for a, b in intervals:
            coverage += max(0, b-max(a, end)); end = max(end, b)
    return coverage >= total


def select_by_coverage(rows, needs, cap=5, token_budget=None):
    if cap < 1 or (token_budget is not None and token_budget < 1): raise ValueError('invalid evidence budget')
    qids = [q['query_id'] for q in needs]
    groups = _need_groups(needs)
    chosen = []; used = 0; best = {q: 0 for q in qids}; uncertain_seen = set()
    remaining = list(rows); reasons = []; costs = {r['chunk_id']: count_tokens(r['text']) for r in rows}
    while remaining and len(chosen) < cap:
        feasible = [r for r in remaining if (token_budget is None or used+costs[r['chunk_id']] <= token_budget) and not _duplicate(r, chosen)]
        if not feasible: break
        def rank(r):
            matches = r['need_matches']
            gain = 0; uncertain_gain = 0; quality = 0
            for group in groups:
                members = group['ids']; weight = group['weight']
                available = max(QUALITY[matches.get(q, 'uncertain')] for q in members)
                covered = max(best[q] for q in members)
                gain += weight*max(0, available-covered)
                quality += weight*available
                if covered == 0 and not any(q in uncertain_seen for q in members) and any(matches.get(q, 'uncertain') == 'uncertain' for q in members):
                    uncertain_gain += weight
            return (gain, r.get('score', 0), -costs[r['chunk_id']], r['chunk_id'])
        row = max(feasible, key=rank)
        if not any(row['need_matches'].get(q, 'uncertain') != 'irrelevant' for q in qids): break
        chosen.append(row); remaining.remove(row); used += costs[row['chunk_id']]
        reasons.append({'chunk_id': row['chunk_id'], 'reason': 'subquery_coverage', 'need_matches': row['need_matches']})
        for q in qids:
            status = row['need_matches'].get(q, 'uncertain')
            best[q] = max(best[q], QUALITY[status])
            if status == 'uncertain': uncertain_seen.add(q)
    # An indivisible node can exceed the selection budget. Extraction already
    # windows full sources; expose the exception instead of returning no sources.
    oversized = False
    if not chosen and rows and token_budget is not None:
        row = min(rows, key=lambda r: costs[r['chunk_id']])
        if costs[row['chunk_id']] > token_budget:
            chosen = [row]; used = costs[row['chunk_id']]; oversized = True
            reasons = [{'chunk_id': row['chunk_id'], 'reason': 'oversized_single_source', 'need_matches': row['need_matches']}]
            for q in qids:
                status = row['need_matches'].get(q, 'uncertain'); best[q] = QUALITY[status]
                if status == 'uncertain': uncertain_seen.add(q)
    selected_ids = {r['chunk_id'] for r in chosen}
    return chosen, {'policy': 'subquery_coverage', 'selection_version': 3, 'distinct_need_groups': len(groups), 'selected': reasons,
        'omitted': [r['chunk_id'] for r in rows if r['chunk_id'] not in selected_ids],
        'covered_need_ids': [q for q in qids if best[q] == 1],
        'partial_need_ids': [q for q in qids if best[q] == .5],
        'uncovered_need_ids': [q for q in qids if best[q] == 0],
        'uncertain_need_ids': [q for q in qids if best[q] == 0 and q in uncertain_seen],
        'budget_omitted': [r['chunk_id'] for r in rows if r['chunk_id'] not in selected_ids and token_budget is not None and used+costs[r['chunk_id']] > token_budget],
        'source_tokens': used, 'source_token_budget': token_budget, 'oversized_single_source': oversized,
        'notice': 'Need matching is relevance, not verified truth or corpus completeness.'}


def judge_and_select(workflow, candidates, queries, ids):
    """Replace relevance+preview selection, reserving mandatory downstream calls."""
    from .workflow import prediction_attempt_reserve
    system = workflow.prompt('subquery-relevance')
    relevance_wire_variant=workflow.config.get('relevance_wire_variant','baseline')
    if relevance_wire_variant not in ('baseline','prompt'):
        raise ValueError('runtime relevance_wire_variant must be baseline or prompt')
    if relevance_wire_variant=='prompt':
        from .relevance_wire import prompt_suffix
        system+='\n\n'+prompt_suffix('prompt')
    limit = workflow.config.get('context_window', 32000)-4000-4600
    reserve = len(ids)+prediction_attempt_reserve(len(ids),workflow.config.get('prediction_batch_size',4))
    from .fact_schedule import time_reserve
    estimate=time_reserve(workflow.model.records if hasattr(workflow.model,'records') else [],len(ids),batch_size=workflow.config.get('prediction_batch_size',4),reserve_prediction=not workflow.config.get('prediction_time_reserved_externally',False))
    fact_waves=math.ceil(len(ids)/workflow.config.get('facts_workers',1))
    downstream_seconds=estimate['prediction_reserve_seconds']+fact_waves*estimate['estimated_call_seconds']
    judgment_deadline=workflow.deadline-downstream_seconds
    relevance_mode=workflow.config.get('relevance_mode','all')
    if relevance_mode not in ('all','over_cap'):raise ValueError('invalid relevance_mode')
    cap=workflow.config.get('evidence_cap',5)
    token_budget=selection_budget(workflow.config)
    judged_candidates={};judgments = []; batches = []; errors = []
    for entity,rows in candidates.items():
        if relevance_mode=='over_cap' and token_budget is None and len(rows)<=cap:
            qids=[q['query_id'] for q in entity_needs(queries,entity)]
            judgments.extend({'entity_id':entity,'candidate_index':i,'window_index':-1,
                              'direct':[],'partial':[],'irrelevant':[],'uncertain':qids}
                             for i in range(len(rows)))
            batches.append({'entity_id':entity,'windows':len(rows),'status':'skipped_candidates_fit_cap'})
        else:judged_candidates[entity]=rows
    task = {'prompt': workflow.inputs.task['prompt'], 'target': workflow.inputs.task['target'],
            'cutoff_date': workflow.inputs.task['cutoff_date']}
    try:
        profile_windows=(getattr(workflow,'profile',None) or {}).get('relevance_max_windows_per_packet')
        max_windows=workflow.config.get('relevance_max_windows_per_packet',profile_windows)
        for i, payload in enumerate(judgment_packets(judged_candidates, queries, system, limit, task=task,
                max_windows=max_windows)):
            # Prepopulate every window conservatively; failed/unjudged windows
            # must prevent an irrelevant judgment in another window deleting it.
            defaults = [{'entity_id': w['entity_id'], 'candidate_index': w['candidate_index'],
                         'window_index': w['window_index'], 'direct': [], 'partial': [], 'irrelevant': [],
                         'uncertain': [q['query_id'] for q in payload['needs'][w['entity_id']]]} for w in payload['windows']]
            record = {'batch': i, 'windows': len(defaults), 'input_tokens': input_tokens(system, payload),
                      'downstream_time_reserve': downstream_seconds,'wire_variant':relevance_wire_variant}
            if workflow.model.requests >= workflow.model.max_requests-reserve or judgment_deadline-time.time() < 10:
                result = defaults; record['status'] = 'uncertain_budget_reserve'
            else:
                try:
                    result = workflow.model.call('need_relevance', system, payload,
                        lambda d: grounded_judgments(d, payload), reserve=reserve, deadline=judgment_deadline)
                    record['status'] = 'judged'
                except (ValueError, RuntimeError, TimeoutError, httpx.TransportError) as exc:
                    result = defaults; record.update(status='uncertain_failure', error_type=type(exc).__name__)
            judgments.extend(result); batches.append(record)
    except ValueError as exc:
        errors.append({'status': 'uncertain_unpacked_sources', 'error': str(exc)})
        # Packing failure can leave a partially judged source. Mark each row
        # uncertain as well so partial input never becomes an absence judgment.
        for e, rows in judged_candidates.items():
            judgments.extend({'entity_id': e, 'candidate_index': j, 'window_index': -1,
                              'uncertain': [q['query_id'] for q in entity_needs(queries, e)]} for j in range(len(rows)))
    kept, relevance = aggregate_judgments(candidates, queries, judgments)
    choices = {}; audits = {}
    for e, rows in kept.items():
        selected, audit = select_by_coverage(rows, entity_needs(queries, e),cap=cap,token_budget=token_budget)
        audits[e] = audit
        choices[e] = {'indices': [rows.index(r) for r in selected], 'rationale': 'Deterministic subquery coverage'}
    write(workflow.output/'need-relevance.json', {'batches': batches, 'errors': errors, 'entities': relevance,
        'judgments': judgments, 'notice': 'Unjudged sources are uncertain, never irrelevant.'})
    write(workflow.output/'need-selection.json', audits)
    workflow.need_selection_audit = audits
    workflow.subquery_needs = {e: entity_needs(queries, e) for e in ids}
    return kept, choices
