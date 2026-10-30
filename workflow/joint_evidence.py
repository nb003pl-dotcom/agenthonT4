"""Joint relevance assessment and grounded fact extraction contracts."""
from copy import deepcopy

from .subquery_evidence import STATUSES, entity_needs, select_by_coverage


PROMPT_SUFFIX = '''
JOINT RELEVANCE + FACT CONTRACT:
In addition to facts, return source_assessments with exactly one row for every
UNIQUE chunk_id listed in joint_assessment_chunk_ids (source windows may repeat a
chunk_id). Use the compact matrix wire format:
{"source_index":0,"matches":"DPU..."}. source_index is the ZERO-BASED index
in joint_assessment_chunk_ids. matches has exactly one character per entry in
information_needs IN THAT ARRAY'S ORDER: D=direct, P=partial, U=uncertain,
I=irrelevant. Every source index must occur exactly once, even if every match is
uncertain. Do not repeat chunk IDs or query IDs in the assessment output.
Return source_assessments as an ARRAY, not an object keyed by IDs.
This assessment is an audit, not
permission to omit material facts. Extract every supported, decision-useful
fact under the normal citation rules, including counterevidence and context.
Use uncertain rather than irrelevant when the relationship is debatable.

FACT PRIORITY AND DOWNSTREAM BUDGET: {budget_notice}
Order facts by distinct decision value across ALL supplied
sources: latest target baseline, current-period driver or explicit guidance,
material counterevidence, then historical trajectory and remaining context.
Do not enumerate successive rows of one historical series as separate facts.
Summarize its relevant trajectory in one concise, accurately cited fact, selecting
the necessary row/header quote IDs. Before adding a second fact from the same
series, retain any material current-period driver from another supplied source.
Keep metric, period, unit and basis distinctions explicit. A related driver is
not the target itself. These priorities do not authorize unsupported conclusions.
'''


def prompt_suffix(profile=None, target_type=None):
    budget=(profile or {}).get('dynamic_evidence_budget') or {}
    cap=budget.get('max_facts')
    if (target_type=='regression' and budget.get('policy')=='dynamic-regression-cap-v1'
            and type(cap) is int and cap>0):
        notice=f'Prediction may retain only the first {cap} facts for this entity.'
    else:
        notice='Prediction has a finite context budget; put decisive, nonredundant evidence first.'
    # Replace only the placeholder: the contract contains literal JSON braces.
    return PROMPT_SUFFIX.replace('{budget_notice}',notice)


def order_facts_for_budget(facts):
    """Present complementary source evidence before redundant same-source rows.

    Source diversity is an ordering heuristic, not a claim of independent truth.
    Every fact and citation is preserved; ties retain extraction order.
    """
    entities=list(dict.fromkeys(f['entity_id'] for f in facts));result=[]
    for entity in entities:
        pending=[f for f in facts if f['entity_id']==entity]
        docs=set();roles=set()
        while pending:
            def rank(f):
                sources={c['doc_id'] for c in f.get('citations',[])}
                role=f.get('evidence_role')
                return (bool(sources-docs),bool(role and role not in roles))
            # The first extracted fact stays the primary anchor.
            chosen=pending[0] if not docs else max(pending,key=rank)
            pending.remove(chosen);result.append(chosen)
            docs.update(c['doc_id'] for c in chosen.get('citations',[]))
            if chosen.get('evidence_role'):roles.add(chosen['evidence_role'])
    return result


def prepare_selection(candidates, queries, *, cap=5, token_budget=None):
    """Select query-diverse sources using retrieval provenance, without an LLM."""
    kept={};choices={};audits={}
    for entity,rows in candidates.items():
        needs=entity_needs(queries,entity);qids=[q['query_id'] for q in needs]
        annotated=[]
        for row in rows:
            hits=set(row.get('query_ids',())) & set(qids)
            annotated.append({**row,'need_matches':{q:('direct' if q in hits else 'uncertain') for q in qids}})
        selected,audit=select_by_coverage(annotated,needs,cap=cap,token_budget=token_budget)
        kept[entity]=annotated
        choices[entity]={'indices':[annotated.index(r) for r in selected],
                         'rationale':'Deterministic lexical query coverage; joint LLM assessment follows'}
        audit={**audit,'policy':'lexical-query-coverage-before-joint-extraction'}
        audits[entity]=audit
    return kept,choices,audits


def add_contract(payload, needs, batch):
    chunk_ids=list(dict.fromkeys(c['chunk_id'] for c in batch))
    result={**payload,'information_needs':deepcopy(needs),
            'joint_assessment_chunk_ids':chunk_ids,'joint_assessment_format':'source-need-matrix-v1'}
    result['joint_assessment_dimensions']={'sources':len(chunk_ids),'needs':len(needs),
        'source_index_base':0,'status_codes':['D','P','U','I']}
    result['instruction']=(result.get('instruction','')+
        ' Return exhaustive source_assessments and grounded facts in the same JSON object.')
    return result


def validate_assessments(data, payload):
    rows=data.get('source_assessments') if isinstance(data,dict) else None
    if not isinstance(rows,list):raise ValueError('source_assessments must be a list')
    expected=payload['joint_assessment_chunk_ids'];needs=[q['query_id'] for q in payload['information_needs']]
    if rows and any(isinstance(r,dict) and 'source_index' in r for r in rows):
        canonical=[]
        codes={'D':'direct','P':'partial','U':'uncertain','I':'irrelevant'}
        for raw in rows:
            if not isinstance(raw,dict):raise ValueError('invalid source assessment row')
            index=raw.get('source_index');matches=raw.get('matches')
            if type(index) is not int or not 0<=index<len(expected):
                raise ValueError('source_index must be an allowed zero-based integer')
            if (not isinstance(matches,str) or len(matches)!=len(needs)
                    or any(code not in codes for code in matches)):
                raise ValueError(f'matches must have exactly {len(needs)} D/P/U/I characters in information_needs order')
            row={'chunk_id':expected[index],**{status:[] for status in STATUSES}}
            for qid,code in zip(needs,matches):row[codes[code]].append(qid)
            canonical.append(row)
        rows=canonical
    if len(rows)!=len(expected) or {r.get('chunk_id') for r in rows if isinstance(r,dict)}!=set(expected):
        raise ValueError('source_assessments must cover every source exactly once')
    result=[]
    for raw in rows:
        assigned=[]
        for status in STATUSES:
            value=raw.get(status)
            if not isinstance(value,list) or any(not isinstance(q,str) for q in value):
                raise ValueError('invalid source assessment partition')
            assigned.extend(value)
        if len(assigned)!=len(set(assigned)) or sorted(assigned)!=sorted(needs):
            raise ValueError('source assessment must partition every need')
        row=deepcopy(raw)
        # A single extraction call cannot substantiate a corpus-level negative.
        row['uncertain'].extend(row['irrelevant']);row['irrelevant']=[]
        result.append(row)
    return result
