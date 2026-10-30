"""Short stage-specific output checks appended after the main semantic prompt."""


_SUFFIXES={
    'facts': '''OUTPUT PREFLIGHT (perform silently before answering): count facts and
verify the count is no greater than the input limits.max_facts. Verify every fact
uses the requested entity and only supplied citation labels. Return one complete JSON object
with exactly facts, conflicts, and unresolved_questions; no prose or
fences.''',
    'supplement_plan': '''OUTPUT PREFLIGHT (perform silently before answering): for
each query, locate its entity_id in the supplied entities, then check that every
one of its 1–3 fact_ids occurs in supplied facts with that same entity_id. Remove
any query that cannot pass this ownership check. Recount the task-level query and
entity limits, then return one complete JSON object with assessment and queries.''',
    'predict': '''OUTPUT PREFLIGHT (perform silently before answering): verify every
allowed_entity_ids member appears exactly once and no other entity appears. For
each row, check every output_requirements.required_per_row field is present, every
fact_id belongs to allowed_fact_ids_by_entity for that row, and
interval.lo <= point_forecast <= interval.hi. Return the complete batch as one
valid JSON object with only the required envelope.''',
}

_JOINT_FACTS='''OUTPUT PREFLIGHT (perform silently before answering): count facts
and verify the count is no greater than input limits.max_facts. Verify every fact
uses the requested entity and only supplied citation labels. Verify
source_assessments is an array covering every zero-based source_index in
joint_assessment_chunk_ids exactly once. Each matches string must have exactly
len(information_needs) characters, each D, P, U or I in information_needs array
order. Return one complete JSON object with exactly source_assessments,
facts, conflicts, and unresolved_questions; no prose or fences.'''


def suffix(stage):
    try:return _SUFFIXES[stage]
    except KeyError:raise ValueError('unsupported preflight stage: '+str(stage)) from None


_EPS_LEVEL_PREDICT='''OUTPUT PREFLIGHT (perform silently before answering): every
allowed_entity_ids member appears exactly once. Every fact_id belongs to that
entity's allowed_fact_ids_by_entity. Return entity_id, eps_forecast, eps_interval,
fact_ids and reasoning for every row. All numeric forecasts and bounds are EPS
levels in USD/share: eps_interval.lo <= eps_forecast <= eps_interval.hi.
Return a JSON object with only predictions as an array; no percentages or
redundant official-target numeric output.'''


def enrich_payload(config,stage,payload):
    """Add redundant machine-readable allow-lists without mutating the caller."""
    if not config.get('structured_output_preflight',False) or stage!='supplement_plan':return payload
    grouped={}
    for fact in payload.get('facts',[]):
        entity=fact.get('entity_id');fid=fact.get('fact_id')
        if isinstance(entity,str) and isinstance(fid,str):grouped.setdefault(entity,[]).append(fid)
    return {**payload,'allowed_fact_ids_by_entity':grouped}


def apply(config,stage,system,payload=None):
    """Append a stage check only for the explicitly enabled experiment."""
    if not config.get('structured_output_preflight',False):return system
    selected=_JOINT_FACTS if stage=='facts' and config.get('joint_relevance_facts') else suffix(stage)
    if stage=='predict' and isinstance(payload,dict) and payload.get('forecast_wire_format')=='eps-level-v1':
        selected=_EPS_LEVEL_PREDICT
    result=system+'\n\n'+selected
    if stage=='predict' and isinstance(payload,dict):
        required=payload.get('output_requirements',{}).get('required_per_row',[])
        if isinstance(required,list) and required:
            result+='\nThe earlier fixed JSON example is not exhaustive for this batch. Each row must contain these exact required field names: '+', '.join(map(str,required))+'.'
    return result
