"""Coherent experimental top-level qualification contract; no semantic verdict."""
import copy

from workflow.qualifier_decision import validate_checks


INSTRUCTION = """
Before the predictions array, write qualification_checks: an array of objects
with exactly entity_id, fact_id, disposition, reason. Cover each supplied
qualification_fact_ids_by_entity ID exactly once for its own entity, and no others.
disposition is adjust_baseline, already_accounted, not_applicable, or uncertain.
reason is one concise sentence of 1-160 characters. Do not add checks to prediction rows.
Match the reported period and accounting basis to the historical EPS baseline.
GAAP EPS can INCLUDE unusual or nonrecurring gains and charges. Being nonrecurring
does NOT mean an item was excluded from GAAP EPS. Use already_accounted only when
supplied evidence explicitly establishes that the baseline already removes this
item; do not assume an adjusted baseline. Inclusion in a reported result alone
does not establish how much recurs in the future. If adjustment inputs or the
period/basis link are missing, use uncertain. Never subtract pre-tax dollars from
after-tax per-share EPS or invent tax/share-count inputs. Use supported drivers
and justified uncertainty when forecasting. No numerical adjustment or direction
is prescribed. These checks are unverified model judgments, not new source facts.
"""


def build_contract(system, payload, expected):
    replacements = {
        'Return only JSON: {"predictions": [...]} with every requested entity exactly once.':
            'Return only JSON: {"qualification_checks": [...], "predictions": [...]} with every requested entity exactly once in predictions.',
        '- Return exactly one JSON object with only the top-level key predictions.':
            '- Return exactly one JSON object with exactly two top-level keys: qualification_checks, then predictions.',
    }
    for old, new in replacements.items():
        if system.count(old) != 1:
            raise ValueError('qualification_system_contract_changed')
        system = system.replace(old, new)
    result = copy.deepcopy(payload)
    result['qualification_fact_ids_by_entity'] = copy.deepcopy(expected)
    req = result['output_requirements']
    req['required_top_level'] = ['qualification_checks', 'predictions']
    req['qualification_checks_schema'] = {
        'type': 'array', 'required_fields': ['entity_id', 'fact_id', 'disposition', 'reason'],
        'dispositions': ['adjust_baseline', 'already_accounted', 'not_applicable', 'uncertain'],
        'reason_max_characters': 160,
        'coverage': 'Exactly the qualification_fact_ids_by_entity IDs, each once for its own entity; [] when none.',
    }
    return system + INSTRUCTION, result


def validate_contract(data, expected):
    if not isinstance(data, dict) or set(data) != {'qualification_checks', 'predictions'}:
        raise ValueError('qualification_exact_top_level_required')
    checks = data['qualification_checks']
    if not isinstance(checks, list):
        raise ValueError('qualification_checks_must_be_array')
    rows = {eid: {'entity_id': eid, 'qualification_checks': []} for eid in expected}
    for check in checks:
        if not isinstance(check, dict) or set(check) != {'entity_id', 'fact_id', 'disposition', 'reason'}:
            raise ValueError('qualification_check_fields')
        eid = check['entity_id']
        if not isinstance(eid, str) or eid not in rows:
            raise ValueError('qualification_wrong_entity')
        rows[eid]['qualification_checks'].append({k: v for k, v in check.items() if k != 'entity_id'})
    _, audit = validate_checks({'predictions': list(rows.values())}, expected)
    before = list(data).index('qualification_checks') < list(data).index('predictions')
    for row in audit:
        row['checks_precede_estimate'] = before
    # The ordinary prediction validator separately verifies its roster and values.
    return {'predictions': copy.deepcopy(data['predictions'])}, audit
