"""An experimental, concise source-qualification decision contract."""
import copy

INSTRUCTION = """
QUALIFICATION CHECK OVERRIDE: The supplied qualification_fact_ids_by_entity lists
verbatim qualifications of reported financial results. Before extending historical
EPS into the forecast, determine whether each qualification changes that baseline.
Add qualification_checks to each affected prediction row, before the forecast
fields: [{"fact_id": "supplied ID", "disposition": "adjust_baseline|already_accounted|not_applicable|uncertain", "reason": "one concise sentence, at most 160 characters"}].
Return each listed ID exactly once for its own entity; unlisted IDs are forbidden.
Other rows may omit qualification_checks or use []. All original prediction fields
remain required. This override only permits this one extra field; do not add essays.
Relate the source period and GAAP/adjusted basis to the historical EPS baseline.
Do not automatically subtract gains or charges: determine recurrence and whether
the baseline already excludes them. Never subtract a pre-tax dollar amount from
after-tax per-share EPS or invent missing tax/share-count inputs. If information
is missing, use uncertain and reflect justified uncertainty instead of fabricating
an exact adjustment. No disposition or forecast direction is mandated. Apply your
assessment consistently to the forecast and choose genuinely relevant fact_ids.
These short decisions are unverified model judgments, not new source facts.
"""

DISPOSITIONS = {'adjust_baseline', 'already_accounted', 'not_applicable', 'uncertain'}


def validate_checks(data, expected):
    if not isinstance(data, dict) or not isinstance(data.get('predictions'), list):
        raise ValueError('qualification_predictions_required')
    cleaned = copy.deepcopy(data); audit = []
    for row in cleaned['predictions']:
        eid = row.get('entity_id')
        if eid not in expected:
            raise ValueError('qualification_wrong_entity')
        keys = list(row)
        positions = [keys.index(k) for k in ('point_forecast', 'underlying_eps', 'interval') if k in row]
        before_estimate = keys.index('qualification_checks') < min(positions) if 'qualification_checks' in row and positions else None
        checks = row.pop('qualification_checks', [])
        if not isinstance(checks, list):
            raise ValueError('qualification_checks_must_be_array')
        seen = set()
        for check in checks:
            if not isinstance(check, dict) or set(check) != {'fact_id', 'disposition', 'reason'}:
                raise ValueError('qualification_check_fields')
            fid = check['fact_id']
            if not isinstance(fid, str) or fid not in expected[eid] or fid in seen:
                raise ValueError('qualification_unknown_duplicate_or_cross_entity_fact')
            if not isinstance(check['disposition'], str) or check['disposition'] not in DISPOSITIONS:
                raise ValueError('qualification_invalid_disposition')
            if not isinstance(check['reason'], str) or not 1 <= len(check['reason'].strip()) <= 160:
                raise ValueError('qualification_reason_1_to_160_characters')
            seen.add(fid)
        if seen != set(expected[eid]):
            raise ValueError('qualification_missing_required_fact_checks')
        audit.append({'entity_id': eid, 'checks': checks, 'checks_precede_estimate': before_estimate,
            'semantic_status': 'model_judgment_unverified'})
    return cleaned, audit
