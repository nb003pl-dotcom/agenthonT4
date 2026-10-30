"""Experimental extraction checks; no inference of table scope or column alignment.

Only used by the fixed-window experiment until quality/budget evidence warrants
runtime adoption. Numeric presence and self-consistency are not entailment.
"""
import math
import re

INSTRUCTION = '''
SOURCE BINDING: Preserve existing fact count and output limits. For each fact add reporting_scope (consolidated|segment|unknown) and comparison_kind (yoy|qoq|ytd_yoy|none|unknown). The document author does not define the scope of every table. Establish scope from the actual section/row headers; never combine a company-wide amount with a segment's growth rate or explanation. If scope is unresolved, use unknown and state the limited assertion. If segment, add segment_name copied from the source and name that segment in the statement. Select quote IDs covering the necessary subject/period headers as well as amounts, within the existing citation cap.
For a numerical comparison, add comparison_value only if it is for the SAME metric, reporting scope, unit/scale and compatible period as value. Add reported_change_percent as a signed relative percentage only if BOTH amounts and a relative change are asserted; otherwise null. Example: current 120, prior 100, up20% => value120, comparison_value100, reported_change_percent20. A decrease is negative. Percentage-point differences, percentage of a total, negative/zero comparison denominators and ambiguous columns are NOT this field: use null rather than guess. Python will check explicit comparable arithmetic. Do not mix rounded billions with unconverted millions.
If the sources give only a growth rate without both operands, preserve it as a supported reported fact with reported_change_percent=null; do not invent missing values. A growth claim can be derived from quoted comparable amounts, but identify it as calculated. Future periods in a valid announcement/guidance remain allowed, labelled as guidance. Do not turn forecast assumptions or task parameters into historical source facts. These fields record your interpretation; do not claim independent verification.
'''


def _numbers(text):
    # Preserve signs, decimal points and valid thousands grouping. Membership
    # alone cannot prove that a number is from the correct row or period.
    return {float(s.replace(',', '')) for s in re.findall(
        r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])', text)}


def validate_bindings(data, catalog):
    facts = data.get('facts')
    if not isinstance(facts, list):
        raise ValueError('facts must be a list')
    checks = []
    for i, fact in enumerate(facts):
        scope = fact.get('reporting_scope')
        if scope not in {'consolidated', 'segment', 'unknown'}:
            raise ValueError(f'fact {i}: reporting_scope is required')
        if fact.get('comparison_kind') not in {'yoy', 'qoq', 'ytd_yoy', 'none', 'unknown'}:
            raise ValueError(f'fact {i}: comparison_kind is required')
        if scope == 'segment':
            name = fact.get('segment_name')
            if not isinstance(name, str) or not name.strip() or name.casefold() not in fact.get('statement', '').casefold():
                raise ValueError(f'fact {i}: name the segment explicitly in segment_name and statement')
        percent = fact.get('reported_change_percent')
        if percent is None:
            checks.append({'fact_index': i, 'status': 'not_arithmetically_checked'})
            continue
        a, b = fact.get('value'), fact.get('comparison_value')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (a, b, percent)) or b <= 0:
            raise ValueError(f'fact {i}: relative change requires finite same-scale operands and positive comparison_value; otherwise use null')
        if fact['comparison_kind'] not in {'yoy', 'qoq', 'ytd_yoy'}:
            raise ValueError(f'fact {i}: relative comparison needs an explicit compatible comparison_kind')
        quotes = []
        for citation in fact.get('citations', []):
            for qid in citation.get('quote_ids', []):
                if qid not in catalog:raise ValueError(f'fact {i}: unknown quote_id')
                quotes.append(catalog[qid]['quote'])
        values = _numbers(' '.join(quotes))
        if a not in values or b not in values:
            raise ValueError(f'fact {i}: operands must occur at the stated numeric scale in chosen quotes; keep original scale or omit ambiguous comparison')
        expected = 100 * (a / b - 1)
        # Allow rounding to an integer percentage; this narrow tolerance is
        # not permission to infer scope/columns, and does not detect every error.
        if abs(expected - percent) > 0.500001:
            raise ValueError(f'fact {i}: reported_change_percent={percent} conflicts with 100*({a}/{b}-1)={expected:.6g}; verify source scope, periods and operands before correcting or omitting the comparison')
        checks.append({'fact_index': i, 'status': 'arithmetic_consistent', 'calculated_percent': expected,
                       'scope_and_period_semantically_verified': False})
    return checks


def screen_bindings(data, catalog):
    """Mark only unsupported facts for existing rejection handling.

    Retains original facts for the audit trail; never fabricates a replacement
    or silently deletes only a contradictory percentage from a statement.
    """
    import copy
    screened = copy.deepcopy(data)
    if not isinstance(screened.get('facts'), list):
        raise ValueError('facts must be a list')
    audit = []
    for i, fact in enumerate(screened['facts']):
        try:
            check = validate_bindings({'facts': [fact]}, catalog)[0]
            check['fact_index'] = i
            audit.append(check)
        except (ValueError, KeyError, TypeError) as exc:
            error = 'source_binding_rejected: ' + str(exc)
            fact['_citation_restore_error'] = error
            audit.append({'fact_index': i, 'status': 'rejected', 'error': error})
    return screened, audit
