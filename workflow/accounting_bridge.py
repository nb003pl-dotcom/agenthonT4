"""Conditional arithmetic over cited, explicitly typed quarterly observations.

No extraction by ticker, tax-rate assumption, future outcome, or automatic
forecast adjustment. Source semantics remain unverified even if arithmetic fits.
"""
import math
import re
from collections import defaultdict

ROLES = {'diluted_eps', 'diluted_earnings_numerator', 'common_net_income',
         'diluted_weighted_average_shares', 'oneoff_gain_pretax',
         'oneoff_cost_pretax', 'oneoff_gain_aftertax', 'oneoff_cost_aftertax'}
MONEY = {'USD': 1, 'USD_thousand': 1000, 'USD_million': 1e6, 'USD_billion': 1e9}
SHARES = {'shares': 1, 'thousand_shares': 1000, 'million_shares': 1e6}

EXTRACT = """
Accounting arithmetic experiment: retain up to limits.max_facts facts. In addition
to the EPS anchor, guidance and material adjustments, retain the EPS reconciliation
numerator and diluted weighted-average shares when actually available. Do not fill
slots with invented data or sacrifice target guidance to redundant history.
Optionally tag each SINGLE quantitative accounting observation with accounting_role:
diluted_eps | diluted_earnings_numerator | common_net_income |
diluted_weighted_average_shares | oneoff_gain_pretax | oneoff_cost_pretax |
oneoff_gain_aftertax | oneoff_cost_aftertax, and accounting_period: YYYYQ1..YYYYQ4.
Use these tags only for consolidated GAAP three-month observations with an explicit
calendar quarter. Never tag six/nine-month cumulative data as a quarter. Preserve
the original displayed amount without scale conversion and set unit exactly to
USD, USD_thousand, USD_million, USD_billion, shares, thousand_shares,
million_shares, or USD/share as appropriate. Cite the period/unit header and row.
EPS reconciliation numerator means earnings actually used to compute diluted EPS;
ordinary net income or income available to common stock is common_net_income unless
the reconciliation explicitly establishes the diluted numerator. Basic/end-period
shares cannot be used as diluted weighted-average shares. Continuing-operations EPS
must not be tagged as total-company diluted_eps. For one-off items preserve tax
basis and use a positive magnitude plus gain/cost role; if tax basis is unclear do
not assign an accounting_role. Distinct items need distinct cited facts. Do not add
an estimated tax rate, normalized earnings, or forecasts to extracted observations.
"""

PREDICT = """
accounting_checks are Python arithmetic diagnostics conditional on extracted source
labels, NOT verified source semantics or forecasts. Recheck their fact_ids for
period, GAAP scope, tax basis and numerator definition. Do not use an inconsistent
identity as a normal earnings baseline. aftertax_item_sensitivities show each item
separately; they are NOT additive (disclosures may overlap), and are not next-quarter
GAAP forecasts. Pretax items without a disclosed after-tax amount remain uncomputed.
Keep forecast assumptions separate and explain their evidence; missing shares or
tax information is uncertainty, not permission to invent exact adjustments.
"""


def metadata(raw):
    role = raw.get('accounting_role')
    period = raw.get('accounting_period')
    if role is None and period is None:
        return {}
    if role not in ROLES or not isinstance(period, str) or not re.fullmatch(r'\d{4}Q[1-4]', period):
        raise ValueError('accounting_role_requires_valid_role_and_calendar_quarter')
    units = {'USD/share'} if role == 'diluted_eps' else SHARES if role == 'diluted_weighted_average_shares' else MONEY
    if raw.get('unit') not in units:
        raise ValueError('accounting_role_has_incompatible_unit')
    return {'accounting_role': role, 'accounting_period': period}


def build_checks(facts):
    groups = defaultdict(list)
    ignored = []
    for f in facts:
        if not f.get('accounting_role'):
            continue
        reason = None
        try:
            metadata(f)
        except (ValueError, TypeError):
            reason = 'invalid_accounting_metadata'
        docs = {c['doc_id'] for c in f.get('citations', [])}
        if not f.get('validation', {}).get('source_located') or len(docs) != 1:
            reason = 'requires_located_single_document_evidence'
        if f.get('reporting_scope') != 'consolidated' or f.get('basis') != 'GAAP':
            reason = 'requires_consolidated_GAAP_observation'
        if type(f.get('value')) not in (int, float) or not math.isfinite(f['value']):
            reason = 'requires_finite_numeric_value'
        if reason:
            ignored.append({'entity_id': f['entity_id'], 'fact_id': f['fact_id'], 'reason': reason})
        else:
            groups[(f['entity_id'], next(iter(docs)), f['accounting_period'])].append(f)
    checks = []
    for (entity, doc, period), rows in sorted(groups.items()):
        by_role = defaultdict(list)
        for f in rows:
            by_role[f['accounting_role']].append(f)
        packet = {'entity_id': entity, 'doc_id': doc, 'period': period,
                  'source_semantics': 'unverified', 'identity': {'status': 'insufficient_evidence'},
                  'aftertax_item_sensitivities': [], 'uncomputed': []}

        def observation(role, scale):
            options = by_role[role]
            values = {f['value'] * scale[f['unit']] for f in options}
            if len(values) != 1:
                packet['uncomputed'].append({'role': role, 'reason': 'conflicting_values' if values else 'not_extracted',
                                             'fact_ids': [f['fact_id'] for f in options]})
                return None
            value = next(iter(values))
            if not math.isfinite(value):
                packet['uncomputed'].append({'role': role, 'reason': 'scaled_value_overflow'})
                return None
            return value, [f['fact_id'] for f in options]

        eps = observation('diluted_eps', {'USD/share': 1})
        shares = observation('diluted_weighted_average_shares', SHARES)
        numerator = observation('diluted_earnings_numerator', MONEY)
        if shares is not None and shares[0] <= 0:
            packet['uncomputed'].append({'role': 'diluted_weighted_average_shares', 'reason': 'nonpositive_shares'})
            shares = None
        if eps and shares and numerator:
            computed = numerator[0] / shares[0]
            # 1 cent allowance accommodates typical displayed EPS rounding.
            residual = computed - eps[0]
            packet['identity'] = {'status': 'arithmetically_consistent' if abs(residual) <= .01 else 'inconsistent',
                'reported_eps': eps[0], 'computed_eps': computed, 'residual': residual,
                'rounding_tolerance_USD_per_share': .01,
                'formula': 'diluted_earnings_numerator / diluted_weighted_average_shares',
                'fact_ids': numerator[1] + shares[1] + eps[1]}
            if not math.isfinite(computed) or not math.isfinite(residual):
                packet['identity']={'status':'arithmetic_overflow','fact_ids':numerator[1]+shares[1]+eps[1]}
        if by_role['common_net_income']:
            packet['uncomputed'].append({'role': 'common_net_income', 'reason': 'not_automatically_diluted_numerator',
                                        'fact_ids': [f['fact_id'] for f in by_role['common_net_income']]})
        for role in ('oneoff_gain_pretax', 'oneoff_cost_pretax', 'oneoff_gain_aftertax', 'oneoff_cost_aftertax'):
            for item in by_role[role]:
                if role.endswith('pretax') or not shares or not eps or item['value'] < 0 or packet['identity']['status'] in ('inconsistent','arithmetic_overflow'):
                    packet['uncomputed'].append({'role': role, 'fact_ids': [item['fact_id']],
                        'reason': 'no_aftertax_amount' if role.endswith('pretax') else 'inconsistent_identity_or_missing_inputs'})
                    continue
                effect = item['value'] * MONEY[item['unit']] / shares[0]
                without_item = eps[0] + (effect if role == 'oneoff_cost_aftertax' else -effect)
                if not math.isfinite(effect) or not math.isfinite(without_item):
                    packet['uncomputed'].append({'role': role, 'fact_ids': [item['fact_id']], 'reason': 'scaled_value_overflow'})
                    continue
                packet['aftertax_item_sensitivities'].append({'role': role, 'per_share_magnitude': effect,
                    'reported_eps_without_this_item_only': without_item,
                    'not_additive_or_a_forecast': True, 'fact_ids': [item['fact_id']] + shares[1] + eps[1]})
        checks.append(packet)
    return {'checks': checks, 'ignored': ignored,
            'notice': 'Arithmetic only. Extraction labels, period alignment and entailment are not certified.'}
