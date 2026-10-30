"""Opt-in accounting-context profile, selected by task family, never ticker.

These instructions improve evidence interpretation; they do not certify
entailment or supply values that are absent from the frozen corpus.
"""
EPS_FAMILIES = {'eps_beat_consensus', 'eps_yoy_direction', 'eps_growth_regression'}


def enabled(config, contract):
    return bool(config.get('financial_scope')) and contract.get('family') in EPS_FAMILIES


QUERY = """
Accounting-context profile: produce four complementary queries per entity:
1. target_history: consolidated company quarterly GAAP diluted earnings per share and income statement, with the latest pre-cutoff quarter.
2. guidance_expectation: target-quarter EPS/revenue/margin or net-interest-income guidance.
3. financial_driver: operating revenues, expenses, provisions and share-count changes.
4. risk_counterevidence: nonrecurring gains/losses, asset/share sales, tax benefits, restructuring charges and EPS excluding special items.
Do not confuse segment net income with consolidated earnings or diluted EPS.
Use company names and the relevant available reporting period; do not invent an event or its amount.
"""

SELECT = """
Accounting-context selection: first secure the consolidated quarterly diluted-EPS/income-statement anchor with its period header. Then retain quantified outlook, material one-off earnings adjustments and distinct operating drivers. Prefer a company-wide summary over a segment profitability table when both mention net income. Segment information is useful as a driver, but cannot replace the consolidated EPS anchor. A comparison may be quarter-on-quarter, year-on-year or year-to-date: select enough header context to distinguish these. Special-item disclosures may be in a different candidate than the EPS row; keep both when available. No relevant evidence in a preview is not proof of absence.
"""

FACTS = """
Accounting-context profile: retain up to four decisive facts, prioritizing (a) the latest pre-cutoff consolidated quarterly GAAP diluted EPS, with a correctly aligned comparison if identifiable; (b) target-quarter quantified guidance; (c) material one-off gains/losses and their disclosed EPS effect; (d) a distinct operating driver. Missing slots may be filled with other supported facts; never invent a baseline or an adjustment.
For EVERY fact add reporting_scope: consolidated|segment|unknown. If segment, also add segment_name and name that segment in the statement. The company document's author is not proof that every table is consolidated. A table under a business division is segment data until explicitly reconciled to the company total. Net income in millions/billions is not EPS in dollars per share.
Read the actual table headers BEFORE assigning periods. Adjacent columns may be current quarter, previous quarter, same quarter last year, two percentage-change columns, and year-to-date totals. Never assume column two means last year. Add comparison_kind: yoy|qoq|ytd_yoy|none|unknown, and comparison_value when asserted. Include header quote IDs when row values alone cannot identify the periods/scope; or use an explicit narrative sentence that does. If alignment cannot be resolved, omit the comparison rather than guess.
Distinguish three/six/nine/twelve months and GAAP/adjusted amounts. Add earnings_character: reported|nonrecurring_adjustment|guidance|operating_driver|unknown. For special items preserve whether amounts are pretax/after-tax and gains/costs; do not subtract a pretax dollar gain directly from after-tax EPS. Forecast normalization happens later and remains an assumption, not an extracted fact.
Keep the factual statement concise. Quote IDs still must locate all asserted amounts and necessary context, and all existing citation limits apply.
"""

PREDICT = """
Accounting-context checks: use consolidated quarterly EPS as the historical level anchor. Segment net income is only a driver; never treat it as total company profit or per-share earnings. Respect comparison_kind and actual quarter labels: QoQ is not YoY. Separate disclosed nonrecurring gains/costs from recurring earnings before extrapolating, but keep the target as future GAAP EPS. Any tax/share-count adjustment not stated in the evidence is a forecast assumption with uncertainty, not a source fact. Do not infer an undisclosed one-off amount. Use the task's prior_year_q_eps for the requested YoY denominator, not an adjacent source-table column.
"""


def query_prompt(prompt, compact):
    if compact:
        prompt = prompt.replace('exactly 3', 'exactly 4').replace('Exactly 3', 'Exactly 4')
    return prompt + '\n' + QUERY


def metadata(raw):
    """Validate optional model metadata without labeling it semantically verified."""
    allowed = {
        'reporting_scope': {'consolidated', 'segment', 'unknown'},
        'comparison_kind': {'yoy', 'qoq', 'ytd_yoy', 'none', 'unknown'},
        'earnings_character': {'reported', 'nonrecurring_adjustment', 'guidance', 'operating_driver', 'unknown'},
        'information_role': {'historical_result','management_guidance','market_expectation','event_risk','definition','unknown'},
    }
    from .accounting_bridge import metadata as accounting_metadata
    result = accounting_metadata(raw)
    for key, values in allowed.items():
        if raw.get(key) is not None:
            if raw[key] not in values: raise ValueError('invalid_' + key)
            result[key] = raw[key]
    if raw.get('segment_name') is not None:
        if not isinstance(raw['segment_name'], str): raise ValueError('invalid_segment_name')
        result['segment_name'] = raw['segment_name']
    if result.get('reporting_scope') == 'segment' and not result.get('segment_name'):
        raise ValueError('segment_fact_requires_segment_name')
    if raw.get('comparison_value') is not None:
        import math
        if type(raw['comparison_value']) not in (float, int) or not math.isfinite(raw['comparison_value']):
            raise ValueError('invalid_comparison_value')
        result['comparison_value'] = raw['comparison_value']
    return result


def arithmetic_issue(fact):
    """Reject only explicit, self-contradictory positive EPS comparisons.

    Both amounts must occur literally in the citations and match the typed
    comparison fields. Negative EPS, mixed monetary scales and ambiguous prose
    remain semantic review cases; no period or source value is guessed here.
    """
    import math
    import re
    a,b=fact.get('value'),fact.get('comparison_value')
    if fact.get('comparison_kind') not in ('yoy','qoq','ytd_yoy'):return None
    if type(a) not in (int,float) or type(b) not in (int,float) or min(a,b)<=0:return None
    if not re.search(r'\bEPS\b|earnings per.*share',str(fact.get('metric','')),re.I):return None
    text=fact.get('statement','')
    if re.search(r'\b(?:million|billion|thousand)\b',text,re.I):return None
    amounts=[float(v) for v in re.findall(r'\$\s*(\d+(?:\.\d+)?)',text)]
    if len(amounts)!=2 or not math.isclose(amounts[0],a) or not math.isclose(amounts[1],b):return None
    quote_values={float(v) for v in re.findall(r'(?<![\w.])\d+\.\d+(?![\w.])',
        ' '.join(c['quote'] for c in fact.get('citations',[])))}
    if a not in quote_values or b not in quote_values:return None
    match=re.search(r'\b(up|down)\s*(?:(\d+(?:\.\d+)?)\s*%\s*)?(?:from|versus)\s*\$',text,re.I)
    if not match:return None
    actual=100*(a/b-1);up=match[1].lower()=='up'
    if (up and a<b) or (not up and a>b):return 'eps_direction_contradicts_explicit_values'
    if match[2] is not None:
        claimed=float(match[2]);places=len(match[2].split('.')[1]) if '.' in match[2] else 0
        if abs(abs(actual)-claimed)>0.5*10**(-places)+1e-6:
            return 'eps_percentage_contradicts_explicit_values'
    return None
