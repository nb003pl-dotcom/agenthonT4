"""Explicit domain-independent research budgets; baseline defaults stay unchanged."""

EXPANDED = {
    'evidence_cap':12,
    'selection_source_token_budget':256000,
    'facts_source_token_budget':100000,
    'facts_max_per_window':12,
    'context_window':196608,
    'max_input_tokens':3000000,
    'max_requests':96,
    'timeout':1200,
    'max_tokens':4000,
}


def apply_budget_profile(config, profile):
    if profile is None:return dict(config)
    if profile!='expanded':raise ValueError('unknown budget profile')
    return {**config,**EXPANDED,'budget_profile':profile}


def fact_limit(config, default):
    value=config.get('facts_max_per_window')
    if value is None:return default
    if type(value) is not int or not 1<=value<=12:raise ValueError('facts_max_per_window must be 1..12')
    return value
