"""Forecast ablation: separate observed rate diagnostics from arbitrary anchors.

This supplies no alternative point prediction or interval. Used by the paired
benchmark only until evidence supports changing the production workflow.
"""
from copy import deepcopy
import math


def without_synthetic_anchor(payload, *, retain_scale=False):
    if payload.get('target_contract', {}).get('family') != 'rate_curve_cross_section':
        return deepcopy(payload)
    result=deepcopy(payload)
    anchors=result.pop('quantitative_anchors',{})
    # Explicit allowlist: exclude zero point, stress bounds, policy-risk floor,
    # forecast sigma and metadata which describes them as predictive estimates.
    fields=('doc_id','doc_date','source_start','source_end','source_header',
            'observations','daily_sigma_bps','business_day_horizon_proxy',
            'policy_step_bps','regime_context')
    result['observed_rate_diagnostics']={e:{k:p[k] for k in fields if k in p}
                                         for e,p in anchors.items()}
    result['rate_diagnostics_notice']=(
        'Historical observations and a business-day horizon approximation only; '
        'no fitted point forecast or calibrated prediction interval is supplied. '
        'Estimate the future yield change from the cited pre-cutoff expectations, '
        'repricing mechanisms and uncertainty. Policy announcements already '
        'occurred; an announced policy step is not a Treasury-yield forecast.')
    if retain_scale:
        scales={}
        for entity,p in anchors.items():
            lo,hi=p.get('lo'),p.get('hi')
            if any(type(v) not in (float,int) or not math.isfinite(v) for v in (lo,hi)) or hi<=lo:
                continue
            # This is the SAME uncertainty scale as the control, not a new fit.
            scales[entity]={'reference_halfwidth_bps':(hi-lo)/2,
                'nominal_interval_level':p.get('level'),
                'empirically_calibrated':False,
                'method':p.get('method'),
                'conditional_inputs':{k:p[k] for k in ('daily_sigma_bps','business_day_horizon_proxy',
                    'policy_uncertainty_bps','policy_step_bps') if k in p}}
        result['conditional_uncertainty_scales']=scales
        result['rate_diagnostics_notice']+=(
            ' conditional_uncertainty_scales retains the control method\'s rough '
            'outcome-dispersion reference, with its assumptions exposed; it is '
            'neither an observed confidence interval nor a calibrated guarantee. '
            'Choose the central forecast separately. Use horizon and event risk '
            'to justify interval width around your forecast; do not narrow it '
            'merely because the zero-change point anchor has been removed.')
    return result
