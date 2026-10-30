"""Experimental release-age alignment; no reference labels or task-ID rules.

Publication age is a calendar proxy, not a claim that every agency has the
same release schedule. Sparse support remains explicit and can cause fallback.
"""
from datetime import date
from statistics import median
import re

from .quantitative import number, distribution


def month_index(value):
    return date.fromisoformat(str(value)[:7] + '-01').year * 12 + int(str(value)[5:7])


def annual_candidates(rows, vintages, *, annual_hint=False):
    """Use only supplied visible columns; preserve the legacy outlier rule."""
    if not annual_hint:
        return []
    changes = {}
    for row in rows:
        available = [(v, number(x)) for v, x in zip(vintages, row[1:]) if number(x) is not None]
        for (_, x), (v, y) in zip(available, available[1:]):
            if y != x:
                changes.setdefault(v, []).append(y - x)
    sizes = {v: median(abs(x) for x in xs) for v, xs in changes.items()}
    typical = median(sizes.values()) if sizes else 0
    return sorted(v for v, size in sizes.items()
                  if len(changes[v]) >= 3 and size > max(1e-9, typical) * 10)


def matched_transitions(rows, vintages, entity, *, as_of, excluded_vintages=()):
    """One age/horizon matched pair per historical reference month.

    Select expected publication dates before checking value availability; this
    avoids choosing a later release merely because a cell is missing. No future
    column is allowed even when its reference month itself is historical.
    """
    start = date.fromisoformat(entity['latest_precutoff_vintage'])
    end = date.fromisoformat(entity['resolving_release_date'])
    cutoff = date.fromisoformat(as_of)
    if start > cutoff or end <= start:
        raise ValueError('invalid_revision_time_contract')
    reference = entity['ref_month']
    start_age = month_index(start.isoformat()) - month_index(reference)
    end_age = month_index(end.isoformat()) - month_index(reference)
    if start_age < 0 or end_age < start_age:
        raise ValueError('invalid_revision_release_age')
    indexed = [(i, date.fromisoformat(v)) for i, v in enumerate(vintages)
               if date.fromisoformat(v) <= cutoff]
    observations = []
    rejected = []
    for row in rows:
        if not re.fullmatch(r'\d{4}-\d{2}', str(row[0])):
            continue
        if row[0] >= reference:
            continue
        candidates = []
        for age, day in ((start_age, start.day), (end_age, end.day)):
            choices = [(i, v) for i, v in indexed if month_index(v.isoformat()) - month_index(row[0]) == age]
            candidates.append(min(choices, key=lambda z: (abs(z[1].day-day), z[1])) if choices else None)
        a, b = candidates
        if a is None or b is None:
            continue
        ai, av = a; bi, bv = b
        if av >= bv:
            continue
        if any(av.isoformat() < v <= bv.isoformat() for v in excluded_vintages):
            rejected.append({'reference_month':row[0], 'reason':'crosses_annual_regime',
                             'from_vintage':av.isoformat(), 'to_vintage':bv.isoformat()})
            continue
        x, y = number(row[ai+1]), number(row[bi+1])
        if x is None or y is None:
            continue
        observations.append({'reference_month':row[0], 'from_vintage':av.isoformat(),
                             'to_vintage':bv.isoformat(), 'from_value':x, 'to_value':y,
                             'delta':y-x, 'start_release_age_months':start_age,
                             'end_release_age_months':end_age})
    return {'start_release_age_months':start_age, 'end_release_age_months':end_age,
            'observations':observations, 'sample_count':len(observations),
            'rejected_transitions':rejected, 'as_of':as_of}


def age_aligned_anchor(rows, vintages, entity, *, as_of, level=.9, annual_hint=False):
    visible = [(i, v) for i, v in enumerate(vintages) if v <= as_of]
    visible_rows = [[r[0], *[r[i+1] for i, _ in visible]] for r in rows]
    visible_vintages = [v for _, v in visible]
    annual = annual_candidates(visible_rows, visible_vintages, annual_hint=annual_hint)
    audit = matched_transitions(visible_rows, visible_vintages, entity,
                                as_of=as_of, excluded_vintages=annual)
    audit['excluded_annual_vintages'] = annual
    ds = [r['delta'] for r in audit['observations']]
    if len(ds) < 3:
        return {'status':'insufficient_age_matched_history', 'alignment':audit}
    delta = float(median(ds))
    baseline = entity['latest_precutoff_estimate']
    band = distribution(baseline+delta, [x-delta for x in ds], level)
    # The task has only up/down. Unchanged historical releases are retained for
    # magnitudes and provide no directional evidence (half mass to each side).
    up = (sum(x>0 for x in ds) + .5*sum(x==0 for x in ds) + .5)/(len(ds)+1)
    return {'status':'available', **band, 'baseline':baseline,
            'estimated_revision_delta':delta, 'observations':len(ds),
            'class_probabilities':{'up':up,'down':1-up}, 'alignment':audit,
            'method':'Calendar release-age and horizon matched median, minimum three samples; zeros retained; annual crossing excluded.',
            'assumption':'Sparse calendar-age analogues, not exact agency release-stage identification or guaranteed interval coverage. Zero revisions contribute neutral direction mass.'}
