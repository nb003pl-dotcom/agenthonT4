"""Fixed equal-weight forecast combination, independent of outcomes.

Averaging interval endpoints does not certify calibration or independence.
No runtime default enables this experimental utility.
"""
import copy
import math
from collections import Counter
from .calculations import derive_eps_growth


def average_forecasts(first, second, contract):
    ids = set(contract['entities'])
    def mapping(rows):
        if len(rows) != len(ids) or {r['entity_id'] for r in rows} != ids:
            raise ValueError('ensemble_roster_mismatch')
        return {r['entity_id']: r for r in rows}
    a, b = mapping(first), mapping(second)
    def mean(x, y):
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y)):
            raise ValueError('ensemble_nonfinite_number')
        return x/2 + y/2
    result = []
    for eid in contract['entities']:
        x, y = a[eid], b[eid]
        r = copy.deepcopy(x)
        r['point_forecast'] = mean(x['point_forecast'], y['point_forecast'])
        for z in (x, y):
            if not z['interval']['lo'] <= z['point_forecast'] <= z['interval']['hi']:
                raise ValueError('ensemble_invalid_component_interval')
        r['interval'] = {k: mean(x['interval'][k], y['interval'][k]) for k in ('lo', 'hi')}
        if contract['family'] == 'eps_growth_regression':
            r['underlying_eps'] = mean(x['underlying_eps'], y['underlying_eps'])
            r['eps_interval'] = {k: mean(x['eps_interval'][k], y['eps_interval'][k]) for k in ('lo', 'hi')}
        if contract['target']['type'] == 'classification':
            labels = contract['target']['labels']
            for z in (x, y):
                probs = z.get('class_probabilities', {})
                if set(probs) != set(labels) or any(type(p) not in (int,float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()) or abs(sum(probs.values())-1) > .02:
                    raise ValueError('ensemble_invalid_probabilities')
            r['class_probabilities'] = {label: mean(x['class_probabilities'][label],y['class_probabilities'][label]) for label in labels}
            total = sum(r['class_probabilities'].values())
            r['class_probabilities'] = {k:v/total for k,v in r['class_probabilities'].items()}
            r['label'] = max(labels,key=r['class_probabilities'].get)
        if contract['target']['type'] == 'ranking':
            if x.get('ranking_order') != y.get('ranking_order'):
                raise ValueError('ensemble_ranking_direction_mismatch')
        if x.get('unit') != y.get('unit'):
            raise ValueError('ensemble_unit_mismatch')
        ordered = list(dict.fromkeys(x['fact_ids'] + y['fact_ids']))
        support = Counter(set(x['fact_ids'])) + Counter(set(y['fact_ids']))
        r['fact_ids'] = sorted(ordered,key=lambda fid:(-support[fid],ordered.index(fid)))
        r['reasoning'] = ('Equal-weight combination of two separately generated forecasts from identical evidence. '
                          f'Component points: {x["point_forecast"]}, {y["point_forecast"]}; interval endpoints are averaged, not confidence intervals for the mean. '
                          'Original assumptions and selected evidence are retained in the component artifacts.')
        result.append(r)
    return derive_eps_growth({'predictions': result},contract)['predictions']
