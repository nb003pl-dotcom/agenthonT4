"""Task-independent audit of raw versus contract-canonical numeric views.

The contract's existing validator owns conversions; this module neither chooses
forecasts nor uses truth. No task/family/entity-specific branching.
"""
import math


def _finite(value):return type(value) in (int,float) and math.isfinite(value)


def numeric_conflicts(raw,canonical,*,relative_tolerance=.01,absolute_tolerance=1e-6):
    original={r.get('entity_id'):r for r in raw.get('predictions',[]) if isinstance(r,dict)}
    issues=[]
    for row in canonical:
        entity=row['entity_id'];before=original.get(entity,{})
        pairs=[('point_forecast',before.get('point_forecast'),row.get('point_forecast'))]
        for key in ('lo','hi'):
            pairs.append(('interval.'+key,(before.get('interval') or {}).get(key),(row.get('interval') or {}).get(key)))
        for field,a,b in pairs:
            if not (_finite(a) and _finite(b)):continue
            tolerance=max(absolute_tolerance,relative_tolerance*max(abs(a),abs(b)))
            if abs(a-b)>tolerance:
                issues.append({'entity_id':entity,'field':field,'reported':a,'canonical':b,
                    'absolute_difference':abs(a-b),'tolerance':tolerance})
    return issues


def require_consistency(issues):
    if issues:
        compact=[{k:r[k] for k in ('entity_id','field','reported','canonical')} for r in issues]
        raise ValueError('numeric_view_conflict: '+str(compact)+
            '. Return the full requested batch with mutually consistent numeric views using the supplied target contract; '
            'do not mechanically repeat incompatible point/interval/underlying values.')
