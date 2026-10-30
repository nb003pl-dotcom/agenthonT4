"""Optional mathematical coherence check, never a realized-outcome check."""
from .prediction import finite

INSTRUCTION = """
MEDIAN CONTRACT: point_forecast is the forecast MEDIAN, not its mean. In the two-label EPS YoY task, a strictly greater-than-50% probability of up requires the median at or above the prior-year EPS; a majority probability of down requires the median at or below it. Do not return a central value on the opposite side. Review the EPS level, class probabilities and explanation together; either may need correction. For the three-label beat/miss/inline task, a modal class below 50% may legitimately differ from the median's class; do NOT force every modal label to match the point. This is internal probability consistency, not knowledge of future outcomes.
"""


def validate_median(rows, inputs):
    # Only the explicit two-label, ordered EPS comparison is checked here.
    # No assumptions about arbitrary label regions or probability intervals.
    if inputs.task.get('family')!='eps_yoy_direction':return rows
    if set(inputs.task['target'].get('labels',[]))!={'up','down'}:return rows
    for row in rows:
        baseline=inputs.entities[row['entity_id']].get('prior_year_q_eps')
        point=row.get('point_forecast');probs=row.get('class_probabilities')
        if not finite(baseline) or not finite(point) or not isinstance(probs,dict):continue
        total=sum(probs.values())
        if not total:continue  # Existing distribution validator owns this case.
        eps=1e-8*max(1,abs(baseline),abs(point))
        contradictory=((probs.get('up',0)/total>.5+1e-8 and point<baseline-eps) or
                       (probs.get('down',0)/total>.5+1e-8 and point>baseline+eps))
        if contradictory:
            raise ValueError('median_probability_inconsistent for '+row['entity_id']+
                ': point_forecast is a median, but it lies opposite the EPS direction assigned more than 50% probability. '+
                'Recheck the EPS level, task prior_year_q_eps, class probabilities and explanation; do not mechanically relabel a valid distribution.')
    return rows
