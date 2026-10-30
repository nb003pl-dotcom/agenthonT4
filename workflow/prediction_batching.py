"""Budgeted joint cross-sectional forecasts; never truncate or drop entities."""
def groups(entities, target_type, fits, mode='fixed', max_joint=12):
    ids=list(entities)
    cap=max_joint if mode=='joint' and target_type in ('ranking','regression') else 4
    pending=[]
    for entity in ids:
        proposed=pending+[entity]
        if pending and (len(proposed)>cap or not fits(proposed)):
            yield pending;pending=[]
        pending.append(entity)
        if not fits(pending):raise ValueError('One entity prediction payload exceeds input budget')
    if pending:yield pending
