"""Past-only pooled position-change regression; no entity/date-specific parameters."""
import numpy as np


def estimate(series, origin, horizon, penalty=10.0):
    """Training labels must have resolved at or before the forecast origin.

    Each series supplies equally spaced, aligned report dates, net contracts and
    open interest. Targets use FIXED origin open interest, including when OI moves.
    """
    if not series or horizon<1 or origin<4:raise ValueError('Invalid history/horizon')
    dates=next(iter(series.values()))['dates']
    for data in series.values():
        if data['dates']!=dates:raise ValueError('Panel dates must be aligned')
        if len(data['net'])!=len(dates) or len(data['oi'])!=len(dates):raise ValueError('Inconsistent series lengths')
        if not np.isfinite(data['net'][:origin+1]).all() or not np.isfinite(data['oi'][:origin+1]).all() or min(data['oi'][:origin+1])<=0:
            raise ValueError('Finite positions and positive open interest required')
    if origin>=len(dates):raise ValueError('Forecast origin outside history')
    def features(data,t):
        net=np.asarray(data['net']);den=data['oi'][t]
        return np.array([net[t]-np.median(net[:t+1]),
                         (net[t]-net[t-4])*horizon/4,
                         (net[t]-net[t-1])*horizon])/den*100
    samples=[(eid,t) for eid in series for t in range(4,origin-horizon+1)]
    if len(samples)<30:return {'status':'insufficient_resolved_training_examples'}
    x=np.array([features(series[e],t) for e,t in samples])
    y=np.array([(series[e]['net'][t+horizon]-series[e]['net'][t])/series[e]['oi'][t]*100 for e,t in samples])
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-6);z=(x-mean)/scale
    beta=np.linalg.solve(z.T@z+penalty*np.eye(x.shape[1]),z.T@(y-y.mean()))
    points={e:float(y.mean()+(features(d,origin)-mean)/scale@beta) for e,d in series.items()}
    return {'status':'available','points':points,'training_examples':len(samples),
            'latest_training_label_index':max(t+horizon for _,t in samples),
            'forecast_origin_index':origin,'forecast_origin_date':dates[origin],
            'horizon_reports':horizon,'penalty':penalty,'feature_names':['position_minus_own_history_median','four_report_momentum_scaled_to_horizon','one_report_momentum_scaled_to_horizon'],
            'standardized_coefficients':beta.tolist(),'intercept':float(y.mean()),
            'assumption':'One pooled linear relationship across heterogeneous markets; overlapping labels, short time history, no calibrated probability or coverage claim.'}


def walk_forward(series,horizon):
    dates=next(iter(series.values()))['dates'];folds=[]
    for origin in range(4,len(dates)-horizon):
        result=estimate(series,origin,horizon)
        if result['status']!='available':continue
        actual={e:(d['net'][origin+horizon]-d['net'][origin])/d['oi'][origin]*100 for e,d in series.items()}
        folds.append({**result,'resolved_target_index':origin+horizon,'resolved_target_date':dates[origin+horizon],
                      'actual_precutoff':actual})
    return folds
