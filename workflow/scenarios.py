"""Experimental numerical scenario mixture. No fitting or reference answers."""
import math
from scipy.optimize import brentq
from scipy.special import ndtr

def mixture_quantiles(components,level=.9):
    if not .5<level<1 or not 2<=len(components)<=5:raise ValueError('Invalid mixture size/level')
    for c in components:
        if any(type(c.get(k)) not in (int,float) or not math.isfinite(c[k]) for k in ('weight','mean','sigma')):
            raise ValueError('Mixture parameters must be finite numbers')
        if c['weight']<=0 or c['sigma']<=0:raise ValueError('Positive weight and outcome sigma required')
    total=sum(c['weight'] for c in components)
    if abs(total-1)>1e-6:raise ValueError('Scenario probabilities must sum to 1')
    def cdf(x):return sum(c['weight']*float(ndtr((x-c['mean'])/c['sigma'])) for c in components)
    lower=min(c['mean']-12*c['sigma'] for c in components);upper=max(c['mean']+12*c['sigma'] for c in components)
    def quantile(p):return float(brentq(lambda x:cdf(x)-p,lower,upper))
    return {'point_forecast':quantile(.5),'interval':{'lo':quantile((1-level)/2),'hi':quantile((1+level)/2)},
            'mean':sum(c['weight']*c['mean'] for c in components),'method':'Median and central quantiles of model-assessed normal outcome scenario mixture; not empirically calibrated'}
