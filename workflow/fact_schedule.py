"""Coverage-first ordering and observable, conservative prediction reserves."""
import math


def order_jobs(jobs, mode):
    if mode == 'sequential': return list(jobs)
    if mode != 'coverage_first': raise ValueError('invalid facts schedule')
    return sorted(jobs,key=lambda j:(j['batch_no'],j['entity_index']))


def time_reserve(records, entity_count, *, reserve_prediction=True,batch_size=4):
    # Estimate from this run, without consulting outcomes or prior task results.
    samples=[r['seconds'] for r in records if r.get('status')=='success'
             and type(r.get('seconds')) in (int,float) and r['seconds']>0]
    samples=sorted(samples[-8:])
    typical=samples[max(0,math.ceil(.75*len(samples))-1)] if samples else 30.0
    typical=max(20.0,min(120.0,typical))
    from .roster_budget import prediction_batch_count
    calls=prediction_batch_count(entity_count,batch_size)
    return {'prediction_calls':calls,'observed_samples':len(samples),
            'estimated_call_seconds':typical,
            'prediction_reserve_seconds':(calls+.5)*typical+10.0 if reserve_prediction else 0.0,
            'method':'recent successful request p75; >=20s/call, half-call repair margin plus 10s publication'}
