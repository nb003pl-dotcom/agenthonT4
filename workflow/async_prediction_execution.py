"""Native asynchronous prediction calls with bounded admission and cleanup."""
import asyncio
import time
from .common import check_time
from .prediction_execution import Reservations


async def execute_async(model,jobs,workers,on_result):
    if type(workers) is not int or workers not in (1,2):
        raise ValueError('prediction_workers must be 1 or 2')
    if not jobs:return []
    check_time(model.deadline)
    ledger=Reservations(model,len(jobs));results={};active={};next_job=0
    async def run(index):
        job=jobs[index]
        try:
            return await model.acall('predict',job['system'],job['payload'],job['validator'],
                                     request_budget=ledger.for_job(index))
        finally:ledger.finish(index)
    try:
        async with asyncio.timeout(max(0,model.deadline-time.time())):
            while next_job<len(jobs) or active:
                while next_job<len(jobs) and len(active)<workers:
                    active[asyncio.create_task(run(next_job))]=next_job;next_job+=1
                finished,_=await asyncio.wait(active,return_when=asyncio.FIRST_COMPLETED)
                # Observe every currently finished failure before launching more.
                for task in sorted(finished,key=lambda task:active[task]):
                    index=active.pop(task);results[index]=task.result()
                check_time(model.deadline)
                on_result([row for i in sorted(results) for row in results[i]])
    finally:
        for task in active:task.cancel()
        if active:await asyncio.gather(*active,return_exceptions=True)
    return [row for i in range(len(jobs)) for row in results[i]]


def execute(model,jobs,workers,on_result):
    try:asyncio.get_running_loop()
    except RuntimeError:pass
    else:raise RuntimeError('Use await execute_async inside an active event loop')
    return asyncio.run(execute_async(model,jobs,workers,on_result))
