"""Bounded independent prediction calls with shared request reservations."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


class Reservations:
    def __init__(self, model, count):
        self.model=model
        self.attempts=[0]*count
        self.done=[False]*count

    def for_job(self,index):
        return Ticket(self,index)

    def finish(self,index):
        with self.model._lock:self.done[index]=True


class Ticket:
    def __init__(self,ledger,index):self.ledger,self.index=ledger,index

    def reserve(self):
        # Model.call holds the SAME admission lock for reserve + admitted.
        return sum(max(0,2-n) for i,n in enumerate(self.ledger.attempts)
                   if i!=self.index and not self.ledger.done[i])

    def admitted(self):
        self.ledger.attempts[self.index]+=1


def execute(model,jobs,workers,on_result):
    """Return results in input order. Never publish a partial successful roster.

    At most two futures are submitted. A failure stops new submissions; already
    running HTTP calls retain the original deadline and join before Model.close.
    The callback runs on the coordinator only, with successful ordered rows.
    """
    if type(workers) is not int or workers not in (1,2):
        raise ValueError('prediction_workers must be 1 or 2')
    if not jobs:return []
    ledger=Reservations(model,len(jobs));results={};next_job=0
    def run(index):
        job=jobs[index]
        try:
            return model.call('predict',job['system'],job['payload'],job['validator'],
                              request_budget=ledger.for_job(index))
        finally:ledger.finish(index)
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='prediction') as pool:
        active={}
        while next_job<len(jobs) and len(active)<workers:
            active[pool.submit(run,next_job)]=next_job;next_job+=1
        try:
            while active:
                finished,_=wait(active,return_when=FIRST_COMPLETED)
                # Validate all finished tasks before submitting new work.
                for future in sorted(finished,key=lambda f:active[f]):
                    index=active.pop(future);results[index]=future.result()
                ordered=[row for i in sorted(results) for row in results[i]]
                on_result(ordered)
                while next_job<len(jobs) and len(active)<workers:
                    active[pool.submit(run,next_job)]=next_job;next_job+=1
        except BaseException:
            for future in active:future.cancel()
            raise
    return [row for i in range(len(jobs)) for row in results[i]]
