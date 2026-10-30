"""Two independent extraction calls per wave; one ordered audit writer."""
import asyncio
import time
import httpx
from .common import write,digest
from .fact_schedule import order_jobs,time_reserve
from .model import input_tokens
from .roster_budget import prediction_attempt_reserve
from .stage_deadline import wall_clock_budget


def single_packets(jobs,prompt,mode):
    return [{'kind':'single','jobs':[j],'original_jobs':[j],'system':prompt,
             'payload':j['payload'],'input_tokens':input_tokens(prompt,j['payload']),
             'bank':None} for j in order_jobs(jobs,mode)]


class FactReservation:
    """Called under Model's admission lock, with one state per active packet."""
    def __init__(self,states,index,entities,covered,prediction_reserve,grouped):
        self.states,self.index=states,index
        self.entities,self.covered=entities,covered
        self.prediction_reserve,self.grouped=prediction_reserve,grouped

    def reserve(self):
        credited=set(self.covered)|self.states[self.index]['entities']
        for state in self.states:
            if state['admitted']:credited.update(state['entities'])
        unseen=self.entities-credited
        reserve=self.prediction_reserve+(0 if self.grouped else len(unseen))
        for index,state in enumerate(self.states):
            if index==self.index or state['terminal']:continue
            # Preserve the peer's first try + one repair, net of its first try
            # already represented by an uncovered-entity reservation.
            counted=0 if self.grouped else len(state['entities'] & unseen)
            reserve+=max(0,2-state['attempts']-counted)
        return reserve

    def admitted(self):
        self.states[self.index]['admitted']=True
        self.states[self.index]['attempts']+=1

    def finished(self):self.states[self.index]['terminal']=True


def packet_validator(packet,validators):
    def validate(data):
        if packet['kind']=='single':
            entity=packet['jobs'][0]['entity']
            return {entity:validators[entity](data)}
        rows=data.get('entity_results') if isinstance(data,dict) else None
        if (not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows)
            or sorted(r.get('entity_id','') for r in rows)!=sorted(validators)):
            raise ValueError('entity_results must cover exactly '+str(list(validators)))
        outputs={};errors=[]
        for row in rows:
            entity=row['entity_id']
            try:outputs[entity]=validators[entity](row)
            except (ValueError,KeyError,TypeError) as exc:
                errors.append(entity+': '+str(exc)[:140])
        if errors:raise ValueError('Entity fact validation: '+'; '.join(errors))
        return outputs
    return validate


async def run_wave(model,calls,deadline,outcomes):
    async def one(index,call):
        try:
            outcomes[index]=await model.acall('facts',call['system'],call['payload'],call['validator'],
                request_budget=call['budget'],deadline=deadline)
        except (ValueError,RuntimeError,TimeoutError,httpx.TransportError) as exc:
            outcomes[index]=exc
        finally:call['budget'].finished()
    tasks=[asyncio.create_task(one(i,c)) for i,c in enumerate(calls)]
    try:await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)


def extract(self,packets,validator_factory,table_reader,bound_eps_audit,*,grouped=False):
    if self.config.get('fact_memory'):raise ValueError('async facts cannot use dependent fact_memory')
    result={'facts':[],'validation':[],'gaps':[],'conflicts':[],'coverage':[],'schedule':[],'source_assessments':[]}
    entities=set(self.inputs.entities);pred_reserve=prediction_attempt_reserve(len(entities),self.config.get('prediction_batch_size',4))
    mode='grouped_coverage_first' if grouped else 'coverage_first'
    if grouped:
        write(self.output/'grouped-facts-plan.json',{'groups':[
            {'entities':[j['entity'] for j in p['jobs']],
             'windows':[{'entity_id':j['entity'],'batch':j['batch_no'],
                         'chunk_ids':[c['chunk_id'] for c in j['batch']]} for j in p['jobs']],
             'kind':p['kind'],'input_tokens':p['input_tokens'],
             'separate_input_tokens':p.get('separate_input_tokens')} for p in packets],
            'prediction_request_reserve':pred_reserve,'source_windows_dropped_by_packing':0})
    write(self.output/'fact-execution.json',{'workers':2,'transport':'native_async_httpx',
        'policy':'two-packet waves; stable planned-order merge','packets':len(packets),
        'grouped':grouped,'prediction_request_reserve':pred_reserve})
    start=0;wave_number=0
    while start<len(packets):
        covered={f['entity_id'] for f in result['facts']}
        wave=packets[start:start+2]
        pending_entities={j['entity'] for p in wave for j in p['jobs']}
        tail_reserve=pred_reserve+(0 if grouped else len(entities-covered-pending_entities))
        # When two repairs cannot fit, run the earlier packet first. Otherwise
        # the fastest network response would win the last repair allowance.
        if self.model.max_requests-self.model.requests-tail_reserve<2*len(wave):wave=wave[:1]
        estimate=time_reserve(self.model.records,len(entities),batch_size=self.config.get('prediction_batch_size',4),reserve_prediction=not self.config.get('prediction_time_reserved_externally',False))
        limit=min(self.deadline,self.model.deadline)-estimate['prediction_reserve_seconds']
        allowance=limit-time.time()
        states=[{'entities':{j['entity'] for j in p['jobs']},'admitted':False,
                 'attempts':0,'terminal':False} for p in wave]
        calls=[];details=[]
        for i,packet in enumerate(wave):
            gi=start+i
            budget=FactReservation(states,i,entities,covered,pred_reserve,grouped)
            audits=[{'entity_id':j['entity'],'batch':j['batch_no'],
                     'chunk_ids':[c['chunk_id'] for c in j['batch']],
                     'reserved_requests':budget.reserve(),**estimate,
                     'facts_allowance_seconds':allowance,'wave':wave_number} for j in packet['jobs']]
            if grouped:
                for a in audits:a['group']=gi
            if allowance<estimate['estimated_call_seconds']+5:
                result['gaps'].extend({**a,'reason':'facts_time_reserved_for_prediction'} for a in audits)
                result['schedule'].extend({**a,'status':'skipped_time_budget'} for a in audits)
                continue
            if packet['bank'] is not None:
                write(self.output/'fact-source-banks'/f'{gi:03d}.json',packet['bank'])
                write(self.output/'fact-source-banks'/f'{gi:03d}-audit.json',packet['bank_audit'])
            for job in packet['original_jobs']:
                for rid,info in job['payload']['table_reads'].items():write(info['read_artifact'],table_reader(rid))
                if not grouped:
                    write(self.output/'facts-inputs'/f"{digest(job['entity'])[:16]}-{job['batch_no']:03d}.json",
                        {'entity_id':job['entity'],'batch':job['batch_no'],
                         'estimated_input_tokens':packet['input_tokens'],
                         'input_limit':self.config.get('context_window',32000)-8600,
                         'output_reserve':4000,'repair_reserve':4600,
                         'source_packing_budget':self.config.get('facts_source_token_budget',16000),
                         'chunk_ids':[c['chunk_id'] for c in job['batch']]})
            local_audit={'enabled':bound_eps_audit['enabled'],'checks':[]}
            validators={j['entity']:validator_factory(j,local_audit) for j in packet['jobs']}
            calls.append({'system':packet['system'],'payload':packet['payload'],
                          'validator':packet_validator(packet,validators),'budget':budget})
            details.append((packet,audits,local_audit))
        outcomes=[None]*len(calls)
        if calls:
            try:
                with wall_clock_budget(max(.001,limit-time.time())):
                    asyncio.run(run_wave(self.model,calls,limit,outcomes))
            except TimeoutError:
                for i,value in enumerate(outcomes):
                    if value is None:outcomes[i]=TimeoutError('fact_wave_deadline')
        for (packet,audits,local_audit),outputs in zip(details,outcomes):
            bound_eps_audit['checks'].extend(local_audit['checks'])
            if bound_eps_audit['enabled']:write(self.output/'source-bound-eps.json',bound_eps_audit)
            if isinstance(outputs,Exception):
                result['gaps'].extend({**a,'reason':'fact_extraction_failed',
                    'error':type(outputs).__name__+': '+str(outputs)[:180]} for a in audits)
                result['schedule'].extend({**a,'status':'extraction_failed'} for a in audits)
                continue
            if outputs is None:raise RuntimeError('fact_wave_missing_terminal_result')
            for job,audit in zip(packet['jobs'],audits):
                item=outputs[job['entity']]
                result['facts'].extend(item['facts']);result['validation'].extend(item['rejected'])
                result['source_assessments'].extend({'entity_id':job['entity'],'batch':job['batch_no'],**row}
                    for row in item.get('source_assessments',[]))
                result['gaps'].extend(item['unresolved_questions']);result['conflicts'].extend(item['conflicts'])
                result['schedule'].append({**audit,'status':'complete','facts':len(item['facts'])})
                result['coverage'].append({**audit,'facts':len(item['facts']),'omitted_by_selection':job['omitted']})
        write(self.output/'facts.json',result['facts'])
        write(self.output/'fact-schedule.json',{'mode':mode,'decisions':result['schedule']})
        start+=len(wave);wave_number+=1
    return result
