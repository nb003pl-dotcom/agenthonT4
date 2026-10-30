"""Pack independent entity windows without changing their source or quote scope."""
import time
import httpx
from .common import write
from .model import input_tokens
from .fact_schedule import order_jobs,time_reserve
from .stage_deadline import wall_clock_budget
from .roster_budget import prediction_attempt_reserve

INSTRUCTION='''\nBATCH CONTRACT: Apply the fact-extraction instructions independently to every entry in entity_batches. Return only {"entity_results":[{"entity_id":"...","source_assessments":[...],"facts":[...],"conflicts":[],"unresolved_questions":[]},...]}, one result per entry. Include source_assessments only when the entry requests the joint relevance + fact contract. Quote IDs are LOCAL to each entry; never borrow a quote, chunk or fact from another entry. Preserve each entry's entity, metric, unit and period. At most three decisive facts per entity, at most 200 characters per statement; omit null optional fields. Keep all entities within the 4000-token response budget. An empty facts list means no supported fact was extracted, not that no evidence exists.'''


def payload(group):
    return {'entity_batches':[{**j['payload'],
        'limits':{**j['payload'].get('limits',{}),'max_facts':3}} for j in group]}


def pack(jobs,system,limit):
    groups=[];current=[]
    for job in order_jobs(jobs,'coverage_first'):
        proposed=current+[job]
        if current and (len(proposed)>3 or job['entity'] in {j['entity'] for j in current}
                        or input_tokens(system,payload(proposed))>limit):
            groups.append(current);current=[]
        if input_tokens(system,payload([job]))>limit:
            raise ValueError('grouped_fact_metadata_exceeds_context_budget')
        current.append(job)
    if current:groups.append(current)
    return groups


def prepare_packets(self,jobs,prompt):
    system=prompt+INSTRUCTION
    limit=self.config.get('context_window',32000)-8600
    source_bank=bool(self.config.get('fact_source_bank'))
    if source_bank:
        from .fact_packing import pack as pack_sources
        packets=pack_sources(jobs,prompt,limit,self.inputs,self.config.get('model',''))
    else:
        packets=[{'kind':'grouped','jobs':g,'original_jobs':g,'system':system,'payload':payload(g),
                  'input_tokens':input_tokens(system,payload(g)),'bank':None}
                 for g in pack(jobs,system,limit)]
    # Validate against the limits actually sent after grouped packing, while
    # retaining original jobs for source-read artifacts and packing audits.
    for packet in packets:
        sent=packet['payload'].get('entity_batches')
        if sent is not None:
            if len(sent)!=len(packet['jobs']):raise ValueError('grouped_payload_job_mismatch')
            packet['jobs']=[{**job,'payload':{**job['payload'],'limits':dict(row['limits'])}}
                            for job,row in zip(packet['jobs'],sent)]
    return packets


def extract(self,jobs,prompt,validator_factory,table_reader):
    limit=self.config.get('context_window',32000)-8600
    packets=prepare_packets(self,jobs,prompt)
    result={'facts':[],'validation':[],'gaps':[],'conflicts':[],'coverage':[],'schedule':[],'source_assessments':[]}
    reserve=prediction_attempt_reserve(len(self.inputs.entities),self.config.get('prediction_batch_size',4))
    write(self.output/'grouped-facts-plan.json',{'groups':[
        {'entities':[j['entity'] for j in p['jobs']], 'windows':[{'entity_id':j['entity'],'batch':j['batch_no'],
         'chunk_ids':[c['chunk_id'] for c in j['batch']]} for j in p['jobs']],
         'kind':p['kind'],'input_tokens':p['input_tokens'],
         'separate_input_tokens':p.get('separate_input_tokens')} for p in packets],
        'prediction_request_reserve':reserve,'source_windows_dropped_by_packing':0})
    for gi,packet in enumerate(packets):
        group=packet['jobs'];system=packet['system']
        if packet['bank'] is not None:
            write(self.output/'fact-source-banks'/f'{gi:03d}.json',packet['bank'])
            write(self.output/'fact-source-banks'/f'{gi:03d}-audit.json',packet['bank_audit'])
        audits=[{'entity_id':j['entity'],'batch':j['batch_no'],'group':gi,
                 'chunk_ids':[c['chunk_id'] for c in j['batch']]} for j in group]
        reason=None
        estimate=time_reserve(self.model.records,len(self.inputs.entities),batch_size=self.config.get('prediction_batch_size',4),reserve_prediction=not self.config.get('prediction_time_reserved_externally',False))
        allowance=self.deadline-time.time()-estimate['prediction_reserve_seconds']
        if self.model.requests>=self.model.max_requests-reserve:reason='facts_request_budget_reserved'
        elif allowance<estimate['estimated_call_seconds']+5:reason='facts_time_reserved_for_prediction'
        if reason:
            result['gaps'].extend({**a,'reason':reason} for a in audits)
            result['schedule'].extend({**a,'status':'skipped','reason':reason} for a in audits)
            continue
        for job in packet['original_jobs']:
            for rid,info in job['payload']['table_reads'].items():write(info['read_artifact'],table_reader(rid))
        validators={j['entity']:validator_factory(j) for j in group}
        def validate(data):
            if packet['kind']=='single':
                entity=group[0]['entity']
                return {entity:validators[entity](data)}
            rows=data.get('entity_results') if isinstance(data,dict) else None
            if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows) or sorted(r.get('entity_id','') for r in rows)!=sorted(validators):
                raise ValueError('entity_results must cover exactly '+str(list(validators)))
            outputs={};errors=[]
            # Visit every entity on every response. Stopping at the first bad
            # entity could give a later one its *first* validation only on the
            # last repair, discarding already-valid entities with the group.
            for row in rows:
                entity=row['entity_id']
                try:outputs[entity]=validators[entity](row)
                except (ValueError,KeyError,TypeError) as exc:
                    errors.append(entity+': '+str(exc)[:140])
            if errors:raise ValueError('Entity fact validation: '+'; '.join(errors))
            return outputs
        original=self.model.deadline
        request_payload=packet['payload']
        if self.config.get('fact_memory',False):
            from .fact_memory import augment
            from .fact_packing import tokens
            system,request_payload,memory_audit=augment(system,request_payload,result['facts'],
                [j['entity'] for j in group],limit,
                lambda s,p:tokens(s,p,self.config.get('model','')))
            write(self.output/'fact-memory'/f'{gi:03d}.json',memory_audit)
        try:
            self.model.deadline=min(original,time.time()+allowance)
            with wall_clock_budget(allowance):
                outputs=self.model.call('facts',system,request_payload,validate,reserve=reserve)
        except (ValueError,RuntimeError,TimeoutError,httpx.TransportError) as exc:
            if isinstance(exc,(TimeoutError,httpx.TransportError)):self.model.close()
            result['gaps'].extend({**a,'reason':'grouped_fact_extraction_failed','error_type':type(exc).__name__} for a in audits)
            result['schedule'].extend({**a,'status':'extraction_failed'} for a in audits)
            continue
        finally:self.model.deadline=original
        for job,audit in zip(group,audits):
            item=outputs[job['entity']]
            result['facts'].extend(item['facts']);result['validation'].extend(item['rejected'])
            result['source_assessments'].extend({'entity_id':job['entity'],'batch':job['batch_no'],**row}
                for row in item.get('source_assessments',[]))
            result['gaps'].extend(item['unresolved_questions']);result['conflicts'].extend(item['conflicts'])
            result['schedule'].append({**audit,'status':'complete','facts':len(item['facts'])})
            result['coverage'].append({**audit,'facts':len(item['facts']),'omitted_by_selection':job['omitted']})
        write(self.output/'facts.json',result['facts'])
        write(self.output/'fact-schedule.json',{'mode':'grouped_coverage_first','decisions':result['schedule']})
    return result
