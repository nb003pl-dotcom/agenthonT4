"""Fact-stage execution, with opt-in fair ordering and forecast-time reserves."""
import json
import time
import httpx
from .common import write,digest
from .facts import fact_payload,extraction_batches,validate_facts,compact
from .model import input_tokens
from .selection import select
from .fact_schedule import order_jobs,time_reserve
from .stage_deadline import wall_clock_budget
from t4_splitter import count_tokens


def extract_stage(self,kept,complete,candidates,ids,model_choices):
    from .workflow import required_fact_reserve,prediction_attempt_reserve
    mode=self.config.get('facts_schedule','sequential')
    adaptive=self.roster_plan['adaptive']
    transient_fallback=bool(self.config.get('facts_transient_fallback',False))
    degraded_fallback=transient_fallback or bool(self.config.get('under25_strategy'))
    if transient_fallback and mode!='sequential':
        raise ValueError('facts_transient_fallback_requires_sequential')
    fallback_eligible=set()
    jobs=[];selected_by_entity={};schedule=[]
    facts=[];validation=[];gaps=[];conflicts=[];coverage=[];selection_audit={};table_cache={}
    prompt=self.prompt('facts-extractor-catalog' if self.config.get('compact_model_outputs') else 'facts-extractor')
    if self.config.get('fact_source_guard'):
        from .fact_source_guard import EXTRACTION_INSTRUCTION
        prompt+='\n'+EXTRACTION_INSTRUCTION
    if self.financial_scope:
        from .financial_scope import FACTS
        prompt+='\n'+FACTS
    reaction=bool(getattr(self,'reaction_evidence',False))
    if reaction:
        from .reaction_evidence import FACTS
        prompt+='\n'+FACTS
    accounting_bridge=self.financial_scope and self.config.get('accounting_bridge',False)
    if accounting_bridge:
        from .accounting_bridge import EXTRACT
        prompt=prompt.replace('up to four decisive facts','up to six decisive facts')+'\n'+EXTRACT
    if self.config.get('joint_relevance_facts'):
        from .joint_evidence import prompt_suffix
        prompt+='\n'+prompt_suffix(getattr(self,'profile',None),self.contract.get('target',{}).get('type'))
    from .output_preflight import apply as apply_output_preflight
    prompt=apply_output_preflight(self.config,'facts',prompt)
    def table_reader(rid):
        if rid not in table_cache:
            table_cache[rid]=self.retrieval.table(rid)
        return table_cache[rid]
    bound_eps_report={'records':[],'conflicts':[]};bound_eps_audit={'enabled':bool(self.config.get('source_bound_eps')),'checks':[]}
    if bound_eps_audit['enabled']:
        from .eps_source_frame import build as build_eps_frame
        started=time.monotonic()
        try:
            allowance=min(5.0,self.deadline-time.time()-2.0)
            if allowance<=0:raise TimeoutError('source_bound_eps_no_time')
            with wall_clock_budget(allowance):bound_eps_report=build_eps_frame(self.inputs)
            bound_eps_audit['status']='parsed'
        except (ValueError,RuntimeError,TimeoutError) as exc:
            bound_eps_audit.update(status='fallback',error_type=type(exc).__name__)
        bound_eps_audit['parse_seconds']=time.monotonic()-started
        write(self.output/'source-bound-eps.json',bound_eps_audit)
    source_priorities=getattr(self,'source_priorities',{})
    source_priority_audit=getattr(self,'source_priority_audit',{'enabled':False,'entities':{}})
    # Optional review must never displace mandatory entity coverage.
    for entity_index,(entity,rows) in enumerate(kept.items()):
        if complete is not None:
            selected=rows
            selection_audit[entity]={'selected':[{'chunk_id':r['chunk_id'],'reason':'complete_authorized_corpus'} for r in selected],
                'omitted':[],'policy':'complete_corpus_no_topk'}
        elif self.policy=='legacy':
            selected=sorted(rows,key=lambda r:(r['relevance_status']!='relevant',-r['score']))[:5]
            selection_audit[entity]={'omitted':[r['chunk_id'] for r in rows if r not in selected],'policy':'legacy_top5'}
        elif getattr(self,'subquery_evidence',False):
            selected=[rows[j] for j in model_choices[entity]['indices']]
            selection_audit[entity]=self.need_selection_audit[entity]
            for q in selection_audit[entity]['uncovered_need_ids']:
                gaps.append({'entity_id':entity,'query_id':q,'reason':'subquery_not_covered_in_selected_sources'})
        elif entity in model_choices:
            choice=model_choices[entity];selected=[rows[j] for j in choice['indices']]
            selection_audit[entity]={'selected':[{'chunk_id':r['chunk_id'],'reason':'model_preview_selection'} for r in selected],
                'omitted':[r['chunk_id'] for r in rows if r not in selected],'rationale':choice['rationale']}
        else:selected,selection_audit[entity]=select(rows,self.contract,cap=self.config.get('evidence_cap',5))
        if source_priority_audit['enabled'] and not getattr(self,'subquery_evidence',False):
            from .source_priority import prioritize
            selected,priority_audit=prioritize(self.inputs,entity,rows,selected,source_priorities)
            source_priority_audit['entities'][entity]=priority_audit
            protected={cid for p in priority_audit['priorities'] for cid in p.get('chunk_ids',[])}
            prior_reasons={r['chunk_id']:r.get('reason','retained_selection') for r in selection_audit[entity].get('selected',[])}
            selection_audit[entity]['selected']=[{'chunk_id':r['chunk_id'],
                'reason':'verified_source_priority' if r['chunk_id'] in protected else prior_reasons.get(r['chunk_id'],'retained_selection')} for r in selected]
            selection_audit[entity]['omitted']=[r['chunk_id'] for r in rows if r not in selected]
            selection_audit[entity]['source_priority']=priority_audit
            write(self.output/'source-priority.json',source_priority_audit)
        omitted=selection_audit[entity]['omitted']
        write(self.output/'selection-audit.json',selection_audit)
        # Reserve output plus room for one schema repair response/message.
        input_limit=self.config.get('context_window',32000)-4000-4600
        from .subquery_evidence import neutral_source_batch
        build_payload_raw=lambda batch:fact_payload(self.inputs.task,self.inputs.entities[entity],neutral_source_batch(batch) if self.config.get('neutral_extraction',getattr(self,'subquery_evidence',False)) else batch,self.output,
            table_reader=table_reader if self.policy!='legacy' else None,contract=self.contract,legacy=self.policy=='legacy')
        def build_payload(batch):
            p=build_payload_raw(batch)
            if self.config.get('compact_model_outputs'):
                from .budgets import fact_limit
                max_facts=fact_limit(self.config,6 if accounting_bridge else 4 if self.financial_scope or reaction else 3)
                p['limits'].update(max_facts=max_facts,quote_max_characters=350)
                p['instruction']+=f' Compact output: retain the {max_facts} most decisive distinct facts, favoring target numeric anchor, driver/guidance and material counterevidence. One short exact citation per fact unless a separate header is necessary. Omit unknown/null optional fields; no essay. This limits the extracted fact set, not the available corpus.'
            if not self.config.get('compact_model_outputs') and self.config.get('facts_max_per_window') is not None:
                from .budgets import fact_limit
                p['limits']['max_facts']=fact_limit(self.config,p['limits']['max_facts'])
            if self.config.get('joint_relevance_facts'):
                from .joint_evidence import add_contract
                p=add_contract(p,self.subquery_needs[entity],batch)
            elif getattr(self,'subquery_evidence',False) and self.config.get('facts_need_context',False):
                from .subquery_evidence import add_need_context
                p=add_need_context(p,self.subquery_needs[entity],batch)
            prior_facts=getattr(self,'supplement_primary_facts',None)
            if prior_facts is not None:
                p['existing_entity_facts']=[{k:f[k] for k in ('fact_id','statement') if k in f}
                    for f in prior_facts if f['entity_id']==entity]
                p['instruction']+=(' Supplemental focused read: use the information needs to extract missing '
                    'explanations, comparability or uncertainty evidence. Existing facts are unverified context, '
                    'not source support. Avoid repeating them; return an empty facts list if this window '
                    'adds no supported information. Cite the provided original source, never the fact summary.')
            if self.config.get('compact_model_outputs'):
                from .citation_catalog import annotate_payload
                p,_=annotate_payload(p)
            return p
        group_instruction=''
        if adaptive:
            from .grouped_facts import INSTRUCTION
            group_instruction=INSTRUCTION
        batches=list(extraction_batches(selected,
            max_tokens=self.config.get('facts_source_token_budget',16000),
            fits=lambda batch:input_tokens(prompt+group_instruction,{'entity_batches':[build_payload(batch)]} if adaptive else build_payload(batch))<=input_limit))
        selected_by_entity[entity]=selected
        for batch_no,batch in enumerate(batches):
            payload=build_payload(batch)
            quote_catalog={}
            if self.config.get('compact_model_outputs'):
                from .citation_catalog import annotate_payload
                _,quote_catalog=annotate_payload(build_payload_raw(batch))
            jobs.append({'entity':entity,'entity_index':entity_index,'batch_no':batch_no,
                         'batch':batch,'payload':payload,'quote_catalog':quote_catalog,'omitted':omitted})
    if self.config.get('decisive_source_readback'):
        self.readback_selected_sources=selected_by_entity
    async_facts=self.config.get('facts_workers',1)==2
    if async_facts:
        from .async_fact_execution import extract, single_packets
        from .fact_validator import validator_for_job
        packets=None if adaptive else single_packets(jobs,prompt,mode)
        if adaptive:
            from .grouped_facts import prepare_packets
            packets=prepare_packets(self,jobs,prompt)
        result=extract(self,packets,
            lambda job,audit:validator_for_job(self,job,bound_eps_report,audit,write_audit=False),
            table_reader,bound_eps_audit,grouped=adaptive)
        facts=result['facts'];validation=result['validation'];gaps.extend(result['gaps'])
        conflicts=result['conflicts'];coverage=result['coverage'];schedule=result['schedule']
        if self.config.get('joint_relevance_facts'):
            self.joint_assessments.extend(result.get('source_assessments',[]))
        if adaptive:mode='grouped_coverage_first'
    elif adaptive:
        from .grouped_facts import extract
        from .fact_validator import validator_for_job
        result=extract(self,jobs,prompt,
            lambda job:validator_for_job(self,job,bound_eps_report,bound_eps_audit),table_reader)
        facts=result['facts'];validation=result['validation'];gaps.extend(result['gaps'])
        conflicts=result['conflicts'];coverage=result['coverage'];schedule=result['schedule']
        if self.config.get('joint_relevance_facts'):
            self.joint_assessments.extend(result.get('source_assessments',[]))
        mode='grouped_coverage_first'
    for job in ([] if adaptive or async_facts else order_jobs(jobs,mode)):
        entity=job['entity'];entity_index=job['entity_index'];batch_no=job['batch_no']
        batch=job['batch'];payload=job['payload'];quote_catalog=job['quote_catalog'];omitted=job['omitted']
        covered={f['entity_id'] for f in facts}
        reserve=required_fact_reserve(len(ids),entity_index,self.config.get('prediction_batch_size',4)) if mode=='sequential' else prediction_attempt_reserve(len(ids),self.config.get('prediction_batch_size',4))+len(set(ids)-covered-{entity})
        audit={'entity_id':entity,'batch':batch_no,'chunk_ids':[c['chunk_id'] for c in batch],
               'reserved_requests':reserve}
        if self.model.requests>=self.model.max_requests-reserve:
            gaps.append({**audit,'reason':'facts_request_budget_reserved'})
            schedule.append({**audit,'status':'skipped_request_budget'})
            if transient_fallback:fallback_eligible.add(entity)
            continue
        allowance=None
        if mode=='coverage_first':
            estimate=time_reserve(self.model.records,len(ids),batch_size=self.config.get('prediction_batch_size',4),reserve_prediction=not self.config.get('prediction_time_reserved_externally',False))
            allowance=self.deadline-time.time()-estimate['prediction_reserve_seconds']
            audit.update(estimate,remaining_seconds=self.deadline-time.time(),facts_allowance_seconds=allowance)
            if allowance<estimate['estimated_call_seconds']+5:
                gaps.append({**audit,'reason':'facts_time_reserved_for_prediction'})
                schedule.append({**audit,'status':'skipped_time_budget'});continue
        for rid,table_info in payload['table_reads'].items():
            table=table_reader(rid)
            write(table_info['read_artifact'],table)
        write(self.output/'facts-inputs'/f'{digest(entity)[:16]}-{batch_no:03d}.json',
              {'entity_id':entity,'batch':batch_no,'estimated_input_tokens':input_tokens(prompt,payload),
               'input_limit':input_limit,'output_reserve':4000,'repair_reserve':4600,
               'source_packing_budget':self.config.get('facts_source_token_budget',16000),
               'chunk_ids':[c['chunk_id'] for c in batch]})
        validation_attempt={'count':0}
        def validate(d):
            validation_attempt['count']+=1
            assessments=None
            if self.config.get('joint_relevance_facts'):
                from .joint_evidence import validate_assessments
                assessments=validate_assessments(d,payload)
            if self.config.get('compact_model_outputs'):
                from .citation_catalog import restore_quotes
                d=restore_quotes(d,quote_catalog,strict=False)
                for f in d.get('facts',[]):f.pop('_citation_completion',None)
                if self.config.get('citation_completion')=='adjacent' or self.config.get('fact_source_guard'):
                    from .citation_completion import complete_adjacent_citation
                    from .fact_source_guard import complete_guarded_source
                    complete_source=complete_adjacent_citation if self.config.get('citation_completion')=='adjacent' else complete_guarded_source
                    completed=[]
                    for f in d.get('facts',[]):
                        if f.get('_citation_restore_error'):
                            completed.append(f);continue
                        updated,completion=complete_source(f,quote_catalog,
                            [e.get('name','') for e in self.inputs.entities.values()])
                        updated['_citation_completion']=completion;completed.append(updated)
                    d={**d,'facts':completed}
            result=validate_facts(d,self.inputs,entity,batch,trust_catalog=self.config.get('compact_model_outputs',False),
                                  source_guard=self.config.get('fact_source_guard',False),
                              max_facts=payload['limits']['max_facts'] if getattr(self,'subquery_evidence',False) else 12)
            if bound_eps_audit['enabled']:
                from .source_bound_eps import bind
                binding=bind(self.inputs,bound_eps_report,entity,batch,result['facts'])
                result['facts']=binding['facts']
                bound_eps_audit['checks'].append({'entity_id':entity,'batch':batch_no,
                    'validation_attempt':validation_attempt['count'],'audit':binding['audit']})
                write(self.output/'source-bound-eps.json',bound_eps_audit)
            from .facts import handle_rejected_window
            result=handle_rejected_window(result,bool(d.get('facts')),
                validation_attempt['count'],entity,batch_no)
            if assessments is not None:
                self.joint_assessments.extend({'entity_id':entity,'batch':batch_no,**row}
                    for row in assessments)
            return result
        original_deadline=self.model.deadline
        try:
            if allowance is None:
                result=self.model.call('facts',prompt,payload,validate,reserve=reserve)
            else:
                self.model.deadline=min(original_deadline,time.time()+allowance)
                with wall_clock_budget(allowance):
                    result=self.model.call('facts',prompt,payload,validate,reserve=reserve)
        except (ValueError,RuntimeError,TimeoutError,httpx.TransportError) as exc:
            if mode=='sequential':
                from .fact_failures import recoverable_fact_failure
                if not transient_fallback or not recoverable_fact_failure(exc):raise
                fallback_eligible.add(entity)
            if isinstance(exc,(TimeoutError,httpx.TransportError)):self.model.close()
            gaps.append({**audit,'reason':'fact_extraction_failed','error':type(exc).__name__+': '+str(exc)[:180]})
            schedule.append({**audit,'status':'extraction_failed'})
            continue
        finally:
            self.model.deadline=original_deadline
            write(self.output/'fact-schedule.json',{'mode':mode,'decisions':schedule})
        schedule.append({**audit,'status':'complete','facts':len(result['facts'])})
        write(self.output/'fact-schedule.json',{'mode':mode,'decisions':schedule})
        facts.extend(result['facts']);validation.extend(result['rejected'])
        gaps.extend(result['unresolved_questions']);conflicts.extend(result['conflicts'])
        coverage.append({'entity_id':entity,'batch':batch_no,'chunk_ids':[c['chunk_id'] for c in batch],
                         'facts':len(result['facts']),'omitted_by_selection':omitted})
        write(self.output/'facts.json',facts)
        if mode=='sequential' and not any(f['entity_id']==entity for f in facts):
            if not any(j['entity']==entity and j['batch_no']>batch_no for j in jobs):
                if degraded_fallback:fallback_eligible.add(entity)
                elif entity not in fallback_eligible:raise ValueError('no_supported_facts_for_'+entity)
    if mode in ('coverage_first','grouped_coverage_first') or degraded_fallback:
        from .source_fallback import source_facts
        queries=json.loads((self.output/'query-plan.json').read_text())['queries']
        for entity in ids:
            if any(f['entity_id']==entity for f in facts):continue
            if mode=='sequential' and entity not in fallback_eligible:continue
            result=source_facts(self.inputs,entity,selected_by_entity[entity],queries)
            facts.extend(result['facts']);validation.extend(result['rejected'])
            gaps.append({'entity_id':entity,'reason':'structured_extraction_unavailable_using_verbatim_excerpts'})
            schedule.append({'entity_id':entity,'status':'verbatim_source_fallback','facts':len(result['facts'])})
    write(self.output/'fact-schedule.json',{'mode':mode,'decisions':schedule})
    missing=[e for e in ids if not any(f['entity_id']==e for f in facts)]
    if missing:raise ValueError('no_supported_facts_for_'+str(missing))
    facts=list({f['fact_id']:f for f in facts}.values());fact_map={f['fact_id']:f for f in facts}
    write(self.output/'facts.json',facts)
    source_tokens=sum(count_tokens(c['text']) for rows_ in candidates.values() for c in rows_)
    fact_tokens=count_tokens(json.dumps(compact(facts),ensure_ascii=False))
    write(self.output/'fact-token-audit.json',{'candidate_source_tokens':source_tokens,
        'fact_package_tokens':fact_tokens,'ratio':fact_tokens/max(1,source_tokens),
        'notice':'Not total workflow savings; short evidence can grow after structuring.'})
    write(self.output/'citation-map.json',{c['citation_id']:c for f in facts for c in f['citations']})
    write(self.output/'fact-validation.json',validation)
    write(self.output/'fact-conflicts.json',conflicts)
    write(self.output/'facts-coverage.json',{'batches':coverage,'gaps':gaps})
    return facts,fact_map,conflicts,gaps
