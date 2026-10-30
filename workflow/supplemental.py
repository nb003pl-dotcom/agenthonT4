"""One bounded evidence-gap/search round; never replaces first-pass facts."""
from copy import copy
import json
import time

from .common import write, read
from .model import input_tokens


def _canonical(text):return ' '.join(str(text).lower().split())


def validate_plan(data, inputs, facts, existing, max_queries=2, max_entities=2, *, require_retrievable=False,
                  omit_invalid_fact_ownership=False):
    if not isinstance(data,dict) or any(k in data for k in ('predictions','point_forecast','label','interval','class_probabilities')):
        raise ValueError('supplement plan must not contain predictions')
    rows=data.get('queries')
    if not isinstance(rows,list) or len(rows)>max_queries:raise ValueError('supplement query cap exceeded')
    known={f['fact_id']:f['entity_id'] for f in facts}
    seen={(e,_canonical(q.get('dense_query',q.get('query',''))))
          for q in existing for e in q.get('entity_ids',list(inputs.entities))}
    result=[];omitted=[];entities=set()
    for i,row in enumerate(rows):
        if not isinstance(row,dict) or any(k in row for k in ('point_forecast','label','interval','class_probabilities')):
            raise ValueError('invalid supplemental query')
        entity=row.get('entity_id');text=row.get('query');refs=row.get('fact_ids')
        if entity not in inputs.entities:raise ValueError('supplement entity not allowed')
        if not isinstance(text,str) or not 0<len(text.strip())<=200:raise ValueError('invalid supplemental query text')
        if not isinstance(refs,list) or not 1<=len(refs)<=3 or any(known.get(fid)!=entity for fid in refs):
            if omit_invalid_fact_ownership:
                omitted.append({'entity_id':entity,'query':str(text or '')[:200],
                                'reason':'fact_ids_not_owned_by_entity'})
                continue
            raise ValueError('supplement fact references must belong to its entity')
        targets=row.get('targets',[])
        if not isinstance(targets,list) or len(targets)>4 or any(not isinstance(t,dict) for t in targets):
            raise ValueError('invalid supplement targets')
        need=row.get('information_need') or text
        if not isinstance(need,str) or not 0<len(need)<=160:raise ValueError('invalid supplemental information need')
        key=(entity,_canonical(text))
        if key in seen:
            omitted.append({'entity_id':entity,'query':text.strip(),'reason':'already_searched_for_entity'})
            continue
        metadata={}
        if require_retrievable:
            if row.get('evidence_stage') not in {'precutoff_observation','precutoff_expectation','definition_or_scope','historical_mechanism'}:
                raise ValueError('supplement must seek pre-cutoff evidence, not a future realized outcome')
            metadata['evidence_stage']=row['evidence_stage']
            for field in ('precutoff_rationale','potential_impact'):
                value=row.get(field)
                if not isinstance(value,str) or not value.strip():
                    raise ValueError('missing or invalid supplemental '+field)
                value=value.strip()
                if len(value)>240:
                    metadata.setdefault('metadata_normalizations',[]).append(
                        {'field':field,'original_characters':len(value),'retained_characters':240})
                metadata[field]=value[:240]
        seen.add(key);entities.add(entity)
        if len(entities)>max_entities:raise ValueError('supplement entity cap exceeded')
        result.append({**metadata,'query_id':f'supp{i+1:03d}','entity_ids':[entity],
            'query_type':'supplemental_gap','dense_query':text.strip(),'lexical_query':text.strip(),
            'purpose':need,'information_need':need,'targets':targets,'based_on_fact_ids':list(dict.fromkeys(refs))})
    return {'assessment':str(data.get('assessment',''))[:600],'queries':result,'omitted_queries':omitted}



def supplemental_candidates(extra, excluded, *, allow_reread=False):
    """Keep fresh hits and at most one selected-source hit per entity."""
    result={};fresh={};revisits={}
    for entity,rows in extra.items():
        kept=[];seen=set();revisited=False
        for row in rows:
            cid=row['chunk_id']
            if cid in seen:continue
            seen.add(cid)
            if cid in excluded.get(entity,set()):
                if not allow_reread or revisited:continue
                revisited=True;revisits.setdefault(entity,[]).append(cid)
            else:fresh[entity]=fresh.get(entity,0)+1
            kept.append(row)
        if kept:result[entity]=kept
    return result,{'novel_candidates':fresh,'revisit_candidates':revisits}


class _AttemptLimit:
    def __init__(self, model, limit, prediction_reserve, nested=None, input_reserve=0):
        self.model=model;self.limit=limit;self.prediction_reserve=prediction_reserve;self.nested=nested;self.reserved_input=input_reserve
    def reserve(self):
        return max(self.prediction_reserve,self.model.max_requests-self.limit,
                   self.nested.reserve() if self.nested is not None else 0)
    def input_reserve(self):
        return max(self.reserved_input,self.nested.input_reserve() if self.nested is not None and hasattr(self.nested,'input_reserve') else 0)
    def admitted(self):
        if self.nested is not None:self.nested.admitted()


class SupplementalModel:
    """Shares real request accounting; retries/repairs consume the same local cap."""
    def __init__(self, model, attempt_cap, prediction_reserve, deadline, input_reserve=0):
        if type(attempt_cap) is not int or attempt_cap<1:raise ValueError('invalid supplemental attempt cap')
        if type(input_reserve) is not int or input_reserve<0:raise ValueError('invalid prediction input reserve')
        self.parent=model;self.start=model.requests;self.limit=model.requests+attempt_cap;self.reserved_input=input_reserve
        self.prediction_reserve=prediction_reserve;self.deadline=min(model.deadline,deadline)
    def __getattr__(self,name):return getattr(self.parent,name)
    @property
    def records(self):
        return [r for r in self.parent.records if r.get('stage','').startswith('supplement_') and r.get('request',0)>self.start]
    def call(self,stage,system,payload,validator,reserve=0,request_budget=None,*,deadline=None):
        budget=_AttemptLimit(self.parent,self.limit,self.prediction_reserve,request_budget,self.reserved_input)
        return self.parent.call('supplement_'+stage,system,payload,validator,
            reserve=max(reserve,self.prediction_reserve),request_budget=budget,
            deadline=min(self.deadline,deadline) if deadline is not None else self.deadline)
    async def acall(self,stage,system,payload,validator,reserve=0,request_budget=None,*,deadline=None):
        budget=_AttemptLimit(self.parent,self.limit,self.prediction_reserve,request_budget,self.reserved_input)
        return await self.parent.acall('supplement_'+stage,system,payload,validator,
            reserve=max(reserve,self.prediction_reserve),request_budget=budget,
            deadline=min(self.deadline,deadline) if deadline is not None else self.deadline)


def _planning_payload(wf,facts,gaps,queries,max_queries,max_entities):
    rows=[]
    for f in facts:
        fields=('fact_id','entity_id','statement','metric','value','unit','period','basis','evidence_role')
        rows.append({**{k:f[k] for k in fields if k in f},
                     'source_context':[c.get('quote','')[:180] for c in f.get('citations',[])[:2]],
                     'source_dates':sorted({c['doc_date'] for c in f.get('citations',[]) if c.get('doc_date')})})
    return {'task':{k:v for k,v in wf.inputs.task.items() if k!='entities'},
            'target_contract':wf.contract,'entities':list(wf.inputs.entities.values()),
            'facts':rows,'gaps':gaps,'existing_query_plan':queries,
            'limits':{'max_queries':max_queries,'max_entities':max_entities,
                      'max_selected_rereads_per_entity':1 if wf.config.get('supplement_reread_selected',False) else 0},
            'notice':'Quotes are incomplete source context, not corpus absence proof. No outcome/reference data is supplied.'}


def _merge_unique(original,new):
    result=list(original);ids={f['fact_id'] for f in original}
    for fact in new:
        if fact['fact_id'] not in ids and fact.get('extraction_method')!='verbatim_source_fallback':
            result.append(fact);ids.add(fact['fact_id'])
    return result


def _prediction_input_reserve(wf,facts):
    from .roster_budget import prediction_batch_count
    groups=prediction_batch_count(len(wf.inputs.entities),wf.config.get('prediction_batch_size',4))
    primary=len(json.dumps({'task':wf.inputs.task,'contract':wf.contract,'facts':facts,
        'anchors':getattr(wf,'numeric_packets',{})},ensure_ascii=False).encode('utf-8'))
    # Two admissions/group (generation+repair), bounded source readback/context,
    # and headroom for up to two entities of supplemental facts.
    return wf.config.get('supplement_prediction_input_reserve',2*primary+groups*200000+80000)


def _publish_merged(root,output,primary,merged,conflicts,gaps,candidates,novel):
    """Stage derived artifacts first; journal root files and roll back on failure."""
    write(output/'merged-facts.json',merged)
    def existing(name,default):return read(root/name) if (root/name).is_file() else default
    old_coverage=existing('facts-coverage.json',{'batches':[]})
    extra_coverage=read(output/'facts-coverage.json') if (output/'facts-coverage.json').is_file() else {'batches':[]}
    updates={'facts-primary.json':primary,
        'citation-map.json':{c['citation_id']:c for f in merged for c in f.get('citations',[])},
        'fact-conflicts.json':conflicts,
        'fact-validation.json':existing('fact-validation.json',[])+(read(output/'fact-validation.json') if (output/'fact-validation.json').is_file() else []),
        'facts-coverage.json':{'batches':old_coverage.get('batches',[])+extra_coverage.get('batches',[]),'gaps':gaps,'scope':'merged_primary_and_supplement'},
        'evidence-stages.json':{'primary_selection':'selection-audit.json','supplement_selection':'supplement/selection-audit.json',
                              'merged_facts':'facts.json','merged_citations':'citation-map.json'}}
    if (root/'fact-token-audit.json').is_file():
        from .facts import compact
        from t4_splitter import count_tokens
        updates['fact-token-audit-primary.json']=existing('fact-token-audit.json',{})
        pool={(e,r['chunk_id']):r for group in (candidates,novel) for e,rows in group.items() for r in rows}
        source_tokens=sum(count_tokens(r['text']) for r in pool.values())
        fact_tokens=count_tokens(json.dumps(compact(merged),ensure_ascii=False))
        updates['fact-token-audit.json']={'candidate_source_tokens':source_tokens,'fact_package_tokens':fact_tokens,
            'ratio':fact_tokens/max(1,source_tokens),'notice':'Union of candidate nodes across both rounds; not total API input usage.'}
    updates['facts.json']=merged  # Publish the main evidence file last.
    backups={name:(root/name).read_bytes() if (root/name).is_file() else None for name in updates}
    try:
        for name,value in updates.items():write(root/name,value)
    except Exception:
        for name,old in backups.items():
            path=root/name
            if old is None:path.unlink(missing_ok=True)
            else:
                restored=path.with_suffix(path.suffix+'.restore');restored.write_bytes(old);restored.replace(path)
        raise


def supplement(wf,facts,fact_map,conflicts,gaps,queries,candidates):
    """Returns merged evidence, or the exact original evidence after any failure."""
    from .workflow import prediction_attempt_reserve
    from .fact_schedule import time_reserve
    output=wf.output/'supplement';output.mkdir(parents=True,exist_ok=True)
    start_requests=wf.model.requests;started=time.monotonic()
    audit={'status':'not_started','max_rounds':1,'max_queries':wf.config.get('supplement_max_queries',2),
           'max_entities':wf.config.get('supplement_max_entities',2),
           'max_new_chunks_per_entity':wf.config.get('supplement_max_chunks',2),
           'max_attempts':wf.config.get('supplement_max_calls',6),'attempts':0,'new_fact_ids':[],
           'max_selected_rereads_per_entity':1 if wf.config.get('supplement_reread_selected',False) else 0,
           'max_chunks_per_entity':wf.config.get('supplement_max_chunks',2)}
    original=(facts,fact_map,conflicts,gaps)
    try:
        if not 1<=audit['max_queries']<=4 or not 1<=audit['max_entities']<=2 or not 1<=audit['max_new_chunks_per_entity']<=2 or not 1<=audit['max_attempts']<=12:
            raise ValueError('invalid supplemental research bounds')
        reserve=prediction_attempt_reserve(len(wf.inputs.entities),wf.config.get('prediction_batch_size',4))
        estimate=time_reserve(wf.model.records,len(wf.inputs.entities),batch_size=wf.config.get('prediction_batch_size',4))
        deadline=min(wf.deadline-estimate['prediction_reserve_seconds'],time.time()+wf.config.get('supplement_timeout',180))
        if deadline-time.time()<15 or wf.model.requests>=wf.model.max_requests-reserve:
            audit['status']='skipped_final_prediction_reserve';return original
        reserved_input=_prediction_input_reserve(wf,facts)
        audit['prediction_input_reserve']=reserved_input
        if getattr(wf.model,'max_input',None) is not None and wf.model.input_tokens+reserved_input>=wf.model.max_input:
            audit['status']='skipped_final_prediction_input_reserve';return original
        model=SupplementalModel(wf.model,audit['max_attempts'],reserve,deadline,input_reserve=reserved_input)
        payload=_planning_payload(wf,facts,gaps,queries,audit['max_queries'],audit['max_entities'])
        system=wf.prompt('supplement-plan')
        from .output_preflight import apply as apply_output_preflight,enrich_payload
        payload=enrich_payload(wf.config,'supplement_plan',payload)
        system=apply_output_preflight(wf.config,'supplement_plan',system)
        if input_tokens(system,payload)>wf.config.get('context_window',32000)-8600:
            audit['status']='skipped_plan_context_budget';return original
        plan=model.call('plan',system,payload,
            lambda d:validate_plan(d,wf.inputs,facts,queries,audit['max_queries'],audit['max_entities'],
                require_retrievable=True,omit_invalid_fact_ownership=bool(wf.config.get('structured_output_preflight'))),reserve=reserve)
        write(output/'query-plan.json',plan)
        extra_queries=plan['queries']
        if not extra_queries:audit['status']='no_additional_query';return original
        extra,audit_retrieval=wf.retrieval.search(extra_queries,top=2,cap=None,duty_anchors=False)
        ids=list(dict.fromkeys(e for q in extra_queries for e in q['entity_ids']))
        previous=getattr(wf,'readback_selected_sources',{})
        excluded={e:{r['chunk_id'] for r in previous.get(e,[])} for e in ids}
        if (wf.output/'selection-audit.json').is_file():
            selected_audit=read(wf.output/'selection-audit.json')
            for e in ids:excluded[e]|={r['chunk_id'] for r in selected_audit.get(e,{}).get('selected',[])}
        reread=bool(wf.config.get('supplement_reread_selected',False))
        novel,candidate_audit=supplemental_candidates({e:extra.get(e,[]) for e in ids},excluded,allow_reread=reread)
        ids=list(novel);audit.update(candidate_audit)
        write(output/'retrieval-candidates.json',novel);write(output/'retrieval-audit.json',audit_retrieval)
        if not ids:audit['status']='no_additional_sources' if reread else 'no_novel_sources';return original
        local=copy(wf);local.output=output;local.model=model;local.deadline=deadline
        local.inputs=copy(wf.inputs);local.inputs.entities={e:wf.inputs.entities[e] for e in ids}
        local.roster_plan={'adaptive':False};local.subquery_evidence=True;local.policy='coverage'
        local.config={**wf.config,'supplemental_retrieval':False,'evidence_cap':audit['max_new_chunks_per_entity'],
                      'facts_workers':1,'facts_schedule':'coverage_first','prediction_time_reserved_externally':True}
        local.supplement_primary_facts=facts if reread else None
        if reread:local.config['facts_need_context']=True
        local.source_priorities={};local.source_priority_audit={'enabled':False,'entities':{}}
        from .subquery_evidence import judge_and_select
        from .fact_stage import extract_stage
        if local.config.get('joint_relevance_facts'):
            from .joint_evidence import prepare_selection
            from .subquery_evidence import entity_needs,selection_budget
            kept,choices,local.need_selection_audit=prepare_selection(novel,extra_queries,
                cap=local.config['evidence_cap'],token_budget=selection_budget(local.config))
            from .joint_source_context import apply_context
            kept,choices=apply_context(local,kept,choices,max_total=audit['max_new_chunks_per_entity'],
                excluded=excluded)
            local.subquery_needs={e:entity_needs(extra_queries,e) for e in ids}
            local.joint_assessments=[]
            write(output/'need-selection.json',local.need_selection_audit)
        else:
            kept,choices=judge_and_select(local,novel,extra_queries,ids)
        audit['selected_revisited_chunks']={e:[kept[e][i]['chunk_id'] for i in choices[e]['indices']
            if kept[e][i]['chunk_id'] in excluded.get(e,set())] for e in ids}
        extra_facts,_,extra_conflicts,extra_gaps=extract_stage(local,kept,None,novel,ids,choices)
        if local.config.get('joint_relevance_facts'):
            write(output/'need-relevance.json',{'mode':'joint_with_fact_extraction',
                'assessments':local.joint_assessments})
        merged=_merge_unique(facts,extra_facts)
        audit['candidate_fact_ids']=[f['fact_id'] for f in merged if f['fact_id'] not in fact_map]
        if not audit['candidate_fact_ids']:audit['status']='no_validated_new_facts';return original
        combined=None
        if hasattr(local,'readback_selected_sources'):
            combined={e:list(rows) for e,rows in previous.items()}
            for e,rows in local.readback_selected_sources.items():
                seen={r['chunk_id'] for r in combined.get(e,[])}
                combined.setdefault(e,[]).extend(r for r in rows if r['chunk_id'] not in seen)
        _publish_merged(wf.output,output,facts,merged,conflicts+extra_conflicts,gaps+extra_gaps,candidates,novel)
        if combined is not None:wf.readback_selected_sources=combined
        audit['new_fact_ids']=audit['candidate_fact_ids']
        audit['status']='completed'
        return merged,{f['fact_id']:f for f in merged},conflicts+extra_conflicts,gaps+extra_gaps
    except Exception as exc:
        audit.update(status='kept_primary_after_failure',error_type=type(exc).__name__,error=str(exc)[:200])
        return original
    finally:
        audit['attempts']=wf.model.requests-start_requests;audit['seconds']=time.monotonic()-started
        write(output/'audit.json',audit)
