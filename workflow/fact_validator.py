"""Existing per-window validator reused by adaptive multi-entity extraction."""
from .common import write
from .facts import validate_facts

def validator_for_job(self, job, bound_eps_report, bound_eps_audit, *, write_audit=True):
    entity=job['entity'];batch=job['batch'];batch_no=job['batch_no'];quote_catalog=job['quote_catalog']
    validation_attempt={'count':0}
    def validate(d):
        validation_attempt['count']+=1
        assessments=None
        if self.config.get('joint_relevance_facts'):
            from .joint_evidence import validate_assessments
            assessments=validate_assessments(d,job['payload'])
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
                              max_facts=job['payload']['limits']['max_facts'] if getattr(self,'subquery_evidence',False) else 12)
        if bound_eps_audit['enabled']:
            from .source_bound_eps import bind
            binding=bind(self.inputs,bound_eps_report,entity,batch,result['facts'])
            result['facts']=binding['facts']
            bound_eps_audit['checks'].append({'entity_id':entity,'batch':batch_no,
                'validation_attempt':validation_attempt['count'],'audit':binding['audit']})
            if write_audit:write(self.output/'source-bound-eps.json',bound_eps_audit)
        if self.config.get('numeric_fact_check')=='warn':
            from .fact_source_guard import numeric_support
            names=[s for e in self.inputs.entities.values() for s in (e.get('name',''),e.get('entity_id',''))]
            # Run after source binding, so the diagnostic describes the actual
            # facts/quotes sent to forecasting, not a replaced model assertion.
            for f in result['facts']:
                f['numeric_support']=numeric_support(f,names)
        from .facts import handle_rejected_window
        result=handle_rejected_window(result,bool(d.get('facts')),
            validation_attempt['count'],entity,batch_no)
        if assessments is not None:
            result['source_assessments']=assessments
        return result
    return validate
