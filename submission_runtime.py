"""Final-only adapter around the restored 10.25 forecasting workflow."""
from copy import deepcopy
import time
from workflow.common import read,write
from workflow.workflow import Workflow
from workflow.reasons import build_reasons,reason_payload


class SubmissionWorkflow(Workflow):
    def __init__(self,config,unit,output):
        core={**config,'max_requests':24,'write_reasons':False}
        total=min(570,float(config.get('timeout') or 570))
        core['timeout']=max(1,total-45)
        super().__init__(core,unit,output)
        total=min(total,float(self.inputs.card.get('agent',{}).get('timeout_sec',total)))
        reserve=min(45,total/4)
        self.submission_deadline=self.started+total
        self.deadline=self.submission_deadline-reserve
        self.model.deadline=self.deadline
        self.config['timeout']=self.deadline-self.started

    def append_submission_reasons(self,answer):
        result={'status':'skipped_budget_or_time'}
        if self.model.requests>=25 or self.submission_deadline-time.time()<=1:
            write(self.output/'reasons-report.json',result)
            return answer
        facts=read(self.output/'facts.json');fact_map={f['fact_id']:f for f in facts}
        draft={r['entity_id']:r for r in read(self.output/'prediction-draft.json')}
        # Explain the published values, retaining the draft's validated fact ownership.
        rows=[{**draft[p['entity_id']],**{k:deepcopy(v) for k,v in p.items() if k!='claims'}}
              for p in answer['entity_predictions']]
        packets=read(self.output/'quantitative-anchors.json')
        contract=read(self.output/'target-contract.json')
        payload=reason_payload(self.inputs,rows,fact_map,packets,contract)
        selected=list(dict.fromkeys(fid for row in rows for fid in row['fact_ids']))
        payload['output_requirements']={
            'top_level_key':'reasons','reason_count':{'minimum':1,'maximum':3},
            'required_per_reason':['entity_ids','premise','mechanism','answer_implication','fact_ids','anchor_entity_ids'],
            'allowed_fact_ids':selected,
            'allowed_anchor_entity_ids':sorted(e for e,p in packets.items() if p.get('status')=='available'),
            'id_arrays':'Use JSON arrays of exact supplied string IDs, never booleans or invented IDs.',
            'citation_source_rule':'Every reason needs at least one supplied fact_id or available anchor_entity_id.'}
        try:
            self.submitted_reasons=self.model.call('reasons',self.prompt('reason-writer'),payload,
                lambda data:build_reasons(data,self.inputs,fact_map,packets),max_attempts=1)
            final=self.publish_reasons(answer,'final_submitted_reasons')
            if final['entity_predictions']!=answer['entity_predictions']:
                raise ValueError('reason_stage_changed_predictions')
            result={'status':'complete','reasons':len(self.submitted_reasons)}
            write(self.output/'reasons-report.json',result)
            return final
        except Exception as exc:
            self.submitted_reasons=[]
            result={'status':'retained_analysis_answer','error_type':type(exc).__name__}
            write(self.output/'reasons-report.json',result)
            return answer

    def run(self):
        answer=super().run()
        self.deadline=self.submission_deadline
        self.model.deadline=self.submission_deadline
        self.model.max_requests=25
        self.config.update(max_requests=25,write_reasons=True,scoring_phase='final')
        try:
            answer=self.append_submission_reasons(answer)
            return answer
        finally:
            self.model.close()
            write(self.output/'run-summary.json',{'status':'complete','scoring_phase':'final',
                'seconds':time.time()-self.started,'model':self.model.summary(),
                'core_request_cap':24,'submitted_reason_request_reserve':1,
                'stages':self.timings,'answer':str(self.output/'answer.json')})
