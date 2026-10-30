"""AgentHon analysis interface v2.0, restricted to the injected House route."""
import argparse,json,os,signal,tempfile,time
from pathlib import Path
from urllib.parse import urlsplit
from workflow.common import read,write


def config_from_environment():
    endpoint=os.environ['MODEL_ENDPOINT'].rstrip('/')
    url=urlsplit(endpoint)
    if (url.scheme not in ('http','https') or not url.hostname or url.path or url.query
        or url.fragment or url.username or url.password):
        raise ValueError('MODEL_ENDPOINT must be a route origin')
    model=os.environ['MODEL_NAME'];token=os.environ['MODEL_TOKEN']
    if not model.strip() or not token.strip():raise ValueError('missing House model or bearer')
    return {'provider':'house','base_url':endpoint+'/v1','model':model,'api_key':token,
        'retrieval_mode':'lexical','laya_mode':'off','policy':'coverage','quantitative_mode':'anchor',
        'under25_strategy':True,'workflow_profile':'typed-evidence-v2','joint_relevance_facts':True,
        'structured_output_preflight':True,'prediction_batch_size':4,'facts_workers':1,'prediction_workers':1,
        'scoring_phase':'final','write_reasons':True,'max_requests':25,'max_tokens':4000,
        'max_input_tokens':None,'enable_thinking':False,'enable_review':False,
        'compact_model_outputs':True,'model_selection':True,'compact_predictions':False,
        'evidence_cap':5,'context_window':32000,'facts_source_token_budget':16000,
        'timeout':570,'rate_interval_floor':True,'decisive_source_readback':True,
        'source_priority':False,'source_bound_eps':False,'eps_stress_interval':False,'facts_schedule':'sequential'}


def complete_roster(answer,task):
    predictions=answer.get('entity_predictions') if isinstance(answer,dict) else None
    if not isinstance(predictions,list):return False
    ids=[p.get('entity_id') for p in predictions if isinstance(p,dict)]
    expected=[p['entity_id'] for p in task.get('entities',[])]
    return bool(expected) and len(ids)==len(predictions)==len(expected) and set(ids)==set(expected) and len(set(ids))==len(ids) and answer.get('task_id')==task.get('task_id')


def execute(task_path,corpus_path,answer_path,factory=None):
    started=time.monotonic();report={'status':'failed'};wf=None
    old_handler=signal.getsignal(signal.SIGALRM)
    with tempfile.TemporaryDirectory(prefix='under25-final-') as tmp:
        work=Path(tmp)/'run'
        try:
            if corpus_path.resolve()!=task_path.parent.resolve()/'corpus':raise ValueError('Wrong mounted corpus')
            if answer_path.resolve().is_relative_to(task_path.parent.resolve()):raise ValueError('Output must not be inside input')
            task=read(task_path)
            if factory is None:
                from submission_runtime import SubmissionWorkflow
                factory=SubmissionWorkflow
            wf=factory(config_from_environment(),task_path.parent,work)
            def expired(signum,frame):raise TimeoutError('unit_finalization_deadline')
            signal.signal(signal.SIGALRM,expired)
            signal.alarm(max(1,int(getattr(wf,'submission_deadline',wf.deadline)-time.time())))
            answer=wf.run();state=read(work/'answer-publication.json')
            if not state.get('validated_complete') or not complete_roster(answer,task):raise ValueError('Unvalidated or incomplete answer')
            write(answer_path,answer);report['status']='complete'
        except Exception as exc:
            report['error_type']=type(exc).__name__
            state=read(work/'answer-publication.json') if (work/'answer-publication.json').exists() else {}
            saved=read(work/'answer.json') if (work/'answer.json').exists() else None
            if state.get('validated_complete') and saved and complete_roster(saved,read(task_path)):
                write(answer_path,saved);report['status']='complete_retained'
            elif answer_path.exists():answer_path.unlink()
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        if wf:
            report['scoring_phase']='final';report['budget']=wf.model.summary()
            report['stages']=[{k:v for k,v in s.items() if k!='error'} for s in wf.timings]
            reason=work/'reasons-report.json'
            report['submitted_reasons']=read(reason) if reason.exists() else {'status':'not_reached'}
    report['seconds']=round(time.monotonic()-started,3)
    write(answer_path.parent/'execution-report.json',report)
    print(json.dumps(report),flush=True)
    return 0 if report['status'].startswith('complete') else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('verb',choices=['analyze'])
    p.add_argument('--task',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();return execute(a.task,a.corpus,a.out)


if __name__=='__main__':raise SystemExit(main())
