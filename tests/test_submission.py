import json,os,time
from pathlib import Path
from types import SimpleNamespace
import pytest


def house(monkeypatch):
 for k,v in {'MODEL_ENDPOINT':'http://house.local:1234','MODEL_NAME':'organizer-pinned-id','MODEL_TOKEN':'secret-test-only'}.items():monkeypatch.setenv(k,v)


def test_submission_is_house_only_under25_final_and_keeps_proxy(monkeypatch):
 from submission_entry import config_from_environment
 house(monkeypatch);monkeypatch.setenv('HTTPS_PROXY','http://organizer-proxy:8080')
 monkeypatch.setenv('T4_SCORING_PHASE','development');monkeypatch.setenv('WORKFLOW_MODEL','participant-override')
 c=config_from_environment()
 assert c['under25_strategy'] is True
 assert c.get('query_plan_context','scanned')=='scanned'
 assert c['workflow_profile']=='typed-evidence-v2' and c['joint_relevance_facts'] is True
 assert c['prediction_batch_size']==4 and c['structured_output_preflight'] is True
 assert c['provider']=='house' and c['model']=='organizer-pinned-id'
 assert c['base_url']=='http://house.local:1234/v1' and c['api_key']=='secret-test-only'
 assert c['scoring_phase']=='final' and c['write_reasons'] is True
 assert c['max_requests']==25 and c['max_tokens']==4000
 assert c['retrieval_mode']=='lexical' and c['laya_mode']=='off'
 assert not c['source_priority'] and not c['source_bound_eps']
 assert os.environ['HTTPS_PROXY']=='http://organizer-proxy:8080'


@pytest.mark.parametrize('endpoint',['https://host/v1','https://host?token=x','https://user:pass@host','file:///tmp/house','https://host#x'])
def test_endpoint_must_be_a_plain_http_origin(monkeypatch,endpoint):
 from submission_entry import config_from_environment
 house(monkeypatch);monkeypatch.setenv('MODEL_ENDPOINT',endpoint)
 with pytest.raises(ValueError):config_from_environment()


def test_runtime_never_exceeds_25_and_appends_reasons_without_changing_predictions(tmp_path,monkeypatch):
 from submission_runtime import SubmissionWorkflow
 from workflow.workflow import Workflow
 from workflow.common import write,read
 original={'task_id':'demo','target_type':'classification','entity_predictions':[{'entity_id':'A','label':'up','point_forecast':1,'interval':{'lo':0,'hi':2}}]}
 seen=[]
 def init(self,config,unit,out):
  assert config['max_requests']==24 and config['write_reasons'] is False
  self.config=config;self.output=out;out.mkdir();self.started=time.time()
  self.deadline=self.started+config['timeout'];self.timings=[]
  self.inputs=SimpleNamespace(card={'agent':{'timeout_sec':600}},entities={'A':{}},task={'task_id':'demo'})
  self.model=SimpleNamespace(requests=24,max_requests=24,deadline=self.deadline,records=[],close=lambda:None,
   summary=lambda:{'requests':self.model.requests,'max_requests':self.model.max_requests})
 def run(self):
  write(self.output/'answer.json',original);write(self.output/'answer-publication.json',{'validated_complete':True})
  write(self.output/'facts.json',[]);write(self.output/'prediction-draft.json',[])
  write(self.output/'quantitative-anchors.json',{});write(self.output/'target-contract.json',{})
  return original
 def final_reasons(self,answer):
  assert self.model.max_requests==25
  self.model.requests+=1;seen.append(self.model.requests)
  result={**answer,'submitted_reasons':[{'reason_id':'r1'}]};write(self.output/'answer.json',result)
  return result
 monkeypatch.setattr(Workflow,'__init__',init);monkeypatch.setattr(Workflow,'run',run)
 monkeypatch.setattr(SubmissionWorkflow,'append_submission_reasons',final_reasons)
 w=SubmissionWorkflow({'max_requests':25,'timeout':570,'write_reasons':True},tmp_path/'unit',tmp_path/'out')
 result=w.run()
 assert seen==[25] and result['entity_predictions']==original['entity_predictions']
 assert w.model.requests==25


def test_cli_refuses_unvalidated_or_partial_answer(tmp_path,monkeypatch):
 from submission_entry import execute
 from workflow.common import write
 house(monkeypatch);unit=tmp_path/'unit';unit.mkdir();(unit/'corpus').mkdir();(unit/'task.json').write_text('{}')
 out=tmp_path/'output/answer.json'
 class Partial:
  def __init__(self,c,u,w):
   self.config=c;self.deadline=time.time()+20;self.output=w;w.mkdir(parents=True)
   self.model=SimpleNamespace(summary=lambda:{});self.timings=[]
  def run(self):
   write(self.output/'answer-publication.json',{'validated_complete':False})
   return {'entity_predictions':[]}
 assert execute(unit/'task.json',unit/'corpus',out,factory=Partial)==1
 assert not out.exists()


@pytest.mark.parametrize('fail_reasons',[False,True])
def test_real_cli_with_house_protocol_and_final_reason_fallback(tmp_path,fail_reasons):
 import subprocess,sys,threading
 from tools.mock_house import serve
 from tests.unit_factory import make_unit
 unit=make_unit(tmp_path/'unit');output=tmp_path/'output/answer.json'
 audit=tmp_path/'calls.jsonl';server=serve(fail_reasons=fail_reasons,audit=audit)
 thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 env={k:v for k,v in os.environ.items() if k.lower() not in {'http_proxy','https_proxy','all_proxy'}}
 env.update(MODEL_ENDPOINT=f'http://127.0.0.1:{server.server_port}',MODEL_NAME='synthetic-house-model',
  MODEL_TOKEN='synthetic-test-token',NO_PROXY='*',PYTHONDONTWRITEBYTECODE='1')
 try:
  proc=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'submission_entry.py'),'analyze',
   '--task',str(unit/'task.json'),'--corpus',str(unit/'corpus'),'--out',str(output)],env=env,capture_output=True,text=True,timeout=90)
 finally:server.shutdown();server.server_close();thread.join(timeout=5)
 assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
 answer=json.loads(output.read_text());report=json.loads((output.parent/'execution-report.json').read_text())
 assert {p['entity_id'] for p in answer['entity_predictions']}=={'AAPL'}
 assert report['scoring_phase']=='final' and report['budget']['requests']<=25
 calls=[json.loads(l) for l in audit.read_text().splitlines()]
 final=[r for r in calls if r['request']['messages'][1]['content'].find('"top_level_key": "reasons"')>=0]
 assert len(final)==1
 assert ('submitted_reasons' in answer) is not fail_reasons
 assert report['submitted_reasons']['status']==('retained_analysis_answer' if fail_reasons else 'complete')
