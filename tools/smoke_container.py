"""Exercise the actual final image on an internal Docker network, without API spend."""
import argparse,json,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tests.unit_factory import make_unit


def docker(args,**kwargs):
 result=subprocess.run(['docker',*args],text=True,capture_output=True,**kwargs)
 if result.returncode:
  raise RuntimeError('Docker command failed: '+result.stdout+'\n'+result.stderr)
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--image',default='under25-agenthon:final-20261027');p.add_argument('--out',type=Path,default=ROOT/'dist/container-smoke')
 a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
 network='under25-smoke-'+uuid.uuid4().hex[:8];server=network+'-house'
 docker(['network','create','--internal',network])
 results=[]
 try:
  for kind,fail in [('classification',False),('regression',False),('ranking',False),('classification',True)]:
   name=kind+('-reason-failure' if fail else '')
   folder=a.out/name;unit=make_unit(folder/'input',kind,3 if kind=='ranking' else 1)
   output=folder/'output';output.mkdir();audit=folder/'audit';audit.mkdir()
   args=['run','-d','--name',server,'--network',network,'--network-alias','house',
    '--entrypoint','python','-v',str(ROOT/'tools/mock_house.py')+':/mock_house.py:ro',
    '-v',str(audit)+':/audit',a.image,'/mock_house.py','--port','8080','--audit','/audit/calls.jsonl']
   if fail:args.append('--fail-reasons')
   docker(args)
   for _ in range(30):
    ready=subprocess.run(['docker','exec',server,'python','-c',"import socket;s=socket.create_connection(('127.0.0.1',8080),1);s.close()"],capture_output=True)
    if ready.returncode==0:break
    time.sleep(0.5)
   try:
    run=docker(['run','--rm','--platform','linux/amd64','--network',network,'--cpus','2','--memory','4g',
     '-e','MODEL_ENDPOINT=http://house:8080','-e','MODEL_NAME=synthetic-house-model','-e','MODEL_TOKEN=synthetic-test-token',
     '-e','HTTP_PROXY=http://unreachable-organizer-proxy:8080','-e','HTTPS_PROXY=http://unreachable-organizer-proxy:8080','-e','NO_PROXY=house',
     '-v',str(unit)+':/input:ro','-v',str(output)+':/output',a.image,'analyze',
     '--task','/input/task.json','--corpus','/input/corpus/','--out','/output/answer.json'],timeout=150)
    (folder/'stdout.log').write_text(run.stdout);(folder/'stderr.log').write_text(run.stderr)
    answer=json.loads((output/'answer.json').read_text());report=json.loads((output/'execution-report.json').read_text())
    assert len(answer['entity_predictions'])==(3 if kind=='ranking' else 1)
    assert ('submitted_reasons' in answer) is not fail
    assert report['budget']['requests']<=25 and report['scoring_phase']=='final'
    calls=[json.loads(l) for l in (audit/'calls.jsonl').read_text().splitlines()]
    assert all(r['path']=='/v1/chat/completions' and r['model']=='synthetic-house-model' for r in calls)
    result={'case':name,'status':'passed','requests':report['budget']['requests'],
     'reason_status':report['submitted_reasons']['status'],'entities':len(answer['entity_predictions'])}
    results.append(result);print(json.dumps(result),flush=True)
   finally:docker(['rm','-f',server])
 finally:
  subprocess.run(['docker','rm','-f',server],capture_output=True)
  docker(['network','rm',network])
 (a.out/'summary.json').write_text(json.dumps({'image':a.image,'network':'internal-only','results':results},indent=2)+'\n')


if __name__=='__main__':main()
