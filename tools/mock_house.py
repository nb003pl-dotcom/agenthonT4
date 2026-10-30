"""Synthetic House-protocol server for local container smoke tests only."""
import argparse,json,re
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path


def respond(payload,fail_reasons=False):
    if 'output_requirements' in payload and payload['output_requirements'].get('top_level_key')=='reasons':
        if fail_reasons:return {'reasons':[]}
        ids=[r['entity_id'] for r in payload['predictions']]
        facts=payload['output_requirements']['allowed_fact_ids']
        return {'reasons':[{'entity_ids':ids,'premise':'Prior-quarter diluted earnings were $1.45 per share.',
            'mechanism':'Stable demand provides a baseline, while currency uncertainty limits confidence.',
            'answer_implication':'The supplied forecasts stay close to consensus with uncertainty around that baseline.',
            'fact_ids':facts[:1],'anchor_entity_ids':[]}]}
    if 'memo_output_requirements' in payload:
        return {'reason_content':[{'entity_id':e,'baseline':{'point':None,'lo':None,'hi':None,'reliability':'unavailable'},
            'direction':'unknown','magnitude':'unknown','numeric_support':'none','point_action':'retain',
            'recommended_point':None,'counterevidence':'The prior observation does not resolve future uncertainty.',
            'uncertainty_action':'retain','fact_ids':payload['allowed_fact_ids_by_entity'][e][:1]}
            for e in payload['allowed_entity_ids']]}
    if 'allowed_fact_ids_by_entity' in payload:
        kind=payload['task']['target']['type'];rows=[]
        for i,(e,ids) in enumerate(payload['allowed_fact_ids_by_entity'].items()):
            row={'entity_id':e,'point_forecast':1.5,'interval':{'lo':1.0,'hi':2.0},
                'unit':payload['target_contract']['entities'][e]['target_unit'],
                'fact_ids':ids[:1],'reasoning':'Synthetic interface-test forecast.'}
            if kind=='classification':row.update(label='inline',class_probabilities={'beat':0.2,'miss':0.2,'inline':0.6})
            if kind=='ranking':row['ranking_order']='descending'
            rows.append(row)
        return {'predictions':rows}
    if 'chunks' in payload:
        text='\n'.join(s['text'] for c in payload['chunks'] for s in c['source_spans'])
        quote=re.search(r'\[\[ (Q\d+) \]\]',text).group(1)
        value={'facts':[{'entity_id':payload['entity']['entity_id'],
            'statement':'Prior-quarter diluted earnings were $1.45 per share.','evidence_role':'target_history',
            'citations':[{'quote_ids':[quote]}]}]}
        if 'joint_assessment_chunk_ids' in payload:
            value['source_assessments']=[{'source_index':i,'matches':'D'*len(payload['information_needs'])}
                for i,_ in enumerate(payload['joint_assessment_chunk_ids'])]
        return value
    return {'query_templates':[{'type':role,'text':text} for role,text in [
        ('target_history','historical diluted earnings per share'),
        ('guidance_expectation','demand expectations and management guidance'),
        ('risk_counterevidence','currency costs and demand uncertainty')]]}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path!='/v1/chat/completions':self.send_error(404);return
        if self.headers.get('Authorization')!='Bearer synthetic-test-token':self.send_error(401);return
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('model')!='synthetic-house-model' or body.get('max_tokens')!=4000 or body.get('tools'):
            self.send_error(400);return
        payload=json.loads(body['messages'][1]['content'])
        result=respond(payload,self.server.fail_reasons)
        if self.server.audit:
            with self.server.audit.open('a') as log:log.write(json.dumps({'path':self.path,'model':body['model'],'request':body,'result':result})+'\n')
        response={'choices':[{'message':{'content':json.dumps(result)},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':100,'total_tokens':200}}
        encoded=json.dumps(response).encode();self.send_response(200)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded)
    def log_message(self,*args):pass


def serve(port=0,fail_reasons=False,audit=None):
    server=ThreadingHTTPServer(('0.0.0.0',port),Handler)
    server.fail_reasons=fail_reasons;server.audit=Path(audit) if audit else None
    return server


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18765);p.add_argument('--fail-reasons',action='store_true');p.add_argument('--audit',type=Path)
    a=p.parse_args();s=serve(a.port,a.fail_reasons,a.audit)
    print(json.dumps({'port':s.server_port}),flush=True);s.serve_forever()
