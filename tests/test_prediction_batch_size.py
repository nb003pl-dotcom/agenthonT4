import pytest


def test_single_entity_prediction_reserves_include_repairs_everywhere():
    from workflow.roster_budget import prediction_attempt_reserve,plan
    from workflow.workflow import required_fact_reserve,prediction_attempt_reserve as workflow_reserve
    from workflow.fact_schedule import time_reserve
    assert prediction_attempt_reserve(8,batch_size=1)==16
    assert workflow_reserve(8,batch_size=1)==16
    assert required_fact_reserve(8,0,batch_size=1)==23
    assert plan(8,96,batch_size=1)['prediction_requests']==8
    assert plan(8,96,batch_size=1)['prediction_attempt_reserve']==16
    timing=time_reserve([],8,batch_size=1)
    assert timing['prediction_calls']==8 and timing['prediction_reserve_seconds']==265
    assert prediction_attempt_reserve(8)==4 and prediction_attempt_reserve(0,batch_size=1)==0


@pytest.mark.parametrize('bad',[0,5,True,1.5,'1'])
def test_prediction_batch_size_rejects_invalid_values(bad):
    from workflow.roster_budget import prediction_batch_size
    with pytest.raises(ValueError):prediction_batch_size({'prediction_batch_size':bad})


def test_supplemental_input_reserve_scales_with_prediction_groups():
    from types import SimpleNamespace
    from workflow.supplemental import _prediction_input_reserve
    wf=SimpleNamespace(config={'prediction_batch_size':1},inputs=SimpleNamespace(entities={str(i):{} for i in range(8)},task={}),contract={})
    solo=_prediction_input_reserve(wf,[])
    wf.config['prediction_batch_size']=4
    grouped=_prediction_input_reserve(wf,[])
    assert solo-grouped==6*200000


@pytest.mark.parametrize('workers',[1,2])
@pytest.mark.parametrize('size',[1,4])
def test_actual_workflow_groups_predictions_and_preserves_entity_fact_ownership(tmp_path,monkeypatch,workers,size):
    import hashlib
    import json
    import shutil
    from workflow.common import ROOT,read,write
    from workflow.model import Model
    from workflow.workflow import Workflow
    from tests.unit_factory import make_unit
    unit=make_unit(tmp_path/'unit')
    task=read(unit/'task.json');task['entities'].append({**task['entities'][0],'entity_id':'OTHER','name':'Other issuer'})
    write(unit/'task.json',task)
    manifest=read(unit/'manifest.json')
    for entry in manifest['files']:
        p=unit/entry['path']
        if entry.get('role')=='corpus':entry['entity_ids']=['AAPL','OTHER']
        if entry['path']=='task.json':
            entry['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();entry['bytes']=p.stat().st_size
    write(unit/'manifest.json',manifest);groups=[]
    def call(model,stage,system,payload,validator,**kwargs):
        model.requests+=1
        if stage=='query':
            value={'queries':[{'query_id':e+str(i),'query_type':role,'entity_ids':[e],
                'dense_query':'earnings EPS','lexical_query':'earnings EPS','purpose':'Find source'}
                for e in payload['allowed_entity_ids'] for i,role in enumerate(['target_history','guidance_expectation','risk_counterevidence'])]}
        elif stage=='facts':
            e=payload['entity']['entity_id'];c=payload['chunks'][0]
            value={'facts':[{'entity_id':e,'statement':'Historical source context.','evidence_role':'target_history',
                'citations':[{'doc_id':c['doc_id'],'chunk_id':c['chunk_id'],'quote':c['source_spans'][0]['text'][:150]}]}]}
        elif stage=='predict':
            groups.append(payload['allowed_entity_ids'])
            assert {e['entity_id'] for e in payload['entities']}==set(payload['allowed_entity_ids'])
            assert set(payload['allowed_fact_ids_by_entity'])==set(payload['allowed_entity_ids'])
            value={'predictions':[{'entity_id':e,'point_forecast':1.5,'interval':{'lo':1.2,'hi':1.8},
                'unit':'USD_per_share','label':'inline','class_probabilities':{'beat':.25,'miss':.25,'inline':.5},
                'fact_ids':ids[:1],'reasoning':'Offline grouping test.'} for e,ids in payload['allowed_fact_ids_by_entity'].items()]}
        else:raise AssertionError(stage)
        return validator(value)
    async def acall(*args,**kwargs):return call(*args,**kwargs)
    monkeypatch.setattr(Model,'call',call);monkeypatch.setattr(Model,'acall',acall)
    config={'provider':'house','model':'offline','retrieval_mode':'lexical','evidence_policy':'legacy',
        'laya_mode':'off','quantitative_mode':'off','write_reasons':False,
        'prediction_workers':workers,'prediction_batch_size':size}
    Workflow(config,unit,tmp_path/'out').run()
    assert sorted(e for group in groups for e in group)==['AAPL','OTHER']
    assert len(groups)==(2 if size==1 else 1)
    assert all(len(group)<=size for group in groups)
