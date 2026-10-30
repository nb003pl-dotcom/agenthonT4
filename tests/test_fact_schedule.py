from types import SimpleNamespace
import pytest
from workflow.fact_schedule import order_jobs,time_reserve
from workflow.source_fallback import source_facts


def test_first_round_covers_entities_before_second_batches():
    jobs=[{'entity_index':e,'batch_no':b} for e in range(3) for b in range(2)]
    assert order_jobs(jobs,'sequential')==jobs
    assert [(j['entity_index'],j['batch_no']) for j in order_jobs(jobs,'coverage_first')]==[(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]
    with pytest.raises(ValueError):order_jobs(jobs,'unknown')


def test_prediction_reserve_uses_current_observed_latency():
    initial=time_reserve([],8)
    observed=time_reserve([{'status':'success','seconds':s} for s in [30,40,50,60]],8)
    assert initial['prediction_calls']==2
    assert observed['prediction_reserve_seconds']>initial['prediction_reserve_seconds']
    assert time_reserve([{'status':'retryable_http','seconds':999}],8)==initial


def test_raw_fallback_keeps_exact_positions_permissions_and_unknown_fields():
    text='Example company consolidated quarterly diluted EPS was $1.80 versus $1.50 a year earlier. '
    chunk={'doc_id':'d','chunk_id':'c','table_refs':[],'text':text,
           'source_spans':[{'start':11,'end':11+len(text),'text':text}]}
    inputs=SimpleNamespace(docs={'d':{'text':' '*11+text,'doc_date':'2024-07-01'}},allowed=lambda e,d:e=='NEW' and d=='d')
    result=source_facts(inputs,'NEW',[chunk],[{'entity_ids':['NEW'],'lexical_query':'quarterly EPS'}])
    f=result['facts'][0];c=f['citations'][0]
    assert f['value'] is None and f['reporting_scope']=='unknown'
    assert f['extraction_method']=='verbatim_source_fallback'
    assert c['quote']==inputs.docs['d']['text'][c['span_start']:c['span_end']]
    assert f['statement']==c['quote']
    with pytest.raises(ValueError):source_facts(inputs,'OTHER',[chunk],[])


@pytest.mark.parametrize('gap',[0,17])
def test_fallback_never_merges_across_authorized_source_spans(gap):
    first='Quarterly income was 120 million. '
    second='The quarterly income comparison was 110 million. '
    raw=first+'x'*gap+second
    start=len(first)+gap
    spans=[{'start':0,'end':len(first),'text':first},
           {'start':start,'end':len(raw),'text':second}]
    chunk={'doc_id':'d','chunk_id':'c','table_refs':[],'text':raw,'source_spans':spans}
    inputs=SimpleNamespace(docs={'d':{'text':raw,'doc_date':'2024-07-01'}},allowed=lambda e,d:e=='E')
    result=source_facts(inputs,'E',[chunk],[{'entity_ids':['E'],'lexical_query':'quarterly income comparison'}])
    assert result['facts']
    for f in result['facts']:
        for c in f['citations']:
            assert any(s['start']<=c['span_start']<c['span_end']<=s['end'] for s in spans)
            assert c['quote']==raw[c['span_start']:c['span_end']]


def test_fallback_can_merge_catalog_pieces_inside_one_source_span():
    raw='Quarterly revenue and operating income remained stable at 120 million. '*9
    chunk={'doc_id':'d','chunk_id':'c','table_refs':[],'text':raw,
           'source_spans':[{'start':0,'end':len(raw),'text':raw}]}
    inputs=SimpleNamespace(docs={'d':{'text':raw,'doc_date':'2024-07-01'}},allowed=lambda e,d:True)
    result=source_facts(inputs,'E',[chunk],[{'entity_ids':['E'],'lexical_query':'quarterly revenue'}])
    c=result['facts'][0]['citations'][0]
    assert len(c['quote'])>320
    assert c['quote']==raw[c['span_start']:c['span_end']]


def test_fact_stage_reserves_forecast_and_records_unprocessed_windows(monkeypatch,tmp_path):
    from workflow.fact_stage import extract_stage
    from workflow.common import write,read
    clock=[1000.0];monkeypatch.setattr('workflow.fact_stage.time.time',lambda:clock[0])
    text='Quarterly consolidated earnings increased. '*1200
    ids=['A','B','C'];calls=[]
    docs={e:{'text':text,'doc_date':'2024-07-01'} for e in ids}
    chunks={e:[{'doc_id':e,'chunk_id':e,'table_refs':[],'text':text,
                'source_spans':[{'start':0,'end':len(text),'text':text}]}] for e in ids}
    inputs=SimpleNamespace(task={'prompt':'Predict earnings'},entities={e:{'entity_id':e} for e in ids},
                           docs=docs,allowed=lambda e,d:e==d)
    class Model:
        records=[];requests=0;max_requests=25;deadline=1110
        def call(self,stage,prompt,payload,validator,reserve):
            e=payload['entity']['entity_id'];calls.append(e);self.requests+=1;clock[0]+=25
            self.records.append({'seconds':25,'status':'success'})
            f={'entity_id':e,'fact_id':'f'+str(len(calls)),'statement':'Source fact',
               'citations':[{'citation_id':'cite'+str(len(calls)),'quote':'Source fact','doc_date':'2024-07-01'}]}
            return {'facts':[f],'rejected':[],'conflicts':[],'unresolved_questions':[]}
    wf=SimpleNamespace(config={'facts_schedule':'coverage_first','facts_source_token_budget':3000},
        output=tmp_path,financial_scope=False,policy='coverage',inputs=inputs,contract={},roster_plan={'adaptive':False},
        deadline=1110,model=Model(),prompt=lambda name:'Extract facts',retrieval=None)
    write(tmp_path/'query-plan.json',{'queries':[]})
    facts,_,_,gaps=extract_stage(wf,chunks,chunks,chunks,ids,{})
    assert calls==['A','B']  # C is not starved by A's second source window.
    assert {f['entity_id'] for f in facts}==set(ids)
    assert any(f.get('extraction_method')=='verbatim_source_fallback' for f in facts if f['entity_id']=='C')
    audit=read(tmp_path/'fact-schedule.json')
    assert any(x['status']=='skipped_time_budget' for x in audit['decisions'])
    assert any(g['reason']=='facts_time_reserved_for_prediction' for g in gaps)
    assert wf.model.deadline==1110 and clock[0]==1050
