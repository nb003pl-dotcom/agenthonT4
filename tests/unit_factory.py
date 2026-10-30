"""Generate synthetic interface-test inputs; no competition tasks or outcomes."""
import hashlib,json
from pathlib import Path


def make_unit(folder,target_type='classification',entity_count=1):
    folder=Path(folder);(folder/'corpus').mkdir(parents=True)
    task={'task_id':'synthetic-eps','schema_version':'3','family':'eps_beat_consensus',
        'target':{'name':'eps_outcome','type':'classification','labels':['beat','miss','inline']},
        'prompt':'Predict synthetic issuer quarterly diluted EPS relative to consensus; cite frozen evidence.',
        'cutoff_date':'2024-03-15','resolution_date':'2024-05-05','interval_level':0.9,
        'corpus_manifest':'manifest.json','entities':[{'entity_id':'AAPL','name':'Synthetic Issuer',
        'consensus_eps':1.5,'threshold_pct':0.05,'unit':'USD_per_share'}]}
    if target_type!='classification':
        task['target']={'name':'synthetic_metric','type':target_type,'unit':'units'}
        task['family']='synthetic_'+target_type
        task['entities']=[{'entity_id':f'SYN{i+1}','name':f'Synthetic Issuer {i+1}','unit':'units'} for i in range(entity_count)]
        if target_type=='ranking':task['ranking_order']='descending'
    doc={'doc_id':'SYNTHETIC_HISTORY','task_id':task['task_id'],'title':'Synthetic earnings history',
        'doc_date':'2024-02-01','text':'Synthetic Issuer reported diluted earnings per share of $1.45 for the prior quarter. Revenue increased by 2% while operating costs remained stable. Management expects demand to remain steady but foreign exchange creates uncertainty.'}
    (folder/'task.json').write_text(json.dumps(task));(folder/'corpus/SYNTHETIC_HISTORY.json').write_text(json.dumps(doc))
    (folder/'card.toml').write_text(f'[task]\nid="{task["task_id"]}"\ntrack="analysis"\ntarget_type="{target_type}"\n[scoring.params]\ntarget_type="{target_type}"\n[agent]\ntimeout_sec=600\n')
    files=[]
    for name,role in [('task.json','input'),('card.toml','input'),('corpus/SYNTHETIC_HISTORY.json','corpus')]:
        p=folder/name;entry={'path':name,'role':role,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        if role=='corpus':entry['entity_ids']=[e['entity_id'] for e in task['entities']]
        files.append(entry)
    (folder/'manifest.json').write_text(json.dumps({'unit_id':task['task_id'],'files':files}))
    return folder
