"""Debug CLI for the local SearchEngine. Use the persistent Python API in agent loops."""
import argparse
import sys

from hybrid_search import SearchEngine, serialize
from search_index import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-root',required=True)
    p.add_argument('--unit')
    sub = p.add_subparsers(dest='command',required=True)
    def search_arguments(parser, top_k):
        parser.add_argument('--query',required=True)
        parser.add_argument('--mode',choices=['hybrid','lexical','dense','exact'],default='hybrid')
        parser.add_argument('--top-k',type=int,default=top_k)
        parser.add_argument('--keyword',action='append',default=[])
        parser.add_argument('--doc-id',action='append',default=[])
        parser.add_argument('--node-type',choices=['text','table','sentence'],action='append',default=[])
        parser.add_argument('--max-response-tokens',type=int,default=8000)
    s = sub.add_parser('search'); search_arguments(s,20)
    s.add_argument('--entity-id')
    s.add_argument('--cursor-file',help='JSON cursor object from an earlier matching search')
    e = sub.add_parser('entity-search'); search_arguments(e,12)
    e.add_argument('--entity-id',required=True)
    q = sub.add_parser('task-search'); search_arguments(q,12)
    q.add_argument('--entity-id',action='append',dest='entity_ids',
                   help='Limit to an entity; repeat for more than one')
    preview = sub.add_parser('preview-search',help='Retrieve O4000 chunks and select one sentence per text hit')
    search_arguments(preview,12)
    preview.add_argument('--entity-id',action='append',dest='entity_ids')
    preview.add_argument('--batch-size',type=int,default=16)
    preview.add_argument('--threads',type=int,default=4)
    preview.add_argument('--debug',action='store_true')
    read_entities = sub.add_parser('read-entity-chunks',help='Read complete chunks grouped by entity')
    read_entities.add_argument('--requests',required=True,help='JSON object with task_id and selections')
    b = sub.add_parser('batch'); b.add_argument('--requests',required=True)
    r = sub.add_parser('read')
    r.add_argument('--chunk-id',action='append',required=True)
    r.add_argument('--view',choices=['agent','source'],default='agent')
    r.add_argument('--before',type=int,default=0); r.add_argument('--after',type=int,default=0)
    r.add_argument('--token-budget',type=int,default=12000)
    t = sub.add_parser('table'); t.add_argument('--ref-id',required=True)
    n = sub.add_parser('read-node')
    n.add_argument('--node-id',required=True)
    n.add_argument('--view',choices=['sentence','window','parent'],default='window')
    n.add_argument('--token-budget',type=int,default=12000)
    args = p.parse_args()
    engine = SearchEngine(args.index_root,args.unit,
                          preview_batch_size=getattr(args,'batch_size',16),
                          preview_threads=getattr(args,'threads',4))
    task = engine.summary['task_id']
    if args.command=='search':
        result = engine.search_evidence({'task_id':task,'query':args.query,'mode':args.mode,'top_k':args.top_k,
                  'keywords':args.keyword,'entity_id':args.entity_id,'filters':{'doc_ids':args.doc_id,'node_types':args.node_type},
                  'max_response_tokens':args.max_response_tokens,'cursor':read_json(args.cursor_file) if args.cursor_file else None})
    elif args.command=='entity-search':
        result = engine.search_entity_evidence({'task_id':task,'query':args.query,'mode':args.mode,'top_k':args.top_k,
                  'keywords':args.keyword,'entity_id':args.entity_id,
                  'filters':{'doc_ids':args.doc_id,'node_types':args.node_type},
                  'max_response_tokens':args.max_response_tokens})
    elif args.command=='task-search':
        request = {'task_id':task,'query':args.query,'mode':args.mode,'top_k':args.top_k,
                   'keywords':args.keyword,'filters':{'doc_ids':args.doc_id,'node_types':args.node_type},
                   'max_response_tokens':args.max_response_tokens}
        if args.entity_ids is not None:
            request['entity_ids'] = args.entity_ids
        result = engine.search_task_entities(request)
    elif args.command=='preview-search':
        request={'task_id':task,'query':args.query,'mode':args.mode,'top_k':args.top_k,
                 'keywords':args.keyword,'filters':{'doc_ids':args.doc_id,'node_types':args.node_type},
                 'max_response_tokens':args.max_response_tokens,'debug':args.debug}
        if args.entity_ids is not None:
            request['entity_ids']=args.entity_ids
        result=engine.search_entity_previews(request)
    elif args.command=='read-entity-chunks':
        result=engine.read_entity_chunks(read_json(args.requests))
    elif args.command=='batch':
        result = engine.search_evidence_batch(read_json(args.requests))
    elif args.command=='read':
        result = engine.read_evidence(task,args.chunk_id,args.view,args.token_budget,args.before,args.after)
    elif args.command=='read-node':
        result = engine.read_node(task,args.node_id,args.view,args.token_budget)
    else:
        result = engine.read_table(task,args.ref_id)
    print(serialize(result))


if __name__=='__main__':
    try:
        main()
    except (ValueError,KeyError,FileNotFoundError) as exc:
        error={'error':str(exc)}
        if hasattr(exc,'details'):
            error['details']=exc.details
        print(serialize(error),file=sys.stderr)
        sys.exit(2)
