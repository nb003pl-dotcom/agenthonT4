"""Verified frozen retrieval replay for controlled downstream experiments."""
from pathlib import Path
from .common import read

class ReplayRetrieval:
    def __init__(self, inputs, run):
        self.inputs=inputs;self.run=Path(run).resolve()
        binding=read(self.run/'input-contract.json')
        if binding['identity']!=inputs.identity or binding['relations']!=inputs.relations:
            raise ValueError('replay input hashes/permissions differ from current unit')
        self.candidates=read(self.run/'retrieval-candidates.json')
        if set(self.candidates)!=set(inputs.entities):raise ValueError('replay entity roster mismatch')
        for entity,rows in self.candidates.items():
            for row in rows:
                if not inputs.allowed(entity,row['doc_id']):raise ValueError('replay wrong entity document')
                for span in row['source_spans']:
                    if inputs.docs[row['doc_id']]['text'][span['start']:span['end']]!=span['text']:
                        raise ValueError('replay source text/offset mismatch')
        self.summary=read(self.run/'preprocessing-report.json')
        self.summary={**self.summary,'replayed_from':str(self.run),'notice':'Reuses verified frozen retrieval, not a full end-to-end latency test.'}
        self.engine=None

    def search(self, queries, top=5, cap=12):
        return self.candidates,{'replayed_from':str(self.run),'queries':read(self.run/'query-plan.json')['queries']}

    def table(self,rid):
        if (self.run/'index/lexical-index.json').exists():
            from .common import digest
            from read_table import read_table_ref
            for rows in self.candidates.values():
                for row in rows:
                    for ref in row.get('table_refs',[]):
                        if ref.get('ref_id')==rid:
                            result=self.run/'index/documents'/digest(row['doc_id'])/'result.json'
                            return read_table_ref(ref,result)
            raise ValueError('replay table is not in verified candidates')
        if self.engine is None:
            from hybrid_search import SearchEngine
            self.engine=SearchEngine(self.run/'index',self.inputs.path)
        return self.engine.read_table(self.inputs.task['task_id'],rid)

    def queries(self):return read(self.run/'query-plan.json')

    def query_budget(self):
        path=self.run/'model-requests.jsonl'
        if not path.exists():raise ValueError('replay missing query budget log')
        records=[__import__('json').loads(line) for line in path.read_text().splitlines()]
        queries=[r for r in records if r['stage']=='query']
        return {'requests':len(queries),'input_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in queries),
                'output_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in queries),
                'source':str(path),'notice':'Inherited original query request charges; replay itself makes no query/embedding calls.'}
