"""Compact structured table-header evidence discovered by preprocessing."""
import json
from pathlib import Path
from types import SimpleNamespace
import re


def _clean(value, limit):
    return " ".join(str(value or "").split())[:limit]


def build_table_header_inventory(retrieval, *, allowed_entity_ids=None, max_tables=80, max_chars=12000):
    if type(max_tables) is not int or max_tables < 0 or type(max_chars) is not int or max_chars < 0:
        raise ValueError('table header inventory limits must be nonnegative integers')
    contexts={};parts={}
    for chunk in retrieval.chunks:
        for ref in chunk.get('table_refs',[]):
            rid=ref.get('ref_id')
            if not rid:continue
            context=_clean(chunk.get('table_context'),320)
            if context:contexts.setdefault(rid,context)
            if chunk.get('node_type')=='table':parts[rid]=parts.get(rid,0)+1
    allowed=set(allowed_entity_ids) if allowed_entity_ids is not None else None
    rows=[]
    for rid,item in retrieval.tables.items():
        ref=item.get('ref',{})
        source=ref.get('source',{})
        did=source.get('doc_id')
        relation=retrieval.inputs.relations.get(did,{})
        header=[]
        for value in ref.get('header',[]):
            value=_clean(value,160)
            if value and value not in header:header.append(value)
        routed=list(dict.fromkeys(relation.get('entity_ids',[])))
        if not did or not header or (allowed is not None and not relation.get('shared',False) and not allowed.intersection(routed)):continue
        rows.append({'ref_id':rid,'doc_id':did,
            'entity_ids':routed,
            'shared':bool(relation.get('shared',False)),
            'header':header,'table_context':contexts.get(rid,''),
            'row_count':ref.get('row_count'),'column_count':ref.get('column_count'),
            'index_parts':parts.get(rid,0)})
    rows.sort(key=lambda r:(r['shared'],not bool(r['entity_ids']),r['doc_id'],r['ref_id']))
    selected=[];used=0
    for row in rows:
        if len(selected)>=max_tables:break
        cost=len(json.dumps(row,ensure_ascii=False,separators=(',',':')))
        if used+cost>max_chars:continue
        selected.append(row);used+=cost
    return {'tables':selected,'audit':{'source_tables':len(rows),'selected_tables':len(selected),
        'omitted_tables':len(rows)-len(selected),'selected_characters':used,
        'max_tables':max_tables,'max_characters':max_chars}}


def load_index_table_header_inventory(index_folder, relations, **limits):
    """Read the exact table refs persisted by a completed lexical index."""
    folder=Path(index_folder)
    tables={};chunks=[]
    for path in sorted((folder/'documents').glob('*/result.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        for node in data.get('result',[]):
            chunks.append(node)
            for ref in node.get('table_refs',[]):
                rid=ref.get('ref_id')
                if rid:tables.setdefault(rid,{'ref':ref,'result_path':path})
    retrieval=SimpleNamespace(inputs=SimpleNamespace(relations=relations),tables=tables,chunks=chunks)
    return build_table_header_inventory(retrieval,**limits)


def _tokens(value):
    return set(re.findall(r'[a-z0-9]+',str(value).casefold().replace('_',' ')))


def enrich_workflow_queries(queries, inventory, *, max_terms=6):
    """Add index header vocabulary to lexical text without changing planning needs."""
    from copy import deepcopy
    result=deepcopy(queries)
    for query in result:
        entities=set(query.get('entity_ids',[]))
        needle=_tokens(query.get('dense_query','')+' '+query.get('purpose',''))
        candidates=[]
        for table in inventory.get('tables',[]):
            if not table.get('shared') and not entities.intersection(table.get('entity_ids',[])):continue
            fields=[]
            for index,field in enumerate(table.get('header',[])):
                overlap=len(needle.intersection(_tokens(field)))
                if overlap or not table.get('shared'):fields.append((-overlap,index,field))
            if fields:candidates.append((table,fields))
        terms=[];refs=[]
        for table,fields in candidates:
            refs.append(table['ref_id'])
            for _,_,field in sorted(fields):
                if field not in terms:terms.append(field)
        if not terms:continue
        terms=terms[:max_terms]
        query['lexical_query']=' '.join([query['lexical_query'],*terms])[:200]
        query['table_header_terms']=terms
        query['table_ref_ids']=list(dict.fromkeys(refs))
    return result
