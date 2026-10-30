"""Lossless, family-independent predict-input wire representations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from xml.etree.ElementTree import Element,SubElement,fromstring,tostring


WIRE_FORMAT='predict-input-compact-v1'
XML_NAME=re.compile(r'^(?![Xx][Mm][Ll])[_A-Za-z][_A-Za-z0-9.-]*$')
ROOT=Path(__file__).resolve().parents[1]


def _window_key(window):
    return window.get('doc_id'),window.get('doc_date')


def _compatible(left,right):
    if _window_key(left)!=_window_key(right):return False
    a,b=left['span_start'],left['span_end'];c,d=right['span_start'],right['span_end']
    if c>b:return False
    overlap=min(b,d)-max(a,c)
    if overlap<=0:return True
    left_at=max(a,c)-a;right_at=max(a,c)-c
    return left['text'][left_at:left_at+overlap]==right['text'][right_at:right_at+overlap]


def _merge(left,right):
    start=min(left['span_start'],right['span_start']);end=max(left['span_end'],right['span_end'])
    chars=[None]*(end-start)
    for window in (left,right):
        offset=window['span_start']-start
        for index,char in enumerate(window['text']):
            position=offset+index
            if 0<=position<len(chars):chars[position]=char
    if any(char is None for char in chars):raise ValueError('source window coordinates contain an uncovered gap')
    return {'doc_id':left.get('doc_id'),'doc_date':left.get('doc_date'),'span_start':start,
            'span_end':end,'text':''.join(chars)}


def compact_payload(payload):
    """Intern overlapping source text while retaining exact window metadata."""
    result=deepcopy(payload);context=result.pop('decisive_source_context',None)
    compact={'wire_format':WIRE_FORMAT,'payload':result}
    if context is None:return compact
    windows=[]
    for entity,items in context.get('by_entity',{}).items():
        for order,window in enumerate(items):
            required=('span_start','span_end','text')
            if any(key not in window for key in required):raise ValueError('source window missing span or text')
            if window['span_end']-window['span_start']!=len(window['text']):
                raise ValueError('source window span length does not match text length')
            windows.append((entity,order,deepcopy(window)))
    registry=[]
    for _,_,window in sorted(windows,key=lambda x:(_window_key(x[2]),x[2]['span_start'],x[2]['span_end'],x[0],x[1])):
        core={key:window.get(key) for key in ('doc_id','doc_date','span_start','span_end','text')}
        merged=False
        for index,item in enumerate(registry):
            if _compatible(item,core):
                registry[index]=_merge(item,core);merged=True;break
        if not merged:registry.append(core)
    registry.sort(key=lambda x:(_window_key(x),x['span_start'],x['span_end']))
    for index,item in enumerate(registry):item['source_id']=f's{index+1}'
    by_entity={entity:[] for entity in context.get('by_entity',{})}
    for entity,order,window in windows:
        matches=[item for item in registry if _window_key(item)==_window_key(window)
                 and item['span_start']<=window['span_start'] and item['span_end']>=window['span_end']
                 and item['text'][window['span_start']-item['span_start']:window['span_end']-item['span_start']]==window['text']]
        if len(matches)!=1:raise ValueError('source window did not map uniquely to registry')
        item=matches[0]
        by_entity[entity].append({'source_id':item['source_id'],'span_start':window['span_start'],
            'span_end':window['span_end'],'selection_reason':window.get('selection_reason'),
            'related_fact_ids':deepcopy(window.get('related_fact_ids',[])),'order':order})
    compact['source_context']={'scope':context.get('scope'),'registry':registry,'by_entity':by_entity}
    return compact


def restore_payload(compact):
    if compact.get('wire_format')!=WIRE_FORMAT:raise ValueError('unknown predict input wire format')
    payload=deepcopy(compact['payload']);source=compact.get('source_context')
    if source is None:return payload
    registry={item['source_id']:item for item in source['registry']}
    by_entity={}
    for entity,refs in source['by_entity'].items():
        restored=[]
        for ref in sorted(refs,key=lambda x:x['order']):
            item=registry[ref['source_id']];start=ref['span_start'];end=ref['span_end']
            text=item['text'][start-item['span_start']:end-item['span_start']]
            restored.append({'doc_id':item.get('doc_id'),'doc_date':item.get('doc_date'),
                'span_start':start,'span_end':end,'text':text,
                'selection_reason':ref.get('selection_reason'),
                'related_fact_ids':deepcopy(ref.get('related_fact_ids',[]))})
        by_entity[entity]=restored
    payload['decisive_source_context']={'scope':source.get('scope'),'by_entity':by_entity}
    return payload


def serialize_compact_json(compact):
    return json.dumps(compact,ensure_ascii=False,separators=(',',':'))


def _xml(parent,key,value):
    node=(SubElement(parent,key) if XML_NAME.fullmatch(key)
          else SubElement(parent,'field',{'name':key}))
    if value is None:node.set('type','null')
    elif isinstance(value,bool):node.set('type','bool');node.text='true' if value else 'false'
    elif isinstance(value,(int,float)):node.set('type','number');node.text=json.dumps(value)
    elif isinstance(value,str):node.text=value
    elif isinstance(value,list):
        node.set('type','array')
        for item in value:_xml(node,'item',item)
    elif isinstance(value,dict):
        node.set('type','object')
        for child_key,child_value in value.items():_xml(node,child_key,child_value)
    else:raise TypeError('unsupported XML value: '+type(value).__name__)
    return node


def serialize_compact_xml(compact):
    if compact.get('wire_format')!=WIRE_FORMAT:raise ValueError('unknown predict input wire format')
    root=Element('predict_input',{'wire_format':WIRE_FORMAT})
    for key,value in compact.items():
        if key!='wire_format':_xml(root,key,value)
    return tostring(root,encoding='unicode',short_empty_elements=True)


def _from_xml(node):
    kind=node.get('type')
    if kind=='null':return None
    if kind=='bool':
        if node.text not in ('true','false'):raise ValueError('invalid XML bool')
        return node.text=='true'
    if kind=='number':return json.loads(node.text or '')
    if kind=='array':
        if any(child.tag!='item' for child in node):raise ValueError('invalid XML array member')
        return [_from_xml(child) for child in node]
    if kind=='object':
        result={}
        for child in node:
            key=child.get('name') if child.tag=='field' else child.tag
            if key is None:raise ValueError('XML field missing name')
            if key in result:raise ValueError('duplicate XML object key: '+key)
            result[key]=_from_xml(child)
        return result
    if kind is not None:raise ValueError('unknown XML type: '+kind)
    if len(node):raise ValueError('untyped XML scalar has children')
    return node.text or ''


def deserialize_compact_xml(content):
    root=fromstring(content)
    if root.tag!='predict_input' or root.get('wire_format')!=WIRE_FORMAT:
        raise ValueError('unknown predict input XML wire format')
    result={'wire_format':WIRE_FORMAT}
    for child in root:
        key=child.get('name') if child.tag=='field' else child.tag
        if key is None:raise ValueError('XML field missing name')
        if key in result:raise ValueError('duplicate XML root key: '+key)
        result[key]=_from_xml(child)
    return result


def encode_input_wire(payload,mode):
    """Losslessly encode a generic model-stage payload."""
    if mode=='original-json':return payload,None
    if mode not in ('compact-json','compact-xml'):
        raise ValueError('unknown prediction input wire: '+str(mode))
    compact=compact_payload(payload)
    content=serialize_compact_json(compact) if mode=='compact-json' else serialize_compact_xml(compact)
    decoded=(json.loads(content) if mode=='compact-json' else deserialize_compact_xml(content))
    if restore_payload(decoded)!=payload:raise ValueError('predict input wire round-trip mismatch')
    original=json.dumps(payload,ensure_ascii=False)
    from .model import RawUserContent
    audit={'mode':mode,'wire_format':WIRE_FORMAT,'round_trip_exact':True,
        'original_characters':len(original),'wire_characters':len(content),
        'character_ratio':len(content)/len(original) if original else None,
        'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'wire_sha256':hashlib.sha256(content.encode()).hexdigest()}
    return RawUserContent(content),audit


def apply_predict_input_wire(system,payload,mode):
    """Apply an optional wire at the final prediction boundary."""
    wire,audit=encode_input_wire(payload,mode)
    if audit is None:return system,wire,None
    guide=(ROOT/'prompts/predict-input-wire-v1.md').read_text()
    return system+'\n\n'+guide,wire,audit
