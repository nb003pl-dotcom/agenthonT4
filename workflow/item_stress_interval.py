"""Uncalibrated non-repetition stress envelope; never alter forecast points."""
from copy import deepcopy
from datetime import date
import math,re
from .contract import build_contract
from .reported_item_sensitivity import parse_sentence,build as source_items


def finite(v):return type(v) in (int,float) and math.isfinite(v)


def bounded_effect(point_eps,historical_eps,effect):
    if not all(finite(v) for v in (point_eps,historical_eps,effect)):raise ValueError('bounded_nonfinite_input')
    clean=historical_eps-effect
    if not finite(clean):raise ValueError('bounded_arithmetic_overflow')
    if effect==0:return 0.0
    direction=1 if effect>0 else -1
    remaining=direction*(point_eps-clean)
    if not finite(remaining):raise ValueError('bounded_arithmetic_overflow')
    return direction*min(abs(effect),max(0.0,remaining))


def expand(inputs,answer,report,*,source_builder=None,source_parser=None,effect_mode='full'):
    if effect_mode not in ('full','bounded'):raise ValueError('invalid_effect_mode')
    source_builder=source_builder or source_items;source_parser=source_parser or parse_sentence
    result=deepcopy(answer);audit=[];contract=build_contract(inputs.task);family=contract['family']
    if family not in ('eps_yoy_direction','eps_growth_regression'):return result,[]
    if answer.get('task_id')!=inputs.task['task_id']:raise ValueError('stress_wrong_task')
    verified={x['item_id']:x for x in source_builder(inputs)['items']}
    if any(verified.get(x['item_id'])!=x for x in report['items']):raise ValueError('stress_item_not_current_source')
    by={r['entity_id']:r for r in result['entity_predictions']}
    if len(by)!=len(result['entity_predictions']) or set(by)!=set(inputs.entities):raise ValueError('stress_wrong_roster')
    for item in report['items']:
        eid,did=item['entity_id'],item['doc_id'];entry={'item_id':item['item_id'],'entity_id':eid}
        if not inputs.allowed(eid,did):raise ValueError('stress_source_not_allowed')
        doc=inputs.docs[did]
        if doc['doc_date']!=item['doc_date'] or doc['doc_date']>inputs.task['cutoff_date']:raise ValueError('stress_source_date')
        s=item['source']
        if type(s['start']) is not int or type(s['end']) is not int or not 0<=s['start']<s['end']<=len(doc['text']) or doc['text'][s['start']:s['end']]!=s['text']:raise ValueError('stress_source_span')
        parsed,why=source_parser(s['text'])
        if why or any(item.get(k)!=v for k,v in parsed.items()):raise ValueError('stress_source_value')
        period=date.fromisoformat(item['period_end'])
        if item['duration_months']!=3 or item['period_precision']!='day' or item['period_end']>doc['doc_date']:raise ValueError('stress_source_period')
        spec=contract['entities'][eid];target=re.fullmatch(r'three months ended (\d{4}-\d{2}-\d{2})',str(spec['target_period']))
        if not target:
            audit.append({**entry,'status':'skipped','reason':'unsupported_target_period'});continue
        target=date.fromisoformat(target[1])
        if (target.year-period.year)*12+target.month-period.month!=3:
            audit.append({**entry,'status':'skipped','reason':'not_immediately_previous_calendar_quarter'});continue
        expected_unit='USD_per_share' if family=='eps_yoy_direction' else 'percent'
        if spec['target_unit']!=expected_unit:raise ValueError('stress_unsupported_unit')
        row=by[eid];p=row.get('point_forecast');band=row['interval']
        if not all(finite(v) for v in [p,band.get('lo'),band.get('hi')]) or not band['lo']<=p<=band['hi']:raise ValueError('stress_invalid_forecast')
        # Recompute effects from source inputs, never trust saved scenario numbers.
        if item.get('source_type')=='disclosed_eps_effect':
            es=item['input_sources']['eps']
            if any(es.get(k)!=item[k] for k in ('entity_id','doc_id','duration_months','period_end','period_precision')):raise ValueError('stress_input_period_mismatch')
            if not finite(item['eps_effect']):raise ValueError('stress_invalid_financial_inputs')
            effects=[('disclosed_diluted_eps_effect',item['eps_effect'])]
        else:
            es=item['input_sources']['eps'];sh=item['input_sources']['shares'];tx=item['input_sources']['tax_check']
            for v in (es,sh,tx):
                if any(v.get(k)!=item[k] for k in ('entity_id','doc_id','duration_months','period_end','period_precision')):raise ValueError('stress_input_period_mismatch')
            den=sh['displayed_value']*{'millions':1e6,'thousands':1e3}[sh['scale']];rate=tx.get('effective_tax_rate')
            if not finite(den) or den<=0 or not finite(rate) or not 0<=rate<=1 or tx.get('status')!='derived' or not tx.get('usable_for_ordinary_sensitivity'):raise ValueError('stress_invalid_financial_inputs')
            effects=[(name,(1 if item['direction']=='gain' else -1)*item['amount_report_currency']*(1-tax)/den) for name,tax in [('zero_tax_assumption',0),('historical_overall_tax_assumption',rate)]]
        baseline=spec['comparison_baseline']
        if family=='eps_growth_regression' and (not finite(baseline) or baseline==0):raise ValueError('stress_invalid_baseline')
        point_eps=p if family=='eps_yoy_direction' else baseline*(1+p/100)
        if effect_mode=='bounded' and (not finite(point_eps) or not finite(es.get('value'))):raise ValueError('bounded_nonfinite_eps')
        before=deepcopy(band);scenarios=[]
        for name,effect in effects:
            used=bounded_effect(point_eps,es['value'],effect) if effect_mode=='bounded' else effect
            delta=used if family=='eps_yoy_direction' else 100*used/baseline
            value=p-delta
            if not finite(value):raise ValueError('stress_nonfinite_result')
            scenario={'assumption':name,'eps_effect':effect,'forecast_if_nonrepeat_assumption':value}
            if effect_mode=='bounded':scenario.update(effect_mode='bounded',point_eps=point_eps,historical_eps=es['value'],historical_eps_without_item=es['value']-effect,remaining_assumed_eps_effect=used,warning='Caps stress under an assumed already-adjusted interpretation; not proof of forecast composition or calibrated coverage.')
            scenarios.append(scenario)
            band['lo']=min(band['lo'],value);band['hi']=max(band['hi'],value)
        audit.append({**entry,'status':'expanded' if before!=band else 'already_contains_stress','before':before,'after':deepcopy(band),'scenarios':scenarios,
           'assumption':'Original point hypothetically includes a repeat of this historical item; isolate its non-repetition. Direct EPS effects use the disclosed value; monetary items assume unchanged shares and two hypothetical tax rates. Not additive, not calibrated coverage, not a new point estimate.'})
    return result,audit
