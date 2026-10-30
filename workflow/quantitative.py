"""Small, cutoff-safe statistical anchors. Never loads labels/reference files.

History is taken from the supplied corpus and official entity relation only.
Every estimate records its source and assumptions; short-history intervals have
no claimed finite-sample coverage guarantee.
"""
from datetime import date
import math
import re
import numpy as np
from scipy.stats import t as student_t

def number(s):
    text=str(s).strip().replace(',','').replace('−','-')
    if text in ('','--','—','N/A'):return None
    if text.startswith('(') and text.endswith(')'):text='-'+text[1:-1]
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?',text):return None
    value=float(text.rstrip('%'))
    return value if math.isfinite(value) else None

def norm(s):return re.sub(r'[^a-z0-9]','',str(s).lower())

def tenor_bucket(text):
    m=re.fullmatch(r'(?i)(\d+)-Year(?:\s+(\d+)-Month)?',str(text))
    return int(m[1])+(1 if m[2] and int(m[2])>0 else 0) if m else norm(text)

def center(history):
    a=np.asarray(history,dtype=float)
    return float(np.median([np.median(a),np.median(a[-3:]),np.mean(a[-6:])]))

def distribution(point, residuals, level=.9, floor=0):
    a=np.asarray(residuals,dtype=float)
    if len(a)<2:return None
    bias=float(np.median(a))
    robust=1.4826*float(np.median(np.abs(a-bias)))
    # Sparse history: do not let MAD collapse for rounded/repeated values.
    sigma=max(robust,float(np.std(a,ddof=1)),float(floor),1e-9)
    df=max(3,len(a)-1)
    half=float(student_t.ppf((1+level)/2,df))*sigma*math.sqrt(1+1/len(a))
    return {'point':float(point),'lo':float(point-half),'hi':float(point+half),
        'sigma':sigma,'residual_count':len(a),'residual_median':bias,'level':level,
        'method':'Student-t predictive scale of pre-cutoff rolling errors; no exchangeability/coverage guarantee'}

def series_anchor(values,level=.9):
    if len(values)<6:return None
    residuals=[values[i]-center(values[:i]) for i in range(4,len(values))]
    if len(residuals)<2:return None
    point=center(values)
    d=distribution(point,residuals,level,floor=float(np.std(values,ddof=1))*.25)
    return {**d,'rolling_mae':float(np.mean(np.abs(residuals))),
        'observations':len(values),'candidates':{'full_median':float(np.median(values)),
            'recent3_median':float(np.median(values[-3:])), 'recent6_mean':float(np.mean(values[-6:]))},
        'selection':'Fixed median combination; no outcome-driven parameter selection'}

def revision_age_diagnostic(rows,vintages,entity,excluded_vintages=()):
    """Compare the same publication ages and horizon, including unchanged values.

    Snapshot column number is not revision age: an already mature observation
    can first appear in column one. This is a transparent sparse-data diagnostic,
    not a calibrated forecast or an outcome-selected replacement for the anchor.
    """
    def month_index(s):return int(s[:4])*12+int(s[5:7])
    reference=entity['ref_month'];start=entity['latest_precutoff_vintage'];end=entity['resolving_release_date']
    start_age=month_index(start)-month_index(reference);end_age=month_index(end)-month_index(reference)
    observations=[]
    for row in rows:
        if not re.fullmatch(r'\d{4}-\d{2}',str(row[0])):continue
        available=[(v,number(x)) for v,x in zip(vintages,row[1:]) if number(x) is not None]
        a=[(v,x) for v,x in available if month_index(v)-month_index(row[0])==start_age]
        b=[(v,x) for v,x in available if month_index(v)-month_index(row[0])==end_age and v not in excluded_vintages]
        if not a or not b:continue
        av,ax=min(a,key=lambda z:abs(int(z[0][8:10])-int(start[8:10])))
        bv,bx=min(b,key=lambda z:abs(int(z[0][8:10])-int(end[8:10])))
        if av>=bv or bv>entity['latest_precutoff_vintage']:continue
        observations.append({'reference_month':row[0],'from_vintage':av,'to_vintage':bv,'delta':bx-ax})
    return {'start_release_age_months':start_age,'end_release_age_months':end_age,
        'observations':observations,'sample_count':len(observations),
        'median_delta':float(np.median([o['delta'] for o in observations])) if observations else None,
        'assumption':'Same calendar publication ages and nearest release day; sparse history, no calibrated interval or guaranteed release-stage alignment. Includes zero revisions; excludes identified annual end vintages.'}

class Quantitative:
    def __init__(self,inputs,contract):
        from table_extraction import extract_tables_with_diagnostics
        self.inputs=inputs;self.contract=contract;self.tables={}
        for did,doc in inputs.docs.items():
            self.tables[did]=extract_tables_with_diagnostics(doc['text'])[0]

    def build(self):
        return {eid:self.entity(eid,e) for eid,e in self.inputs.entities.items()}

    def entity(self,eid,e):
        family=self.contract['family'];level=self.contract['interval_level']
        options=[]
        for did,tables in self.tables.items():
            if not self.inputs.allowed(eid,did):continue
            for table in tables:
                header=table['header'];keys=[norm(h) for h in header];rows=table['rows']; result=None
                if family=='auction_demand' and 'bidtocover' in keys and 'term' in keys:
                    col=keys.index('bidtocover');term=keys.index('term')
                    ordered=[r for r in rows if tenor_bucket(r[term])==tenor_bucket(e.get('tenor')) and r[0]<=self.contract['cutoff_date']]
                    values=[number(r[col]) for r in sorted(ordered,key=lambda r:r[0])]
                    if values and all(v is not None for v in values):result=series_anchor(values,level)
                elif family=='cpi_component_nowcast' and keys[0]=='month' and norm(e.get('name')) in keys:
                    col=keys.index(norm(e['name']));ordered=sorted([r for r in rows if r[0]<e['ref_month']],key=lambda r:r[0])
                    values=[number(r[col]) for r in ordered]
                    if values and all(v is not None for v in values):
                        result=series_anchor(values,level)
                        if result and re.search(r'(?i)gasoline',e.get('name','')):
                            result['contemporaneous_price_driver']=self.gasoline_driver(e,ordered,col,level)
                elif family=='positioning_shift' and all(k in keys for k in ('reportdate','openinterest','noncommnet')):
                    result=self.positioning(e,table,level)
                elif family=='macro_revision_direction' and keys[0]=='referencemonth':
                    result=self.revisions(e,did,table,level)
                elif family=='rate_curve_cross_section' and norm(e.get('series_fred')) in keys and keys[0]=='date':
                    result=self.rates(e,did,table,level)
                if result:
                    options.append({**result,'doc_id':did,'source_start':table['start'],'source_end':table['end'],
                        'source_header':header,'doc_date':self.inputs.docs[did]['doc_date']})
        if not options and family=='rate_curve_cross_section':
            for did,doc in self.inputs.docs.items():
                if self.inputs.allowed(eid,did):
                    context=self.rate_context(e,doc['text'],level)
                    if context:options.append({**context,'doc_id':did,'source_start':0,'source_end':len(doc['text']),'doc_date':doc['doc_date']})
        if not options:return {'status':'unsupported_or_insufficient_history','unit':self.contract['entities'][eid]['target_unit'],
            'notice':'No suitable frozen numeric history; use evidence-grounded model scenarios, not an invented fitted model.'}
        best=max(options,key=lambda r:(r.get('observations',0),r['doc_date']))
        if family=='rate_curve_cross_section':
            contexts=[self.rate_context(e,d['text'],level) for did,d in self.inputs.docs.items() if self.inputs.allowed(eid,did)]
            best['regime_context']=[c['regime_context'] for c in contexts if c]
        return {'status':'available','unit':self.contract['entities'][eid]['target_unit'],**best,
            'family':family,'information_source':'frozen official corpus only; no outcome access'}

    def positioning(self,e,table,level):
        keys=[norm(h) for h in table['header']];i_net=keys.index('noncommnet');i_oi=keys.index('openinterest')
        rows=sorted([r for r in table['rows'] if r[0]<=self.contract['cutoff_date']],key=lambda r:r[0])
        net=[number(r[i_net]) for r in rows];oi=[number(r[i_oi]) for r in rows]
        if len(rows)<12 or any(v is None for v in net+oi) or any(v<=0 for v in oi):return None
        # Determine forecast report dates from the literal task statement, rather
        # than the reporting/publication resolution date (often a later Monday).
        target_date=re.search(r'(?i)\bto\s+(?:the\s+)?(20\d{2}-\d{2}-\d{2})\s+COT report',self.inputs.task['prompt'])
        if target_date is None:return None
        horizon=max(1,round((date.fromisoformat(target_date[1])-date.fromisoformat(rows[-1][0])).days/7))
        if horizon>12:return None
        def forecast(n):
            recent=net[max(0,n-6):n+1]
            revert=float(np.median(recent))-net[n]
            momentum=(net[n]-net[max(0,n-4)])*horizon/max(1,min(4,n))*.5
            return float(np.median([0,revert,momentum]))/oi[n]*100
        residuals=[(net[i+horizon]-net[i])/oi[i]*100-forecast(i) for i in range(6,len(net)-horizon)]
        if len(residuals)<3:return None
        # Entity-provided starting OI must agree with the table; otherwise refuse
        # to apply a moving-denominator change in place of the task definition.
        starts=[v for k,v in e.items() if k.startswith('open_interest_')]
        if starts and abs(starts[0]-oi[-1])>1e-6:return None
        d=distribution(forecast(len(net)-1),residuals,level)
        return {**d,'observations':len(net),'horizon_weeks':horizon,'fixed_start_open_interest':oi[-1],
            'current_net_contracts':net[-1],'rolling_mae':float(np.mean(np.abs(residuals))),
            'assumption':'Median of no-change, recent-level reversion and half-strength four-week momentum.'}

    def gasoline_driver(self,e,cpi_rows,column,level):
        """Ridge nowcast from frozen pump prices, with past-only rolling checks."""
        history={r[0]:number(r[column]) for r in cpi_rows};target=e['ref_month']
        for did,tables in self.tables.items():
            if not self.inputs.allowed(e['entity_id'],did):continue
            for table in tables:
                if [norm(h) for h in table['header']]!=['weekending','usdpergallon']:continue
                monthly={}
                for row in table['rows']:
                    value=number(row[1])
                    if value is not None and row[0]<=self.contract['cutoff_date']:
                        monthly.setdefault(row[0][:7],[]).append(value)
                means={m:float(np.mean(v)) for m,v in monthly.items()}
                def previous(m):
                    year,month=map(int,m.split('-'));return f'{year-1 if month==1 else year:04d}-{12 if month==1 else month-1:02d}'
                changes={m:100*(v/means[previous(m)]-1) for m,v in means.items() if previous(m) in means and means[previous(m)]>0}
                pairs=[(m,changes[m],history[m]) for m in sorted(history) if m in changes and history[m] is not None and m<target]
                if len(pairs)<6 or target not in changes:return {'status':'insufficient_comparable_history'}
                def predict(train,x):
                    xs=np.array([r[1] for r in train]);ys=np.array([r[2] for r in train]);xm=float(xs.mean());ym=float(ys.mean());scale=max(float(xs.std()),1e-6)
                    z=(xs-xm)/scale;beta=float(np.dot(z,ys-ym)/(np.dot(z,z)+1.0))
                    return ym+beta*(x-xm)/scale
                errors=[];baseline_errors=[];folds=[]
                for j in range(4,len(pairs)):
                    month,x,y=pairs[j];estimate=predict(pairs[:j],x)
                    past=[v for m,v in sorted(history.items()) if m<month and v is not None]
                    baseline=center(past);errors.append(y-estimate);baseline_errors.append(y-baseline)
                    folds.append({'month':month,'training_months':[r[0] for r in pairs[:j]],'predicted':estimate,'observed_precutoff':y,'baseline':baseline})
                point=predict(pairs,changes[target]);d=distribution(point,errors,level)
                if d is None:return {'status':'insufficient_rolling_errors'}
                mae=float(np.mean(np.abs(errors)));base_mae=float(np.mean(np.abs(baseline_errors)))
                return {**d,'status':'available','source_doc_id':did,'pump_mom_percent':changes[target],
                    'monthly_pump_means':means,'training_pairs':pairs,'rolling_folds':folds,
                    'rolling_mae':mae,'univariate_rolling_mae':base_mae,'preferred_by_pre_cutoff_mae':mae<base_mae,
                    'method':'Fixed standardized ridge with intercept and alpha=1; rolling-origin validation. No future CPI labels.',
                    'assumption':'Weekly regular-gas pump averages proxy all-type gasoline prices; seasonal adjustments and incomplete months create model risk. Sparse errors do not guarantee interval coverage.'}
        return {'status':'no_allowed_pump_price_table'}

    def revisions(self,e,did,table,level):
        doc=self.inputs.docs[did]
        if doc.get('series_id') and doc['series_id']!=e.get('series_id'):return None
        header=table['header'];vintages=[h.removeprefix('as_of_') for h in header[1:]]
        if any(not re.fullmatch(r'\d{4}-\d{2}-\d{2}',v) for v in vintages):return None
        values=[[number(v) for v in r[1:]] for r in table['rows']]
        target=next((i for i,r in enumerate(table['rows']) if r[0]==e.get('ref_month')),None)
        if target is None:return None
        observed=[(j,v) for j,v in enumerate(values[target]) if v is not None]
        if not observed or abs(observed[-1][1]-e['latest_precutoff_estimate'])>1e-6:return None
        # Some releases repeat a vintage within a month. Use changes in values,
        # not all adjacent copies, and keep stages/ages explicitly observable.
        pairs=[]
        for ri,row in enumerate(values):
            available=[(j,v) for j,v in enumerate(row) if v is not None]
            for stage,((a,x),(b,y)) in enumerate(zip(available,available[1:]),1):
                if y!=x:pairs.append({'delta':y-x,'stage':stage,'vintage':vintages[b],'reference_month':table['rows'][ri][0]})
        if len(pairs)<3:return None
        # A coordinated, exceptionally large vintage jump is an annual regime
        # candidate, only when the source itself mentions annual/benchmark revisions.
        annual_hint=bool(re.search(r'(?i)annual|benchmark',doc['text']))
        by_vintage={v:[p['delta'] for p in pairs if p['vintage']==v] for v in vintages}
        medians={v:float(np.median(np.abs(ds))) for v,ds in by_vintage.items() if ds}
        typical=float(np.median(list(medians.values()))) if medians else 0
        annual={v for v,m in medians.items() if annual_hint and len(by_vintage[v])>=3 and m>max(1e-9,typical)*10}
        routine=[p for p in pairs if p['vintage'] not in annual]
        next_stage=len(observed)
        matched=[p['delta'] for p in routine if abs(p['stage']-next_stage)<=1]
        if len(matched)<3:matched=[p['delta'] for p in routine]
        if len(matched)<3:return None
        # Same-month staged releases cannot justify pretending a July second
        # estimate is an initial August estimate. Audit the observable proxy.
        latest=date.fromisoformat(e['latest_precutoff_vintage']); resolving=date.fromisoformat(e['resolving_release_date'])
        steps=max(1,(resolving.year-latest.year)*12+resolving.month-latest.month)
        delta=float(np.median(matched))*steps
        scale_residuals=[(x-float(np.median(matched)))*math.sqrt(steps) for x in matched]
        d=distribution(e['latest_precutoff_estimate']+delta,scale_residuals,level)
        p_up=float(np.mean(np.asarray(matched)>0))
        p_up=(p_up*len(matched)+.5)/(len(matched)+1)
        return {**d,'observations':len(matched),'baseline':e['latest_precutoff_estimate'],'estimated_revision_delta':delta,
            'class_probabilities':{'up':p_up,'down':1-p_up},'revision_steps_proxy':steps,
            'next_stage_proxy':next_stage,'excluded_annual_vintages':sorted(annual),
            'release_age_diagnostic':revision_age_diagnostic(table['rows'],vintages,e,annual),
            'assumption':'Routine empirical revisions, smoothed direction frequencies; stage/window proxies are uncertain.'}

    def rate_context(self,e,text,level):
        if 'Treasury' not in text or 'percent' not in text:return None
        # Explicit source wording is passed as a regime discriminator. No fitted
        # future outcome or fixed calendar-specific direction is smuggled in.
        step=re.findall(r'(?i)(\d+) basis point (?:increase|reduction)|by (\d+) basis points',text)
        changes=[float(a or b) for a,b in step]
        if not changes:return None
        stress=max(changes)
        excerpts=[p.strip() for p in text.split('\n\n') if any(w in p.lower() for w in ('positioning context','curve shape','trailing context','monetary policy'))]
        return {'point':0.0,'lo':-2*stress,'hi':2*stress,'observations':0,'level':level,
            'policy_step_bps':stress,'regime_context':excerpts,
            'assumption':'Sparse-history stress scenario, NOT a statistically calibrated 90% band. Direction must account for what is already priced and maturity exposure.',
            'method':'Pre-cutoff policy-step and curve-regime diagnostics'}

    def rates(self,e,did,table,level):
        keys=[norm(h) for h in table['header']];col=keys.index(norm(e['series_fred']))
        rows=sorted([r for r in table['rows'] if r[0]<=self.contract['cutoff_date']],key=lambda r:r[0])
        values=[number(r[col]) for r in rows]
        if len(values)<5 or any(v is None for v in values):return None
        daily=np.diff(values)*100
        business_days=max(1,round((self.contract.get('horizon_calendar_days') or 30)*5/7))
        daily_sigma=float(np.std(daily,ddof=1))
        steps=re.findall(r'(?i)(?:by|another|raised[^.]{0,20}by|lowered[^.]{0,20}by)\s+(\d+)\s+basis points',self.inputs.docs[did]['text'])
        policy_risk=max([float(x)/2 for x in steps]+[25.0])
        sigma=math.sqrt(daily_sigma*daily_sigma*business_days+policy_risk*policy_risk)
        half=float(student_t.ppf((1+level)/2,max(3,len(daily)-1)))*sigma
        return {'point':0.0,'lo':-half,'hi':half,'sigma':sigma,'level':level,'observations':len(values),
            'daily_sigma_bps':daily_sigma,'business_day_horizon_proxy':business_days,'policy_uncertainty_bps':policy_risk,
            'assumption':'No-change point anchor; short-window diffusive scale plus explicit policy-repricing model risk, not a calibrated guarantee.',
            'method':'Numerical scale check; LLM must supply scenario direction, not mechanical policy-to-yield translation.'}

def combine(rows,packets,mode):
    """A predeclared blend experiment; no selection from the resolved answers."""
    audit=[]
    for row in rows:
        packet=packets.get(row['entity_id'],{})
        if packet.get('status')!='available':continue
        before={'point_forecast':row.get('point_forecast'),'interval':dict(row['interval'])}
        if mode=='blend' and type(row.get('point_forecast')) in (float,int):
            # Rates use a scale check, not an arbitrary zero shrinkage override.
            if packet.get('family')!='rate_curve_cross_section':
                row['point_forecast']=.5*row['point_forecast']+.5*packet['point']
            if 'class_probabilities' in packet and 'class_probabilities' in row:
                row['class_probabilities']={l:.5*p+.5*packet['class_probabilities'].get(l,0) for l,p in row['class_probabilities'].items()}
                row['label']=max(row['class_probabilities'],key=row['class_probabilities'].get)
        # Statistical uncertainty is outcome dispersion, not a mean's SE. Do not
        # shrink a model's wider evidence-grounded band. Values remain auditable.
        if mode=='blend':
            p=row.get('point_forecast',packet['point']);half=(packet['hi']-packet['lo'])/2
            row['interval']={'lo':min(row['interval']['lo'],p-half),'hi':max(row['interval']['hi'],p+half)}
        audit.append({'entity_id':row['entity_id'],'mode':mode,'before':before,
            'after':{'point_forecast':row.get('point_forecast'),'interval':row['interval']},'packet_source':packet.get('doc_id')})
    return rows,audit


def rate_interval_floor(rows, packets, contract):
    """Keep rate point estimates; preserve the existing frozen risk-scale width.

    This is an explicit candidate policy, not calibrated 90% coverage. Other
    families, missing scales and incompatible units remain unchanged. Applying
    it repeatedly is idempotent, including after an optional model review.
    """
    from .contract import canonical_unit
    if contract.get('family') != 'rate_curve_cross_section':
        return []
    audit=[]
    for row in rows:
        eid=row['entity_id'];packet=packets.get(eid,{})
        target_unit=contract.get('entities',{}).get(eid,{}).get('target_unit')
        if packet.get('status')!='available':
            audit.append({'entity_id':eid,'applied':False,'reason':'no_frozen_scale'});continue
        if (not target_unit or canonical_unit(target_unit) != canonical_unit(packet.get('unit'))
            or packet.get('level') != contract.get('interval_level')):
            audit.append({'entity_id':eid,'applied':False,'reason':'scale_contract_mismatch'});continue
        numbers=[row.get('point_forecast'),packet.get('lo'),packet.get('hi')]
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in numbers) or numbers[2]<=numbers[1]:
            audit.append({'entity_id':eid,'applied':False,'reason':'invalid_scale'});continue
        point,lo,hi=numbers;half=(hi-lo)/2;before=dict(row['interval'])
        row['interval']={**before,'lo':min(before['lo'],point-half),'hi':max(before['hi'],point+half)}
        audit.append({'entity_id':eid,'applied':before!=row['interval'],'before':before,'after':dict(row['interval']),
                      'half_width':half,'source_doc_id':packet.get('doc_id'),
                      'method':packet.get('method'),'assumption':packet.get('assumption'),
                      'notice':'Frozen numerical risk-scale floor; no claim of calibrated coverage. Point and claims unchanged.'})
    return audit
