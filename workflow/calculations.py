"""Deterministic numeric checks; no execution of model-provided Python."""
import ast
import operator
from .prediction import finite

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,
     ast.Gt:operator.gt,ast.GtE:operator.ge,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Eq:operator.eq}


def expression(source, values):
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return n.value
        if isinstance(n,ast.Name) and finite(values.get(n.id)):return values[n.id]
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:return OPS[type(n.op)](visit(n.left),visit(n.right))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -visit(n.operand)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='abs' and len(n.args)==1 and not n.keywords:return abs(visit(n.args[0]))
        if isinstance(n,ast.Compare):
            left=visit(n.left)
            for op,right in zip(n.ops,n.comparators):
                if type(op) not in OPS:raise ValueError('unsupported_comparison')
                value=visit(right)
                if not OPS[type(op)](left,value):return False
                left=value
            return True
        raise ValueError('unsupported_expression_node')
    return visit(ast.parse(source,mode='eval').body)


def recompute_labels(inputs, rows):
    audit=[]
    if inputs.task['target']['type']!='classification':return audit
    for row in rows:
        entity=inputs.entities[row['entity_id']]
        conditions={label:entity.get(label+'_condition') for label in inputs.task['target'].get('labels',[])}
        if not all(isinstance(v,str) for v in conditions.values()):
            audit.append({'entity_id':row['entity_id'],'status':'no_explicit_machine_conditions'});continue
        # This calculator binds reported_eps only. Other task families may
        # supply conditions over a different future variable; never guess that
        # its numerical meaning is EPS, or fail an otherwise valid probability
        # decision merely because this optional diagnostic cannot evaluate it.
        values={k:v for k,v in entity.items() if finite(v)}
        names=set()
        for condition in conditions.values():
            names.update(n.id for n in ast.walk(ast.parse(condition,mode='eval')) if isinstance(n,ast.Name))
        unbound=names-set(values)-{'reported_eps','abs'}
        if unbound:
            probabilities=row.get('class_probabilities')
            if probabilities is not None:
                labels=list(conditions)
                if (not isinstance(probabilities,dict) or set(probabilities)!=set(labels)
                    or any(not finite(p) or not 0<=p<=1 for p in probabilities.values())
                    or abs(sum(probabilities.values())-1)>.02):
                    raise ValueError('invalid_class_probabilities_at_publication')
                row['label']=max(labels,key=lambda label:probabilities[label])
            audit.append({'entity_id':row['entity_id'],'status':'condition_diagnostic_not_evaluated',
                'reason':'unbound_future_or_task_variables','unbound_variables':sorted(unbound),
                'published_label':row['label'],'probability_decision_retained':probabilities is not None})
            continue
        # The task owns expression names. The model must explicitly supply the
        # underlying reported_eps estimate, not a class-probability interval.
        if not finite(row.get('point_forecast')):raise ValueError('underlying_estimate_required_for_conditions')
        values['reported_eps']=row['point_forecast']
        matched=[label for label,condition in conditions.items() if expression(condition,values)]
        if len(matched)!=1:raise ValueError('classification_conditions_not_unique')
        before=row['label']
        probabilities=row.get('class_probabilities')
        if probabilities is not None:
            # Conditions label a REALIZED outcome. A forecast median can lie in
            # a narrow middle band even when either tail has more probability.
            labels=list(conditions)
            if (not isinstance(probabilities,dict) or set(probabilities)!=set(labels)
                or any(not finite(p) or not 0<=p<=1 for p in probabilities.values())
                or abs(sum(probabilities.values())-1)>.02):
                raise ValueError('invalid_class_probabilities_at_publication')
            row['label']=max(labels,key=lambda label:probabilities[label])
            status='probability_decision'
        else:
            # Preserve the legacy deterministic path when no distribution exists.
            row['label']=matched[0];status='recomputed'
        audit.append({'entity_id':row['entity_id'],'status':status,'before':before,'after':row['label'],
                      'point_condition_label':matched[0],
                      'point_and_probability_disagree':row['label']!=matched[0],
                      'point_forecast':row['point_forecast'],'conditions':conditions})
    return audit


def derive_eps_growth(data, contract):
    """Model estimates future EPS; Python computes growth in the task's units."""
    if contract.get('family')!='eps_growth_regression':return data
    for row in data.get('predictions',[]):
        eid=row.get('entity_id');spec=contract['entities'].get(eid)
        if spec is None:continue  # Roster validator reports unexpected entities.
        baseline=spec['comparison_baseline'];eps=row.get('underlying_eps');band=row.get('eps_interval',{})
        if not finite(baseline) or baseline==0:raise ValueError('EPS growth needs a nonzero task comparison baseline')
        if not isinstance(band,dict):raise ValueError(f'{eid}: eps_interval must be an object {{lo,hi}} in USD/share, not null or a list')
        if not finite(eps) or not finite(band.get('lo')) or not finite(band.get('hi')):
            raise ValueError(f'{eid}: supply finite underlying_eps and eps_interval={{lo,hi}}, ALL in USD/share; got underlying_eps={eps}, eps_interval={band}. Return the full requested roster.')
        if not band['lo']<=eps<=band['hi']:
            raise ValueError(f"{eid}: EPS point {eps} is outside eps_interval [{band['lo']}, {band['hi']}] USD/share. Require eps_interval.lo <= underlying_eps <= eps_interval.hi. Recheck this entity's EPS forecast and bounds; do not repeat the same incompatible values. Python computes YoY growth using TASK prior_year_q_eps={baseline}; do not substitute the latest quarter's YoY rate. Return the full requested roster.")
        convert=lambda x:100*(x/baseline-1)
        row['point_forecast']=convert(eps);a,b=convert(band['lo']),convert(band['hi'])
        row['interval']={'lo':min(a,b),'hi':max(a,b)};row['unit']='percent'
        row['calculation']={'baseline_eps':baseline,'forecast_eps':eps,'formula':'100 * (forecast_eps / baseline_eps - 1)'}
    return data
