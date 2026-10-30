"""Opt-in pre-announcement evidence profile, keyed by task family, not company.

Improves information selection, not a mapping of good earnings to positive
returns. Market expectations may be unobserved in the frozen corpus.
"""


def enabled(config, contract):
    return bool(config.get('reaction_evidence')) and contract.get('family')=='post_earnings_reaction'


QUERY = """
Pre-announcement reaction profile: derive each entity's retrieval needs from the task and use the
available six-to-eight query capacity. Prioritize predictive inputs in this order:
1. latest already disclosed consolidated revenue level and year-over-year change, plus consolidated operating income level and year-over-year change, with the complete comparison period;
2. an expectation proxy published before the cutoff, such as management guidance or a sourced analyst/investor expectation, keeping those source types distinct;
3. entity-specific quantified segment or unit drivers. When the business mechanism has both, use separate rows for volume and price or unit-economics rather than a generic revenue row; also consider margin, cost, or one-off drivers;
4. material company-specific risk or counterevidence.
Treat the first group as historical derived change measures, not raw levels.
Prefer derived changes and expectation gaps over redundant raw levels. An outcome-shaped query may
use remaining capacity only after the pre-cutoff predictive needs above are covered; it cannot replace
them. The corpus cutoff and citation checks remain the authority on admissible evidence. A single
pre-event share price is neither return history nor a measured market expectation.
Every derived metric must include its base measure and operation, and the overall plan must be
entity-specific rather than a copied generic list.
Infer the latest already disclosed fiscal period from the event date and entity calendar. Never label
the quarter being announced as a historical actual. Historical results must come from the previous earnings release
and therefore precede the pending quarter; if exact fiscal dates are uncertain, name
that previous disclosed quarter rather than inventing calendar dates. Do not spend a row restating a
target definition that the task already provides.
"""

SELECT = """
Pre-announcement reaction selection: secure quantitative consolidated results and profit/margin drivers with complete period headers. Preserve operating-income outlook as well as sales guidance. Prefer quantified, company-specific counterevidence over inventory/payment/stock-volatility boilerplate. Distinguish annual and quarterly results and consolidated versus segment profits. Cash balances or historical repurchases are not forward guidance just because they support a company. Search previews are incomplete; choose the best complementary source windows, not a predicted label. Do not discard relevant negative evidence or invent consensus expectations.
"""

FACTS = """
Pre-announcement reaction facts: retain up to four decisive facts in this window. Prioritize (1) latest disclosed consolidated sales and profit changes with period comparison; (2) a distinct quantified margin/cost/segment operating driver; (3) target-period management guidance, including income or expense ranges as well as sales; (4) a sourced market-expectations measure or material counterevidence. Empty slots can use other relevant facts, not invented values. Do not spend all slots on boilerplate risks while a quantitative operating result is available.
For each fact add information_role: historical_result|management_guidance|market_expectation|event_risk|definition|unknown. Historical cash, revenues, repurchases and dated share prices are NOT management_guidance. Management's own guidance is NOT market_expectation; that role needs an actual cited analyst/investor expectation. Preserve missing consensus or prior-event return history as unresolved_questions rather than infer them from a share-price level.
Also add reporting_scope: consolidated|segment|unknown; segment facts need segment_name. Read actual headers before assigning period and comparison_period; add comparison_kind: yoy|qoq|ytd_yoy|none|unknown and comparison_value only when explicitly supported. Never relabel annual or year-to-date changes as quarterly. Cite enough row/header context to identify the amounts and periods. Short factual statements only; forecasts about the pending release are not extracted facts.
"""

PREDICT = """
Pre-announcement reaction task: distinguish operating performance, public management guidance, market expectations and the as-yet-unknown announcement. Historical growth alone does not prove a positive abnormal return: it may be priced in. Weigh margin/cost momentum and material counterevidence; reason about what the next release could change relative to the available expectation proxy. If no cited consensus exists, state that gap; never invent a consensus or claim an observed earnings surprise. Guidance and historical cash are not interchangeable.
Consider plausible positive and negative announcement surprises before assigning probabilities. Missing information is uncertainty, not evidence that the narrow flat band is most probable. Use the task's exact event window, benchmark subtraction and class thresholds. The point and 90% interval are abnormal return in percent, not EPS/revenue growth or label confidence. Do not infer daily volatility from a single share-price level. Do not use or recall the pending release's realized results, buybacks/dividends, or subsequent stock returns. Preserve the existing concise JSON contract and this entity's allowed fact IDs; add no extra output fields.
"""


def query_prompt(prompt, compact):
    return prompt+'\n'+QUERY


def coverage(facts, entity_ids):
    """Metadata inventory only, never a semantic correctness certificate."""
    return {eid:{'roles':sorted({f['information_role'] for f in facts if f['entity_id']==eid and f.get('information_role')}),
                 'fact_ids':[f['fact_id'] for f in facts if f['entity_id']==eid],
                 'semantic_status':'unverified'} for eid in entity_ids}
