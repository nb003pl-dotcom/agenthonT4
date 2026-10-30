"""Deterministic, task-agnostic evidence ledger for forecast requests."""
from collections import OrderedDict


def _time_relation(period, entity_contract):
    if period and period == entity_contract.get("target_period"):
        return "target_period"
    if period and period == entity_contract.get("comparison_period"):
        return "comparison_period"
    return "historical" if period else "unspecified"


def _item(fact, entity_contract):
    return {
        "fact_ids": [fact["fact_id"]],
        "metric": fact.get("metric"),
        "value": fact.get("value"),
        "unit": fact.get("unit"),
        "period": fact.get("period"),
        "time_relation": _time_relation(fact.get("period"), entity_contract),
        "basis": fact.get("basis"),
        "evidence_role": fact.get("evidence_role"),
        "statement": fact.get("statement"),
        "target_metric_match": fact.get("metric") == entity_contract.get("target_name"),
        "target_unit_match": fact.get("unit") == entity_contract.get("target_unit"),
        "derivation": fact.get("derivation"),
    }


def _deduplicate(items):
    grouped = OrderedDict()
    for item in items:
        key = tuple(item.get(name) for name in (
            "metric", "value", "unit", "period", "basis", "statement"
        ))
        if key in grouped:
            grouped[key]["fact_ids"].extend(
                fact_id for fact_id in item["fact_ids"]
                if fact_id not in grouped[key]["fact_ids"]
            )
        else:
            grouped[key] = item
    return list(grouped.values())


def _missing_inputs(entity, gaps, has_facts):
    kept = []
    for gap in gaps or []:
        if isinstance(gap, dict):
            if gap.get("entity_id") not in (None, entity):
                continue
            if has_facts and gap.get("reason") == "all_facts_rejected_after_one_repair":
                continue
        kept.append(gap)
    return kept


def build_typed_evidence(contract, facts, anchors, conflicts, gaps, entity_ids):
    ledgers = {}
    target = contract.get("target", {})
    for entity in entity_ids:
        entity_contract = {**contract.get("entities", {}).get(entity, {}), "target_name": target.get("name")}
        entity_facts = [fact for fact in facts if fact.get("entity_id") == entity]
        observations = []
        derived = []
        qualifiers = []
        for fact in entity_facts:
            item = _item(fact, entity_contract)
            (derived if fact.get("derivation") else observations).append(item)
            if fact.get("evidence_role") == "risk_counterevidence":
                qualifiers.append({"fact_ids": [fact["fact_id"]], "statement": fact.get("statement")})
        prior = anchors.get(entity)
        priors = []
        if prior:
            priors.append({"kind": "quantitative_anchor", **prior})
        if entity_contract.get("comparison_baseline") is not None:
            priors.append({
                "kind": "comparison_baseline",
                "value": entity_contract["comparison_baseline"],
                "unit": entity_contract.get("target_unit"),
                "period": entity_contract.get("comparison_period"),
            })
        ledgers[entity] = {
            "entity_id": entity,
            "target": {
                "metric": target.get("name"), "type": target.get("type"),
                "unit": entity_contract.get("target_unit"),
                "period": entity_contract.get("target_period"),
                "basis": entity_contract.get("basis"),
            },
            "priors": priors,
            "observations": _deduplicate(observations),
            "derived_values": _deduplicate(derived),
            "qualifiers": qualifiers,
            "conflicts": [row for row in (conflicts or [])
                          if not isinstance(row, dict) or row.get("entity_id") in (None, entity)],
            "missing_inputs": _missing_inputs(entity, gaps, bool(entity_facts)),
        }
    return ledgers
