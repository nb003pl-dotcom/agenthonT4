"""Family-independent evidence budgeting for prediction payloads."""
from copy import deepcopy


POLICY = "dynamic-regression-cap-v1"


def _filter_ledger(ledger, keep_by_entity):
    for entity, sections in ledger.items():
        keep = keep_by_entity.get(entity, set())
        for key, items in list(sections.items()):
            if not isinstance(items, list):
                continue
            retained = []
            for item in items:
                if not isinstance(item, dict) or "fact_ids" not in item:
                    retained.append(item)
                    continue
                ids = [fid for fid in item.get("fact_ids", []) if fid in keep]
                if ids:
                    retained.append({**item, "fact_ids": ids})
            sections[key] = retained


def apply_dynamic_evidence_budget(payload, config):
    """Cap oversized regression ledgers while preserving other target types exactly."""
    policy = (config or {}).get("policy")
    if policy != POLICY:
        raise ValueError("unknown dynamic evidence budget policy")
    cap = (config or {}).get("max_facts")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError("max_facts must be a positive integer")

    result = deepcopy(payload)
    target_type = result.get("target_contract", {}).get("target", {}).get("type")
    allowed = result.get("allowed_fact_ids_by_entity", {})
    before = sum(len(ids) for ids in allowed.values())
    audit = {
        "policy": policy,
        "target_type": target_type,
        "max_facts": cap,
        "before_fact_references": before,
        "after_fact_references": before,
        "removed_fact_references": 0,
        "entities_trimmed": [],
    }
    if target_type != "regression":
        audit["status"] = "not_applicable_target_type"
        return result, audit

    keep_by_entity = {}
    for entity, ids in allowed.items():
        retained = ids[:cap]
        result["allowed_fact_ids_by_entity"][entity] = retained
        keep_by_entity[entity] = set(retained)
        if len(retained) < len(ids):
            audit["entities_trimmed"].append(entity)
    _filter_ledger(result.get("evidence_ledger", {}), keep_by_entity)
    after = sum(len(ids) for ids in result.get("allowed_fact_ids_by_entity", {}).values())
    audit.update(status="applied" if after < before else "within_budget",
                 after_fact_references=after, removed_fact_references=before-after)
    return result, audit

