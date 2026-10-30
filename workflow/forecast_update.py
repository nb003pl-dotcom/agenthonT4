"""Profile-gated forecast update request construction."""
from .typed_evidence import build_typed_evidence


FORECAST_UPDATE_CONTRACT = {
    "output_contract": "unchanged",
    "procedure": [
        "Choose an explicit prior from the supplied priors or state that no usable prior exists.",
        "List only evidence-bound updates; every update must cite existing fact_ids.",
        "Treat observations, derived values and priors as distinct epistemic types.",
        "Do not give duplicate observations additional weight.",
        "Do not invent a numeric adjustment from qualitative evidence.",
        "Do not narrow an interval solely because qualitative evidence agrees with the point direction.",
        "Return the original official prediction schema only; the update procedure belongs in concise reasoning.",
    ],
}


def validate_forecast_envelope(data):
    if not isinstance(data, dict) or not isinstance(data.get("predictions"), list):
        raise ValueError('Return a top-level JSON object {"predictions": [...]} with predictions as an array; '
                         'do not use entities or entity IDs as top-level keys. Return the full batch and all required fields.')


def build_forecast_update_payload(profile_name, payload, contract, facts, anchors, conflicts, gaps):
    if profile_name != "typed-evidence-v2":
        return payload
    result = dict(payload)
    for key in ("facts", "conflicts", "gaps", "quantitative_anchors", "anchor_instruction"):
        result.pop(key, None)
    group = payload.get("allowed_entity_ids", [])
    result["output_requirements"] = {
        **payload.get("output_requirements", {}),
        "top_level_schema": {"predictions": "array of prediction rows"},
        "entity_ids_exactly_once": list(group),
        "fact_ids_per_row": {"min": 1, "max": 3, "unique": True},
        "reasoning_max_characters": 240,
    }
    result["evidence_ledger"] = build_typed_evidence(
        contract, facts, anchors or {}, conflicts or [], gaps or [], group,
    )
    result["forecast_update_contract"] = FORECAST_UPDATE_CONTRACT
    return result
