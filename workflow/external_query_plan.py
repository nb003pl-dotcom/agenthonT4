"""Load offline query-coverage plans for a fresh workflow retrieval run."""
import json
from copy import deepcopy
from pathlib import Path


QUERY_TYPES = {
    "target_history", "guidance_expectation", "financial_driver",
    "macro_market_driver", "risk_counterevidence", "definition_baseline",
}


def _purpose(targets):
    parts = []
    for target in targets or []:
        metric = str(target.get("metric", "")).strip()
        period = str(target.get("time", "")).strip()
        text = " | ".join(x for x in (metric, period) if x)
        if text and text not in parts:
            parts.append(text)
    return "; ".join(sorted(parts))[:400]


def load_external_query_plan(path, task_id, entity_ids, *, preserve_needs=False):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("unit") != task_id:
        raise ValueError("external query plan unit mismatch")
    rows = data.get("subqueries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("external query plan requires nonempty subqueries")
    allowed = set(entity_ids)
    seen = set()
    queries = []
    for row in rows:
        query_id = row.get("subquery_id")
        routed = row.get("entity_ids")
        query_type = row.get("query_type")
        query = row.get("query")
        if not isinstance(query_id, str) or not query_id or query_id in seen:
            raise ValueError("external query plan has invalid or duplicate subquery_id")
        seen.add(query_id)
        if not isinstance(routed, list) or not routed:
            raise ValueError("external query plan query requires entity_ids")
        if any(entity not in allowed for entity in routed):
            raise ValueError("external query plan references unknown entity")
        if query_type not in QUERY_TYPES:
            raise ValueError("external query plan has invalid query_type")
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 300:
            raise ValueError("external query plan has invalid query text")
        text = query.strip()
        header_terms = row.get("table_header_terms", [])
        if not isinstance(header_terms, list) or any(not isinstance(x, str) or not x.strip() for x in header_terms):
            raise ValueError("external query plan has invalid table_header_terms")
        lexical = " ".join([text, *(x.strip() for x in header_terms)]).strip()[:200]
        query_need = {}
        if preserve_needs:
            targets = row.get("targets", [])
            if not isinstance(targets, list) or any(not isinstance(t, dict) for t in targets):
                raise ValueError("external query plan has invalid targets")
            query_need = {"targets": deepcopy(targets)}
            for field in ("priority", "information_need", "metric", "period", "basis", "unit", "scope", "constraints"):
                if field in row: query_need[field] = deepcopy(row[field])
        queries.append({
            **query_need,
            "query_id": query_id,
            "entity_ids": list(dict.fromkeys(routed)),
            "query_type": query_type,
            "dense_query": text,
            "lexical_query": lexical,
            "purpose": _purpose(row.get("targets")) or text[:400],
        })
    missing = allowed - {entity for query in queries for entity in query["entity_ids"]}
    if missing:
        raise ValueError("external query plan omits entities: " + ",".join(sorted(missing)))
    return {"queries": queries, "source_unit": data["unit"], "source_path": str(path.resolve())}
