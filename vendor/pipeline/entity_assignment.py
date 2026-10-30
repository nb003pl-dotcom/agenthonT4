"""Deterministic document-to-entity assignment for T4 search indexes."""
from __future__ import annotations

import re
import unicodedata


ASSIGNMENT_VERSION = "entity-assignment-v2-tenor"
SCOPES = ("entity", "multi_entity", "unassigned")


def assignment_configuration():
    return {
        "version": ASSIGNMENT_VERSION,
        "strong_fields": ["cik", "ticker", "series_id"],
        "title_fallback": "normalized_exact_entity_name_phrase",
        "strong_confidence": 1.0,
        "title_confidence": 0.8,
        "content_matching": False,
    }


def _cik(value):
    digits = re.sub(r"\D", "", str(value or ""))
    return digits.zfill(10) if digits and len(digits) <= 10 else None


def _code(value):
    value = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    return value or None


def _phrase(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = "".join(" " if unicodedata.category(char)[0] in {"P", "S", "Z"} else char
                    for char in value)
    return " ".join(value.split())


def _entities(task):
    entities = task.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("task.entities must be a list")
    result = []
    seen = set()
    for entity in entities:
        entity_id = entity.get("entity_id") if isinstance(entity, dict) else None
        if not isinstance(entity_id, str) or not entity_id.strip() or entity_id in seen:
            raise ValueError("Missing or duplicate entity_id")
        seen.add(entity_id)
        result.append(entity)
    return result


def _title_matches(entities, title):
    normalized = _phrase(title)
    if not normalized:
        return set()
    padded = f" {normalized} "
    matches = set()
    for entity in entities:
        name = _phrase(entity.get("name"))
        if len(name) > 1 and f" {name} " in padded:
            matches.add(entity["entity_id"])
    return matches


def _tenor_matches(entities, title):
    """Conservative Treasury title matching; never inspect table/body history.

    Exact numeric boundaries prevent 2Y/20Y collisions. Instrument type is
    required in the title, so generic company names cannot acquire this rule.
    """
    if not re.search(r'\bTreasury\b', str(title), re.I):
        return set()
    terms = re.findall(r'\b(\d+)[ -]+(Year|Month|Week)[ -]+(Note|Bond|Bill|TIPS)\b', str(title), re.I)
    if len(terms) != 1:
        return set()
    number, unit, instrument = terms[0]
    matches = set()
    for entity in entities:
        tenor = re.fullmatch(r'(\d+)[ -]+(Year|Month|Week)', str(entity.get('tenor', '')), re.I)
        if tenor and (int(tenor[1]), tenor[2].lower()) == (int(number), unit.lower()):
            name = str(entity.get('name', ''))
            if re.search(r'\b' + re.escape(instrument) + r'\b', name, re.I):
                matches.add(entity['entity_id'])
    return matches


def assign_document(task, doc):
    """Return one auditable assignment using metadata only, never document content."""
    if not isinstance(doc, dict) or not isinstance(doc.get("doc_id"), str) or not doc["doc_id"]:
        raise ValueError("Missing doc_id")
    entities = _entities(task)
    signal_matches = []

    doc_cik = _cik(doc.get("cik"))
    if doc_cik:
        matches = {e["entity_id"] for e in entities if _cik(e.get("cik")) == doc_cik}
        if matches:
            signal_matches.append(("cik_exact_match", matches))

    doc_ticker = _code(doc.get("ticker"))
    if doc_ticker:
        matches = {e["entity_id"] for e in entities
                   if doc_ticker in {_code(e.get("ticker")), _code(e["entity_id"])}}
        if matches:
            signal_matches.append(("ticker_exact_match", matches))

    doc_series = _code(doc.get("series_id"))
    if doc_series:
        matches = set()
        for entity in entities:
            explicit = _code(entity.get("series_id"))
            entity_id = _code(entity["entity_id"])
            if explicit == doc_series or (not explicit and
                    (entity_id == doc_series or entity_id.startswith(doc_series + "_"))):
                matches.add(entity["entity_id"])
        if matches:
            signal_matches.append(("series_id_exact_match", matches))

    diagnostics = []
    title_matches = _title_matches(entities, doc.get("title"))
    tenor_matches = _tenor_matches(entities, doc.get('title'))
    if signal_matches:
        distinct = {tuple(sorted(matches)) for _, matches in signal_matches}
        if len(distinct) != 1:
            details = ", ".join(f"{basis}={sorted(matches)}" for basis, matches in signal_matches)
            raise ValueError(f"Conflicting strong entity assignments for {doc['doc_id']}: {details}")
        entity_ids = sorted(signal_matches[0][1])
        basis = [name for name, _ in signal_matches]
        confidence = 1.0
        if title_matches and title_matches != set(entity_ids):
            diagnostics.append("title_matches_ignored_due_to_strong_assignment")
    elif title_matches:
        entity_ids = sorted(title_matches)
        basis = ["title_exact_name_match"]
        confidence = 0.8
    elif tenor_matches:
        entity_ids = sorted(tenor_matches)
        basis = ['treasury_title_tenor_instrument_match']
        confidence = 0.8
    else:
        entity_ids = []
        basis = []
        confidence = 0.0

    scope = "unassigned" if not entity_ids else "entity" if len(entity_ids) == 1 else "multi_entity"
    return {
        "doc_id": doc["doc_id"],
        "entity_ids": entity_ids,
        "scope": scope,
        "assignment_basis": basis,
        "assignment_confidence": confidence,
        "diagnostics": diagnostics,
    }


def assign_documents(task, docs):
    if not isinstance(docs, dict):
        raise ValueError("docs must be a mapping keyed by doc_id")
    result = {}
    for doc_id in sorted(docs):
        doc = docs[doc_id]
        if doc.get("doc_id") != doc_id:
            raise ValueError("Document mapping key differs from doc_id")
        result[doc_id] = assign_document(task, doc)
    return result
