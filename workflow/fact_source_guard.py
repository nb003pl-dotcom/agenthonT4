"""Narrow numeric provenance guard; no semantic or outcome claims.

Reject a quantified assertion only if every attached quote lacks quantities.
Non-quantitative statements and calculations with quoted operands are retained.
The source may contain the value elsewhere; this is a citation insufficiency,
not a conclusion that the assertion is false. Optional, pending quality tests.
"""
import sys
from .common import ROOT

sys.path.insert(0, str(ROOT / 'vendor/official'))
from qfbench2_track_analysis.numeric import figures


EXTRACTION_INSTRUCTION = """
Source-grounding experiment: establish each fact's period and scope from the quoted source, never by copying the task's future target period. A past report can contain future guidance, declared terms or scheduled events: retain these when explicitly supported, preserving words such as expects, tentative, subject to, and not yet announced. Do not turn a previous period's value or a repeated historical practice into an already-announced future value. Such extrapolation belongs in prediction, not extracted facts. A component or segment metric is not a company-wide total; an expectation or policy is not a realized outcome. If a table row lacks its period/scope header, cite the header too or omit the unresolved assertion. Task-supplied values are already in task metadata; do not re-create them as corpus facts with unrelated citations. These constraints also apply to qualitative facts with null value.
"""


def numeric_source_issue(fact):
    statement = fact.get('statement', '')
    quotes = [c.get('quote', '') for c in fact.get('citations', [])]
    # Preserve genuinely extractive statements even where date/number parsing
    # treats line breaks differently (e.g. "one-tenth of one\npercent").
    normalized = ' '.join(statement.split())
    if not normalized or any(normalized in ' '.join(q.split()) for q in quotes):
        return None
    if figures(statement) and not any(figures(q) for q in quotes):
        return ('quantified_fact_without_quantified_citation: chosen quotes contain no '
                'amounts supporting this assertion; choose the actual quantity and '
                'metric/period context or omit the fact. Task values already have '
                'their task source; do not attach unrelated corpus citations.')
    return None


def complete_guarded_source(fact, catalog, names=()):
    """Only repair a flagged fact, within its previously read source window."""
    if not numeric_source_issue(fact):
        return fact, {'changed': False, 'reason': 'guard_not_triggered', 'semantic_status': 'unverified'}
    from .citation_completion import complete_adjacent_citation
    result, audit = complete_adjacent_citation(fact, catalog, names)
    return result, {**audit, 'trigger': 'quantified_fact_without_quantified_citation'}
