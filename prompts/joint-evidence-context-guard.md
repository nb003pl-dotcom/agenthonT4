EVIDENCE SCOPE: This is extraction of evidence for a forecast, not a request for
the future realized answer. The target period may be unobserved at the cutoff.
That does not make earlier observations, comparison baselines, mechanisms,
definitions, constraints, or risk evidence irrelevant. Extract material supported
pre-cutoff facts even when no exact target-period observation or guidance exists.
The information_needs are search directions, not an exhaustive fact checklist or
a requirement that a source prove the future outcome. Assess a need as partial or
uncertain when the source provides useful but indirect or different-period context.
Do not suppress historical facts merely because relevance to an exact future need
is indirect. Preserve each fact's actual metric, period, basis and units; never
relabel an earlier observation as the target period or invent missing guidance.
An empty facts list is valid when no useful supported facts exist in these windows;
absence of future results alone is not sufficient justification for an empty list.

JOINT TASK ORDER: Produce the facts before classifying source relevance. Start the
JSON object with facts, then conflicts, unresolved_questions, and place
source_assessments LAST. First read the provided source passages for task-useful
supported historical observations and context, using their supplied quote IDs.
Then assess relevance to each information need. A partial or uncertain relevance
label must not erase a fact already supported by those passages. This ordering
does not require a nonempty fact list when the provided source is truly unusable.

TARGET AND SOURCE SCOPE: Prioritize directly observed historical values of the
target metric, or the underlying levels and denominator when the target is a
change, growth rate, ratio, or other derived measure, before generic drivers.
Resolve each numeric observation against all relevant table header levels:
entity or component, metric, unit, period and comparison basis. A source authored
by the target entity may contain subgroup or component tables. Name that narrower
scope explicitly in the fact statement; never present a component total as the
whole entity, mix columns from different components, or mix quarterly and
cumulative periods. If the necessary header or scope cannot be resolved, state
that limitation and omit the ambiguous numeric assertion instead of guessing.
