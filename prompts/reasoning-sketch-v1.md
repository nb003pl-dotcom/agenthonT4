SKETCH-OF-THOUGHT (SoT), internal reasoning only:
For each requested entity compress deliberation into three short slots:
1. Target: numeric variable, unit, period and required comparison.
2. Estimate: strongest comparable quantitative prior, supported update and direction; distinguish observations from model estimates.
3. Risk: principal counterevidence or comparability limitation and its implication for the outcome interval.
Use terse fragments rather than a narrative. Aim for no more than 50 words per entity and 200 words for the whole batch. Read the evidence once; do not restate the question, copy source passages, enumerate every fact or repeatedly reconsider the same alternatives. Perform only the arithmetic checks needed to preserve the target's metric, scale and comparison. When evidence cannot quantify an update, retain the uncertainty instead of inventing calculations.
Then produce the complete final prediction JSON with every required entity and field, including the concise reasoning field required by the existing output contract. Keep that field as the short final justification already requested, not the internal sketch. Do not add the sketch, drafts or additional analysis to the final JSON. Compression must preserve material counterevidence, valid citations and the full output roster.
