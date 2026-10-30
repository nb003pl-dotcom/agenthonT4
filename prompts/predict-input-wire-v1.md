PREDICT INPUT WIRE: The user message uses exactly one of these equivalent encodings:

- Original JSON: the task payload appears directly.
- `predict-input-compact-v1` JSON: the original non-source fields are under `payload`. `source_context.registry` stores source text once; `source_context.by_entity` maps each entity to source IDs and its original spans, selection reasons, and related fact IDs.
- `predict-input-compact-v1` XML: the same compact object appears under `<predict_input>`. Elements marked `type="object"`, `type="array"`, `type="number"`, `type="bool"`, or `type="null"` retain their JSON types; array members are `<item>`.

These encodings do not change evidence meaning or permissions. A source registry entry is shared storage, not extra evidential weight. Entity source references do not create new fact IDs. Follow the prediction output contract in the input and return the same required JSON prediction envelope for every encoding.
