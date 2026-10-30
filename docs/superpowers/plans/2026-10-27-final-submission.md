# Under25 final submission implementation plan

**Goal:** Package the restored 10.25 analyst behind the published AgentHon analysis CLI.

**Architecture:** Keep `under25/` immutable and copy its runtime into this separate directory. Add a House-only adapter that runs the historical workflow with a 24-request core allowance and reserves one request for final submitted reasons. Use the official Docker verb and a narrow build context containing only code and permitted tokenizer resources.

**Constraints:** linux/amd64; interface_version=2.0; no Docker VOLUME; use MODEL_ENDPOINT/NAME/TOKEN unchanged; inherit organizer proxy variables; max 25 requests and 4000 output tokens per call. Do not ship public tasks, runs, outcomes, keys, snapshots or model weights.

- [x] Add failing tests for House configuration, input/output validation, complete-roster publication, reason budgeting and reason failure retention.
- [x] Add the submission adapter, CLI, and minimal runtime memo helper; retain the historical forecast pipeline.
- [x] Add Dockerfile, descriptor generation and official pack instructions. Validate descriptors through the official contract parser.
- [x] Run focused and historical regression tests; build linux/amd64; exercise the actual container CLI against a mock House endpoint and immutable exemplar inputs.
- [x] Verify package contents, image label, architecture, no volumes and no vendor API/proxy overrides; export the image and source checksums.
