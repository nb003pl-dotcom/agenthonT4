# NLI judge — serving backend (public documentation)

## Executive summary (read this first)

The Track 4 faithfulness judge can run in two places: **in-process**, loading the pinned DeBERTa
ensemble locally, or **against a served endpoint** that answers a small HTTP contract. Both compute
the same thing — serving changes *where* inference runs, never *what* it computes; the model pins,
the ensemble mean, `tau_citation` and `faithfulness_threshold` are identical in both. This page
documents the contract and the environment knobs, because the client half ships in this repository
(`ServedNLIJudge` and `build_judge` in `faithfulness/judge.py`) and you may want to point it at a
model server of your own while developing. **The organiser deployment is not published here**: the
container image, the cluster manifests and the endpoint's address and credentials are operational
detail, they are not part of the participant contract, and nothing in this directory is runnable.

---

## The two backends

| Backend | Selected by | What it does |
|---|---|---|
| `local` (default) | `T4_JUDGE_BACKEND` unset or `local` | Loads each pinned ensemble member in-process. Needs the weights cached locally and `transformers` installed. |
| `served` | `T4_JUDGE_BACKEND=served` | Delegates each `(model_id, premise, hypothesis)` scoring call over HTTP. The ensemble mean stays client-side. |

`build_judge` raises on any other value — there is no silent fallback, in either direction, because
a judge that quietly becomes something else is how a faithfulness gate stops being a gate.

## Environment knobs (local development only)

These are **not** part of the container-environment contract in
[`../../SUBMISSION_CLI.md`](../../SUBMISSION_CLI.md); the harness does not inject them. They exist
so you can run the shipped client against your own server.

| Variable | Meaning |
|---|---|
| `T4_JUDGE_BACKEND` | `local` (default) or `served` |
| `T4_JUDGE_URL` | Base URL of the endpoint, no trailing slash. Required when the backend is `served`. |
| `T4_JUDGE_TOKEN` | Optional bearer token, sent as `Authorization: Bearer <token>` |

## The endpoint contract

A server is compatible if it answers these two routes. This is the contract `ServedNLIJudge`
speaks; read that class for the authoritative request and response handling.

```
GET  /health   -> {"status": "ok", "models": [<model_id>, ...]}
POST /entail   {"model_id": ..., "premise": ..., "hypothesis": ...}
               -> {"entailment": <float in [0, 1]>, "model_id": ...}
```

One request scores one ensemble member on one premise/hypothesis pair. No scoring logic lives
server-side: thresholding, the ensemble mean and the roster denominator are all client-side, in
this repository, where you can read them.

## Comparing two backends

`faithfulness/judge_equivalence.py` runs every claim of one or more `(unit, answer)` pairs through
both backends and reports per-claim deltas, the maximum absolute delta, and the number of
**τ-crossings** — claims whose supported/unsupported status differs between the two. It asserts
nothing; it measures. The τ-crossing count is the number that matters: a delta that flips no claim
past `tau_citation` cannot change any unit's score.

```bash
python -m faithfulness.judge_equivalence \
  --pair units/t4-EXAMPLE-eps-beat:/tmp/answer.json \
  --served-url http://localhost:8080 \
  --out /tmp/equivalence_report.json
```

## What is not here, and why

The judge-serving image, its dependency pins, and the Kubernetes deployment and service that run it
for official scoring are organiser infrastructure. They carry an image registry, a namespace and a
bearer-token secret, none of which is participant-facing, and none of which changes a single score.
Publishing them would add an operational surface to a starter kit without adding anything you could
use. What matters for reproducibility — the model pins, the ensemble mean, the thresholds and the
roster denominator — is all in `faithfulness/judge.py` and `qfbench2_track_analysis/`.
