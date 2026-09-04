# AI Revenue Recovery Agent

**Track 03 — AI Revenue Recovery.** Solo hackathon build, MVP deadline
Sept 5. Full scope and architecture: [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).

> **Draft status:** started early (day ~7 of the build), not day 9 — will
> keep growing alongside the build order below. Sections marked **TBD** are
> genuinely not written yet, not forgotten.

## Pitch

An AI revenue recovery agent that identifies money at risk, chooses the
highest-value recovery action, and executes it safely — maximizing
recovered revenue while enforcing compliance. Every layer exists to make
the system's output a measurable business number, not a generated message.

## Architecture — what's built vs. pending

```
Customer/Event Ingestion              ⚠ not built — batch pipeline reads events.csv directly
        ↓
Revenue Risk Engine                   ⚠ not built — no prioritization stage yet
        ↓
ML Recovery Prediction                ✅ XGBoost (primary) + Logistic Regression (baseline)
        ↓
Intervention / Expected-Value Policy  ✅ EV per allowed action
        ↓
AI Decision Agent                     ✅ Gemini (live, verified) / Groq (untested) / rule-based fallback
        ↓
Hard Compliance Gate                  ⚠ not built — build-order step 5 (override authority, quiet
                                          hours, forced escalation, stop-on-recovery)
        ↓
Execution Simulator                   ✅ (folded into outcome simulator below — see note)
        ↓
Outcome / Recovery Simulator          ✅ samples against ground_truth.csv, same method as generate_data.py
        ↓
Audit Ledger + Metrics Engine         ✅ SQLite, composite (transaction_id, policy_name) key
        ↓
Dashboard ("Revenue Recovery          ✅ React + Vite — KPIs, policy comparison, compliance panel,
 Control Center")                        per-case trace drilldown
```

Note on "Execution Simulator": the brief separates *execute* (log the
message as sent) from *outcome* (sample whether it worked). This build
folds them into one step — `app/outcome.py` records the executed action
(= the agent's selected action, since no hard gate exists yet to override
it) and the sampled outcome together. Worth splitting cleanly once the hard
gate exists and `executed_action` can actually differ from `selected_action`.

What's currently missing is exactly the two things that require the hard
gate: **quiet_hour_violations** (null in every report — the gate would
enforce this) and the **deliberate override demo** (brief §19 step 6,
build-order step 10) — today's compliance demo (see `DEMO_SCRIPT.md`)
shows the *eligibility filter* restricting the candidate set, not the gate
overriding an already-made choice.

## Running it

```
# Backend (from backend/, with the venv created per requirements.txt)
backend/.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

# Frontend
cd frontend && npm install && npm run dev
# -> http://localhost:5173
```

`GEMINI_API_KEY` (optional — falls back to a deterministic rule-based
provider without it) goes in a `.env` file at the repo root or in `backend/`.
See `DEMO_SCRIPT.md` for a repeatable click-through of the compliance cases.

## ML pipeline

Single model, `intervention` as a categorical feature (not per-action
models) — a documented simplification, see Limitations. Proper 80/20
train/test split (`random_state=42`, stratified), evaluated only on the
held-out 800 rows.

| Model | Test ROC-AUC | Accuracy | Precision | Recall | Train ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (baseline) | 0.7676 | 0.704 | 0.661 | 0.596 | 0.7838 |
| XGBoost (tuned, + interaction features) | 0.7658 | 0.700 | 0.661 | 0.578 | 0.8004 |

XGBoost hyperparameters found via 5-fold `StratifiedKFold` `GridSearchCV`
on the train split only (`max_depth=3, n_estimators=100, learning_rate=0.03,
min_child_weight=1`) — the original untuned model overfit badly (train
ROC-AUC 0.92 vs test 0.75); this closed that gap to a healthy ~0.03.

**Optimality gap** (model's top-ranked action vs. ground truth's actually-
best action, scored on the held-out 800 rows only, `ground_truth.csv` never
touched during training): 45.75% exact match, mean regret 0.057 when
matched action differs. Action coverage: only 3 of 5 actions (`retry_link`,
`discount_offer`, `escalate_to_human`) are ever the model's top pick;
`reminder` and `no_action` never are. See Limitations for why this is a
real ceiling, not an unaddressed bug.

**Controlled experiment** (adding `intervention × failure_reason` /
`intervention × lifetime_value_bucket` interaction features — the ones that
correspond to real interaction terms in `generate_data.py`'s true model —
vs. a `max_depth=4` capacity check):

| Variant | Test ROC-AUC | Optimality Match | Mean Regret | Action Coverage |
|---|---:|---:|---:|---:|
| A — baseline (depth 3, no interactions) | 0.7636 | 38.5% | 0.0752 | 2/5 |
| **B — + interaction features (adopted)** | **0.7658** | **45.75%** | **0.0572** | **3/5** |
| C — depth 4 only | 0.7637 | 38.4% | 0.0741 | 3/5* |

\* C's third action (`discount_offer`) was picked only 5/800 times —
negligible, not real coverage. B materially improved ranking and coverage;
C barely moved anything — this isolates the coverage gap as a feature-
representation problem, not a tree-depth capacity limit. B was promoted to
the primary model (`ml/train.py`); full numbers in
`ml/artifacts/experiment_interactions_report.json`.

## Policy layer

`EV = P(recovery | customer, context, action) × recoverable_amount −
intervention_cost − discount_cost` (`policy/ev.py`), computed only over
actions the **compliance eligibility filter** allows (`policy/compliance.py`
— opted-out → `no_action` only; ≥2 contacts in 24h → `escalate_to_human`/
`no_action` only; independently re-derived from raw state and verified
byte-identical to the dataset's own `allowed_interventions` column across
all 4,000 rows). The AI decision agent (`policy/agent.py`) then picks and
explains from the EV-ranked eligible candidates only — structured JSON,
validated against the allowed list, corrected to top-EV with a flagged note
if it ever returns something off-list (tested directly, not just assumed:
an invented action, a real-but-ineligible action, and a missing `action`
key were all caught and corrected).

Provider-agnostic: Gemini (live, verified working end-to-end with a real
key) → Groq (implemented, untested — no key available) → deterministic
rule-based fallback, auto-selected by which API key is set. A
`ResilientProvider` wrapper catches any primary-provider failure and
degrades to the rule-based fallback for that one decision, tagging the
audit trail honestly rather than pretending an LLM ran.

## Audit ledger & baseline experiment

SQLite (`app/db.py`), primary key `(transaction_id, policy_name)` — lets
multiple policies log a result for the same transaction without clobbering
each other. Every row: full input snapshot, per-action predicted
probabilities, EV-ranked candidates, compliance state, agent decision +
reasoning, and a simulated outcome (sampled against `ground_truth.csv`'s
true probability for the selected action, same functional form
`generate_data.py` itself uses).

**Baseline experiment** (build-order step 8), full 4,000-transaction batch,
identical population across all three policies (verified, not assumed):

| Policy | ₹ Recovered | Recovery Rate (count) | Recovery Rate (₹-weighted) |
|---|---:|---:|---:|
| do_nothing | ₹314,135.94 | 18.47% | 18.40% |
| generic_reminder | ₹716,243.43 | 42.78% | 41.96% |
| **ai_agent** | **₹855,495.75** | **49.65%** | **50.11%** |

**Revenue at risk: ₹1,707,076.43.** Incremental recovery of the agent:
**+₹541,359.81** vs. do-nothing, **+₹139,252.32** vs. generic reminder.

Two recovery-rate definitions are reported side by side deliberately —
count-based (recovered transactions / total) and ₹-weighted (₹ recovered /
₹ at risk) will not numerically match each other (recovered transactions
don't skew the same amounts as unrecovered ones), and that's expected, not
an inconsistency. Both are labeled everywhere they're shown (API and
dashboard) to avoid it reading as a bug.

## Limitations (honest, not deferred to the end)

- **All financial outcomes are simulated** against a synthetic ground-truth
  model, not real customer behavior. The optimality-gap framing is a way to
  be rigorous *within* that constraint, not a substitute for real-world
  validation.
- **Single model with `intervention` as a feature**, not separate per-action
  models — reasonable for MVP scope, but a real trade-off in how cleanly it
  isolates each intervention's true effect.
- **LR and XGBoost are essentially tied** (0.7676 vs. 0.7658 test ROC-AUC) —
  not a modeling failure. `generate_data.py`'s true model *is* a linear
  combination of features passed through a sigmoid (a correctly-specified
  logistic form) plus independent Gaussian noise (σ=0.5) added directly to
  the pre-sigmoid score. A linear model is the right family for this
  generating process; XGBoost's extra flexibility buys nothing because the
  irreducible noise dominates, not underfit signal.
- **`reminder` and `no_action` never win as the model's top pick**, in any
  variant tested. Traced to the generator's own math:
  `escalate_to_human`'s `+0.6` action-base score is unconditional (no
  penalty term ever reduces it); `reminder` is a flat `+0.1` with zero
  interaction terms. Even `retry_link`'s worst case (`-0.2`) and
  `discount_offer`'s worst case (`-0.1`) land above `reminder`'s constant —
  so `reminder` can only become the true-best action via the σ=0.5 noise
  term alone, not learnable signal. `no_action`'s `-1.5` base is so far
  negative it's essentially never true-best regardless of noise. This
  directly confirms the "largely noise-driven, not learnable" hypothesis
  behind the controlled experiment above. Separately: once expected value
  (which prices in `escalate_to_human`'s ₹40 cost) rather than raw
  probability drives the choice, `reminder` *does* win regularly — 134/4,000
  times in the full-batch run — because it's cheap, not because the model
  ranks it highest. Two different mechanisms; worth not conflating them.
- **The 45.75% optimality-match ceiling is structural, not an unaddressed
  bug.** The true model's independent per-candidate noise term means even a
  perfect estimate of the deterministic component can't recover the
  noise-driven cases — this was verified, not assumed, via the controlled
  interaction-feature experiment above (see full numbers in
  `ml/artifacts/experiment_interactions_report.json`).
- **Compliance rules are a representative subset** (contact caps, opt-out;
  quiet hours and forced-escalation-after-M-failures exist in the brief but
  need the hard gate, step 5, to enforce) — not a complete regulatory
  implementation.
- **`batch_id` is a label, not part of the ledger's primary key.** The key
  is `(transaction_id, policy_name)` by design (so multiple policies can
  share a transaction) — but that means a *different-sized* batch run over
  overlapping transactions silently reassigns those rows' `batch_id`,
  fragmenting an earlier batch's grouping. Caught live: a 50-row smoketest
  overlapping `baseline_experiment_full` shifted its reported ₹ recovered
  by exactly ₹2,732.23 (the smoketest's `do_nothing` contribution). Fixed
  by re-running the full experiment as the authoritative pass — confirmed
  fully deterministic (identical numbers reproduced). `/metrics` also warns
  explicitly (`_policy_warning`) if a query spans multiple policies without
  a `policy_name` filter, rather than silently summing incompatible arms.
  Operational takeaway: don't run smaller-scope test batches against a
  batch_id you care about keeping intact.
- **Gemini model IDs drift faster than expected.** `gemini-2.0-flash-lite`
  (and even `gemini-2.5-flash-lite`/`gemini-2.5-flash`) 404'd against a
  freshly-issued key — Google's error body explained dated/numbered
  snapshots get cut off from new API keys as newer generations ship.
  Pinned to `gemini-3.5-flash-lite` (confirmed live), with one automatic
  retry against `gemini-flash-lite-latest` (the rolling alias — immune to
  this specific failure mode) before the `ResilientProvider` wrapper drops
  to the deterministic fallback. Tested against a real 404, not assumed.

## API reference

Implemented: `POST /batch/run`, `POST /batch/baseline_experiment`,
`GET /batch/{batch_id}`, `GET /batch/{batch_id}/compare`,
`GET /audit/{transaction_id}`, `GET /metrics`, `POST /recover/{transaction_id}`,
`GET /health`. Not implemented: `POST /events` (no ingestion stage exists
yet — the batch runner reads `events.csv` directly).

## TBD

- Hard compliance gate (build-order step 5) + deliberate override demo
  (step 10)
- Revenue Risk Engine / prioritization stage
- `POST /events` ingestion
- 5-minute demo video
- Razorpay integration (if included — small authenticity add-on only, per
  brief §20)
