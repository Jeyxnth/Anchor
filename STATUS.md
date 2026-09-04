# Anchor — Project Status Report

**Generated:** 2026-09-04, compiled directly against the current codebase, live database, and fresh test runs (not from memory or prior summaries). Where a figure came from re-running code today rather than reading a saved artifact, that's stated explicitly. Nothing below is rounded for presentation — figures are copied verbatim from JSON reports, live API responses, or SQL query output.

**Methodology note:** ML metrics (§3) are read from the committed evaluation artifacts (`backend/ml/artifacts/*.json`), produced by a deterministic, fixed-`random_state` pipeline against the current `events.csv`/`ground_truth.csv` — verified in sync (see §2) rather than re-trained live, to check the numbers that are actually deployed in `xgb_recovery_model.joblib` rather than a fresh run that could drift. Everything else — provider behavior (§5), compliance filter (§6), baseline experiment (§8), ledger behavior (§7) — was re-executed live today against the running backend and is reported from that live output.

---

## 1. Architecture as-built

The pipeline that actually exists in code today, in call order:

```
events.csv (batch source)
    -> ML prediction: predict_all_candidates()      [policy/pipeline.py]
         XGBoost model scores ALL 5 actions per row (P(recovery | customer, context, action))
    -> Compliance ELIGIBILITY FILTER: eligible_actions()   [policy/compliance.py]
         bounds the candidate set BEFORE ranking/agent — see §4/§6
    -> Expected-value ranking: rank_candidates()      [policy/ev.py]
         EV = P(recovery) x amount - intervention_cost - discount_cost, sorted desc
    -> Decision agent: provider.decide()              [policy/agent.py]
         Gemini / Groq / RuleBasedFallbackProvider, wrapped in ResilientProvider
    -> Validator (2nd compliance layer): validate_decision()   [policy/agent.py]
         re-checks the returned action against the SAME eligible list; corrects if invalid
    -> Execution (simulated, no dispatch) + Outcome simulator: simulate_outcome()  [app/outcome.py]
         samples against ground_truth.csv's true probability for the selected action
    -> Audit ledger: record_decision_trace()          [app/ledger.py -> SQLite]
    -> Dashboard (React) via FastAPI (/metrics, /batch/{id}, /audit/{id}, /batch/{id}/compare)
```

### Differences from the brief's diagram (`PROJECT_BRIEF.md` §4), and why

| Brief's stage | As built | Why |
|---|---|---|
| Customer/Event Ingestion | No `POST /events` endpoint. Batch runner reads `events.csv` directly. | Deliberately deferred — see §11. |
| Revenue Risk Engine (separate prioritization stage) | Does not exist as a separate stage. | Never built; the EV ranking effectively does amount-aware prioritization per case, but there's no separate cross-case "which cases first" prioritizer. |
| Hard Compliance Gate (deterministic, **after** the agent, override authority) | Not built as a separate third layer. Two layers exist instead: an eligibility **pre**-filter (before the agent) and a validator (after the agent, but only checking format/allowed-list, not full gate semantics like quiet hours). | Deliberate scope decision — see §6 and §11 for the full reasoning. |
| Execution Simulator (separate stage) | Folded into the outcome simulator — no message content is actually generated/logged separately; `executed_action` is just `selected_action` copied over (no gate exists to make them diverge). | Simplification: with no gate, execution and outcome are the same event. |

Everything else (ML prediction, EV policy, decision agent, audit ledger, dashboard) matches the brief's intended shape.

---

## 2. Data — `generate_data.py`

**Row count:** 4,000 events (`N_EVENTS = 4000`), across 1,500 seeded customers (`N_CUSTOMERS = 1500`); 1,372 of those 1,500 actually appear in the generated events (the rest were never sampled). Simulation window: 60 days from 2026-06-01, seed `RNG = np.random.default_rng(42)`.

**Customer fields:** `customer_id`, `customer_age_days`, `lifetime_value` (lognormal), `average_order_value` (lognormal, clipped 100–20,000), plus an internal-only latent `_quality ~ N(0,1)` that seeds everything else but is never written to `events.csv`.

**Transaction fields:** `transaction_id`, `amount` (lognormal around the customer's AOV, clipped 50–50,000), `event_type` (`payment_failed` 60% / `checkout_abandoned` 40%), `failure_reason` (5 categories, probability shifted by `_quality`), `timestamp`, `time_of_day`, `day_of_week`.

**Contact/compliance fields:** `attempts_so_far` (contacts in the trailing 24h), `last_contact_time`, `opted_out` (grows with `previous_payment_failures`, capped at 15% probability per event), `allowed_interventions` (precomputed compliance-state column, used only as a regression check — see §6).

**Dynamic history fields (evolve per customer across events):** `previous_payment_failures`, `previous_success_rate`, `previous_recovery_successes`, `days_since_last_purchase`.

**Outcome fields:** `recovered`, `recovered_amount`, `time_to_recovery_hours`.

### Noise and interaction terms

The hidden scoring function (`true_recovery_probability()`) is a logistic-transformed linear score:

```
score = 0.8*quality + reason_effect + event_effect + action_base
        + discount_interaction + retry_interaction
        + fatigue_effect + amount_effect
        + 1.2*(previous_success_rate - 0.5)
        + noise
prob = sigmoid(score), clipped to [0.01, 0.99]
```

- `noise ~ N(0, 0.5)` — drawn **fresh per action**, not once per event, so the 5 candidate probabilities for one event are not just base+constant; each gets independent noise.
- **Real interaction terms** (the only two `if action == ...` blocks in the generator):
  - `discount_offer`: +0.6 if `failure_reason == price_hesitation`, else +0.0; +0.3 if `lifetime_value < 5000`, else −0.1.
  - `retry_link`: +0.5 if `failure_reason in (otp_timeout, gateway_error)`, else −0.2.
- **Not interaction terms** (additive-only, same for every action): `event_effect`, `previous_success_rate` term, `fatigue_effect` (−0.25 × attempts_so_far), `amount_effect` (−0.15 if amount > 10,000). `escalate_to_human` and `no_action` and `reminder` have **no** interaction term at all — flat base effects (0.6, −1.5, 0.1 respectively).
- **Tested and found NOT real:** `features.py`'s own comment states this explicitly — no interaction feature was engineered for `event_type` or `previous_success_rate`, "doing so would just give the model 3200 rows to fit noise against a signal that doesn't exist," because neither is multiplied by `action` anywhere in the generator.

### Sync check (verified live today)

- `events.csv`: 4,000 rows (4,001 lines incl. header). `ground_truth.csv`: 4,000 rows, same.
- `cut -d, -f1` on both files, diffed: **identical transaction_id order**, row-for-row, both files.
- `git log` on `backend/data/generate_data.py`, `events.csv`, `ground_truth.csv`: all three have exactly **one** commit (`f8bb3c7`) touching them, and `git status`/`git diff --stat` shows **zero** uncommitted changes to any of the three. They are provably generated together and untouched since.

---

## 3. ML pipeline — full results

Split: `train_test_split(test_size=0.2, random_state=42, stratify=y)` → **3,200 train / 800 test**.

### Promoted model (XGBoost, variant B — interaction features), on the held-out test set

| Metric | Value |
|---|---|
| Accuracy | 0.7 |
| Precision | 0.660958904109589 |
| Recall | 0.5778443113772455 |
| F1 | 0.6166134185303515 |
| ROC-AUC | 0.7658020868134974 |

Confusion matrix (rows=true, cols=pred; `[[TN, FP], [FN, TP]]`):
```
[[367,  99],
 [141, 193]]
```
Best hyperparameters (from 5-fold CV grid search on TRAIN only): `learning_rate=0.03, max_depth=3, min_child_weight=1, n_estimators=100`. Best CV ROC-AUC: 0.7743420044530034.

### Logistic Regression (baseline, plain features — not retested with interactions), on the held-out test set

| Metric | Value |
|---|---|
| Accuracy | 0.70375 |
| Precision | 0.6611295681063123 |
| Recall | 0.5958083832335329 |
| F1 | 0.6267716535433071 |
| ROC-AUC | 0.7676428259361106 |

Confusion matrix:
```
[[364, 102],
 [135, 199]]
```
LR train ROC-AUC: 0.7837625274456597 (vs. test 0.7676428259361106 — small gap).

LR and XGBoost are a near-tie on ROC-AUC (0.7676 vs. 0.7658) — LR is marginally *higher* on this single metric. XGBoost is still the primary model because the optimality-gap metric (below) is what actually matters for this task (picking the best *action*, not just predicting recovery), and that's where XGBoost + interaction features pulls ahead.

### The overfitting finding and fix

First (untuned) XGBoost pass: **train ROC-AUC 0.92 vs. test ROC-AUC 0.75** — a ~0.17 gap, diagnosed as memorization on only 3,200 training rows after one-hot expansion. Fixed with a guarded 5-fold `StratifiedKFold` CV grid search (train-only) over `max_depth ∈ {2,3,4}`, `n_estimators ∈ {100,200,300}`, `learning_rate ∈ {0.03,0.05,0.1}`, `min_child_weight ∈ {1,5,10}`, scored on ROC-AUC.

Post-fix, on the **final promoted model** (variant B, interaction features, same hyperparameters as the plain-feature search converged to):

| | Accuracy | ROC-AUC |
|---|---|---|
| Train | 0.718125 | 0.8003858953096378 |
| Test | 0.7 | 0.7658020868134974 |

Gap: 0.0346 ROC-AUC — the code's own threshold for "notable gap, worth investigating" is 0.08; this is well under it. Verdict logged by the pipeline: *"small gap — looks reasonable, not memorizing."*

### Three-way experiment table (`experiment_interactions_report.json`, exact numbers)

All three share the identical tuned hyperparameters above; only ONE thing changes per variant.

| Variant | Test ROC-AUC | Optimality match rate | Mean regret (all) | Mean regret (mismatched) | Actions ever top-1 | Action coverage counts |
|---|---|---|---|---|---|---|
| **A** — baseline (depth=3, plain features) | 0.7635501529130581 | 38.5% | 0.075158 | 0.12220813008130083 | 2/5 | retry_link 236, reminder 0, discount_offer 0, escalate_to_human 564, no_action 0 |
| **B** — interaction features (**promoted**) | 0.7658020868134974 | 45.75% | 0.057182375 | 0.10540529953917051 | 3/5 | retry_link 172, reminder 0, discount_offer 99, escalate_to_human 529, no_action 0 |
| **C** — depth=4, plain features | 0.7636850761995323 | 38.375% | 0.0740775 | 0.12020689655172413 | 3/5 | retry_link 247, reminder 0, discount_offer 5, escalate_to_human 548, no_action 0 |

Conclusion (matches `train.py`'s own comment, confirmed by the numbers): adding the two real interaction features moved the optimality match rate from 38.5%→45.75% and cut mean regret ~24%. Adding tree depth alone (variant C) barely moved anything (38.5%→38.375%, i.e. no real change) — this was a **feature-representation gap**, not a tree-capacity gap.

### Optimality gap — promoted model, full numbers

- **n_events evaluated:** 800 (held-out test set)
- **Optimality match rate: 45.75%** (0.4575)
- **Mean regret (all): 0.057182375** (expected-recovery-probability units)
- **Mean regret (mismatches only): 0.10540529953917051**
- Median regret (all): 0.010050000000000003
- Max regret: 0.547
- Action coverage (model's own top-1 pick across all 800 test rows): `retry_link 172, reminder 0, discount_offer 99, escalate_to_human 529, no_action 0` — 3/5 actions ever picked as top-1.

### Reminder-unlearnability finding — verified live and explained

The model **never** picks `reminder` as its top-1 argmax on the held-out test set, in any of the three variants (0/800, every time). This is real, and I verified it isn't simply because reminder is never actually optimal:

Computing the true-best action directly from `ground_truth.csv` (5 `true_prob_*` columns, argmax per row, all 4,000 rows):

| Action | True-best count | % |
|---|---|---|
| escalate_to_human | 1,267 | 31.675% |
| retry_link | 1,251 | 31.275% |
| discount_offer | 1,195 | 29.875% |
| **reminder** | **287** | **7.175%** |
| no_action | 0 | 0% |

Restricted to just the 800-row held-out test set (from `optimality_gap_details.csv`): `discount_offer 235, escalate_to_human 260, reminder 62, retry_link 243` — **reminder is truly best in 62/800 = 7.75%** of test cases, yet the model's own top-1 pick is reminder in exactly 0 of them.

**The underlying math:** `reminder`'s `action_base = 0.1` is flat — it has **no** interaction term anywhere in `true_recovery_probability()` (only `retry_link` and `discount_offer` get `if action == ...` bonuses). Compare bases: `escalate_to_human = 0.6` (also flat, no interaction — so it unconditionally beats reminder's 0.1 by 0.5 before any noise), `retry_link = 0.5` (worst case 0.3 with the −0.2 penalty, still triple reminder's base), `discount_offer = 0.3` (worst case ~0.2, still double). Since each of the 5 candidate probabilities per event draws its own **independent** `N(0, 0.5)` noise, reminder can only win when its own noise draw happens to be favorably large *and* the structurally-stronger competitors (especially `escalate_to_human`, which has no downside scenario) happen to draw unfavorable noise simultaneously — a real but minority (~7%) joint event, not a deterministic pattern. A classifier trained on binary recovered/not-recovered labels at 3,200 rows and ~0.77 test ROC-AUC learns the dominant, low-noise structural pattern (retry/discount/escalate usually beat reminder) very reliably, but has no realistic way to also detect the noise-driven minority tail where reminder specifically wins — that would require near-perfect modeling of a σ=0.5 Gaussian draw, far beyond what this data size and AUC level support. This is a property of the *data-generating process interacting with a fixed-size training set*, not a training bug.

(Separately, in the live production batch under the rule-based EV policy — §8 — `reminder` **is** selected 134/4000 times. That's not a contradiction: EV ranking factors in cost and amount, not just raw predicted probability, and 3.35% selection is broadly consistent with reminder being genuinely-best in the ~7% ballpark once compliance-restricted cases are excluded.)

---

## 4. Expected-value policy layer

Formula as implemented (`policy/ev.py`, identical in both `ev.py` and `generate_data.py`):

```
expected_value = predicted_probability * recoverable_amount - intervention_cost - discount_cost
```

Cost constants (shared between the generator and the policy layer — not reinvented separately):

| Action | `intervention_cost` | `discount_cost` |
|---|---|---|
| retry_link | ₹2 | — |
| reminder | ₹1 | — |
| discount_offer | ₹0 | 10% of `recoverable_amount` (`DISCOUNT_RATE = 0.10`) |
| escalate_to_human | ₹40 | — |
| no_action | ₹0 | — |

**Eligibility filtering happens before the agent runs — confirmed by code order**, not just by comment: `pipeline.py`'s `build_decision_input()` calls `eligible_actions(state)` and `rank_candidates(..., eligible=eligible)` to build the `candidates` list *before* constructing the `DecisionInput` object; `decide_for_row()` then passes that already-filtered `DecisionInput` into `provider.decide()`. The LLM/rule-based provider never receives the full 5-action set for a restricted case — it only ever sees `decision_input.allowed_actions()`, which is already the filtered list.

---

## 5. Decision agent — provider distribution (critical, verified live)

**Short answer: Gemini is not firing right now.** As of this test (2026-09-04), every real-LLM attempt fell back to `rule_based_fallback`. This is diagnosed below — it's a live upstream availability issue, not a code defect, and the fallback path itself worked exactly as designed (100% graceful, 0 crashes, every fallback fully explained in the audit trail).

### What was actually in the ledger before I touched anything today

Most recent pre-existing batch (`baseline_experiment_d6a2fda4c420`, from this session's earlier screenshot-verification runs): `ai_agent` policy, 200/200 rows, **100% `rule_based_fallback`**. This is expected and not evidence of a problem — the dashboard's "use real LLM" toggle defaults to **off**, and none of those runs turned it on.

### Live test run (just executed, to answer this question definitively)

`POST /batch/run {"n": 15, "use_real_llm": true, "batch_id": "status_report_live_llm_test"}` → the endpoint-level summary claims `"provider":"gemini"` (this only reflects which provider was *configured*, not what happened per row — `ResilientProvider.name` is always the primary's name even when every individual call fell back).

**Actual per-row result, queried directly from the ledger:**

| `agent_provider` | Count |
|---|---|
| `rule_based_fallback` | **15 / 15 (100%)** |
| `gemini` | 0 |

Every one of the 15 rows carries a `validation_note` recording exactly what happened. Breakdown of the 15 failures:
- **3** × `ReadTimeout` ("read timeout=20")
- **12** × `HTTPError('503 Server Error: Service Unavailable ... gemini-3.5-flash-lite:generateContent')`

### Diagnosis (root-caused live, not assumed)

1. **API key and connectivity are fine:** `GET /v1beta/models` → 200 OK in 0.45s. `gemini-3.5-flash-lite` and `gemini-flash-lite-latest` (the code's pinned model and its documented fallback) both appear in the live model list.
2. **Direct `generateContent` probes, bypassing the app entirely:**
   - `gemini-3.5-flash-lite` → `ReadTimeout` after 30.23s.
   - `gemini-flash-lite-latest` (the code's own fallback target) → **also** `ReadTimeout`, 30.78s. So retrying against the fallback model wouldn't have helped either — both are affected identically.
   - `gemini-2.5-flash-lite` (deprecated) → fast 404 in 0.73s: *"This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemini-3.5-flash-lite..."* — confirms the deprecation note already in `agent.py` is current and accurate.
3. **Control probes to isolate scope** — `gemini-2.5-flash` and `gemini-2.0-flash` (both deprecated, non-flash-lite) → fast 404s (1.0s, 0.33s), not hangs. So the API itself is responsive; it's specifically the flash-lite tier's `generateContent` endpoint that's timing out/503-ing right now.

**Conclusion:** this looks like a live, transient Google-side capacity/availability issue on the Flash-Lite tier at the moment of testing — not a bug in this codebase. The 404-only retry logic in `GeminiProvider.decide()` correctly did **not** retry these (retrying a 503/timeout against the same two model IDs wouldn't help), and `ResilientProvider` correctly caught every failure and degraded to the rule-based fallback, exactly as designed.

**Gemini has worked before:** `backend/policy/example_decision_trace.json` (git-tracked, generated 2026-08-27 22:32 IST) has `"provider": "gemini"` — a real successful call, at a time when `gemini-2.5-flash-lite` (the model pinned then) was still live. The code was already updated in response to that model's later deprecation; what's failing today is availability, layered on top of an integration that has been confirmed working end-to-end at least once.

**Groq** is not a currently-usable fallback — no `GROQ_API_KEY` has ever been configured in this project (only `GEMINI_API_KEY` exists in `.env`), and `GroqProvider` has never been exercised against a live key (`agent.py`'s own docstring says the same).

**Recommendation:** re-test immediately before recording the demo — this may well have cleared by then. If it hasn't, the honest options are (a) demo on `rule_based_fallback`, which is a fully-designed, transparently-labeled path, not a degraded one, or (b) provision a `GROQ_API_KEY` as an untested-but-available second real-LLM path.

---

## 6. Compliance — both layers

### Layer 1 — eligibility pre-filter (`policy/compliance.py`), runs BEFORE the agent

```python
if opted_out:                        return ["no_action"]
if attempts_so_far >= 2:              return ["escalate_to_human", "no_action"]
else:                                  return ALL_INTERVENTIONS  # all 5
```
`MAX_CONTACTS_PER_24H = 2`, mirroring `generate_data.py`'s own compliance-state logic exactly (deliberately re-derived from raw `(opted_out, attempts_so_far)` state, not trusted from the dataset's precomputed `allowed_interventions` column — a live system would have to compute this itself).

**Verified live today:** `verify_against_dataset()` run against the current `events.csv` (all 4,000 rows): **`{'n_checked': 4000, 'n_mismatches': 0}`** — the independently-derived rule set matches the dataset's own compliance-state column exactly, on every row, right now.

### Layer 2 — validator (`agent.py`'s `validate_decision()`), runs AFTER the agent

Every raw LLM/agent response is checked: `action` must be in `decision_input.allowed_actions()` (the **same** eligibility-filtered list from Layer 1) and `reason` must be non-empty. If either check fails, the decision is corrected to the top-EV eligible candidate, `valid` is set to `False`, and a `validation_note` records exactly what was wrong. This is not a separate mock path — `GeminiProvider.decide()` and `GroqProvider.decide()` both call this exact function on their real API responses.

**Synthetic-injection test — re-run live today, exact output:**

```
Transaction: TXN000000 (real model predictions, real EV ranking)
Allowed actions: ['retry_link', 'escalate_to_human', 'reminder', 'discount_offer', 'no_action']
EV-ranked candidates:
    retry_link         EV=Rs 509.95
    escalate_to_human  EV=Rs 479.23
    reminder           EV=Rs 452.17
    discount_offer     EV=Rs 369.70
    no_action          EV=Rs 308.11

Injected (staged, not a live API call): {'action': 'send_free_product', 'reason': 'Sounds generous and should improve customer goodwill.'}

Result:
  action:           'retry_link'  (corrected to the top-EV eligible candidate)
  valid:            False
  validation_note:  Model returned invalid/incomplete decision (action='send_free_product',
                     allowed=['retry_link', 'escalate_to_human', 'reminder', 'discount_offer',
                     'no_action']); corrected to top EV candidate 'retry_link'.
```
The script's own asserts (`valid is False`, `action == candidates[0].action`) both passed.

### The Hard Compliance Gate — deliberately not built as a separate third layer

Stating this plainly, as asked: the brief's §9 "hard compliance gate" — a deterministic layer with override authority that runs *after* the agent and can visibly block a recommendation — **was not built**. The two mechanisms above satisfy the parts of it that matter for what's actually demoable:

- Contact-cap and opt-out enforcement are **fully** covered by Layer 1 — and arguably *more* strongly than a post-hoc gate would give: the agent is never even offered the disallowed action as a candidate, so there's nothing to override after the fact. It's structurally impossible, not caught-and-corrected.
- The validator (Layer 2) gives the same "never trust the model's raw output" guarantee the brief wants from the gate, for the specific failure mode of an invented/off-list action.

What a real hard gate would add that neither layer covers today: **quiet hours** (not implemented anywhere — `quiet_hour_violations` is always `null`), **forced escalation after M cumulative failed attempts** (Layer 1's `attempts_so_far >= 2` is a *contact-count-in-24h* check, not a cumulative-failure counter — related but not the same rule), and **stop-on-confirmed-recovery** (not applicable — there's no multi-touch retry loop to stop mid-sequence; one event is scored once).

One more honest nuance: the brief's demo narrative is explicitly *"Agent recommends: reminder → Compliance check: opted_out=true → **BLOCKED** → Final action: no_action"* — a **post**-decision override. The current design doesn't produce that sequence, because Layer 1 restricts the candidate set **before** the agent runs — the agent is structurally never even given `reminder` as an option for an opted-out customer. This is a stronger guarantee (impossible-to-misbehave vs. caught-after-misbehaving), but it changes the demo story from "watch it get blocked" to "watch the option never exist in the first place." `DEMO_SCRIPT.md` already states this distinction honestly to whoever presents it.

---

## 7. Audit ledger

**Primary key:** `(transaction_id, policy_name)` — composite, not `transaction_id` alone.

**The composite-key fix (historical, done):** the original schema had `transaction_id` as the sole primary key. Because `run_baseline_experiment()` logs three policies (`ai_agent`, `do_nothing`, `generic_reminder`) against the **same transaction_ids** under one `batch_id`, `INSERT OR REPLACE` on a `transaction_id`-only key meant each policy's write **silently overwrote the previous policy's row** for that transaction — only the last-run policy's decisions actually survived per transaction, breaking `compare_policies()` without raising any error. Fixed via `migrate_composite_key.py` (rebuilds the table with the composite key, backfills all pre-existing rows with `policy_name='ai_agent'` since that was the only policy ever logged before this fix; the migration ran before outcome simulation existed, so its documented `'unsimulated_legacy_row'` sentinel path was never actually exercised against real data — every row was immediately overwritten by the next real batch run anyway).

**A related, currently-UNFIXED bug, found live while preparing this report:** `batch_id` is *still not part of the primary key*. `INSERT OR REPLACE` matches on `(transaction_id, policy_name)` only — so a **later** batch run that touches the same transaction+policy pairs silently overwrites and orphans **every earlier `batch_id`** that held those rows, even fully-successful, previously-valid ones. I demonstrated this live: I ran a full 4,000-row baseline experiment (`batch_id=status_report_full_4000`), then re-ran it again for a reproducibility check (`batch_id=status_report_full_4000_repro_check`) — the second run's writes overwrote the first's rows entirely. `GET /batch/status_report_full_4000` now returns 404, minutes after it completed successfully. The same thing happened to the pre-existing `baseline_experiment_016b3939eb9b` (a legitimate Aug 28 run) and — critically — to **`baseline_experiment_full`, the exact batch_id `DEMO_SCRIPT.md` tells the presenter to load**, which currently returns 404 and does not exist in the ledger at all. See §12 for the practical demo implication.

### Schema — what's logged vs. what stays NULL

Populated on every row logged via `batch.py`/`main.py` (verified across 4,000+ current rows): `batch_id`, `customer_id`, `event_type`, `failure_reason`, `amount`, `input_event_json`, `predicted_probabilities_json` (empty `{}` for baseline policies, which don't consult the model — by design, not a gap), `candidates_ev_ranked_json` (empty `[]` for baselines, same reason), `eligible_actions_json`, `compliance_state_json`, `selected_action`, `agent_reason`, `agent_provider`, `agent_valid`, `executed_action`, `outcome_recovered`, `outcome_recovered_amount`, `stopping_reason`, `created_at`.

Always `NULL` today, and why:
- `gate_status`, `gate_reason`, `final_action` — the hard compliance gate (§6) was never built, so there's nothing to write here. Not a bug; a placeholder for a deliberately-descoped feature.
- `agent_validation_note` — `NULL` only when `agent_valid=1` (the response was fine as returned); populated whenever `agent_valid=0`.
- `time_to_recovery_hours` — `NULL` only when `outcome_recovered=0` (not recovered); a legitimate per-row null, not a gap.
- `quiet_hour_violations` — not a ledger column at all; it's a `/metrics`-computed field, always `null` in the API response, same reason as the gate columns.

---

## 8. Baseline experiment — authoritative numbers

**Batch:** ran fresh today, full 4,000-row population, all three policies, currently live under `batch_id = status_report_full_4000_repro_check` (see §7/§12 for why the first run's `batch_id` no longer resolves — this is the same-numbers re-run that superseded it).

- `same_transaction_population_across_policies`: **true**
- **₹ at risk: ₹1,707,076.43**

| Policy | ₹ recovered | Recovery rate (count-based) | Recovery rate (₹-weighted) | Incremental vs. ai_agent |
|---|---|---|---|---|
| do_nothing | ₹314,135.94 | 18.47% | 18.40% | — |
| generic_reminder | ₹716,243.43 | 42.78% | 41.96% | — |
| **ai_agent** | **₹855,495.75** | **49.65%** | **50.11%** | — |

**Incremental recovery of ai_agent vs. each baseline:**
- vs. do_nothing: **+₹541,359.81**
- vs. generic_reminder: **+₹139,252.32**

*(Recovery-rate labeling, exactly as the code defines it: `recovery_rate` = count-based = recovered transactions / total transactions. `revenue_weighted_recovery_rate` = ₹ recovered / ₹ at risk. These do not numerically match by construction — recovered transactions skew toward different amounts than unrecovered ones — that's expected, not an inconsistency.)*

**Provider for this run:** 100% `rule_based_fallback` (4,000/4,000) — `use_real_llm=false`, matching the dashboard's default.

### Reproducibility — verified, not assumed

Ran the identical experiment twice independently (`status_report_full_4000`, then `status_report_full_4000_repro_check`): the `results` dict came back **byte-identical** both times (fixed `OUTCOME_SIM_SEED=4242`, fixed rule-based provider — fully deterministic). Additionally, `DEMO_SCRIPT.md` — written on an earlier date, referencing a since-overwritten `batch_id` — independently quotes the exact same incremental figures (+₹541,359.81 / +₹139,252.32), matching today's fresh run digit-for-digit. That's cross-session confirmation, not just within-session repetition.

**Caveat on reproducibility:** this determinism holds for `use_real_llm=false`. A real-LLM run would not be bit-identical run-to-run (the LLM's phrasing/occasional deviation isn't seeded), though the EV-ranked candidate set and compliance filtering underneath it always would be.

---

## 9. Dashboard — current state

| Panel | Data source |
|---|---|
| Header search ("Jump to transaction") | `GET /audit/{id}` on submit, opens the trace modal |
| Sidebar | Nav-only, no data (search was moved to the header in this session's redesign) |
| KPI row — Revenue at Risk / Revenue Recovered / Recovery Rate | `GET /metrics?batch_id=&policy_name=ai_agent` |
| KPI row — Lift vs. Best Baseline | `GET /batch/{id}/compare` (min of `incremental_recovery_of_ai_agent_vs_each_baseline`) |
| Recovery Funnel (₹ at risk → recovered → still at risk) | same `/metrics` call as above; "still at risk" is client-computed (`at_risk - recovered`) |
| Policy Comparison (condensed bars) | same `/batch/{id}/compare` call as the Lift KPI |
| Compliance Panel | `/metrics` → `.compliance` + `.stopping_reason_breakdown` |
| Policy Comparison — Baseline Experiment (detailed 3-card block) | same `/compare` data as the condensed bars, plus its own Load/Run controls |
| Compliance Demo Cases | static, hardcoded 3 transaction IDs; each button calls `/audit/{id}` |
| Recent Decisions table | `GET /batch/{id}?policy_name=ai_agent` |
| Trace modal | `GET /audit/{id}` (defaults to `policy_name='ai_agent'` if unspecified) |

`api.js`'s `metrics()` call defaults `policyName` to `"ai_agent"` unless explicitly nulled — the dashboard's headline numbers are always the live agent's own numbers, never a meaningless cross-policy sum, by construction.

### Gap vs. the original brief: no "Recovery by Intervention" panel

The brief (§13) asks for a standalone "Recovery by intervention" breakdown panel. That panel **existed** earlier in this project's UI history and was **removed** during a redesign pass earlier in this session (replaced by the condensed Policy Comparison bars, per an explicit request at the time). The underlying data (`metrics().recovery_by_intervention`) is still fully computed and fetched by the frontend — it's sitting in React state — but has **no visible UI surface right now**. If a judging rubric checks for this literally, it's currently missing from the live dashboard.

### "Pending"/"TBD" audit

Grep-verified across the current `App.jsx`: exactly **one** live "pending" label remains — the Compliance Panel's **"Quiet-hour violations"** row renders a literal dotted-underline "pending" badge, because `compliance.quiet_hour_violations` is always `null` from the backend (the hard gate isn't built — see §6). Nothing else renders "pending," "TBD," "coming soon," "not yet," or "not implemented" — in particular, the trace modal's former "Hard Compliance Gate — pending" section was fully replaced with a neutral compliance-callout during this session's modal redesign and no longer references `gate_status` at all.

---

## 10. Known bugs found and fixed, in order

1. **Overfitting (ML pipeline).** First untuned XGBoost pass: train ROC-AUC 0.92 vs. test 0.75. Fixed with a guarded 5-fold CV grid search (train-only). Final promoted-model gap: train 0.8004 vs. test 0.7658 (0.0346, under the code's own 0.08 "notable" threshold). Full detail in §3.

2. **Composite-key / cross-policy overwrite bug.** Original `transaction_id`-only PK meant the three baseline-experiment policies clobbered each other's rows via `INSERT OR REPLACE`. Fixed by migrating to `(transaction_id, policy_name)`. **A related bug in the same area is still live and unfixed** — see §7/§12: `batch_id` isn't part of the key either, so later batches still silently overwrite earlier ones' rows.

3. **Gemini model deprecation.** `gemini-2.0-flash-lite` (original default) 404'd; `gemini-2.5-flash-lite` and `gemini-2.5-flash` were found to have *also* since 404'd during today's investigation. Repinned to `gemini-3.5-flash-lite`, added a one-retry-on-404 fallback to the rolling `gemini-flash-lite-latest` alias. This fix is unrelated to — and does not resolve — the separate, current availability issue documented in §5.

4. **PowerShell/console encoding.** `demo_validator_injection.py`'s `print()` statements originally used em-dashes, which mojibake'd to `�` under the Windows console codepage. Fixed by switching to plain ASCII hyphens in every `print()` call; verified clean under both git-bash and PowerShell.

5. **Recent Decisions column misalignment** *(not previously reported)*. The status/provider badge pair shared one flexbox container, so a longer badge (e.g. "Not Recovered") shifted the next badge's starting position row-to-row; the amount column's fixed width didn't reliably keep its right edge straight either. Fixed by converting the row to a fixed-column CSS grid (`36px / 1fr / 100px / 120px / 128px / 168px`) shared identically by a new header row and every data row, with status and provider each in their own grid cell.

6. **"Opt-outs respected" misleading denominator** *(not previously reported in these terms)*. Displayed as a fraction against a synthetic `Math.max(1, …)` denominator to dodge divide-by-zero, so a batch with zero opted-out customers showed a misleading "0 / 1." Fixed to show "0 opted-out customers this batch" when zero, "N of N respected" otherwise, with the bar filled 0% or 100% directly (this metric is always empty or fully satisfied by construction — the eligibility filter never fails to restrict an opted-out case).

7. **"Lift vs. Best Baseline" always green** *(not previously reported in these terms)*. On a small/noisy sample the agent can genuinely underperform the best baseline, but the KPI tone was hardcoded green regardless of sign. Fixed to apply green/success only when the raw incremental value is `> 0`; otherwise muted/red.

8. **Nonsensical signed phrasing.** A negative incremental value rendered as "agent recovers +₹-726.29 more." Fixed to phrase conditionally on sign ("...+X more" vs. "...X less," via `Math.abs()`).

Nothing else surfaced during this report's investigation that wasn't already reported to you in the sessions where it was fixed, except the two items above (6, 7 restated with full context) and the still-open batch_id-overwrite issue (§7/§12), which is new.

---

## 11. Explicit scope decisions (not bugs, not gaps)

- **`POST /events` ingestion endpoint deferred.** No separate risk-prioritization/ingestion stage exists; the batch runner reads `events.csv` directly. Sufficient for a synthetic-data MVP and demo; a live system would need this to accept real-time webhook events.
- **Stopping-reason granularity limited to `recovered` / `not_recovered`.** The pipeline scores one event per transaction, not a multi-touch retry sequence over time — there's no underlying process yet for richer stopping reasons (max-attempts-reached, opted-out-mid-sequence, etc.) to report on.
- **Hard Compliance Gate not built as a separate layer.** The eligibility pre-filter (before the agent) plus the validator (after the agent) already give a deterministic, provably-impossible-to-violate guarantee for the rules that exist (opt-out, contact-cap); the additional behaviors a real gate would add (quiet hours, forced escalation after cumulative failures, stop-on-recovery, and the brief's specific post-hoc block-and-override demo narrative) were descoped for time within the solo build, not built and hidden. Full reasoning in §6.

---

## 12. Currently broken, uncertain, or untested — read this before building the demo

1. **Batch_id overwrite bug (live, unfixed — the most important item here).** Any `batch_id` you plan to reference in the demo can silently stop existing the moment a later batch run touches overlapping transactions. **Confirmed right now:** `DEMO_SCRIPT.md`'s own referenced batch_id, `baseline_experiment_full`, returns 404 and has zero rows in the ledger as of this report. **Practical fix:** don't rely on loading a specific historical batch_id — click "Run Baseline Experiment" fresh immediately before/during the demo (the numbers are proven deterministic and will reproduce identically, per §8), and prefer `/audit/{transaction_id}` (not batch_id-scoped) for the three demo transaction lookups where possible.

2. **Gemini is not firing as of this report (2026-09-04).** 0/15 real calls succeeded in a live test batch; both the pinned model and its documented fallback are timing out/503-ing. Diagnosed as a live upstream availability issue (§5), not a code defect — but genuinely uncertain whether it clears by demo time. **Re-test immediately before recording.** Groq exists as a coded alternative but has never been exercised live — treat it as unverified, not a proven backup.

3. **"Recovery by Intervention" has no dashboard panel right now** (§9) — data is fetched but not displayed. If the brief/rubric checks for this literally, it's currently absent from the live UI.

4. **Quiet-hour violations are always null/pending** on the dashboard — genuinely not implemented (§6/§11), not a display bug.

5. **None of this session's work is committed to git.** Only 4 commits exist total (all dated Aug 26–27). Everything from Aug 28 onward — including the composite-key `ledger.py` change itself, the entire frontend redesign, `demo_validator_injection.py`, `DEMO_SCRIPT.md`, and `README.md` — is uncommitted working-tree state (`git status` confirms this live). If anything happens to the working directory before a commit, none of it is safely recorded in version control.

6. **Minor dead code.** `frontend/src/lib/utils.js` (a `cn()` class-merge helper) and the `clsx`/`tailwind-merge` npm dependencies are unreferenced leftovers from a UI-component library that was fully removed earlier in this session. Harmless, but never cleaned up.

7. **ML numbers were read from saved artifacts, not re-trained live for this report** (deliberately — re-training risks a library-version-driven numeric drift from what's actually deployed in `xgb_recovery_model.joblib`; reading the artifact checks the model that's really in production). The compliance filter (§6) and the full baseline experiment (§8) *were* re-executed live today as the more relevant checks. If you want a live re-verified training run on top of this, that's still open.

8. **Unexplained row count in a historical batch.** `baseline_experiment_016b3939eb9b` (Aug 28) logged 3,760 rows per policy, not the full 4,000. I don't have a recorded reason for that specific number — flagging as an open question rather than guessing at one. (This batch is also now unreachable by its `batch_id`, per item 1 above.)
