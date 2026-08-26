# AI Revenue Recovery Agent — Project Brief

**Hackathon Track:** Track 03 — AI Revenue Recovery
**Scope:** Payment failures + checkout abandonment, unified into one recovery engine. Overdue receivables and multi-language/voice are explicitly out of scope for MVP.
**Builder:** Solo. Prior ML experience. Timeline: MVP complete by Sept 5.
**Principle:** One recovery engine, done with real measurement and enforced compliance, beats several shallow workflows.

---

## 1. Pitch

> An AI revenue recovery agent that identifies money at risk, chooses the highest-value recovery action, and executes it safely — maximizing recovered revenue while enforcing compliance.

Flow: **Detect → Predict → Decide → Gate → Recover → Measure**

## 2. Problem

Payments fail and checkouts get abandoned for many different reasons — insufficient funds, OTP timeouts, gateway errors, price hesitation. Right now that revenue is either ignored, or every failure gets the same generic reminder regardless of whether that's the right (or even a useful) response. The gap isn't detecting failure — it's knowing *which* intervention is worth taking for *this* customer, and proving that the answer was worth more than doing nothing or doing the same thing for everyone.

## 3. Why this approach

The project must not read as "an LLM that sends payment reminders." Every layer exists to make the system's output a measurable business number, not a generated message:

- A **real trained model** estimates recovery probability *per candidate intervention* — not just "will this recover" but "will this recover if we send X."
- An **expected-value calculation** turns those probabilities into a business decision, not a vibe.
- The **LLM chooses among a fixed, explicit set of allowed actions** and explains itself — it does not invent actions or bypass policy.
- A **deterministic compliance gate has final authority** over the LLM, and can override it — this is proven in the demo, not claimed in a README.
- A **baseline comparison** (do nothing / generic intervention / this agent) proves the system outperforms the obvious alternatives, using only numbers that came out of the actual experiment.

## 4. Architecture

```
Customer/Event Ingestion
        ↓
Revenue Risk Engine          (prioritizes cases: amount × risk × urgency)
        ↓
ML Recovery Prediction        (P(recovery | customer, context, intervention) — per action)
        ↓
Intervention / Expected-Value Policy   (EV per allowed action)
        ↓
AI Decision Agent             (LLM — chooses among allowed actions, explains why)
        ↓
Hard Compliance Gate          (deterministic code — can override/block the agent)
        ↓
Execution Simulator           (simulated send: retry link / reminder / discount / escalate / no-op)
        ↓
Outcome / Recovery Simulator  (did it actually recover, and how much)
        ↓
Audit Ledger + Metrics Engine
        ↓
Dashboard ("Revenue Recovery Control Center")
```

The LLM sits in the middle of this pipeline, not at the top of it. It never has the final word — the compliance gate does.

## 5. Data schema (synthetic — primary data source for MVP)

Generate a few thousand synthetic events (this is cheap — a script runtime cost, not a project-risk cost; don't hesitate on volume). Critically: **include realistic noise and overlap, not deterministic rules** like "insufficient_funds always recovers at 70%." Recovery probability should be a noisy function of *multiple* interacting features, so the ML task is genuinely non-trivial.

**Customer:**
`customer_id`, `customer_age_days`, `lifetime_value`, `previous_payment_failures`, `previous_success_rate`, `average_order_value`, `previous_recovery_successes`, `days_since_last_purchase`

**Transaction:**
`transaction_id`, `amount`, `event_type` (`payment_failed` | `checkout_abandoned`), `failure_reason`, `timestamp`, `time_of_day`, `day_of_week`

**Contact/compliance:**
`attempts_so_far`, `last_contact_time`, `opted_out`

**Intervention (the action taken):**
`retry_link` | `reminder` | `discount_offer` | `escalate_to_human` | `no_action`

**Outcome:**
`recovered` (bool), `recovered_amount`, `time_to_recovery`

**Important design point — counterfactual labels:** because this data is synthetic, you can (and should) generate a *true underlying recovery probability for every candidate intervention* per customer (a noisy function of their features), then sample the actually-observed outcome for whichever intervention gets chosen. This gives you something you could never have with real-world data: ground truth to check whether your policy picked the actually-best action, not just whether it beat a baseline on average. Use this to compute an "optimality gap" metric alongside your headline recovery numbers — it's a strong, hard-to-fake rigor signal for judges.

**Razorpay test mode** (optional, secondary): a small number of real `payment.failed` webhook events from a Razorpay Test Mode account can seed a few authentic-looking cases for the demo video. This is a credibility polish item — cap it at roughly a day of effort, and never let it become a dependency for the core pipeline. Check Razorpay's current docs for exact test card/VPA values.

## 6. ML methodology

**Target:** recovery probability **conditional on intervention** — not a single "will this recover" score.

**Practical build recommendation:** rather than training a separate model per intervention (more moving parts, more to evaluate, more that can go subtly wrong solo), train **one model with intervention type as a categorical feature**, and score the same customer row once per candidate intervention to get a probability for each action. This is standard practice for this kind of problem (an uplift-style setup) and is meaningfully less work to build, train, and validate correctly in a 10-day solo timeline while still satisfying "recovery probability conditional on intervention."

- **Models:** XGBoost as primary, Logistic Regression as a clean, interpretable baseline.
- **Split:** proper train/test split (don't evaluate on training data — an easy mistake under time pressure).
- **Report:** precision, recall, confusion matrix, and (given the counterfactual labels above) the optimality-gap metric. Do not optimize only for the most flattering-looking numbers — an honest, slightly-less-perfect metric with a clear explanation is worth more to a judge than a suspiciously clean one.

## 7. Expected-value / intervention selection

For each candidate intervention, compute:

```
Expected Value = P(recovery | customer, context, intervention)
                 × recoverable_amount
                 − intervention_cost
                 − discount_cost (if applicable)
```

The formula can be refined during implementation, but this expected-value framing must be the actual mechanism deciding what "best" means — not an LLM's unstructured judgment call. The set of *allowed* candidate interventions for a given case is determined by the compliance/context state (e.g., if already contacted twice today, `retry_link`/`reminder` aren't even offered as candidates) — this keeps the search space bounded before the agent even runs.

## 8. Agent design

The LLM receives structured input:
- customer context
- ML predictions (probability per candidate intervention)
- expected value per candidate intervention
- compliance state
- contact history

And returns structured JSON only:

```json
{ "action": "retry_link", "reason": "Highest expected value among allowed actions; low prior contact frequency." }
```

Constraints:
- `action` must come from the explicit allowed-action list for this case — never an invented action.
- The LLM is a **decision-and-explanation layer**, not the final authority.
- Provider-agnostic interface, structured JSON output enforced regardless of which model sits behind it. Use a free-tier API (Gemini Flash-Lite via Google AI Studio, or Groq) — no paid API is required at this project's scale.

## 9. Compliance gate (deterministic code — final authority)

Rules:
- Max N contact attempts per customer per rolling 24h window.
- No contact outside allowed hours (e.g. 9am–8pm).
- Immediate, permanent stop if `opted_out = true`.
- Stop immediately on confirmed recovery.
- After M failed attempts, force `escalate_to_human` rather than continuing automated retries.
- Any action violating policy is blocked, full stop — regardless of the agent's reasoning or expected-value score.

**This override must be visibly demonstrated in the demo**, e.g.:

> Agent recommends: `reminder` → Compliance check: `opted_out = true` → **BLOCKED** → Final action: `no_action`

> Agent recommends: `retry_link` → Compliance check: already contacted twice in 24h → **BLOCKED** → Final action: `escalate_to_human`

This is one of the highest-value moments in the whole demo — it's the difference between "the AI is well-behaved" (a claim) and "the AI cannot misbehave" (a proof).

## 10. Execution simulator

Simulates sending the chosen, gate-approved action (retry link / reminder / discount offer / escalation / no-op). Nothing is actually delivered anywhere — this is logged as if sent, with the message content generated but not dispatched.

## 11. Outcome / recovery simulator

Determines whether the simulated intervention actually "recovered" the revenue, sampling from the same underlying (noisy) probability model used to generate the synthetic ground truth — this is what produces your `recovered`, `recovered_amount`, and `time_to_recovery` fields per case.

## 12. Audit system

Every case must be fully reconstructable after the fact. Log, per transaction:
- input event
- model prediction(s) and confidence/probabilities per candidate intervention
- candidate actions considered
- selected action
- agent reasoning
- compliance checks performed and their results
- blocked/allowed decision
- executed action
- outcome and recovered amount
- stopping reason
- timestamps throughout

## 13. Dashboard — "Revenue Recovery Control Center"

Not a generic admin panel — organize around money and compliance:

**Top KPI cards:** Revenue at Risk (₹) · Revenue Recovered (₹) · Recovery Rate · Incremental Recovery vs. baseline

**Recovery by intervention:** retry / reminder / discount / human escalation — performance breakdown per action type

**Compliance panel:** blocked actions count, opt-outs respected, contact-limit violations, quiet-hour violations, target compliance violations = **0** (state this explicitly)

**Recent decisions table:** customer, amount/risk, predicted recovery, selected action, compliance result, outcome

**Stopping-reason breakdown**

## 14. Evaluation methodology

- Held-out test split for the ML model; report precision/recall/confusion matrix honestly.
- Optimality-gap metric using the synthetic ground-truth probabilities (see §5/§6) — how close did the policy get to the actually-best action per case.
- Batch-level financial metrics: ₹ at risk, ₹ recovered, recovery rate, incremental recovery vs. baseline.
- Zero compliance violations, demonstrated not asserted.

## 15. Baseline experiment (required)

Run the same synthetic batch through three policies and report actual results from each — never fabricate or approximate these numbers:

1. **Do nothing** → ₹X at risk → ₹0 recovered
2. **Generic intervention for everyone** (e.g. always send a plain reminder) → ₹X at risk → ₹Y recovered
3. **AI Recovery Agent** (full pipeline) → ₹X at risk → ₹Z recovered

Report: total recovered, recovery rate, incremental recovered revenue, uplift vs. each baseline. This is the single most important piece of evidence that the system does something beyond "generate messages." Protect the time to build it properly.

## 16. API / backend structure (adjust as needed during implementation)

```
POST /events                    # ingest a payment_failed / checkout_abandoned event
POST /recover/{transaction_id}  # run the full pipeline for one case
POST /batch/run                 # run a full synthetic batch (and baselines)
GET  /batch/{batch_id}          # batch-level results
GET  /audit/{transaction_id}    # full decision trace for one case
GET  /metrics                   # dashboard-facing aggregate metrics
```

## 17. Tech stack

- **Backend:** FastAPI
- **ML:** XGBoost (primary), Logistic Regression (baseline)
- **Database:** SQLite (sufficient for MVP)
- **Frontend:** React + Vite
- **Agent:** any free-tier LLM API (Gemini Flash-Lite via Google AI Studio, or Groq) behind a provider-agnostic interface, structured JSON output enforced. No paid API required.

## 18. MVP deliverables

1. Synthetic event generator (with counterfactual per-intervention ground truth)
2. ML training/evaluation pipeline (proper split, honest metrics)
3. Intervention-conditional recovery prediction
4. Expected-recovery-value calculation
5. Constrained AI decision agent (structured JSON, fixed action set)
6. Deterministic compliance engine
7. Execution simulator
8. Outcome/recovery simulator
9. Audit ledger
10. Revenue Recovery Control Center dashboard
11. Baseline comparison (do nothing / generic / agent)
12. A deliberately demonstrated compliance-override case
13. README (architecture, metrics, limitations)
14. 5-minute demo video

## 19. Demo flow (structure around money, not features)

1. Batch arrives → "₹X revenue at risk."
2. Agent prioritizes the riskiest cases.
3. Walk through one customer: failure reason, per-intervention recovery probabilities, expected values.
4. Show the agent choosing an intervention, with its reasoning.
5. Show the compliance gate approving it.
6. Show a **second** case where the agent is **blocked** by compliance — this is the standout moment.
7. Run the full batch.
8. Show final results: ₹ at risk, ₹ recovered, recovery rate, compliance violations (zero), per-intervention performance.
9. Show the baseline comparison — do nothing vs. generic vs. agent — side by side.

The demo should make it obvious this recovers measurable revenue, not that it generates plausible-sounding messages.

## 20. Limitations (state these honestly in the README — judges respect this more than silence)

- All financial outcomes are simulated against a synthetic ground-truth model, not real customer behavior — the counterfactual/optimality-gap framing is a way to be rigorous *within* that constraint, not a substitute for real-world validation.
- The intervention-conditional model uses a single model with intervention as a feature rather than separate per-action models — a reasonable simplification for MVP scope, with a real trade-off in how well it isolates each intervention's true effect (a known limitation of this modeling choice, worth naming rather than hiding).
- Razorpay integration (if included) is a small authenticity add-on, not a validated production integration.
- Compliance rules are a representative subset (contact caps, quiet hours, opt-out, escalation), not a complete regulatory implementation.

## 21. Build order

1. **Synthetic data generator** — schema + counterfactual per-intervention probabilities + sampled outcomes. Get this right first; everything depends on it.
2. **ML pipeline** — train, proper split, honest eval report (precision/recall/confusion matrix/optimality gap).
3. **Expected-value policy layer** — turn per-intervention probabilities into an EV ranking among allowed actions.
4. **Decision agent** — structured JSON in/out, fixed action set, reasoning string.
5. **Compliance gate** — enforced code, with explicit tests for each override case (opt-out, contact cap, quiet hours, escalation).
6. **Execution + outcome simulators.**
7. **Audit ledger.**
8. **Baseline experiment** — do nothing / generic / agent, run on the same batch.
9. **Dashboard** — KPI cards, per-intervention breakdown, compliance panel, recent decisions, stopping reasons.
10. **Deliberate compliance-override demo case** — confirm it behaves and is easy to show live.
11. **README** — architecture, metrics, limitations.
12. **Record demo video**, structured around §19.
