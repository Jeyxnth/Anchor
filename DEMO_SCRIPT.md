# Demo Script — Compliance & Validation Cases

Repeatable click-through path for the compliance-restricted cases plus the
validator-injection moment, for live recording. No transaction-ID hunting
required — the dashboard's **Compliance Demo Cases** panel has one-click
buttons for all three IDs below, or use the **Jump to transaction** search
box in the header.

## Prerequisites

1. Backend running: `backend/.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000`
2. Dashboard running: `cd frontend && npm run dev`, open http://localhost:5173
3. A terminal in `backend/`, venv activated (or use the full path shown
   above) — needed for Case 4.
4. The ledger needs at least the three transactions below already logged
   under `policy_name='ai_agent'`. Either:
   - Click **Run Batch (agent only)** with rows ≥ 2200 (TXN002166 is at
     index 2166), or
   - Load the existing full run: type `baseline_experiment_full` into the
     Policy Comparison panel's batch-id box and click **Load** (this also
     populates the comparison panel — good to do first anyway, see below).

## Order

### 0. Set the scene — Policy Comparison panel
Load `baseline_experiment_full` (or run a fresh baseline experiment). Point
at the three policy cards and the "AI Agent recovers +₹541,359.81 more than
do_nothing, and +₹139,252.32 more than generic_reminder" callout. This is
the number that matters — everything after this is *why* the agent gets
there.

### 1. TXN000000 — Normal case (baseline, for contrast)
Click its button in **Compliance Demo Cases** (or search it).
**Proves:** all 5 actions eligible; the EV ranking — not raw probability —
drives the pick. `escalate_to_human` has the *highest* predicted recovery
probability (35.5%) but loses to `retry_link` (35.0%) once its ₹40 cost is
priced in (EV ₹509.95 vs ₹479.23). Cost-aware, not just probability-aware.

### 2. TXN000025 — Opted-out customer
Click its button.
**Proves:** `opted_out=true` → `eligible_actions=['no_action']`. Scroll to
the trace's **"Compliance effect — before / after"** section: without
compliance, the top-EV pick would have been `retry_link` (EV ₹67.71) — the
system would have tried to contact an opted-out customer. Compliance
restricts the candidate set to `no_action` *before* the agent ever runs,
so that's what actually happened. This is the before/after moment — point
at it directly, it's the clearest single screen for "compliance actually
did something here."

### 3. TXN002166 — Contact-capped customer
Click its button.
**Proves:** `attempts_so_far ≥ 2` in the trailing 24h → eligible actions
narrow to `['escalate_to_human', 'no_action']`. Same before/after section:
unconstrained top pick would again have been `retry_link` (EV ₹78.72) —
another contact attempt on an already-over-the-cap customer — but
compliance restricted it to `escalate_to_human` (EV ₹43.09), which is what
was actually chosen.

### 4. Validator injection — off-list LLM response, caught live
Switch to the terminal:
```
backend/.venv/Scripts/python.exe policy/demo_validator_injection.py
```
This reuses TXN000000's real model predictions and real EV ranking, then
feeds a **staged** (not live-API) LLM response that invents an action —
`"send_free_product"` — never a real intervention. Narrate as it prints:
the real EV-ranked candidates first (same numbers as Case 1), then the
fake response, then `agent.validate_decision()` — the *exact* function
every real Gemini/Groq response passes through, not a separate mock path —
catching it, correcting to the top-EV eligible candidate (`retry_link`),
and flagging `valid=False` with a `validation_note` naming exactly what
happened. The script asserts both outcomes before printing its final line,
so a silent failure can't slip through unnoticed.
**Proves:** the agent's raw action choice is never trusted at face value —
structurally impossible for an LLM (or a compromised one) to make the
system execute an action outside the allowed set.

## Honest scope note (say this out loud, don't let a question catch it)

Cases 2–3 demonstrate the compliance **eligibility filter** — it bounds the
candidate set *before* the agent runs, and the before/after view makes that
concrete. They are not yet the brief's "agent recommends X → compliance
blocks it → final action is Y" override moment (§19 step 6) — that needs
the **hard compliance gate** (build-order step 5: quiet hours, forced
escalation, stop-on-recovery, override authority), which isn't built yet.
When it is, this script gets a **Case 5** showing an actual post-decision
override, distinct from both the eligibility-filter cases above and the
validator-injection case (which guards against a *malformed* response, not
a *compliant-but-overridden* one).

## Fallback if the ledger doesn't have these transactions loaded

`GET /audit/TXN000025?policy_name=ai_agent` (and similarly for the other
two IDs) works directly against the API regardless of what's currently
loaded in the dashboard's Recent Decisions table — useful as a backup if
something in the UI state gets into a weird spot mid-recording.
