# Demo Script — Final

## Headline numbers (say these early and often)

**Batch:** 4,000 transactions · ₹17,07,076.43 total revenue at risk

| Policy | ₹ Recovered | Recovery Rate |
|---|---|---|
| Do Nothing | ₹3,14,135.94 | 18.4% |
| Generic Reminder | ₹6,44,189.16 | 37.7% |
| **AI Agent** | **₹7,46,318.78** | **43.5%** |

**AI Agent recovers +₹4,32,182.84 more than doing nothing, and +₹1,02,129.62 more than a generic reminder to everyone.**

## Order

### 0. Set the scene — Policy Comparison panel
Click **Run Baseline Experiment** (this always runs the full 4,000-row dataset, independent of the rows field in the header). Point at the three cards and say the numbers above out loud. *"This is the number that matters — everything after this is why the agent gets there."*

### 1. TXN000224 — Normal case (baseline, for contrast)
Click it in Compliance Demo Cases. 
**Say:** All 5 actions are eligible here. Escalate to Human has the *highest* predicted recovery probability (56.0% vs. Retry Link's 55.0%), but the system picks Retry Link instead — because Escalate costs ₹40 to execute and Retry Link costs ₹2. Once that's priced in, Retry Link's expected value (₹629.45) beats Escalate's (₹602.69). *"This isn't just picking the most likely outcome — it's making a genuine cost-aware business decision."*

### 2. TXN000025 — Opted-out customer
Click it. Scroll to **Compliance Effect — Before/After**.
**Say:** This customer opted out. Without compliance, the system's top pick would have been Retry Link (EV ₹67.71) — it would have tried to contact someone who told us to stop. Compliance restricts the eligible set to *only* No Action before the agent even runs, so that's what actually happens. *"The agent isn't being told not to do this — it's structurally never offered the option."*

### 3. TXN002166 — Contact-capped customer (now a stacked-restrictions example)
Click it. Same before/after panel — note the panel itself only shows two states (Unconstrained vs. the actual/executed outcome); the middle number below is spoken context, not something on screen.
**Say:** Already contacted twice in 24 hours — *and* this event happens to land at 3am, outside contact hours. Two independent rules apply here from different angles: unconstrained, the top pick would be Retry Link (EV ₹78.72, visible on the panel) — another attempt on someone already at the cap. The contact-cap rule *alone* would already restrict this to Escalate to Human (EV ₹43.09 — this part isn't on the panel, it's worth saying out loud). But quiet hours applies *on top of* that, and independently forces it all the way to No Action (EV ₹29.14, the panel's actual/executed side) — even the human hand-off waits until contact hours resume. *"This is what defense-in-depth looks like: two separate rules, checked independently, agreeing on the same restriction from different reasons — not one rule doing double duty."* (This is TXN002166's real behavior: it's the only row in the full 4,000-transaction dataset with 2+ prior contact attempts, and it happens to also be a quiet-hour case — a coincidence of the data, not a staged example.)

### 4. NEW — Quiet hours, with the freshness exception
Find and open any `checkout_abandoned` case flagged as quiet-hour restricted (any customer contacted for the first time, outside 9am–8pm, on an abandoned checkout — Compliance Panel shows the live count of these).
**Say:** *"We also enforce quiet hours — no automated outreach outside 9am to 8pm. But we made one deliberate, reasoned exception: if someone's payment fails right now, mid-transaction, they're provably awake and present — treating that the same as a cold 2am marketing text would be wrong. So the very first response to an active payment failure is allowed regardless of the hour. A checkout that was abandoned hours ago doesn't get that exception, because there's no evidence anyone's still there — and neither does a second or third follow-up attempt on anything."* Then show one `payment_failed` case with `attempts_so_far == 0` where all 5 actions stayed eligible despite the hour — contrast it against a `checkout_abandoned` case restricted at the same hour.
**Compliance Panel numbers to cite:** 0 target compliance violations across 4,000 cases. Opt-outs: 100% respected. Quiet-hour restricted: a real, meaningful count (not near-zero, not near-total) — cite whatever the panel currently shows.

### 5. Validator injection — off-list LLM response, caught live
Terminal:
```
backend/.venv/Scripts/python.exe policy/demo_validator_injection.py
```
**Say, as it prints:** real EV-ranked candidates first (same numbers as Case 1), then a staged fake response inventing an action (`send_free_product`) that was never a real intervention, then `validate_decision()` — *the exact function every real Gemini/Groq response passes through* — catching it, correcting to the top-EV eligible candidate, flagging it invalid. *"The agent's raw output is never trusted at face value — this is structurally impossible to bypass, whether the model hallucinates or is compromised."*

### 6. Recovery by Intervention (if time allows)
Point at the breakdown: Retry Link, Reminder, Discount Offer, and Escalate to Human each recovering in the 50–57% range — genuinely close to each other. *"The system isn't just defaulting to one favorite action — it's making a real per-case call, and each action type performs well when it's actually the right fit."*

### 7. Close — back to the Policy Comparison panel
Restate the headline numbers one more time as the closing beat: **+₹4,32,182.84 vs. doing nothing, +₹1,02,129.62 vs. a generic policy, zero compliance violations across every one of the 4,000 cases.**

## Honest scope note — say this proactively, don't wait to be asked

*"Compliance here is two deterministic code layers, not a prompt instruction: an eligibility filter that removes forbidden actions before the agent ever sees them, and a validator that re-checks the agent's actual output against that same list and corrects it if it's ever wrong — we tested that directly with the injection you just saw. We considered a third, after-the-fact override layer and deliberately didn't build it, because there's no case in this design where a properly filtered, on-list decision still needs blocking afterward — a third layer would prove nothing the first two haven't already proven."*

## If Gemini's real-LLM toggle is live and working when you record

Show one small batch (10-15 rows) with "use real LLM" on, and narrate the Provider column honestly — if it's a mix of `gemini` and `rule_based_fallback`, that's a *better* demo moment than a clean run: *"Google's API had a hiccup mid-batch there — watch it degrade to a rule-based fallback instead of the whole pipeline stalling. That's the resilience wrapper working exactly as designed."*

## Fallback if the ledger doesn't have these transactions loaded

`GET /audit/TXN000025?policy_name=ai_agent` (and the other two IDs) works directly against the API regardless of dashboard state — useful if something in the UI gets into a weird spot mid-recording.
