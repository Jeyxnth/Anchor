# Revenue Recovery Control Center (dashboard)

React + Vite frontend for the AI Revenue Recovery Agent (`../PROJECT_BRIEF.md`,
build-order steps 7/9). Talks to the FastAPI backend in `../backend/app`.

## Run

Backend first (from `backend/`, with the venv):

```
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Then, from this directory:

```
npm install   # first time only
npm run dev
```

Open http://localhost:5173. Click **Run Batch** to score N rows of
`events.csv` through the DECIDE-stage pipeline (model → EV ranking →
compliance-eligibility filter → agent decision) and log each one to the
audit ledger (`backend/app/ledger.db`, SQLite). Click any row in **Recent
Decisions** for the full per-case trace.

By default batch runs use the deterministic `rule_based_fallback` provider
(fast, free). Check "use real LLM" to route through `agent.get_provider()`
instead (Gemini/Groq, auto-detected from `backend/.env`) — slower, one API
call per row.

## What's real vs. pending

Several dashboard fields are `null` and labeled "pending" rather than
faked — they depend on pipeline stages that don't exist yet (build-order
steps 5, 6, 8): the hard compliance gate, execution/outcome simulators, and
the baseline comparison. Everything else (revenue at risk, per-intervention
predicted-recovery breakdown, compliance-eligibility stats, per-case EV
ranking and agent reasoning) is computed from real pipeline output, not
mocked.

## Known limitation

The ledger's `decisions` table keys on `transaction_id` alone, so a second
batch run over overlapping transactions overwrites the first (latest
decision per transaction wins — there's no per-batch history). Fine for the
current single-pipeline audit trail; will need revisiting once the
baseline experiment (step 8) needs 3 parallel result sets (do-nothing /
generic / agent) for the *same* transactions side by side.
