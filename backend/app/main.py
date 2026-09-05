"""
FastAPI backend — brief §16/§17. Exposes the audit ledger and dashboard
metrics over HTTP so the React dashboard (frontend/) has something to call.

Endpoints implemented now:
    POST /batch/run                   run one policy over N events.csv rows, log to ledger
    POST /batch/baseline_experiment   run all 3 policies (ai_agent/do_nothing/generic_reminder)
                                        over the same N rows under one batch_id (build-order step 8)
    GET  /batch/{batch_id}            all traces for one batch (optional ?policy_name=)
    GET  /batch/{batch_id}/compare    side-by-side policy comparison for one batch_id
    GET  /audit/{transaction_id}      full decision trace for one case (optional ?policy_name=)
    GET  /metrics                     dashboard-facing aggregate metrics (optional ?batch_id=&policy_name=)
    POST /recover/{transaction_id}    run the live pipeline for one case (real LLM
                                        provider by default), simulate its outcome, and log it

Not implemented yet (things they'd depend on don't exist): POST /events
(ingestion — no separate risk-prioritization stage exists yet, the batch
runner reads events.csv directly), and anything gate-shaped (build-order
step 5 — the hard compliance gate with override authority).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import load_events  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy"))
from pipeline import decide_for_row, load_model  # noqa: E402
from agent import get_provider  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import WRITE_LOCK, get_connection, init_db  # noqa: E402
from ledger import compare_policies, compute_metrics, get_trace, list_batch, record_decision_trace  # noqa: E402
from batch import run_batch as _run_batch  # noqa: E402
from batch import run_baseline_experiment as _run_baseline_experiment  # noqa: E402
from outcome import load_ground_truth, new_rng, simulate_outcome  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = FastAPI(title="AI Revenue Recovery Agent — API", version="0.1.0")

# Local dev only: Vite's default port. Tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None      # lazy-loaded singleton, avoids reloading the joblib artifact per request
_events_df = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def get_events_df():
    global _events_df
    if _events_df is None:
        _events_df = load_events(str(DATA_DIR / "events.csv"))
    return _events_df


@app.on_event("startup")
def _startup():
    init_db()


class BatchRunRequest(BaseModel):
    n: Optional[int] = None          # None = all rows in events.csv
    use_real_llm: bool = False        # False (default): RuleBasedFallbackProvider for speed/cost
    batch_id: Optional[str] = None


class BaselineExperimentRequest(BaseModel):
    n: Optional[int] = None
    use_real_llm: bool = False
    batch_id: Optional[str] = None


@app.post("/batch/run")
def batch_run(req: BatchRunRequest):
    provider = get_provider() if req.use_real_llm else None  # None -> batch.py's own fallback default
    summary = _run_batch(n=req.n, provider=provider, batch_id=req.batch_id)
    return summary


@app.post("/batch/baseline_experiment")
def baseline_experiment(req: BaselineExperimentRequest):
    provider = get_provider() if req.use_real_llm else None
    return _run_baseline_experiment(n=req.n, batch_id=req.batch_id, provider=provider)


@app.get("/batch/{batch_id}")
def batch_results(batch_id: str, policy_name: Optional[str] = None):
    conn = get_connection()
    traces = list_batch(conn, batch_id, policy_name=policy_name)
    conn.close()
    if not traces:
        raise HTTPException(status_code=404, detail=f"No decisions found for batch_id={batch_id!r}")
    return {"batch_id": batch_id, "policy_name": policy_name, "n_cases": len(traces), "decisions": traces}


@app.get("/batch/{batch_id}/compare")
def batch_compare(batch_id: str):
    conn = get_connection()
    result = compare_policies(conn, batch_id)
    conn.close()
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/audit/{transaction_id}")
def audit_trace(transaction_id: str, policy_name: Optional[str] = None):
    conn = get_connection()
    trace = get_trace(conn, transaction_id, policy_name=policy_name)
    conn.close()
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"No decision logged for transaction_id={transaction_id!r}"
                   + (f", policy_name={policy_name!r}" if policy_name else ""),
        )
    return trace


@app.get("/metrics")
def metrics(batch_id: Optional[str] = None, policy_name: Optional[str] = None):
    conn = get_connection()
    m = compute_metrics(conn, batch_id=batch_id, policy_name=policy_name)
    conn.close()
    return m


@app.post("/recover/{transaction_id}")
def recover(transaction_id: str, batch_id: str = "live"):
    """Runs the live DECIDE pipeline for one transaction_id from events.csv,
    using the real auto-detected LLM provider (get_provider(), resilient
    fallback included) — unlike /batch/run, which defaults to the
    deterministic provider for cost/speed reasons. Simulates and logs the
    outcome too, under policy_name='ai_agent', same as a batch run."""
    events_df = get_events_df()
    matches = events_df.loc[events_df["transaction_id"] == transaction_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"transaction_id={transaction_id!r} not found in events.csv")
    row = matches.iloc[0]

    trace = decide_for_row(get_model(), row, provider=get_provider())

    ground_truth = load_ground_truth()
    rng = new_rng()  # a fresh seeded RNG per call — see outcome.py; a single live case doesn't
                      # need the cross-policy alignment a batch run cares about
    true_prob = ground_truth[transaction_id][trace["agent_decision"]["action"]]
    outcome = simulate_outcome(rng, true_prob, row["amount"])

    with WRITE_LOCK:  # see the note on WRITE_LOCK in app/db.py
        conn = get_connection()
        record_decision_trace(conn, batch_id, "ai_agent", row, trace, outcome)
        conn.commit()
        conn.close()
    return {**trace, "outcome": outcome}


@app.get("/health")
def health():
    return {"status": "ok"}
