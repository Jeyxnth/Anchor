"""
Run policies over a batch of events.csv rows and persist every trace +
simulated outcome to the audit ledger. Three policies exist:

  ai_agent          — the real pipeline: model -> EV ranking -> compliance
                       eligibility filter -> AI decision agent (or its
                       rule-based fallback). policy/pipeline.py.
  do_nothing        — baseline: always no_action, ignoring the model entirely.
  generic_reminder  — baseline: always reminder if compliance allows it,
                       else no_action (never escalates or offers a discount —
                       that would no longer be "generic"). Still respects the
                       SAME compliance-eligibility filter as ai_agent — a
                       baseline that ignored compliance wouldn't be a fair
                       comparison, and this project doesn't ship one.

All three log outcome_recovered / outcome_recovered_amount, sampled from
ground_truth.csv via app/outcome.py — build-order step 6, wired in here.
run_baseline_experiment() runs all three under one shared batch_id so
ledger.compare_policies() can report them side by side (build-order step 8).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import load_events  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy"))
from pipeline import decide_for_row, load_model  # noqa: E402
from agent import RuleBasedFallbackProvider  # noqa: E402
from compliance import eligible_actions, state_from_event_row  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import WRITE_LOCK, get_connection, init_db  # noqa: E402
from ledger import compare_policies, record_decision_trace  # noqa: E402
from outcome import load_ground_truth, new_rng, simulate_outcome  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _baseline_trace(row: pd.Series, selected_action: str, eligible: list[str], reason: str, provider_tag: str) -> dict:
    """Builds a trace dict shaped like pipeline.decide_for_row()'s output,
    but for a fixed-rule baseline policy that never consults the model —
    predicted_probabilities/candidates_ev_ranked are empty, honestly, rather
    than backfilled with numbers the policy never looked at."""
    return {
        "transaction_id": row["transaction_id"],
        "customer_context": {
            "customer_age_days": int(row["customer_age_days"]),
            "lifetime_value": float(row["lifetime_value"]),
            "previous_success_rate": float(row["previous_success_rate"]),
            "average_order_value": float(row["average_order_value"]),
            "previous_recovery_successes": int(row["previous_recovery_successes"]),
            "days_since_last_purchase": int(row["days_since_last_purchase"]),
        },
        "transaction_context": {
            "amount": float(row["amount"]),
            "event_type": row["event_type"],
            "failure_reason": row["failure_reason"],
            "time_of_day": int(row["time_of_day"]),
            "day_of_week": int(row["day_of_week"]),
        },
        "compliance_state": {
            "opted_out": bool(row["opted_out"]),
            "attempts_so_far_24h": int(row["attempts_so_far"]),
            "last_contact_time": None if pd.isna(row.get("last_contact_time")) else row["last_contact_time"],
        },
        "predicted_probabilities_all_actions": {},
        "eligible_actions": eligible,
        "candidates_ev_ranked": [],
        "agent_decision": {
            "action": selected_action, "reason": reason, "provider": provider_tag,
            "valid": True, "validation_note": None,
        },
    }


def run_batch(n: Optional[int] = None, provider=None, batch_id: Optional[str] = None,
              db_path=None) -> dict:
    """policy_name='ai_agent'. Runs the real DECIDE pipeline on the first `n`
    rows of events.csv (None = all 4000), simulates each outcome, and writes
    every row to the ledger."""
    batch_id = batch_id or f"batch_{uuid.uuid4().hex[:12]}"
    provider = provider or RuleBasedFallbackProvider()

    events_df = load_events(str(DATA_DIR / "events.csv"))
    if n is not None:
        events_df = events_df.head(n)

    model = load_model()
    ground_truth = load_ground_truth()
    rng = new_rng()

    with WRITE_LOCK:
        conn = init_db(db_path) if db_path else init_db()

    n_ok, n_failed = 0, 0
    try:
        for _, row in events_df.iterrows():
            try:
                # ML prediction, EV ranking, and the agent/LLM call all
                # happen here, OUTSIDE WRITE_LOCK. This is deliberate and
                # load-bearing: provider.decide() can make a real network
                # call (Gemini/Groq) that's allowed to run for a while, and
                # in practice has been observed to hang well past its own
                # `timeout=` value (see agent.py / STATUS.md §5) — a stuck
                # socket read never raises, so code after it (including a
                # lock release) never runs either. WRITE_LOCK must never
                # wrap anything that can block indefinitely; it now only
                # ever wraps the actual DB write below, which is always
                # fast and always local.
                trace = decide_for_row(model, row, provider=provider)
                true_prob = ground_truth[row["transaction_id"]][trace["agent_decision"]["action"]]
                outcome = simulate_outcome(rng, true_prob, row["amount"])
                with WRITE_LOCK:
                    record_decision_trace(conn, batch_id, "ai_agent", row, trace, outcome)
                    conn.commit()
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 - one bad row shouldn't kill the whole batch
                n_failed += 1
                print(f"  [batch {batch_id} / ai_agent] row {row.get('transaction_id')} failed: {exc!r}")
    finally:
        conn.close()

    return {"batch_id": batch_id, "policy_name": "ai_agent", "n_cases": n_ok, "n_failed": n_failed,
            "provider": provider.name}


def run_baseline_batch(policy_name: str, n: Optional[int] = None, batch_id: Optional[str] = None,
                        db_path=None) -> dict:
    """policy_name in {'do_nothing', 'generic_reminder'} — fixed-rule
    policies, no model/agent call, same compliance filter and outcome
    simulator as ai_agent."""
    if policy_name not in ("do_nothing", "generic_reminder"):
        raise ValueError(f"unknown baseline policy_name={policy_name!r}")

    batch_id = batch_id or f"batch_{uuid.uuid4().hex[:12]}"
    events_df = load_events(str(DATA_DIR / "events.csv"))
    if n is not None:
        events_df = events_df.head(n)

    ground_truth = load_ground_truth()
    rng = new_rng()

    with WRITE_LOCK:
        conn = init_db(db_path) if db_path else init_db()

    n_ok, n_failed = 0, 0
    try:
        for _, row in events_df.iterrows():
            try:
                state = state_from_event_row(row)
                eligible = eligible_actions(state)

                if policy_name == "do_nothing":
                    action, reason = "no_action", "Baseline policy: always no_action, regardless of eligibility."
                else:  # generic_reminder
                    if "reminder" in eligible:
                        action, reason = "reminder", "Baseline policy: always reminder when compliance allows it."
                    else:
                        action, reason = "no_action", (
                            "Baseline policy: reminder not eligible under current compliance state "
                            f"(eligible={eligible}); falls back to no_action rather than escalate or "
                            "discount, which would no longer be a 'generic' policy."
                        )

                trace = _baseline_trace(row, action, eligible, reason, provider_tag=f"baseline_{policy_name}")
                true_prob = ground_truth[row["transaction_id"]][action]
                outcome = simulate_outcome(rng, true_prob, row["amount"])
                # No LLM call on this path (fixed-rule baseline), but locked
                # the same way as run_batch() for consistency and so a large
                # baseline run never holds the file lock for its whole
                # duration either — see the note in run_batch() above.
                with WRITE_LOCK:
                    record_decision_trace(conn, batch_id, policy_name, row, trace, outcome)
                    conn.commit()
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                print(f"  [batch {batch_id} / {policy_name}] row {row.get('transaction_id')} failed: {exc!r}")
    finally:
        conn.close()

    return {"batch_id": batch_id, "policy_name": policy_name, "n_cases": n_ok, "n_failed": n_failed}


def run_baseline_experiment(n: Optional[int] = None, batch_id: Optional[str] = None,
                             provider=None, db_path=None) -> dict:
    """Runs all three policies over the SAME batch_id (so they're logged
    side by side under the composite (transaction_id, policy_name) key) and
    returns ledger.compare_policies()'s report. Build-order step 8."""
    batch_id = batch_id or f"baseline_experiment_{uuid.uuid4().hex[:12]}"

    summaries = [
        run_batch(n=n, provider=provider, batch_id=batch_id, db_path=db_path),
        run_baseline_batch("do_nothing", n=n, batch_id=batch_id, db_path=db_path),
        run_baseline_batch("generic_reminder", n=n, batch_id=batch_id, db_path=db_path),
    ]

    conn = get_connection(db_path) if db_path else get_connection()
    comparison = compare_policies(conn, batch_id)
    conn.close()

    return {"batch_id": batch_id, "run_summaries": summaries, "comparison": comparison}


if __name__ == "__main__":
    import json as _json

    result = run_baseline_experiment()
    print(_json.dumps(result, indent=2))
