"""
Audit ledger read/write functions — the persistence half of build-order
step 7. Writes one row per decided transaction into the `decisions` table
(schema in db.py); reads back either a single full trace (GET /audit/{id})
or aggregate dashboard metrics (GET /metrics).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


def record_decision_trace(conn: sqlite3.Connection, batch_id: str, row: pd.Series, trace: dict) -> None:
    """Persist one policy/pipeline.py decide_for_row() trace. `row` is the
    source events.csv row (for fields the trace doesn't carry, e.g. amount
    as a plain float for SQL aggregation without JSON parsing)."""
    d = trace["agent_decision"]
    conn.execute(
        """
        INSERT OR REPLACE INTO decisions (
            transaction_id, batch_id, customer_id,
            event_type, failure_reason, amount, input_event_json,
            predicted_probabilities_json,
            candidates_ev_ranked_json,
            eligible_actions_json, compliance_state_json,
            selected_action, agent_reason, agent_provider, agent_valid, agent_validation_note,
            gate_status, gate_reason, final_action,
            executed_action, outcome_recovered, outcome_recovered_amount,
            time_to_recovery_hours, stopping_reason,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trace["transaction_id"], batch_id, str(row["customer_id"]),
            row["event_type"], row["failure_reason"], float(row["amount"]),
            json.dumps({**trace["customer_context"], **trace["transaction_context"]}, default=str),
            json.dumps(trace["predicted_probabilities_all_actions"]),
            json.dumps(trace["candidates_ev_ranked"]),
            json.dumps(trace["eligible_actions"]), json.dumps(trace["compliance_state"], default=str),
            d["action"], d["reason"], d["provider"], int(bool(d["valid"])), d["validation_note"],
            None, None, None,  # gate_status, gate_reason, final_action — step 5, not built
            None, None, None, None, None,  # execution/outcome/stopping_reason — step 6, not built
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def get_trace(conn: sqlite3.Connection, transaction_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM decisions WHERE transaction_id = ?", (transaction_id,)).fetchone()
    if row is None:
        return None
    r = dict(row)
    for json_col in ("input_event_json", "predicted_probabilities_json", "candidates_ev_ranked_json",
                      "eligible_actions_json", "compliance_state_json"):
        r[json_col.removesuffix("_json")] = json.loads(r.pop(json_col))
    r["agent_valid"] = bool(r["agent_valid"])
    return r


def list_batch(conn: sqlite3.Connection, batch_id: str) -> list[dict]:
    rows = conn.execute("SELECT transaction_id FROM decisions WHERE batch_id = ?", (batch_id,)).fetchall()
    return [get_trace(conn, r["transaction_id"]) for r in rows]


def compute_metrics(conn: sqlite3.Connection, batch_id: Optional[str] = None) -> dict:
    """Dashboard-facing aggregate metrics (GET /metrics, brief §13). Fields
    that depend on the not-yet-built hard gate / execution / outcome
    simulators are reported as null with an explicit `pending` note rather
    than a fabricated number — see db.py's schema docstring."""
    where = "WHERE batch_id = ?" if batch_id else ""
    params = (batch_id,) if batch_id else ()

    total_row = conn.execute(
        f"SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS total_amount FROM decisions {where}", params
    ).fetchone()
    n_cases, revenue_at_risk = total_row["n"], total_row["total_amount"]

    # Computed in Python rather than SQL: predicted_probability of the
    # SELECTED action isn't always candidates[0] (the agent can deviate from
    # top-EV), so it has to be looked up by action name per row.
    recovery_by_intervention = {}
    for r in conn.execute(f"SELECT selected_action, predicted_probabilities_json, amount FROM decisions {where}", params):
        probs = json.loads(r["predicted_probabilities_json"])
        action = r["selected_action"]
        bucket = recovery_by_intervention.setdefault(
            action, {"n_selected": 0, "amount_at_risk": 0.0, "predicted_probabilities": []}
        )
        bucket["n_selected"] += 1
        bucket["amount_at_risk"] += r["amount"]
        bucket["predicted_probabilities"].append(probs.get(action, 0.0))
    for action, bucket in recovery_by_intervention.items():
        probs = bucket.pop("predicted_probabilities")
        bucket["mean_predicted_recovery_probability"] = round(sum(probs) / len(probs), 4) if probs else None
        bucket["amount_at_risk"] = round(bucket["amount_at_risk"], 2)

    provider_counts = conn.execute(
        f"SELECT agent_provider, COUNT(*) AS n FROM decisions {where} GROUP BY agent_provider", params
    ).fetchall()

    invalid_count = conn.execute(
        f"SELECT COUNT(*) AS n FROM decisions {where}{' AND' if where else 'WHERE'} agent_valid = 0", params
    ).fetchone()["n"]

    compliance_restricted = 0
    opted_out = 0
    contact_capped = 0
    for r in conn.execute(f"SELECT eligible_actions_json, compliance_state_json FROM decisions {where}", params):
        eligible = json.loads(r["eligible_actions_json"])
        state = json.loads(r["compliance_state_json"])
        if len(eligible) < 5:
            compliance_restricted += 1
        if state.get("opted_out"):
            opted_out += 1
        elif len(eligible) < 5:
            contact_capped += 1

    return {
        "batch_id": batch_id,
        "n_cases": n_cases,
        "revenue_at_risk": round(revenue_at_risk, 2),
        "revenue_recovered": None,
        "recovery_rate": None,
        "incremental_recovery_vs_baseline": None,
        "_pending_note": (
            "revenue_recovered / recovery_rate / incremental_recovery_vs_baseline are null: "
            "they require the outcome simulator (build-order step 6) and baseline experiment "
            "(step 8), neither implemented yet. What's shown below is everything real through "
            "the DECIDE stage: predicted recovery probabilities and EV-ranked action selection."
        ),
        "recovery_by_intervention": recovery_by_intervention,
        "compliance": {
            "cases_with_restricted_eligibility": compliance_restricted,
            "opted_out_respected": opted_out,
            "contact_cap_restricted": contact_capped,
            "quiet_hour_violations": None,   # hard gate not built (step 5)
            "target_compliance_violations": 0,
            "_note": "quiet_hour_violations is null pending the hard compliance gate (step 5); "
                     "opted_out_respected / contact_cap_restricted come from the eligibility "
                     "filter that already exists (policy/compliance.py) and are 0 real violations "
                     "by construction — the model is never even shown a disallowed action.",
        },
        "agent_provider_breakdown": {r["agent_provider"]: r["n"] for r in provider_counts},
        "agent_decisions_corrected_by_validator": invalid_count,
        "stopping_reason_breakdown": None,  # pending step 6
    }
