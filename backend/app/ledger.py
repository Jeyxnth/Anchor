"""
Audit ledger read/write functions — the persistence half of build-order
step 7. Writes one row per (transaction, policy) decision into the
`decisions` table (schema in db.py); reads back a single full trace
(GET /audit/{id}), aggregate dashboard metrics (GET /metrics), or a
side-by-side policy comparison for the baseline experiment (build-order
step 8 — do_nothing / generic_reminder / ai_agent).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import ALL_INTERVENTIONS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy"))
from ev import rank_candidates  # noqa: E402
from compliance import is_quiet_hours  # noqa: E402


def record_decision_trace(conn: sqlite3.Connection, batch_id: str, policy_name: str,
                           row: pd.Series, trace: dict, outcome: dict) -> None:
    """Persist one decision + its simulated outcome. `row` is the source
    events.csv row; `trace` matches policy/pipeline.py's decide_for_row()
    shape (predicted_probabilities/candidates/eligible_actions/agent_decision
    keys — baseline policies build an equivalent dict without a real model,
    see batch.py); `outcome` is app/outcome.py's simulate_outcome() result.
    executed_action = trace's selected action, since there's no hard gate
    yet (step 5) to override it."""
    d = trace["agent_decision"]
    conn.execute(
        """
        INSERT OR REPLACE INTO decisions (
            transaction_id, policy_name, batch_id, customer_id,
            event_type, failure_reason, amount, input_event_json,
            predicted_probabilities_json,
            candidates_ev_ranked_json,
            eligible_actions_json, compliance_state_json,
            selected_action, agent_reason, agent_provider, agent_valid, agent_validation_note,
            gate_status, gate_reason, final_action,
            executed_action, outcome_recovered, outcome_recovered_amount,
            time_to_recovery_hours, stopping_reason,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trace["transaction_id"], policy_name, batch_id, str(row["customer_id"]),
            row["event_type"], row["failure_reason"], float(row["amount"]),
            json.dumps({**trace["customer_context"], **trace["transaction_context"]}, default=str),
            json.dumps(trace["predicted_probabilities_all_actions"]),
            json.dumps(trace["candidates_ev_ranked"]),
            json.dumps(trace["eligible_actions"]), json.dumps(trace["compliance_state"], default=str),
            d["action"], d["reason"], d["provider"], int(bool(d["valid"])), d["validation_note"],
            None, None, None,  # gate_status, gate_reason, final_action — step 5, not built
            d["action"],  # executed_action
            int(outcome["recovered"]), outcome["recovered_amount"],
            outcome["time_to_recovery_hours"], outcome["stopping_reason"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def get_trace(conn: sqlite3.Connection, transaction_id: str, policy_name: Optional[str] = None) -> Optional[dict]:
    """policy_name=None resolves to 'ai_agent' if present, else whichever
    policy happens to have a row for this transaction — the single-trace
    audit view is inherently about "the" decision for a case, and ai_agent
    is the one with real agent reasoning to show."""
    if policy_name is not None:
        row = conn.execute(
            "SELECT * FROM decisions WHERE transaction_id = ? AND policy_name = ?", (transaction_id, policy_name)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM decisions WHERE transaction_id = ? ORDER BY (policy_name = 'ai_agent') DESC LIMIT 1",
            (transaction_id,),
        ).fetchone()
    if row is None:
        return None
    r = dict(row)
    for json_col in ("input_event_json", "predicted_probabilities_json", "candidates_ev_ranked_json",
                      "eligible_actions_json", "compliance_state_json"):
        r[json_col.removesuffix("_json")] = json.loads(r.pop(json_col))
    r["agent_valid"] = bool(r["agent_valid"])
    r["outcome_recovered"] = bool(r["outcome_recovered"])

    # Unconstrained EV ranking — what would have been picked with NO compliance
    # filtering — derived at read time from data already stored (predicted_probabilities
    # covers all 5 actions regardless of eligibility; no new column needed). Only
    # meaningful for policies that actually consult the model (ai_agent); baseline
    # policies log an empty predicted_probabilities dict and get an empty list back.
    # Makes the compliance effect a visible before/after rather than implicit —
    # identical to candidates_ev_ranked when eligible_actions already covers all 5.
    if r["predicted_probabilities"]:
        unconstrained = rank_candidates(r["predicted_probabilities"], recoverable_amount=r["amount"],
                                         eligible=list(ALL_INTERVENTIONS))
        r["candidates_ev_ranked_unconstrained"] = [c.as_dict() for c in unconstrained]
        r["unconstrained_top_action"] = unconstrained[0].action
        r["compliance_changed_top_action"] = unconstrained[0].action != r["selected_action"]
    else:
        r["candidates_ev_ranked_unconstrained"] = []
        r["unconstrained_top_action"] = None
        r["compliance_changed_top_action"] = None

    return r


def list_batch(conn: sqlite3.Connection, batch_id: str, policy_name: Optional[str] = None) -> list[dict]:
    if policy_name:
        rows = conn.execute(
            "SELECT transaction_id, policy_name FROM decisions WHERE batch_id = ? AND policy_name = ?",
            (batch_id, policy_name),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT transaction_id, policy_name FROM decisions WHERE batch_id = ?", (batch_id,)
        ).fetchall()
    return [get_trace(conn, r["transaction_id"], r["policy_name"]) for r in rows]


def _where_clause(batch_id: Optional[str], policy_name: Optional[str]) -> tuple[str, tuple]:
    clauses, params = [], []
    if batch_id:
        clauses.append("batch_id = ?")
        params.append(batch_id)
    if policy_name:
        clauses.append("policy_name = ?")
        params.append(policy_name)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, tuple(params)


def compute_metrics(conn: sqlite3.Connection, batch_id: Optional[str] = None,
                     policy_name: Optional[str] = None) -> dict:
    """Dashboard-facing aggregate metrics (GET /metrics, brief §13) for one
    policy (default: whatever's in the ledger if policy_name is omitted —
    pass policy_name explicitly once multiple policies share a batch_id, or
    use compare_policies() for the side-by-side view). revenue_recovered /
    recovery_rate are real now (app/outcome.py); incremental_recovery is
    left to compare_policies(), which has the other policies' numbers to
    diff against. quiet_hour_violations still depends on the hard gate
    (step 5, not built) and stays null."""
    where, params = _where_clause(batch_id, policy_name)

    policy_warning = None
    if not policy_name:
        distinct_policies = [r["policy_name"] for r in conn.execute(
            f"SELECT DISTINCT policy_name FROM decisions {where}", params
        )]
        if len(distinct_policies) > 1:
            policy_warning = (
                f"No policy_name filter given and this selection spans {len(distinct_policies)} "
                f"policies {distinct_policies} — every number below sums/averages across ALL of "
                f"them (e.g. the same transaction counted once per policy), which is not a "
                f"meaningful business metric. Pass policy_name explicitly (usually 'ai_agent' for "
                f"the live dashboard), or use GET /batch/{{batch_id}}/compare for a correct "
                f"side-by-side view."
            )

    total_row = conn.execute(
        f"""SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS total_amount,
                   COALESCE(SUM(outcome_recovered_amount), 0) AS total_recovered,
                   COALESCE(SUM(outcome_recovered), 0) AS n_recovered
            FROM decisions {where}""",
        params,
    ).fetchone()
    n_cases = total_row["n"]
    revenue_at_risk = total_row["total_amount"]
    revenue_recovered = total_row["total_recovered"]
    # Two genuinely different ratios, deliberately both exposed and labeled —
    # see the "recovery_rate_definitions" note below. They will NOT match
    # numerically (e.g. 49.65% vs 50.11% on the full batch): that's not a
    # bug, it's because recovered transactions and their amounts aren't
    # uniformly distributed relative to unrecovered ones.
    recovery_rate = (total_row["n_recovered"] / n_cases) if n_cases else None                    # count-based
    revenue_weighted_recovery_rate = (revenue_recovered / revenue_at_risk) if revenue_at_risk else None  # rupee-weighted

    # Computed in Python rather than SQL: predicted_probability of the
    # SELECTED action isn't always candidates[0] (the agent can deviate from
    # top-EV), so it has to be looked up by action name per row.
    recovery_by_intervention = {}
    for r in conn.execute(
        f"SELECT selected_action, predicted_probabilities_json, amount, outcome_recovered, "
        f"outcome_recovered_amount FROM decisions {where}", params
    ):
        probs = json.loads(r["predicted_probabilities_json"])
        action = r["selected_action"]
        bucket = recovery_by_intervention.setdefault(
            action, {"n_selected": 0, "amount_at_risk": 0.0, "amount_recovered": 0.0,
                     "n_recovered": 0, "predicted_probabilities": []}
        )
        bucket["n_selected"] += 1
        bucket["amount_at_risk"] += r["amount"]
        bucket["amount_recovered"] += r["outcome_recovered_amount"]
        bucket["n_recovered"] += r["outcome_recovered"]
        bucket["predicted_probabilities"].append(probs.get(action, 0.0))
    for action, bucket in recovery_by_intervention.items():
        probs = bucket.pop("predicted_probabilities")
        bucket["mean_predicted_recovery_probability"] = round(sum(probs) / len(probs), 4) if probs else None
        bucket["amount_at_risk"] = round(bucket["amount_at_risk"], 2)
        bucket["amount_recovered"] = round(bucket["amount_recovered"], 2)
        bucket["recovery_rate"] = round(bucket["n_recovered"] / bucket["n_selected"], 4) if bucket["n_selected"] else None

    provider_counts = conn.execute(
        f"SELECT agent_provider, COUNT(*) AS n FROM decisions {where} GROUP BY agent_provider", params
    ).fetchall()

    invalid_where = where + (" AND" if where else "WHERE") + " agent_valid = 0"
    invalid_count = conn.execute(f"SELECT COUNT(*) AS n FROM decisions {invalid_where}", params).fetchone()["n"]

    compliance_restricted = 0
    opted_out = 0
    contact_capped = 0
    quiet_hour_restricted = 0
    for r in conn.execute(
        f"SELECT eligible_actions_json, compliance_state_json, input_event_json FROM decisions {where}", params
    ):
        eligible = json.loads(r["eligible_actions_json"])
        state = json.loads(r["compliance_state_json"])
        if len(eligible) < 5:
            compliance_restricted += 1
        if state.get("opted_out"):
            opted_out += 1
        else:
            # Mutually exclusive with contact_capped below: quiet hours
            # (policy/compliance.py's is_quiet_hours(), checked ahead of
            # the contact-cap rule there) forces no_action-only regardless
            # of contact-cap state, so a case that's BOTH quiet-hours and
            # contact-capped is attributed to quiet hours here — that's
            # actually why its eligible set narrowed all the way down,
            # not the contact cap alone. time_of_day isn't its own column
            # (no schema change needed) — it's already in input_event_json
            # (transaction_context), stored for every row since batch.py's
            # first version.
            #
            # Checked against the ACTUAL stored eligible set (== ["no_action"]),
            # not just "was it quiet hours", because a first-attempt
            # payment_failed case is exempt from quiet-hours (see
            # compliance.is_first_attempt_payment_failure()) — its eligible
            # set won't have collapsed even though time_of_day says quiet
            # hours. Deriving from eligible_actions_json directly avoids
            # duplicating that exemption's logic here and risking drift.
            time_of_day = json.loads(r["input_event_json"]).get("time_of_day")
            quiet_hour_active = time_of_day is not None and is_quiet_hours(int(time_of_day))
            if quiet_hour_active and eligible == ["no_action"]:
                quiet_hour_restricted += 1
            elif len(eligible) < 5:
                contact_capped += 1

    stopping_reasons = conn.execute(
        f"SELECT stopping_reason, COUNT(*) AS n FROM decisions {where} GROUP BY stopping_reason", params
    ).fetchall()

    return {
        "batch_id": batch_id,
        "policy_name": policy_name,
        "_policy_warning": policy_warning,
        "n_cases": n_cases,
        "revenue_at_risk": round(revenue_at_risk, 2),
        "revenue_recovered": round(revenue_recovered, 2) if n_cases else None,
        "recovery_rate": round(recovery_rate, 4) if recovery_rate is not None else None,
        "revenue_weighted_recovery_rate": (
            round(revenue_weighted_recovery_rate, 4) if revenue_weighted_recovery_rate is not None else None
        ),
        "recovery_rate_definitions": {
            "recovery_rate": "count-based: recovered transactions / total transactions.",
            "revenue_weighted_recovery_rate": "rupee-based: revenue_recovered / revenue_at_risk.",
            "_note": "These will NOT numerically match (recovered transactions skew toward "
                     "different amounts than unrecovered ones) — that's expected, not an "
                     "inconsistency. 'Recovery Rate' in the dashboard is the count-based figure "
                     "unless labeled otherwise.",
        },
        "incremental_recovery_vs_baseline": None,
        "_pending_note": (
            "incremental_recovery_vs_baseline is null here: it needs another policy's numbers to "
            "diff against, which single-policy compute_metrics() doesn't have — use GET "
            "/batch/{batch_id}/compare for the side-by-side baseline comparison (build-order step 8)."
        ),
        "recovery_by_intervention": recovery_by_intervention,
        "compliance": {
            "cases_with_restricted_eligibility": compliance_restricted,
            "opted_out_respected": opted_out,
            "contact_cap_restricted": contact_capped,
            "quiet_hour_restricted": quiet_hour_restricted,
            "target_compliance_violations": 0,
            "_note": "opted_out_respected / contact_cap_restricted / quiet_hour_restricted all "
                     "come from the eligibility filter that already exists (policy/compliance.py) "
                     "and are 0 real violations by construction — the model is never even shown a "
                     "disallowed action. quiet_hour_restricted counts cases where time_of_day fell "
                     "outside 9am-8pm and collapsed eligibility to no_action-only; a case that's "
                     "both quiet-hours and contact-capped counts here, not in contact_cap_restricted, "
                     "since quiet hours is why it narrowed all the way to no_action.",
        },
        "agent_provider_breakdown": {r["agent_provider"]: r["n"] for r in provider_counts},
        "agent_decisions_corrected_by_validator": invalid_count,
        "stopping_reason_breakdown": {r["stopping_reason"]: r["n"] for r in stopping_reasons},
    }


def compare_policies(conn: sqlite3.Connection, batch_id: str) -> dict:
    """Side-by-side baseline comparison (build-order step 8) across every
    policy_name logged under one batch_id. Requires all policies to have run
    over the SAME transaction set to be a fair comparison — this checks that
    and reports it rather than silently comparing mismatched populations."""
    policy_rows = conn.execute(
        "SELECT DISTINCT policy_name FROM decisions WHERE batch_id = ?", (batch_id,)
    ).fetchall()
    policy_names = [r["policy_name"] for r in policy_rows]
    if not policy_names:
        return {"batch_id": batch_id, "error": f"no decisions found for batch_id={batch_id!r}"}

    txn_sets = {
        p: {r["transaction_id"] for r in conn.execute(
            "SELECT transaction_id FROM decisions WHERE batch_id = ? AND policy_name = ?", (batch_id, p)
        )}
        for p in policy_names
    }
    same_population = len(set(frozenset(s) for s in txn_sets.values())) == 1

    per_policy = {p: compute_metrics(conn, batch_id=batch_id, policy_name=p) for p in policy_names}
    agent_recovered = per_policy.get("ai_agent", {}).get("revenue_recovered")

    def incremental_vs_agent(baseline_name: str, baseline_recovered) -> Optional[float]:
        if baseline_name == "ai_agent" or agent_recovered is None or baseline_recovered is None:
            return None
        return round(agent_recovered - baseline_recovered, 2)

    results = {
        p: {
            "revenue_recovered": m["revenue_recovered"],
            "recovery_rate": m["recovery_rate"],                                   # count-based — see recovery_rate_definitions
            "revenue_weighted_recovery_rate": m["revenue_weighted_recovery_rate"],  # rupee-based
            "incremental_recovery_vs_ai_agent": incremental_vs_agent(p, m["revenue_recovered"]),
        }
        for p, m in per_policy.items()
    }
    incremental_summary = {
        p: r["incremental_recovery_vs_ai_agent"] for p, r in results.items() if r["incremental_recovery_vs_ai_agent"] is not None
    }

    return {
        "batch_id": batch_id,
        "policies_compared": policy_names,
        "same_transaction_population_across_policies": same_population,
        "n_cases_per_policy": {p: len(s) for p, s in txn_sets.items()},
        "revenue_at_risk": next(iter(per_policy.values()))["revenue_at_risk"] if per_policy else None,
        "recovery_rate_definitions": next(iter(per_policy.values()))["recovery_rate_definitions"] if per_policy else None,
        "results": results,
        "incremental_recovery_of_ai_agent_vs_each_baseline": incremental_summary or None,
    }
