"""
End-to-end DECIDE-stage pipeline for a single case — wires together:

  ML model (per-candidate recovery probability)
    -> compliance-based candidate filter (policy/compliance.py)
    -> expected-value ranking (policy/ev.py)
    -> LLM decision agent (policy/agent.py)

Returns a full trace dict — every number that fed the decision, not just the
final action — which is the shape the audit ledger (build-order step 7,
not implemented yet) will persist.

Scope note: this covers Detect -> Predict -> Decide only. It does NOT yet
apply the hard compliance gate with override authority (step 5) or
execute/simulate the outcome (steps 6-7) — those come next.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import (  # noqa: E402
    ALL_INTERVENTIONS,
    FEATURE_COLUMNS_WITH_INTERACTIONS,
    rows_for_all_interventions,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compliance import eligible_actions, state_from_event_row  # noqa: E402
from ev import rank_candidates  # noqa: E402
from agent import DecisionInput, get_provider  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "ml" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_recovery_model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_all_candidates(model_pipeline, row: pd.Series) -> dict:
    """P(recovery | customer, context, action) for every action in
    ALL_INTERVENTIONS — unconstrained by compliance eligibility. Eligibility
    filtering happens afterwards, in build_decision_input()."""
    variants = rows_for_all_interventions(row, with_interactions=True)[FEATURE_COLUMNS_WITH_INTERACTIONS]
    probs = model_pipeline.predict_proba(variants)[:, 1]
    return dict(zip(ALL_INTERVENTIONS, probs))


def build_decision_input(row: pd.Series, predicted_probabilities: dict) -> DecisionInput:
    state = state_from_event_row(row)
    eligible = eligible_actions(state)
    candidates = rank_candidates(
        predicted_probabilities, recoverable_amount=float(row["amount"]), eligible=eligible
    )

    customer_context = {
        "customer_age_days": int(row["customer_age_days"]),
        "lifetime_value": float(row["lifetime_value"]),
        "previous_success_rate": float(row["previous_success_rate"]),
        "average_order_value": float(row["average_order_value"]),
        "previous_recovery_successes": int(row["previous_recovery_successes"]),
        "days_since_last_purchase": int(row["days_since_last_purchase"]),
    }
    transaction_context = {
        "amount": float(row["amount"]),
        "event_type": row["event_type"],
        "failure_reason": row["failure_reason"],
        "time_of_day": int(row["time_of_day"]),
        "day_of_week": int(row["day_of_week"]),
    }
    last_contact = row.get("last_contact_time")
    compliance_state = {
        "opted_out": bool(state.opted_out),
        "attempts_so_far_24h": state.attempts_so_far,
        "last_contact_time": None if pd.isna(last_contact) else last_contact,
    }

    return DecisionInput(
        transaction_id=row["transaction_id"],
        customer_context=customer_context,
        transaction_context=transaction_context,
        compliance_state=compliance_state,
        candidates=candidates,
    )


def decide_for_row(model_pipeline, row: pd.Series, provider=None) -> dict:
    predicted_probabilities = predict_all_candidates(model_pipeline, row)
    decision_input = build_decision_input(row, predicted_probabilities)
    provider = provider or get_provider()
    decision = provider.decide(decision_input)

    return {
        "transaction_id": decision_input.transaction_id,
        "customer_context": decision_input.customer_context,
        "transaction_context": decision_input.transaction_context,
        "compliance_state": decision_input.compliance_state,
        "predicted_probabilities_all_actions": {k: round(float(v), 4) for k, v in predicted_probabilities.items()},
        "eligible_actions": decision_input.allowed_actions(),
        "candidates_ev_ranked": [c.as_dict() for c in decision_input.candidates],
        "agent_decision": {
            "action": decision.action,
            "reason": decision.reason,
            "provider": decision.provider,
            "valid": decision.valid,
            "validation_note": decision.validation_note,
        },
    }
