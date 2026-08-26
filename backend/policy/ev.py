"""
Expected-value computation — brief §7.

    Expected Value = P(recovery | customer, context, intervention)
                      x recoverable_amount
                      - intervention_cost
                      - discount_cost (if applicable)

Cost constants are the SAME ones already defined in generate_data.py
(INTERVENTION_COST, DISCOUNT_RATE) rather than a second, inconsistent set
invented here — these represent the assumed operational cost of sending
each action type (a retry link SMS/email costs a little, a human escalation
costs a lot, a discount costs a % of the order). The brief says this formula
"can be refined during implementation" — reusing the existing constants is
the least arbitrary choice available, not a claim that these are the "true"
real-world costs.
"""

from __future__ import annotations

from dataclasses import dataclass

INTERVENTION_COST = {
    "retry_link": 2,
    "reminder": 1,
    "discount_offer": 0,  # cost modeled separately, as a % of amount, below
    "escalate_to_human": 40,
    "no_action": 0,
}
DISCOUNT_RATE = 0.10  # 10% of amount, cost incurred only when discount_offer is used


@dataclass
class CandidateAction:
    action: str
    predicted_probability: float  # P(recovery | customer, context, action) from the ML model
    recoverable_amount: float
    intervention_cost: float
    discount_cost: float
    expected_value: float

    def as_dict(self) -> dict:
        # explicit float(...) casts: predicted_probability/expected_value are
        # often numpy.float64 (they flow from model.predict_proba()), which
        # json.dump can't serialize natively and would otherwise silently
        # stringify via a default=str fallback — round() alone doesn't fix
        # that, it stays numpy.float64.
        return {
            "action": self.action,
            "predicted_probability": round(float(self.predicted_probability), 4),
            "recoverable_amount": round(float(self.recoverable_amount), 2),
            "intervention_cost": round(float(self.intervention_cost), 2),
            "discount_cost": round(float(self.discount_cost), 2),
            "expected_value": round(float(self.expected_value), 2),
        }


def compute_expected_value(action: str, predicted_probability: float, recoverable_amount: float) -> CandidateAction:
    intervention_cost = INTERVENTION_COST[action]
    discount_cost = DISCOUNT_RATE * recoverable_amount if action == "discount_offer" else 0.0
    ev = predicted_probability * recoverable_amount - intervention_cost - discount_cost
    return CandidateAction(
        action=action,
        predicted_probability=predicted_probability,
        recoverable_amount=recoverable_amount,
        intervention_cost=intervention_cost,
        discount_cost=discount_cost,
        expected_value=ev,
    )


def rank_candidates(predicted_probabilities: dict[str, float], recoverable_amount: float,
                     eligible: list[str]) -> list[CandidateAction]:
    """
    Build the EV-ranked candidate list, restricted to `eligible` actions
    (the output of policy/compliance.py's eligible_actions()). Sorted
    highest expected value first — this ordering, not the LLM's unstructured
    judgment, is what "best" means per brief §7.
    """
    candidates = [
        compute_expected_value(action, predicted_probabilities[action], recoverable_amount)
        for action in eligible
    ]
    candidates.sort(key=lambda c: c.expected_value, reverse=True)
    return candidates
