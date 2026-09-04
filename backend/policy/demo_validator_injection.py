"""
Live demo script — validator injection (DEMO_SCRIPT.md, Case 4).

Uses a REAL DecisionInput (real model predictions, real EV ranking, real
compliance eligibility) for TXN000000 — the same transaction the "normal
case" demo beat already uses, so this reuses a familiar reference point
rather than introducing a new one — then feeds validate_decision() a
synthetic LLM response that invents an action outside the allowed list
("send_free_product", never a real intervention). Shows, live: the model
never trusts a raw LLM action choice at face value; an off-list response is
caught, corrected to the top-EV eligible candidate, and flagged (valid=False
+ a validation_note) rather than silently accepted or silently swallowed.

This exercises the EXACT function GeminiProvider.decide() and
GroqProvider.decide() call on real API responses (agent.py's
validate_decision()) — not a separate mock path. What you're watching here
is the real guardrail, staged with a synthetic input because provoking a
real LLM into inventing an action on command isn't reliable to do live.

Run from backend/:  .venv\\Scripts\\python.exe policy\\demo_validator_injection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import load_events  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import build_decision_input, load_model, predict_all_candidates  # noqa: E402
from agent import validate_decision  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEMO_TXN_ID = "TXN000000"

# The invented action — deliberately not `discount_offer` or any other real
# intervention name, so it's unambiguous that this is a fabricated action,
# not merely an ineligible-but-real one (that's a different failure mode,
# also caught by validate_decision() — see the other two cases the code
# handles, exercised in the original off-list correction test but not
# staged here to keep this one demo beat focused).
FAKE_LLM_RESPONSE = {
    "action": "send_free_product",
    "reason": "Sounds generous and should improve customer goodwill.",
}


def main():
    print("=" * 70)
    print("VALIDATOR INJECTION DEMO - synthetic off-list LLM response")
    print("=" * 70)

    events_df = load_events(str(DATA_DIR / "events.csv"))
    row = events_df.loc[events_df["transaction_id"] == DEMO_TXN_ID].iloc[0]
    model = load_model()

    predicted_probabilities = predict_all_candidates(model, row)
    decision_input = build_decision_input(row, predicted_probabilities)

    print(f"\nTransaction: {DEMO_TXN_ID}  (real model predictions, real EV ranking)")
    print(f"Allowed actions: {decision_input.allowed_actions()}")
    print("EV-ranked candidates:")
    for c in decision_input.candidates:
        print(f"    {c.action:<18} EV=Rs {c.expected_value:.2f}")

    print(f"\nStaged LLM response (NOT a real API call - this is the injected input):")
    print(f"    {FAKE_LLM_RESPONSE}")
    print(f"\n'{FAKE_LLM_RESPONSE['action']}' is not in the allowed list above - it's an invented action.")

    print("\nRunning agent.validate_decision() - the exact function every real")
    print("Gemini/Groq response is passed through before anything downstream sees it:")
    decision = validate_decision(FAKE_LLM_RESPONSE, decision_input, provider_name="gemini",
                                  raw_response=str(FAKE_LLM_RESPONSE))

    print(f"\n  action:           {decision.action!r}  (corrected to the top-EV eligible candidate)")
    print(f"  valid:            {decision.valid}")
    print(f"  validation_note:  {decision.validation_note}")

    assert decision.valid is False, "expected this injection to be caught"
    assert decision.action == decision_input.candidates[0].action, "expected correction to top-EV candidate"
    print("\nCaught, corrected, and flagged - confirmed programmatically, not just by eye.")


if __name__ == "__main__":
    main()
