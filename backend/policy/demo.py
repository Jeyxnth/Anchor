"""
Runnable demonstration of the EV policy layer (step 3 of the build order)
on real rows from events.csv. Not the compliance GATE demo (build-order
step 10) — that needs the hard gate (step 5), which doesn't exist yet, and
shows the agent's recommendation being overridden. This demo shows the
narrower thing that exists today: the candidate list ITSELF being bounded
before the agent ever runs, for an opted-out customer and a contact-capped
customer, plus a normal unrestricted case for contrast.

Run from backend/:  .venv\\Scripts\\python.exe policy\\demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import load_events  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compliance import verify_against_dataset  # noqa: E402
from pipeline import decide_for_row, load_model  # noqa: E402
from agent import get_provider  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Hand-picked to show the three eligibility regimes:
DEMO_TXN_IDS = [
    "TXN000000",  # normal case: opted_out=False, attempts_so_far=0 -> all 5 actions eligible
    "TXN000025",  # opted_out=True -> only no_action eligible
    "TXN002166",  # attempts_so_far=2 (>= cap) -> only escalate_to_human/no_action eligible
]


def main():
    events_df = load_events(str(DATA_DIR / "events.csv"))

    print("=" * 70)
    print("Compliance-filter regression check (independently-derived rules")
    print("vs. the dataset's own allowed_interventions column, all 4000 rows)")
    print("=" * 70)
    check = verify_against_dataset(events_df)
    print(check)

    print(f"\n{'=' * 70}\nLoading trained model: ml/artifacts/xgb_recovery_model.joblib\n{'=' * 70}")
    model = load_model()

    # Provider is env-driven and auto-detected (Gemini > Groq > rule-based
    # fallback — see agent.get_provider()). Wrapped in ResilientProvider by
    # default: if the real LLM call fails (network hiccup, rate limit,
    # malformed response), it degrades to RuleBasedFallbackProvider for that
    # one decision rather than crashing the demo, and the returned
    # AgentDecision.provider still honestly reports which path actually ran.
    provider = get_provider()
    underlying = getattr(provider, "primary", provider)
    print(f"LLM provider in use: {underlying.name}  "
          f"(resilient={'yes, falls back to rule_based_fallback on API failure' if underlying is not provider else 'no (rule_based_fallback has nothing to fall back to)'})")

    for txn_id in DEMO_TXN_IDS:
        row = events_df.loc[events_df["transaction_id"] == txn_id].iloc[0]
        trace = decide_for_row(model, row, provider=provider)

        print(f"\n{'-' * 70}\n{txn_id}  (customer {row['customer_id']}, "
              f"{row['event_type']}/{row['failure_reason']}, amount Rs {row['amount']:.2f})")
        print(f"Compliance state : {trace['compliance_state']}")
        print(f"Eligible actions : {trace['eligible_actions']}")
        print("Candidates (EV-ranked):")
        for c in trace["candidates_ev_ranked"]:
            print(f"    {c['action']:<18} P(recover)={c['predicted_probability']:.1%}  "
                  f"EV=Rs {c['expected_value']:.2f}  "
                  f"(cost=Rs {c['intervention_cost']:.2f} + discount=Rs {c['discount_cost']:.2f})")
        d = trace["agent_decision"]
        print(f"Agent decision   : action={d['action']!r}  provider={d['provider']!r}  valid={d['valid']}")
        print(f"Reason           : {d['reason']}")
        if d["validation_note"]:
            print(f"Validation note  : {d['validation_note']}")

    # Save one full trace as a worked example for the README / audit-shape reference
    example_row = events_df.loc[events_df["transaction_id"] == "TXN000000"].iloc[0]
    example_trace = decide_for_row(model, example_row, provider=provider)
    out_path = Path(__file__).resolve().parent / "example_decision_trace.json"
    with open(out_path, "w") as f:
        json.dump(example_trace, f, indent=2, default=str)
    print(f"\nSaved full example trace: {out_path}")


if __name__ == "__main__":
    main()
