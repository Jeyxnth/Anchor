"""
Candidate-eligibility filter — the FIRST of two compliance touchpoints
described in the brief (§7, §9). This one runs BEFORE the EV ranking and
the LLM, and exists to bound the search space: it decides which actions are
even offered as candidates for a given case.

This is NOT the full "hard compliance gate" (brief §9 / build-order step 5).
That gate runs AFTER the LLM has picked an action from the eligible set
built here, has final override authority, and additionally enforces quiet
hours, forced escalation after M total failed attempts, and stop-on-recovery
— none of which are implemented yet. Building that gate, and the deliberate
override demo it enables, is still a separate step. What's here only
answers "what's even allowed to be considered."

The rule set mirrors generate_data.py's own compliance-state logic exactly
(opted_out -> no_action only; >=2 contacts in the trailing 24h window ->
escalate_to_human/no_action only), deliberately re-derived here from raw
state rather than trusting a pre-computed `allowed_interventions` column —
that column only exists because this dataset is synthetic; a live system
has to compute this itself from actual contact history. `verify_against_dataset()`
below is a regression check that this independent re-derivation matches the
dataset's column exactly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import ALL_INTERVENTIONS  # noqa: E402

MAX_CONTACTS_PER_24H = 2  # matches generate_data.py: attempts_so_far >= 2 -> restricted


@dataclass
class ComplianceState:
    opted_out: bool
    attempts_so_far: int  # contacts already made to this customer in the trailing 24h window


def eligible_actions(state: ComplianceState) -> list[str]:
    """
    Returns the bounded candidate list for this case. Order is not
    meaningful — EV ranking (policy/ev.py) decides priority among these.
    """
    if state.opted_out:
        return ["no_action"]
    if state.attempts_so_far >= MAX_CONTACTS_PER_24H:
        return ["escalate_to_human", "no_action"]
    return list(ALL_INTERVENTIONS)


def compute_attempts_in_window(contact_timestamps: list[datetime], now: datetime,
                                window_hours: int = 24) -> int:
    """
    Forward-looking helper for when this runs against a real contact-history
    store instead of a pre-computed events.csv row: counts prior contacts
    within the trailing `window_hours` of `now`. Not exercised by the
    synthetic pipeline today (events.csv already carries attempts_so_far),
    but the compliance gate (step 5) and the live API will need this.
    """
    window_start = now - timedelta(hours=window_hours)
    return sum(1 for t in contact_timestamps if t > window_start)


def state_from_event_row(row) -> ComplianceState:
    """Build a ComplianceState from an events.csv row (or equivalent dict/Series)."""
    opted_out = row["opted_out"]
    if isinstance(opted_out, str):
        opted_out = opted_out.strip().lower() == "true"
    return ComplianceState(opted_out=bool(opted_out), attempts_so_far=int(row["attempts_so_far"]))


def verify_against_dataset(events_df) -> dict:
    """
    Sanity check: for every row in events_df, does eligible_actions() computed
    independently from (opted_out, attempts_so_far) match the dataset's own
    `allowed_interventions` column exactly? Returns a small summary dict;
    raises if there's a real mismatch (would mean our rule re-derivation is
    wrong, not that the synthetic data is wrong).
    """
    mismatches = []
    for idx, row in events_df.iterrows():
        state = state_from_event_row(row)
        computed = set(eligible_actions(state))
        expected = set(row["allowed_interventions"].split(";"))
        if computed != expected:
            mismatches.append({
                "transaction_id": row["transaction_id"],
                "computed": sorted(computed),
                "expected": sorted(expected),
            })

    if mismatches:
        raise AssertionError(
            f"{len(mismatches)} rows disagree with dataset's allowed_interventions "
            f"column — compliance.py's rules don't match generate_data.py. "
            f"First mismatch: {mismatches[0]}"
        )

    return {"n_checked": len(events_df), "n_mismatches": 0, "status": "compliance.py rules verified identical to dataset"}
