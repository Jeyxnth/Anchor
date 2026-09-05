"""
Candidate-eligibility filter — the FIRST of two compliance touchpoints
described in the brief (§7, §9). This one runs BEFORE the EV ranking and
the LLM, and exists to bound the search space: it decides which actions are
even offered as candidates for a given case.

This is NOT the full "hard compliance gate" (brief §9 / build-order step 5).
That gate runs AFTER the LLM has picked an action from the eligible set
built here, has final override authority, and additionally enforces forced
escalation after M total failed attempts and stop-on-recovery — neither of
which is implemented yet. Building that gate, and the deliberate override
demo it enables, is still a separate step. What's here only answers "what's
even allowed to be considered."

Quiet hours (9am-8pm, i.e. time_of_day 9-20 inclusive) ARE enforced here as
of 2026-09-05 — outside that window, eligible actions collapse to
`["no_action"]`, same treatment as opted_out, checked independently of and
in addition to the opt-out/contact-cap rules below (see eligible_actions()).
This is quick to add here specifically because events.csv already carries
`time_of_day` per event — no new data or live clock needed for the
synthetic-batch pipeline.

Exception, also added 2026-09-05 (same day): a first-attempt
payment_failed event (event_type == "payment_failed" and
attempts_so_far == 0) is exempt from the quiet-hours collapse — it's an
automated response to a session the customer is actively in (their card
was just declined), not outbound contact we initiated cold. Opt-out and
contact-cap still apply to it unchanged; only the quiet-hours restriction
is skipped. See is_first_attempt_payment_failure().

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
QUIET_HOURS_START = 9   # 9am — first hour contact IS allowed
QUIET_HOURS_END = 20    # 8pm — last hour contact IS allowed (i.e. allowed window is [9, 20] inclusive)


@dataclass
class ComplianceState:
    opted_out: bool
    attempts_so_far: int  # contacts already made to this customer in the trailing 24h window
    time_of_day: int  # hour of day the event occurred, 0-23 (events.csv's own `time_of_day` column)
    event_type: str = ""  # events.csv's own `event_type` column ("payment_failed" / "checkout_abandoned")


def is_quiet_hours(time_of_day: int) -> bool:
    return not (QUIET_HOURS_START <= time_of_day <= QUIET_HOURS_END)


def is_first_attempt_payment_failure(state: ComplianceState) -> bool:
    """
    The very first automated response to a payment failure is exempt from
    quiet-hours: it's a reaction to an active session the customer just had
    (their card was just declined), not cold outreach initiated by us. Only
    applies when this is the first contact attempt (attempts_so_far == 0) —
    any follow-up (attempts_so_far >= 1) is no longer "responding to the
    live moment" and reverts to normal quiet-hours treatment. Added
    2026-09-05, same day as quiet-hours itself.
    """
    return state.event_type == "payment_failed" and state.attempts_so_far == 0


def eligible_actions(state: ComplianceState) -> list[str]:
    """
    Returns the bounded candidate list for this case. Order is not
    meaningful — EV ranking (policy/ev.py) decides priority among these.

    Opt-out and quiet-hours are both independent, unconditional collapses
    to no_action-only — checked ahead of the contact-cap rule so quiet
    hours overrides even the escalate_to_human allowance that a contact-
    capped-but-daytime case would otherwise get. This is layered ON TOP of
    the existing opt-out/contact-cap checks, not a replacement for either.

    Exception: a first-attempt payment_failed case (see
    is_first_attempt_payment_failure()) is exempt from the quiet-hours
    collapse specifically — opt-out and contact-cap still apply to it
    exactly as to any other case.
    """
    if state.opted_out:
        return ["no_action"]
    if is_quiet_hours(state.time_of_day) and not is_first_attempt_payment_failure(state):
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
    return ComplianceState(
        opted_out=bool(opted_out),
        attempts_so_far=int(row["attempts_so_far"]),
        time_of_day=int(row["time_of_day"]),
        event_type=str(row["event_type"]),
    )


def verify_against_dataset(events_df) -> dict:
    """
    Sanity check against the dataset's own `allowed_interventions` column —
    but that column is generate_data.py's OPT-OUT/CONTACT-CAP logic only;
    it was computed before quiet-hours existed as a rule anywhere in this
    project, so it was never meant to reflect quiet-hours restriction and
    can't be used to verify that rule (see the module docstring and
    STATUS.md for why quiet hours is a live-system rule layered on top,
    not a retroactive relabeling of the static dataset).

    So this check compares against the dataset column using ONLY the
    opt-out/contact-cap portion of eligible_actions() (time_of_day fixed
    to a guaranteed-non-quiet-hour value, 12 = noon, so the quiet-hours
    branch can never fire here) — still a real, meaningful regression
    check that the pre-existing rules haven't drifted — plus a SEPARATE,
    self-contained check that the quiet-hours rule itself does what it
    claims directly against events.csv's own `time_of_day` column (no
    dataset column to compare against, since none carries this rule).

    The first-attempt-payment_failed exemption (see
    is_first_attempt_payment_failure()) means "is_quiet_hours(time_of_day)"
    no longer implies "must collapse to no_action" on its own — rows in
    that exempted subset are checked against the OPPOSITE expectation
    (full eligible set, contact-cap permitting) instead.
    """
    mismatches = []
    quiet_hour_mismatches = []
    n_quiet_hour_rows = 0
    n_quiet_hour_exempted = 0
    for idx, row in events_df.iterrows():
        opted_out = row["opted_out"]
        if isinstance(opted_out, str):
            opted_out = opted_out.strip().lower() == "true"
        pre_quiet_hours_state = ComplianceState(
            opted_out=bool(opted_out), attempts_so_far=int(row["attempts_so_far"]), time_of_day=12,
        )
        computed = set(eligible_actions(pre_quiet_hours_state))
        expected = set(row["allowed_interventions"].split(";"))
        if computed != expected:
            mismatches.append({
                "transaction_id": row["transaction_id"],
                "computed": sorted(computed),
                "expected": sorted(expected),
            })

        if is_quiet_hours(int(row["time_of_day"])):
            n_quiet_hour_rows += 1
            live_state = state_from_event_row(row)
            live_eligible = set(eligible_actions(live_state))
            if not opted_out:
                if is_first_attempt_payment_failure(live_state):
                    n_quiet_hour_exempted += 1
                    # exempted: quiet-hours must NOT have restricted this row —
                    # it should equal whatever eligible_actions() would give at
                    # a neutral (non-quiet-hour) time, i.e. the contact-cap-only
                    # outcome, which at attempts_so_far==0 is the full set.
                    non_quiet_state = ComplianceState(
                        opted_out=opted_out, attempts_so_far=live_state.attempts_so_far,
                        time_of_day=12, event_type=live_state.event_type,
                    )
                    if live_eligible != set(eligible_actions(non_quiet_state)):
                        quiet_hour_mismatches.append((row["transaction_id"], "exempted-but-restricted"))
                elif live_eligible != {"no_action"}:
                    quiet_hour_mismatches.append((row["transaction_id"], "not-exempt-but-not-restricted"))

    if mismatches:
        raise AssertionError(
            f"{len(mismatches)} rows disagree with dataset's allowed_interventions "
            f"column (opt-out/contact-cap rules) — compliance.py's rules don't match "
            f"generate_data.py. First mismatch: {mismatches[0]}"
        )
    if quiet_hour_mismatches:
        raise AssertionError(
            f"{len(quiet_hour_mismatches)} quiet-hour rows did not match expected "
            f"treatment (restricted vs. exempted). First: {quiet_hour_mismatches[0]}"
        )

    return {
        "n_checked": len(events_df),
        "n_mismatches": 0,
        "n_quiet_hour_rows": n_quiet_hour_rows,
        "n_quiet_hour_exempted": n_quiet_hour_exempted,
        "n_quiet_hour_restricted": n_quiet_hour_rows - n_quiet_hour_exempted,
        "status": "compliance.py rules verified identical to dataset (opt-out/contact-cap) "
                  "and quiet-hours self-consistency (incl. first-attempt payment_failed "
                  "exemption) verified separately (no dataset column exists for it, by "
                  "design — see docstring above)",
    }
