"""
Synthetic data generator — AI Revenue Recovery Agent

Generates events (payment_failed / checkout_abandoned) tied to a pool of
customers with persistent, evolving history. For every event, an intervention
is chosen (simulating a historical randomized policy, for good training
coverage across actions), and an outcome is sampled from a hidden,
noisy, intervention-conditional recovery-probability function.

Two output files:
  - events.csv       : everything a real system would observe. Use this to
                        train/evaluate your model and build the pipeline.
  - ground_truth.csv : the TRUE recovery probability under every candidate
                        intervention, per event. This does NOT exist in the
                        real world — it's for offline evaluation only
                        (optimality-gap metric, baseline simulation). Never
                        feed this into model training; that would be leakage.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_CUSTOMERS = 1500
N_EVENTS = 4000
SIM_DAYS = 60
START_DATE = datetime(2026, 6, 1)

FAILURE_REASONS = ["insufficient_funds", "otp_timeout", "gateway_error", "bank_decline", "price_hesitation"]
EVENT_TYPES = ["payment_failed", "checkout_abandoned"]
INTERVENTIONS = ["retry_link", "reminder", "discount_offer", "escalate_to_human", "no_action"]

INTERVENTION_COST = {
    "retry_link": 2,
    "reminder": 1,
    "discount_offer": 0,   # cost modeled separately as % of amount below
    "escalate_to_human": 40,
    "no_action": 0,
}
DISCOUNT_RATE = 0.10  # 10% of amount, cost when discount_offer is used


# ---------------------------------------------------------------------------
# 1. Customer pool — baseline attributes that persist across events
# ---------------------------------------------------------------------------

def generate_customers(n):
    customer_ids = [f"CUST{i:05d}" for i in range(n)]
    customer_age_days = RNG.integers(1, 1500, size=n)
    lifetime_value = RNG.lognormal(mean=8.5, sigma=1.1, size=n)  # roughly hundreds to lakhs
    average_order_value = np.clip(RNG.lognormal(mean=5.8, sigma=0.6, size=n), 100, 20000)
    # underlying "quality" latent trait correlated with several fields, for realistic overlap
    quality = RNG.normal(0, 1, size=n)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_age_days": customer_age_days,
        "lifetime_value": lifetime_value.round(2),
        "average_order_value": average_order_value.round(2),
        "_quality": quality,  # internal only, used to seed dynamic history below
    })
    return df


# ---------------------------------------------------------------------------
# 2. Event schedule — assign events to customers across the simulation window
# ---------------------------------------------------------------------------

def generate_event_schedule(customers, n_events):
    # sample customers with replacement, weighted so higher-quality customers
    # appear a bit less often (fewer failures) — adds realistic imbalance
    weights = np.exp(-0.3 * customers["_quality"].values)
    weights = weights / weights.sum()
    chosen_idx = RNG.choice(len(customers), size=n_events, p=weights, replace=True)

    offsets_days = RNG.uniform(0, SIM_DAYS, size=n_events)
    timestamps = [START_DATE + timedelta(days=float(d)) for d in offsets_days]

    schedule = customers.iloc[chosen_idx].reset_index(drop=True).copy()
    schedule["timestamp"] = timestamps
    schedule = schedule.sort_values("timestamp").reset_index(drop=True)
    return schedule


# ---------------------------------------------------------------------------
# 3. Walk through chronologically, building dynamic per-customer history
# ---------------------------------------------------------------------------

def build_events(schedule):
    history = {}  # customer_id -> dict of running state
    rows = []

    for i, row in schedule.iterrows():
        cid = row["customer_id"]
        ts = row["timestamp"]
        quality = row["_quality"]

        h = history.setdefault(cid, {
            "previous_payment_failures": 0,
            "previous_successes": 0,
            "previous_recovery_successes": 0,
            "last_purchase_date": ts - timedelta(days=int(RNG.integers(5, 120))),
            "last_contact_time": None,
            "attempts_in_window": [],  # list of contact timestamps
            "opted_out": False,
        })

        # ---- transaction-level fields ----
        event_type = RNG.choice(EVENT_TYPES, p=[0.6, 0.4])
        # failure reason correlated loosely with customer quality
        if quality > 0.5:
            reason_p = [0.10, 0.30, 0.30, 0.10, 0.20]  # higher-quality: more transient issues
        elif quality < -0.5:
            reason_p = [0.40, 0.10, 0.10, 0.20, 0.20]  # lower-quality: more hard declines
        else:
            reason_p = [0.25, 0.20, 0.20, 0.15, 0.20]
        failure_reason = RNG.choice(FAILURE_REASONS, p=reason_p)

        amount = float(np.clip(
            RNG.lognormal(mean=np.log(max(row["average_order_value"], 100)), sigma=0.35),
            50, 50000
        ))

        days_since_last_purchase = max((ts - h["last_purchase_date"]).days, 0)

        total_prior = h["previous_payment_failures"] + h["previous_successes"]
        previous_success_rate = (h["previous_successes"] / total_prior) if total_prior > 0 else 0.5

        # ---- compliance state at time of event ----
        window_start = ts - timedelta(hours=24)
        h["attempts_in_window"] = [t for t in h["attempts_in_window"] if t > window_start]
        attempts_so_far = len(h["attempts_in_window"])
        last_contact_time = h["last_contact_time"]

        # opt-out risk grows mildly with how often we've bothered them
        if not h["opted_out"]:
            opt_out_prob = min(0.01 + 0.015 * h["previous_payment_failures"], 0.15)
            if RNG.random() < opt_out_prob:
                h["opted_out"] = True
        opted_out = h["opted_out"]

        time_of_day = ts.hour
        day_of_week = ts.weekday()

        # ---- determine allowed interventions given compliance state ----
        if opted_out:
            allowed = ["no_action"]
        elif attempts_so_far >= 2:
            allowed = ["escalate_to_human", "no_action"]
        else:
            allowed = INTERVENTIONS.copy()

        # historical policy: pick uniformly among allowed actions (randomized
        # logging policy -> unbiased training coverage across actions)
        chosen_intervention = RNG.choice(allowed)

        # ---- hidden true recovery-probability function, per candidate action ----
        true_probs = {}
        for action in INTERVENTIONS:
            true_probs[action] = true_recovery_probability(
                quality=quality,
                failure_reason=failure_reason,
                event_type=event_type,
                previous_success_rate=previous_success_rate,
                attempts_so_far=attempts_so_far,
                amount=amount,
                lifetime_value=row["lifetime_value"],
                action=action,
            )

        # ---- sample the actually observed outcome for the chosen action ----
        p_chosen = true_probs[chosen_intervention]
        recovered = bool(RNG.random() < p_chosen)
        recovered_amount = round(amount, 2) if recovered else 0.0
        time_to_recovery_hours = round(float(RNG.exponential(6.0)), 2) if recovered else None

        # ---- update running history for next event of this customer ----
        if recovered:
            h["previous_successes"] += 1
            h["previous_recovery_successes"] += 1
            h["last_purchase_date"] = ts
        else:
            h["previous_payment_failures"] += 1
        if chosen_intervention != "no_action":
            h["attempts_in_window"].append(ts)
            h["last_contact_time"] = ts

        transaction_id = f"TXN{i:06d}"

        rows.append({
            "transaction_id": transaction_id,
            "customer_id": cid,
            "customer_age_days": int(row["customer_age_days"]),
            "lifetime_value": round(float(row["lifetime_value"]), 2),
            "previous_success_rate": round(previous_success_rate, 3),
            "average_order_value": round(float(row["average_order_value"]), 2),
            "previous_recovery_successes": h["previous_recovery_successes"] - (1 if recovered else 0),
            "days_since_last_purchase": days_since_last_purchase,
            "amount": round(amount, 2),
            "event_type": event_type,
            "failure_reason": failure_reason,
            "timestamp": ts.isoformat(),
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "attempts_so_far": attempts_so_far,
            "last_contact_time": last_contact_time.isoformat() if last_contact_time else None,
            "opted_out": opted_out,
            "allowed_interventions": ";".join(allowed),
            "intervention": chosen_intervention,
            "recovered": recovered,
            "recovered_amount": recovered_amount,
            "time_to_recovery_hours": time_to_recovery_hours,
        })

        # ground truth stashed separately per row (matched by transaction_id)
        rows[-1]["_true_probs"] = true_probs

    events_df = pd.DataFrame(rows)

    # fix the previous_payment_failures column properly (state *before* this event)
    events_df["previous_payment_failures"] = None
    per_cust_counter = {}
    for idx, r in events_df.iterrows():
        cid = r["customer_id"]
        events_df.at[idx, "previous_payment_failures"] = per_cust_counter.get(cid, 0)
        if not r["recovered"]:
            per_cust_counter[cid] = per_cust_counter.get(cid, 0) + 1

    return events_df


def true_recovery_probability(quality, failure_reason, event_type, previous_success_rate,
                               attempts_so_far, amount, lifetime_value, action):
    """
    Hidden scoring function -> logistic-transformed probability.
    Combines customer quality, context, and the intervention itself with
    interaction terms, plus noise, so labels are NOT deterministic rules.
    """
    reason_effect = {
        "insufficient_funds": -0.3,
        "otp_timeout": 0.4,
        "gateway_error": 0.5,
        "bank_decline": -0.6,
        "price_hesitation": -0.1,
    }[failure_reason]

    event_effect = -0.2 if event_type == "checkout_abandoned" else 0.1

    action_base = {
        "retry_link": 0.5,
        "reminder": 0.1,
        "discount_offer": 0.3,
        "escalate_to_human": 0.6,
        "no_action": -1.5,
    }[action]

    # interaction: discount matters more for price_hesitation and lower lifetime_value
    discount_interaction = 0.0
    if action == "discount_offer":
        discount_interaction += 0.6 if failure_reason == "price_hesitation" else 0.0
        discount_interaction += 0.3 if lifetime_value < 5000 else -0.1

    # interaction: retry_link works best for transient technical failures
    retry_interaction = 0.0
    if action == "retry_link":
        retry_interaction += 0.5 if failure_reason in ("otp_timeout", "gateway_error") else -0.2

    # fatigue: more prior attempts this window -> diminishing returns
    fatigue_effect = -0.25 * attempts_so_far

    # amount effect: very large amounts slightly harder to recover automatically
    amount_effect = -0.15 if amount > 10000 else 0.0

    noise = RNG.normal(0, 0.5)

    score = (
        0.8 * quality
        + reason_effect
        + event_effect
        + action_base
        + discount_interaction
        + retry_interaction
        + fatigue_effect
        + amount_effect
        + 1.2 * (previous_success_rate - 0.5)
        + noise
    )

    prob = 1 / (1 + np.exp(-score))
    return float(np.clip(prob, 0.01, 0.99))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    customers = generate_customers(N_CUSTOMERS)
    schedule = generate_event_schedule(customers, N_EVENTS)
    events_df = build_events(schedule)

    ground_truth_rows = []
    for idx, r in events_df.iterrows():
        gt = {"transaction_id": r["transaction_id"]}
        for action, p in r["_true_probs"].items():
            gt[f"true_prob_{action}"] = round(p, 4)
        ground_truth_rows.append(gt)
    ground_truth_df = pd.DataFrame(ground_truth_rows)

    events_out = events_df.drop(columns=["_true_probs"])

    events_out.to_csv("events.csv", index=False)
    ground_truth_df.to_csv("ground_truth.csv", index=False)

    # ---- sanity summary ----
    print(f"Generated {len(events_out)} events across {events_out['customer_id'].nunique()} customers")
    print(f"\nEvent type distribution:\n{events_out['event_type'].value_counts()}")
    print(f"\nFailure reason distribution:\n{events_out['failure_reason'].value_counts()}")
    print(f"\nIntervention distribution:\n{events_out['intervention'].value_counts()}")
    print(f"\nOverall recovery rate: {events_out['recovered'].mean():.3f}")
    print(f"\nRecovery rate by intervention:\n{events_out.groupby('intervention')['recovered'].mean()}")
    print(f"\nOpted-out customers: {events_out.groupby('customer_id')['opted_out'].max().sum()} / {events_out['customer_id'].nunique()}")
    print(f"\nTotal amount at risk (all events): Rs {events_out['amount'].sum():,.2f}")
    print(f"Total amount recovered (as-logged, mixed random policy): Rs {events_out['recovered_amount'].sum():,.2f}")


if __name__ == "__main__":
    main()
