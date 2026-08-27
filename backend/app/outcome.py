"""
Execution + outcome simulator — build-order step 6.

Simulates whether the chosen action actually recovered the revenue, sampled
against ground_truth.csv's true recovery probability for that action — the
SAME functional form generate_data.py itself used to produce the dataset's
own `recovered` / `recovered_amount` / `time_to_recovery_hours` columns:

    recovered = bool(rng.random() < true_prob_for_chosen_action)
    recovered_amount = amount if recovered else 0.0
    time_to_recovery_hours = rng.exponential(6.0) if recovered else None

This reads ground_truth.csv, which is fine here — it's evaluation/simulation
code, not model training. The rule that's held throughout this project is
narrower and still holds: ground_truth.csv is never used as a training
feature or joined into the classifier's training set (see ml/train.py).
Using it to sample a simulated outcome for an already-decided action is
exactly what the brief's §11 spec asks the outcome simulator to do.

Nothing here recomputes true_recovery_probability() from scratch — it can't:
that function's `quality` input is a hidden per-customer latent variable
never written to events.csv (by design, per generate_data.py's docstring).
ground_truth.csv's precomputed true_prob_<action> columns are the only
faithful source for these probabilities outside generate_data.py itself.

Reproducibility / comparison note: a policy's full batch run creates its own
RNG via `np.random.default_rng(OUTCOME_SIM_SEED)` at the start of that run,
then draws in fixed events.csv row order. Every policy therefore uses the
IDENTICAL draw sequence at each row index — a lightweight form of common
random numbers — so that when two policies pick the SAME action for a
transaction, they get the SAME simulated outcome, and observed differences
in the baseline comparison come from differences in WHICH actions get
chosen, not from independent sampling noise. This isn't a fully paired
design (once two policies' chosen actions diverge for a transaction, their
subsequent per-row draw counts can drift out of lockstep for later rows,
since a recovered outcome consumes one extra exponential() draw) — but it's
a real, honest, fully reproducible improvement over independent seeding,
not a claim of perfect pairing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

OUTCOME_SIM_SEED = 4242  # distinct from ml/train.py's RANDOM_STATE=42 and generate_data.py's seed=42,
                          # to keep "which RNG produced which number" unambiguous when debugging.

GROUND_TRUTH_PATH = Path(__file__).resolve().parents[1] / "data" / "ground_truth.csv"

_ground_truth_cache: Optional[dict] = None


def load_ground_truth(path: Path | str = GROUND_TRUTH_PATH) -> dict:
    """{transaction_id: {action: true_probability}}, columns true_prob_<action> -> <action>."""
    global _ground_truth_cache
    if _ground_truth_cache is not None:
        return _ground_truth_cache

    df = pd.read_csv(path)
    prob_cols = [c for c in df.columns if c.startswith("true_prob_")]
    lookup = {}
    for _, row in df.iterrows():
        lookup[row["transaction_id"]] = {c.removeprefix("true_prob_"): float(row[c]) for c in prob_cols}
    _ground_truth_cache = lookup
    return lookup


def new_rng() -> np.random.Generator:
    """Fresh RNG for one policy's full batch run — see module docstring."""
    return np.random.default_rng(OUTCOME_SIM_SEED)


def simulate_outcome(rng: np.random.Generator, true_prob: float, amount: float) -> dict:
    """One outcome draw, same functional form as generate_data.py's build_events().
    stopping_reason is a deliberate simplification: this project scores one
    event per transaction rather than a multi-touch retry sequence, so there's
    no richer stopping taxonomy (max-attempts-reached, opted-out-mid-sequence,
    etc. — brief §12's full list) to report yet; only the two outcomes that
    are actually possible today are used."""
    recovered = bool(rng.random() < true_prob)
    return {
        "recovered": recovered,
        "recovered_amount": round(float(amount), 2) if recovered else 0.0,
        "time_to_recovery_hours": round(float(rng.exponential(6.0)), 2) if recovered else None,
        "stopping_reason": "recovered" if recovered else "not_recovered",
    }
