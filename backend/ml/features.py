"""
Shared feature definitions and preprocessing for the recovery-prediction model.

This module is intentionally the single source of truth for "what the model
sees" — it's imported by train.py (training), the optimality-gap evaluator,
and later by the expected-value policy layer / FastAPI app so that scoring a
candidate intervention at inference time uses exactly the same feature
construction as training. Never import ground_truth.csv here — this module
only ever touches events.csv-shaped data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The fixed action set. Order matters only for display; never rely on it
# for anything semantic.
ALL_INTERVENTIONS = ["retry_link", "reminder", "discount_offer", "escalate_to_human", "no_action"]

TARGET = "recovered"

# Raw numeric features, observed at decision time (all pre-outcome).
NUMERIC_FEATURES = [
    "customer_age_days",
    "lifetime_value",
    "previous_success_rate",
    "average_order_value",
    "previous_recovery_successes",
    "days_since_last_purchase",
    "amount",
    "time_of_day",
    "day_of_week",
    "attempts_so_far",
    "previous_payment_failures",
    # log1p-transformed versions of the heavy-tailed (lognormal) money fields —
    # helps the linear baseline in particular; harmless for the tree model.
    "log_amount",
    "log_lifetime_value",
    "log_average_order_value",
]

BOOLEAN_FEATURES = ["opted_out"]

CATEGORICAL_FEATURES = ["event_type", "failure_reason", "intervention"]

FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

# Explicit interaction features — added only where generate_data.py's
# true_recovery_probability() actually has an interaction term tied to
# `action`:
#   - discount_offer × failure_reason==price_hesitation   (+0.6)
#   - discount_offer × lifetime_value<5000                (+0.3 / -0.1)
#   - retry_link × failure_reason in (otp_timeout, gateway_error)  (+0.5 / -0.2)
# event_type and previous_success_rate are purely additive in the true model
# (no term multiplies them by `action`), so no interaction feature is added
# for them — doing so would just give the model 3200 rows to fit noise
# against a signal that doesn't exist.
INTERACTION_FEATURES = ["intervention_x_failure_reason", "intervention_x_lifetime_bucket"]

CATEGORICAL_FEATURES_WITH_INTERACTIONS = CATEGORICAL_FEATURES + INTERACTION_FEATURES
FEATURE_COLUMNS_WITH_INTERACTIONS = FEATURE_COLUMNS + INTERACTION_FEATURES

LIFETIME_VALUE_BUCKET_THRESHOLD = 5000  # matches generate_data.py's discount_interaction cutoff exactly

# Columns that exist in events.csv but must NEVER be used as model features —
# either because they are post-outcome (leak the label) or are identifiers /
# raw timestamps not meant to be fed in directly.
LEAKAGE_OR_ID_COLUMNS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "last_contact_time",
    "allowed_interventions",
    "recovered",
    "recovered_amount",
    "time_to_recovery_hours",
]


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = engineer_features(df)
    df = add_interaction_features(df)  # cheap; only used if FEATURE_COLUMNS_WITH_INTERACTIONS is selected
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and normalize dtypes. Safe to call on any subset
    of rows (e.g. counterfactual copies built for scoring candidate actions)."""
    df = df.copy()

    # opted_out arrives as a real bool from pandas already in most cases, but
    # be defensive in case it was read from CSV as the string "True"/"False".
    if df["opted_out"].dtype == object:
        df["opted_out"] = df["opted_out"].astype(str).str.strip().str.lower() == "true"
    df["opted_out"] = df["opted_out"].astype(int)

    df["log_amount"] = np.log1p(df["amount"].astype(float))
    df["log_lifetime_value"] = np.log1p(df["lifetime_value"].astype(float))
    df["log_average_order_value"] = np.log1p(df["average_order_value"].astype(float))

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the explicit intervention-dependent interaction columns (see
    INTERACTION_FEATURES above). Must be called AFTER `intervention` is in
    its final value for the row — in particular, rows_for_all_interventions()
    below re-calls this after swapping `intervention`, since these columns
    are string concatenations that depend on it.
    """
    df = df.copy()
    lifetime_bucket = np.where(
        df["lifetime_value"].astype(float) < LIFETIME_VALUE_BUCKET_THRESHOLD, "lt5000", "ge5000"
    )
    df["intervention_x_failure_reason"] = (
        df["intervention"].astype(str) + "__" + df["failure_reason"].astype(str)
    )
    df["intervention_x_lifetime_bucket"] = df["intervention"].astype(str) + "__" + lifetime_bucket
    return df


def build_preprocessor(scale_numeric: bool, categorical_features: list[str] = CATEGORICAL_FEATURES) -> ColumnTransformer:
    """
    scale_numeric=True  -> for Logistic Regression (needs standardized inputs)
    scale_numeric=False -> for XGBoost (tree splits are scale-invariant)
    categorical_features -> pass CATEGORICAL_FEATURES_WITH_INTERACTIONS to
    include the engineered interaction columns.
    """
    numeric_pipeline = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES + BOOLEAN_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def make_pipeline(estimator, scale_numeric: bool, categorical_features: list[str] = CATEGORICAL_FEATURES) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric, categorical_features=categorical_features)),
            ("model", estimator),
        ]
    )


def rows_for_all_interventions(row: pd.Series, with_interactions: bool = False) -> pd.DataFrame:
    """
    Given a single event row (already feature-engineered), return a 5-row
    DataFrame — one copy of the row per candidate intervention in
    ALL_INTERVENTIONS, with the `intervention` column swapped out each time.

    This is how we get "P(recovery | customer, context, X)" for every
    candidate action X from a single model that takes intervention as a
    categorical feature, instead of training five separate models.

    with_interactions=True re-derives intervention_x_* columns for each of
    the 5 swapped rows (they must change when intervention changes — a plain
    copy would leave them stuck at the original row's actual intervention).
    """
    variants = pd.concat([row.to_frame().T] * len(ALL_INTERVENTIONS), ignore_index=True)
    variants["intervention"] = ALL_INTERVENTIONS
    if with_interactions:
        variants = add_interaction_features(variants)
    return variants
