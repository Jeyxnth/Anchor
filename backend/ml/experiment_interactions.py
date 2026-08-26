"""
Controlled experiment — follow-up to train.py.

Tests two bounded variants against the existing tuned XGBoost baseline
(depth=3, n_estimators=100, learning_rate=0.03, min_child_weight=1) to see
whether either closes the "model never recommends reminder/discount_offer/
no_action as top pick" coverage gap seen in the baseline optimality-gap
check:

  (B) intervention_x_failure_reason / intervention_x_lifetime_bucket added
      as explicit features, hyperparams unchanged from the tuned baseline.
      These mirror the ONLY interaction terms that actually exist in
      generate_data.py's true_recovery_probability() — event_type and
      previous_success_rate are additive-only there, so no interaction
      feature is added for them (would just fit noise on 3200 rows).

  (C) max_depth=4, everything else identical to the tuned baseline, no
      interaction features. One specific check, not a search — isolates
      whether more per-tree capacity alone closes the gap.

Deliberately NOT a hyperparameter search. Ground truth is read only for
evaluation (never as a training feature), same as every other script here.

Run from backend/:  .venv\\Scripts\\python.exe ml\\experiment_interactions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    CATEGORICAL_FEATURES_WITH_INTERACTIONS,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_WITH_INTERACTIONS,
    TARGET,
    load_events,
    make_pipeline,
)
from train import (  # noqa: E402
    ARTIFACT_DIR,
    EVENTS_PATH,
    GROUND_TRUTH_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    compute_optimality_gap,
)

# Hyperparams held fixed at the already-tuned baseline config found by the
# grid search in train.py. Only ONE thing changes per variant below.
BASELINE_XGB_PARAMS = dict(
    max_depth=3,
    n_estimators=100,
    learning_rate=0.03,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def run_variant(name, pipeline, X_train, y_train, X_test, y_test, events_df, idx_test,
                 ground_truth_df, feature_columns, with_interactions):
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    opt_summary, opt_details = compute_optimality_gap(
        pipeline, events_df, idx_test, ground_truth_df,
        feature_columns=feature_columns, with_interactions=with_interactions,
    )
    print(f"\n>>> {name}: test ROC-AUC = {auc:.4f}")
    return pipeline, {
        "test_roc_auc": float(auc),
        "optimality_match_rate": opt_summary["optimality_match_rate"],
        "mean_gap_all": opt_summary["mean_gap_all"],
        "mean_gap_when_mismatched": opt_summary["mean_gap_when_mismatched"],
        "n_actions_ever_top_pick": opt_summary["n_actions_ever_top_pick"],
        "action_coverage_counts": opt_summary["action_coverage_counts"],
    }, opt_details


def main():
    events_df = load_events(str(EVENTS_PATH))
    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)
    y = events_df[TARGET].astype(int)

    # Identical split to train.py: same index array + same stratify labels +
    # same random_state => identical row membership, regardless of which
    # feature-column set we slice afterwards. This makes every variant below
    # directly comparable to the saved baseline.
    _, _, y_train, y_test, idx_train, idx_test = train_test_split(
        events_df[FEATURE_COLUMNS], y, events_df.index,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    results = {}
    fitted = {}

    # ---- Reference: baseline retrained fresh from the tuned config (not
    # reloaded from artifacts/ — this script must stay a valid, reproducible
    # standalone comparison even after the artifact is overwritten below with
    # whichever variant wins). Same params, same split => same result as the
    # grid search in train.py converged to.
    X_train_base = events_df.loc[idx_train, FEATURE_COLUMNS]
    X_test_base = events_df.loc[idx_test, FEATURE_COLUMNS]
    baseline_pipeline = make_pipeline(XGBClassifier(**BASELINE_XGB_PARAMS), scale_numeric=False)
    fitted["A_xgb_baseline_depth3"], results["A_xgb_baseline_depth3"], _ = run_variant(
        "A_xgb_baseline_depth3", baseline_pipeline, X_train_base, y_train, X_test_base, y_test,
        events_df, idx_test, ground_truth_df,
        feature_columns=FEATURE_COLUMNS, with_interactions=False,
    )

    # ---- Variant B: interaction features, depth unchanged ----
    X_train_int = events_df.loc[idx_train, FEATURE_COLUMNS_WITH_INTERACTIONS]
    X_test_int = events_df.loc[idx_test, FEATURE_COLUMNS_WITH_INTERACTIONS]
    interaction_pipeline = make_pipeline(
        XGBClassifier(**BASELINE_XGB_PARAMS),
        scale_numeric=False,
        categorical_features=CATEGORICAL_FEATURES_WITH_INTERACTIONS,
    )
    fitted["B_xgb_interaction_features"], results["B_xgb_interaction_features"], _ = run_variant(
        "B_xgb_interaction_features", interaction_pipeline, X_train_int, y_train, X_test_int, y_test,
        events_df, idx_test, ground_truth_df,
        feature_columns=FEATURE_COLUMNS_WITH_INTERACTIONS, with_interactions=True,
    )

    # ---- Variant C: max_depth=4, no interaction features ----
    depth4_params = dict(BASELINE_XGB_PARAMS)
    depth4_params["max_depth"] = 4
    depth4_pipeline = make_pipeline(XGBClassifier(**depth4_params), scale_numeric=False)
    fitted["C_xgb_depth4"], results["C_xgb_depth4"], _ = run_variant(
        "C_xgb_depth4", depth4_pipeline, X_train_base, y_train, X_test_base, y_test,
        events_df, idx_test, ground_truth_df,
        feature_columns=FEATURE_COLUMNS, with_interactions=False,
    )

    # ---- Comparison table ----
    print(f"\n{'=' * 90}\nComparison: A (tuned baseline) vs B (interaction features) vs C (depth=4)\n{'=' * 90}")
    header = f"{'variant':<28}{'test ROC-AUC':>14}{'opt match rate':>16}{'mean regret':>14}{'actions covered':>17}"
    print(header)
    for key in ["A_xgb_baseline_depth3", "B_xgb_interaction_features", "C_xgb_depth4"]:
        r = results[key]
        print(f"{key:<28}{r['test_roc_auc']:>14.4f}{r['optimality_match_rate']:>15.1%}"
              f"{r['mean_gap_all']:>14.4f}{r['n_actions_ever_top_pick']:>14}/5")
        print(f"    action_coverage_counts: {r['action_coverage_counts']}")

    with open(ARTIFACT_DIR / "experiment_interactions_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {ARTIFACT_DIR / 'experiment_interactions_report.json'}")

    return results, fitted


if __name__ == "__main__":
    main()
