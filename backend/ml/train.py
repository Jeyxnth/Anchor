"""
ML pipeline — step 2 of the build order.

Trains a single intervention-conditional recovery model (XGBoost primary,
Logistic Regression baseline), evaluates it honestly on a held-out split,
and — using ground_truth.csv (evaluation-only, never a training feature) —
computes the optimality gap: for each held-out event, does the model's
top-ranked candidate intervention match the actually-best intervention
under the hidden true probabilities?

XGBoost is trained WITH the explicit intervention_x_failure_reason /
intervention_x_lifetime_bucket interaction features (see features.py) — a
controlled experiment (experiment_interactions.py) showed this materially
improves ranking quality over the plain-feature baseline at the same
hyperparams (optimality match rate 38.5%→45.8%, mean regret -24%, action
coverage 2/5→3/5), while a depth=4 retrain with the same plain features
barely moved any of those numbers — so this was a feature-representation
gap, not a tree-capacity one. See experiment_interactions.py and
artifacts/experiment_interactions_report.json for the full comparison.
Logistic Regression is left on the plain feature set (not retested with
interactions — out of scope for that experiment; the LR/XGBoost near-tie
documented below predates the interaction features and still stands on
ROC-AUC).

Run from backend/:  .venv\\Scripts\\python.exe ml\\train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    ALL_INTERVENTIONS,
    CATEGORICAL_FEATURES_WITH_INTERACTIONS,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_WITH_INTERACTIONS,
    TARGET,
    load_events,
    make_pipeline,
    rows_for_all_interventions,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

EVENTS_PATH = DATA_DIR / "events.csv"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def evaluate_classifier(name: str, y_true, y_pred, y_proba) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    report = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }

    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"Accuracy : {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall   : {report['recall']:.4f}")
    print(f"F1       : {report['f1']:.4f}")
    print(f"ROC-AUC  : {report['roc_auc']:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred; [[TN, FP], [FN, TP]]):")
    print(cm)
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["not_recovered", "recovered"]))

    return report


def compute_optimality_gap(model_pipeline, events_df: pd.DataFrame, test_idx: pd.Index,
                            ground_truth_df: pd.DataFrame, feature_columns=None,
                            with_interactions: bool = False) -> dict:
    """
    For every held-out event: score all 5 candidate interventions with the
    trained model, take the model's argmax, compare it to the argmax of the
    TRUE (ground-truth) probabilities for that same event. `gap` is the
    expected-recovery-probability regret of following the model's pick
    instead of the true-best pick (0 when they match).

    feature_columns/with_interactions let this be reused for the
    interaction-feature model variant (see experiment_interactions.py) —
    default behavior (None, False) matches the baseline model exactly.
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    gt_by_txn = ground_truth_df.set_index("transaction_id")
    prob_cols = {a: f"true_prob_{a}" for a in ALL_INTERVENTIONS}

    records = []
    for idx in test_idx:
        row = events_df.loc[idx]
        txn_id = row["transaction_id"]
        if txn_id not in gt_by_txn.index:
            continue  # shouldn't happen, but don't let a bad join crash the run

        variants = rows_for_all_interventions(row, with_interactions=with_interactions)[feature_columns]
        pred_probs = model_pipeline.predict_proba(variants)[:, 1]
        pred_by_action = dict(zip(ALL_INTERVENTIONS, pred_probs))
        model_best_action = max(pred_by_action, key=pred_by_action.get)

        gt_row = gt_by_txn.loc[txn_id]
        true_by_action = {a: gt_row[prob_cols[a]] for a in ALL_INTERVENTIONS}
        true_best_action = max(true_by_action, key=true_by_action.get)

        gap = true_by_action[true_best_action] - true_by_action[model_best_action]

        records.append({
            "transaction_id": txn_id,
            "model_best_action": model_best_action,
            "true_best_action": true_best_action,
            "match": model_best_action == true_best_action,
            "gap": gap,
        })

    results = pd.DataFrame(records)
    coverage_counts = results["model_best_action"].value_counts().to_dict()
    summary = {
        "n_events": len(results),
        "optimality_match_rate": float(results["match"].mean()),
        "mean_gap_all": float(results["gap"].mean()),
        "mean_gap_when_mismatched": float(results.loc[~results["match"], "gap"].mean()) if (~results["match"]).any() else 0.0,
        "median_gap_all": float(results["gap"].median()),
        "max_gap": float(results["gap"].max()),
        "action_coverage_counts": {a: int(coverage_counts.get(a, 0)) for a in ALL_INTERVENTIONS},
        "n_actions_ever_top_pick": int((results["model_best_action"].value_counts() > 0).sum()),
    }

    print(f"\n{'=' * 70}\nOptimality-gap evaluation (held-out set, vs. ground_truth.csv)\n{'=' * 70}")
    print(f"Events evaluated       : {summary['n_events']}")
    print(f"Model matched true-best: {summary['optimality_match_rate']:.1%}")
    print(f"Mean regret (all)      : {summary['mean_gap_all']:.4f}  (expected-recovery-probability units)")
    print(f"Mean regret (mismatches only): {summary['mean_gap_when_mismatched']:.4f}")
    print(f"Median regret (all)    : {summary['median_gap_all']:.4f}")
    print(f"Max regret             : {summary['max_gap']:.4f}")
    print(f"Actions ever picked as top-1: {summary['n_actions_ever_top_pick']}/5  {summary['action_coverage_counts']}")

    print("\nWhat the model picks vs. what was actually best (confusion, held-out set):")
    action_confusion = pd.crosstab(results["true_best_action"], results["model_best_action"],
                                    rownames=["true_best"], colnames=["model_picked"])
    print(action_confusion)

    return summary, results


def main():
    print(f"Loading events from {EVENTS_PATH}")
    events_df = load_events(str(EVENTS_PATH))
    print(f"{len(events_df)} events loaded, {events_df['customer_id'].nunique()} unique customers")

    X = events_df[FEATURE_COLUMNS]
    y = events_df[TARGET].astype(int)

    print(f"\nOverall recovery rate: {y.mean():.3f}  (positive class prevalence)")
    print("Recovery rate by intervention (as-logged, historical randomized policy):")
    print(events_df.groupby("intervention")[TARGET].mean().sort_values(ascending=False))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, events_df.index,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"\nTrain/test split: {len(X_train)} train / {len(X_test)} test "
          f"(stratified on target, random_state={RANDOM_STATE})")
    print(f"Train positive rate: {y_train.mean():.3f} | Test positive rate: {y_test.mean():.3f}")

    # ---- Logistic Regression baseline ----
    lr_pipeline = make_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        scale_numeric=True,
    )
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_report = evaluate_classifier("Logistic Regression (baseline)", y_test, lr_pred, lr_proba)
    lr_train_auc = roc_auc_score(y_train, lr_pipeline.predict_proba(X_train)[:, 1])
    print(f"(LR train ROC-AUC: {lr_train_auc:.4f} vs test {lr_report['roc_auc']:.4f} — "
          f"{'small gap' if (lr_train_auc - lr_report['roc_auc']) < 0.08 else 'notable gap'})")

    # ---- XGBoost primary model ----
    # Small guarded hyperparameter search (cross-validated on TRAIN only —
    # the test set is never touched here) to control overfitting: the data
    # is only 3200 training rows after one-hot expansion, and an
    # untuned deep/many-tree XGBoost memorizes it (see the first pass of this
    # pipeline: train ROC-AUC 0.92 vs test 0.75). Shallower, more regularized
    # trees generalize better on data this size.
    base_estimator = XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_search_pipeline = make_pipeline(base_estimator, scale_numeric=False)
    param_grid = {
        "model__max_depth": [2, 3, 4],
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__min_child_weight": [1, 5, 10],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        xgb_search_pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
    )
    print(f"\n{'=' * 70}\nXGBoost hyperparameter search (5-fold CV on TRAIN set only)\n{'=' * 70}")
    grid_search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"Best params    : {grid_search.best_params_}")

    # ---- Promote to the interaction-feature variant ----
    # The grid search above (plain features) is what discovered these
    # hyperparams and is kept as the reproducible record of that search.
    # experiment_interactions.py then showed that retraining this exact
    # config with intervention_x_failure_reason / intervention_x_lifetime_
    # bucket added (and hyperparams otherwise UNCHANGED — no re-search)
    # materially improves ranking quality: optimality match rate 38.5%→45.8%,
    # mean regret -24%, action coverage 2/5→3/5, ROC-AUC roughly flat. That
    # interaction-feature model is what actually gets evaluated/saved below.
    best_params = {k.replace("model__", ""): v for k, v in grid_search.best_params_.items()}
    X_train_int = events_df.loc[idx_train, FEATURE_COLUMNS_WITH_INTERACTIONS]
    X_test_int = events_df.loc[idx_test, FEATURE_COLUMNS_WITH_INTERACTIONS]
    xgb_pipeline = make_pipeline(
        XGBClassifier(
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1, **best_params,
        ),
        scale_numeric=False,
        categorical_features=CATEGORICAL_FEATURES_WITH_INTERACTIONS,
    )
    xgb_pipeline.fit(X_train_int, y_train)
    xgb_pred = xgb_pipeline.predict(X_test_int)
    xgb_proba = xgb_pipeline.predict_proba(X_test_int)[:, 1]
    xgb_report = evaluate_classifier("XGBoost (primary, tuned + interaction features)", y_test, xgb_pred, xgb_proba)
    xgb_report["best_params"] = grid_search.best_params_
    xgb_report["best_cv_roc_auc"] = grid_search.best_score_
    xgb_report["feature_set"] = "FEATURE_COLUMNS_WITH_INTERACTIONS (see experiment_interactions.py for why)"

    # ---- Overfitting sanity check: compare test metrics to train metrics ----
    xgb_train_pred = xgb_pipeline.predict(X_train_int)
    xgb_train_proba = xgb_pipeline.predict_proba(X_train_int)[:, 1]
    train_auc = roc_auc_score(y_train, xgb_train_proba)
    train_acc = accuracy_score(y_train, xgb_train_pred)
    print(f"\n{'=' * 70}\nOverfitting check (XGBoost)\n{'=' * 70}")
    print(f"Train accuracy: {train_acc:.4f}  vs  Test accuracy: {xgb_report['accuracy']:.4f}")
    print(f"Train ROC-AUC : {train_auc:.4f}  vs  Test ROC-AUC : {xgb_report['roc_auc']:.4f}")
    gap_note = "small gap — looks reasonable, not memorizing" if (train_auc - xgb_report["roc_auc"]) < 0.08 \
        else "notable train/test gap — worth investigating before trusting this model further"
    print(f"-> {gap_note}")

    # ---- Optimality gap, using ground_truth.csv (eval-only, held-out set) ----
    print(f"\nLoading ground truth (evaluation-only) from {GROUND_TRUTH_PATH}")
    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)
    opt_summary, opt_details = compute_optimality_gap(
        xgb_pipeline, events_df, idx_test, ground_truth_df,
        feature_columns=FEATURE_COLUMNS_WITH_INTERACTIONS, with_interactions=True,
    )

    # ---- Persist artifacts ----
    joblib.dump(xgb_pipeline, ARTIFACT_DIR / "xgb_recovery_model.joblib")
    joblib.dump(lr_pipeline, ARTIFACT_DIR / "lr_recovery_model.joblib")

    eval_report = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "logistic_regression": lr_report,
        "logistic_regression_train_roc_auc": lr_train_auc,
        "xgboost": xgb_report,
        "xgboost_train_vs_test": {"train_accuracy": train_acc, "train_roc_auc": train_auc},
        "optimality_gap": opt_summary,
    }
    with open(ARTIFACT_DIR / "eval_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)
    opt_details.to_csv(ARTIFACT_DIR / "optimality_gap_details.csv", index=False)

    print(f"\nSaved: {ARTIFACT_DIR / 'xgb_recovery_model.joblib'}")
    print(f"Saved: {ARTIFACT_DIR / 'lr_recovery_model.joblib'}")
    print(f"Saved: {ARTIFACT_DIR / 'eval_report.json'}")
    print(f"Saved: {ARTIFACT_DIR / 'optimality_gap_details.csv'}")


if __name__ == "__main__":
    main()
