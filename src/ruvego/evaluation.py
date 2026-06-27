from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold


@dataclass(frozen=True)
class CVConfig:
    folds: int
    repeats: int


def choose_cv_config(y: pd.Series, requested_folds: int, requested_repeats: int) -> CVConfig:
    min_class = int(y.value_counts().min())
    folds = max(2, min(requested_folds, min_class))
    repeats = requested_repeats
    if len(y) < 150:
        repeats = max(repeats, 5)
    return CVConfig(folds=folds, repeats=repeats)


def cross_validate_oof(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    estimator_factory: Callable[[int], object],
    dataset_name: str,
    folds: int,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    y_array = y.to_numpy()
    proba_sum = np.zeros(len(y_array), dtype=float)
    proba_count = np.zeros(len(y_array), dtype=float)
    fold_rows = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y_array), start=1):
        model = estimator_factory(seed + fold_idx)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(X_train, y_train)
        valid_proba = model.predict_proba(X_valid)[:, 1]

        proba_sum[valid_idx] += valid_proba
        proba_count[valid_idx] += 1

        fold_metrics = compute_metrics(y_valid.to_numpy(), valid_proba, threshold=0.5)
        fold_metrics.update(
            {
                "dataset": dataset_name,
                "fold": fold_idx,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
            }
        )
        fold_rows.append(fold_metrics)

    oof_proba = proba_sum / np.maximum(proba_count, 1)
    if np.any(proba_count == 0):
        missing = int(np.sum(proba_count == 0))
        raise RuntimeError(f"{dataset_name}: {missing} rows did not receive OOF predictions.")

    threshold_report = find_best_threshold(y_array, oof_proba)
    oof = pd.DataFrame(
        {
            "dataset": dataset_name,
            "Variant_ID": ids.reset_index(drop=True),
            "Label": y_array,
            "oof_proba": oof_proba,
            "oof_pred_0_5": (oof_proba >= 0.5).astype(int),
            "oof_pred_best_threshold": (oof_proba >= threshold_report["best_threshold"]).astype(int),
            "oof_prediction_count": proba_count.astype(int),
        }
    )
    oof["label_confidence"] = np.where(oof["Label"] == 1, oof["oof_proba"], 1 - oof["oof_proba"])
    oof["uncertain_zone"] = oof["oof_proba"].between(0.40, 0.60)

    fold_metrics_df = pd.DataFrame(fold_rows)
    return oof, fold_metrics_df, threshold_report


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> dict:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = compute_metrics(y_true, proba, threshold=float(threshold))
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    best = sorted(
        rows,
        key=lambda row: (
            row["macro_f1"],
            row["pathogenic_recall"],
            row["pathogenic_precision"],
            -abs(row["threshold"] - 0.5),
        ),
        reverse=True,
    )[0]
    at_default = compute_metrics(y_true, proba, threshold=0.5)
    return {
        "best_threshold": float(best["threshold"]),
        "best_macro_f1": float(best["macro_f1"]),
        "best_pathogenic_f1": float(best["pathogenic_f1"]),
        "best_pathogenic_precision": float(best["pathogenic_precision"]),
        "best_pathogenic_recall": float(best["pathogenic_recall"]),
        "default_macro_f1": float(at_default["macro_f1"]),
        "default_pathogenic_f1": float(at_default["pathogenic_f1"]),
        "default_pathogenic_precision": float(at_default["pathogenic_precision"]),
        "default_pathogenic_recall": float(at_default["pathogenic_recall"]),
        "threshold_grid": rows,
    }


def compute_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "pathogenic_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "benign_f1": float(f1_score(y_true, pred, pos_label=0, zero_division=0)),
        "pathogenic_precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "pathogenic_recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "benign_precision": float(precision_score(y_true, pred, pos_label=0, zero_division=0)),
        "benign_recall": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        metrics["pr_auc"] = float(average_precision_score(y_true, proba))
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        metrics["pr_auc"] = np.nan
        metrics["roc_auc"] = np.nan
    return metrics


def summarize_oof(dataset_name: str, oof: pd.DataFrame, threshold_report: dict) -> dict:
    y_true = oof["Label"].to_numpy()
    proba = oof["oof_proba"].to_numpy()
    best_threshold = threshold_report["best_threshold"]
    row = compute_metrics(y_true, proba, threshold=best_threshold)
    row.update(
        {
            "dataset": dataset_name,
            "best_threshold": best_threshold,
            "rows": len(oof),
            "uncertain_zone_rows": int(oof["uncertain_zone"].sum()),
            "mean_label_confidence": float(oof["label_confidence"].mean()),
        }
    )
    return row
