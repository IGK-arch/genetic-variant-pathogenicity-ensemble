from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "tmp" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from .data import ID_COL, LABEL_COL, split_xy
from .features import ensure_dense, transformed_feature_names
from .models import make_model_pipeline


def save_pr_curve(oof: pd.DataFrame, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(oof["Label"], oof["oof_proba"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{oof['dataset'].iloc[0]} precision-recall curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_calibration_curve(oof: pd.DataFrame, out_path: Path) -> None:
    n_bins = min(10, max(3, int(np.sqrt(len(oof)))))
    prob_true, prob_pred = calibration_curve(
        oof["Label"], oof["oof_proba"], n_bins=n_bins, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed pathogenic frequency")
    ax.set_title(f"{oof['dataset'].iloc[0]} calibration")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_low_confidence(oof: pd.DataFrame, out_path: Path, fraction: float = 0.08) -> pd.DataFrame:
    n_rows = min(len(oof), max(20, int(round(len(oof) * fraction))))
    report = oof.sort_values("label_confidence", ascending=True).head(n_rows)
    report.to_csv(out_path, index=False)
    return report


def save_feature_importance(model, out_path: Path, top_n: int = 80) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    feature_names = transformed_feature_names(preprocessor)

    rows = []
    named_estimators = getattr(classifier, "named_estimators_", {})
    for model_name, estimator in named_estimators.items():
        importance = _extract_importance(estimator)
        if importance is None or len(importance) != len(feature_names):
            continue
        importance = np.asarray(importance, dtype=float)
        total = float(np.nansum(np.abs(importance)))
        if total > 0:
            importance = importance / total
        for feature, value in zip(feature_names, importance):
            rows.append({"model": model_name, "feature": feature, "importance": float(value)})

    if not rows:
        frame = pd.DataFrame(columns=["feature", "mean_importance", "models_used"])
        frame.to_csv(out_path, index=False)
        return frame

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("feature", as_index=False)
        .agg(mean_importance=("importance", "mean"), models_used=("model", "nunique"))
        .sort_values("mean_importance", ascending=False)
        .head(top_n)
    )
    summary.to_csv(out_path, index=False)
    return summary


def _extract_importance(estimator):
    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_
    if hasattr(estimator, "get_feature_importance"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return estimator.get_feature_importance()
    return None


def save_near_duplicate_report(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    fitted_model,
    out_path: Path,
    top_n: int = 100,
) -> pd.DataFrame:
    if len(X) < 2:
        frame = pd.DataFrame()
        frame.to_csv(out_path, index=False)
        return frame

    matrix = fitted_model.named_steps["preprocess"].transform(X)
    matrix = ensure_dense(matrix)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)

    seen = set()
    rows = []
    y_values = y.reset_index(drop=True)
    id_values = ids.reset_index(drop=True)
    for left_idx in range(len(X)):
        right_idx = int(indices[left_idx, 1])
        pair = tuple(sorted((left_idx, right_idx)))
        if pair in seen:
            continue
        seen.add(pair)
        similarity = 1.0 - float(distances[left_idx, 1])
        rows.append(
            {
                "left_variant_id": id_values.iloc[left_idx],
                "right_variant_id": id_values.iloc[right_idx],
                "left_label": int(y_values.iloc[left_idx]),
                "right_label": int(y_values.iloc[right_idx]),
                "same_label": bool(y_values.iloc[left_idx] == y_values.iloc[right_idx]),
                "cosine_similarity": similarity,
            }
        )

    frame = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False).head(top_n)
    frame.to_csv(out_path, index=False)
    return frame


def adversarial_panel_shift(
    master: pd.DataFrame,
    panel: pd.DataFrame,
    panel_name: str,
    seed: int,
    folds: int = 5,
) -> dict:
    panel_ids = set(panel[ID_COL].astype(str))
    master_reference = master.loc[~master[ID_COL].astype(str).isin(panel_ids)].copy()

    master_X, _, _ = split_xy(master_reference)
    panel_X, _, _ = split_xy(panel)
    X = pd.concat([master_X, panel_X], ignore_index=True)
    y = pd.Series([0] * len(master_X) + [1] * len(panel_X), name="is_panel")

    min_class = int(y.value_counts().min())
    n_splits = max(2, min(folds, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.zeros(len(y), dtype=float)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        model = make_model_pipeline(
            X.iloc[train_idx],
            seed=seed + 1000 + fold_idx,
            fast=True,
            include_optional=False,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]

    auc = float(roc_auc_score(y, proba))
    return {
        "panel": panel_name,
        "master_reference_rows": len(master_reference),
        "panel_rows": len(panel),
        "overlap_removed_from_master": len(master) - len(master_reference),
        "adversarial_auc": auc,
        "shift_flag_auc_gt_0_6": bool(auc > 0.6),
    }
