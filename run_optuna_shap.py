#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.context")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ruvego.data import DATASET_FILES, ID_COL, PANEL_NAMES, discover_data_dir, load_datasets, split_xy
from ruvego.evaluation import choose_cv_config, cross_validate_oof, summarize_oof
from ruvego.features import ensure_dense, transformed_feature_names
from ruvego.models import make_model_pipeline, optional_model_availability


MODEL_WEIGHT_NAMES = ("lightgbm", "catboost", "hist_gbdt", "extra_trees", "random_forest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune LightGBM/CatBoost parameters and ensemble weights with Optuna, then generate SHAP reports."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_optuna"))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_FILES), default=list(DATASET_FILES))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=995380)
    parser.add_argument("--fast", action="store_true", help="Use lighter base estimators for quicker tuning.")
    parser.add_argument("--shap-sample-size", type=int, default=400)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument(
        "--reuse-foundation-dir",
        type=Path,
        default=Path("outputs_full"),
        help="Directory with existing panel foundation models. Used when present to avoid retraining them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    availability = optional_model_availability()
    if not availability.lightgbm or not availability.catboost or not availability.optuna:
        raise RuntimeError(
            "Optuna tuning requires lightgbm, catboost, and optuna. "
            f"Current availability: {availability}"
        )

    data_dir = args.data_dir.resolve() if args.data_dir else discover_data_dir(ROOT)
    output_dir = args.output_dir.resolve()
    dirs = prepare_output_dirs(output_dir)
    datasets = load_datasets(data_dir)
    master = datasets["MASTER"]

    run_config = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "datasets": args.datasets,
        "trials": args.trials,
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "fast": args.fast,
        "shap_sample_size": args.shap_sample_size,
        "optional_model_availability": availability.__dict__,
        "objective": "threshold-tuned OOF Macro F1",
        "tuned_parameters": {
            "lightgbm": [
                "n_estimators",
                "learning_rate",
                "num_leaves",
                "max_depth",
                "min_child_samples",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
            ],
            "catboost": [
                "iterations",
                "learning_rate",
                "depth",
                "l2_leaf_reg",
                "random_strength",
                "bagging_temperature",
                "border_count",
            ],
            "ensemble_weights": list(MODEL_WEIGHT_NAMES),
        },
    }
    write_json(dirs["reports"] / "optuna_run_config.json", run_config)

    summary_rows = []
    for dataset_idx, dataset_name in enumerate(args.datasets):
        print(f"\n=== Optuna tuning: {dataset_name} ===", flush=True)
        frame = datasets[dataset_name]
        X, y, ids = split_xy(frame)
        if dataset_name in PANEL_NAMES:
            X = add_foundation_feature(
                X=X,
                panel_ids=ids,
                master=master,
                panel_name=dataset_name,
                seed=args.seed + dataset_idx * 100,
                fast=args.fast,
                output_model_dir=dirs["models"],
                reuse_foundation_dir=args.reuse_foundation_dir,
            )

        cv_config = choose_cv_config(y, requested_folds=args.folds, requested_repeats=args.repeats)
        cv_seed = args.seed + dataset_idx * 1000
        print(
            f"Rows={len(X)}, pathogenic={int(y.sum())}, benign={int((1 - y).sum())}, "
            f"CV={cv_config.folds}x{cv_config.repeats}, trials={args.trials}",
            flush=True,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed + dataset_idx),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=max(5, args.trials // 4)),
            study_name=f"{dataset_name}_gbdt_weight_tuning",
        )

        def objective(trial: optuna.Trial) -> float:
            model_params, model_weights = suggest_trial_config(trial, fast=args.fast)

            def estimator_factory(seed: int):
                return make_model_pipeline(
                    X,
                    seed=seed,
                    fast=args.fast,
                    include_optional=True,
                    model_params=model_params,
                    model_weights=model_weights,
                )

            oof, _, threshold_report = cross_validate_oof(
                X=X,
                y=y,
                ids=ids,
                estimator_factory=estimator_factory,
                dataset_name=dataset_name,
                folds=cv_config.folds,
                repeats=cv_config.repeats,
                seed=cv_seed,
            )
            summary = summarize_oof(dataset_name, oof, threshold_report)
            trial.set_user_attr("best_threshold", threshold_report["best_threshold"])
            trial.set_user_attr("macro_f1", summary["macro_f1"])
            trial.set_user_attr("pr_auc", summary["pr_auc"])
            trial.set_user_attr("roc_auc", summary["roc_auc"])
            trial.set_user_attr("pathogenic_recall", summary["pathogenic_recall"])
            trial.set_user_attr("pathogenic_precision", summary["pathogenic_precision"])
            return float(summary["macro_f1"])

        study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
        save_study_outputs(study, dataset_name, dirs)

        best_params, best_weights = split_flat_params(study.best_trial.params)
        write_json(
            dirs["params"] / f"{dataset_name}_best_config.json",
            {
                "dataset": dataset_name,
                "best_value_macro_f1": study.best_value,
                "best_trial_number": study.best_trial.number,
                "model_params": best_params,
                "model_weights": best_weights,
                "cv_seed": cv_seed,
                "trial_user_attrs": study.best_trial.user_attrs,
            },
        )

        print(f"Best {dataset_name} Macro F1={study.best_value:.4f}", flush=True)
        print(f"Best weights: {best_weights}", flush=True)

        best_oof, best_fold_metrics, best_threshold_report = run_best_cv(
            X=X,
            y=y,
            ids=ids,
            dataset_name=dataset_name,
            model_params=best_params,
            model_weights=best_weights,
            folds=cv_config.folds,
            repeats=cv_config.repeats,
            seed=cv_seed,
            fast=args.fast,
        )
        best_summary = summarize_oof(dataset_name, best_oof, best_threshold_report)
        best_summary["best_trial_number"] = study.best_trial.number
        best_summary["optuna_objective_macro_f1"] = study.best_value
        summary_rows.append(best_summary)

        best_oof.to_csv(dirs["predictions"] / f"{dataset_name}_optuna_oof_predictions.csv", index=False)
        best_fold_metrics.to_csv(dirs["metrics"] / f"{dataset_name}_optuna_fold_metrics.csv", index=False)
        pd.DataFrame(best_threshold_report["threshold_grid"]).to_csv(
            dirs["metrics"] / f"{dataset_name}_optuna_threshold_grid.csv",
            index=False,
        )
        write_json(
            dirs["metrics"] / f"{dataset_name}_optuna_threshold_report.json",
            {key: value for key, value in best_threshold_report.items() if key != "threshold_grid"},
        )

        final_model = make_model_pipeline(
            X,
            seed=args.seed + dataset_idx * 3000 + 999,
            fast=args.fast,
            include_optional=True,
            model_params=best_params,
            model_weights=best_weights,
        )
        final_model.fit(X, y)
        joblib.dump(final_model, dirs["models"] / f"{dataset_name}_optuna_final_model.joblib")

        if not args.skip_shap:
            save_shap_reports(
                model=final_model,
                X=X,
                dataset_name=dataset_name,
                output_dir=dirs["shap"],
                sample_size=args.shap_sample_size,
                seed=args.seed + dataset_idx,
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(dirs["metrics"] / "optuna_cv_summary.csv", index=False)
    print("\n=== Optuna tuned CV summary ===")
    cols = ["dataset", "macro_f1", "pr_auc", "roc_auc", "best_threshold", "pathogenic_recall", "pathogenic_precision"]
    print(summary_df[cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def suggest_trial_config(trial: optuna.Trial, fast: bool) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    lightgbm_params = {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 160 if fast else 300, 610 if fast else 1100, step=50),
        "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.01, 0.12, log=True),
        "num_leaves": trial.suggest_int("lgbm_num_leaves", 12, 96),
        "max_depth": trial.suggest_categorical("lgbm_max_depth", [-1, 3, 4, 5, 6, 7, 8, 9, 10]),
        "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 5, 90),
        "subsample": trial.suggest_float("lgbm_subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.55, 1.0),
        "reg_alpha": trial.suggest_float("lgbm_reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-3, 30.0, log=True),
    }
    catboost_params = {
        "iterations": trial.suggest_int("cat_iterations", 160 if fast else 300, 610 if fast else 1000, step=50),
        "learning_rate": trial.suggest_float("cat_learning_rate", 0.01, 0.12, log=True),
        "depth": trial.suggest_int("cat_depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1e-2, 30.0, log=True),
        "random_strength": trial.suggest_float("cat_random_strength", 0.0, 3.0),
        "bagging_temperature": trial.suggest_float("cat_bagging_temperature", 0.0, 3.0),
        "border_count": trial.suggest_categorical("cat_border_count", [32, 64, 128, 254]),
    }
    weights = {
        name: trial.suggest_float(f"weight_{name}", 0.05, 3.0, log=True)
        for name in MODEL_WEIGHT_NAMES
    }
    return {"lightgbm": lightgbm_params, "catboost": catboost_params}, weights


def split_flat_params(params: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    lightgbm = {}
    catboost = {}
    weights = {}
    for key, value in params.items():
        if key.startswith("lgbm_"):
            lightgbm[key.removeprefix("lgbm_")] = value
        elif key.startswith("cat_"):
            catboost[key.removeprefix("cat_")] = value
        elif key.startswith("weight_"):
            weights[key.removeprefix("weight_")] = float(value)
    return {"lightgbm": lightgbm, "catboost": catboost}, weights


def run_best_cv(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    dataset_name: str,
    model_params: dict[str, dict[str, Any]],
    model_weights: dict[str, float],
    folds: int,
    repeats: int,
    seed: int,
    fast: bool,
):
    def estimator_factory(model_seed: int):
        return make_model_pipeline(
            X,
            seed=model_seed,
            fast=fast,
            include_optional=True,
            model_params=model_params,
            model_weights=model_weights,
        )

    return cross_validate_oof(
        X=X,
        y=y,
        ids=ids,
        estimator_factory=estimator_factory,
        dataset_name=dataset_name,
        folds=folds,
        repeats=repeats,
        seed=seed,
    )


def add_foundation_feature(
    X: pd.DataFrame,
    panel_ids: pd.Series,
    master: pd.DataFrame,
    panel_name: str,
    seed: int,
    fast: bool,
    output_model_dir: Path,
    reuse_foundation_dir: Path | None,
) -> pd.DataFrame:
    foundation_path = None
    if reuse_foundation_dir:
        candidate = reuse_foundation_dir / "models" / f"{panel_name}_foundation_model.joblib"
        if candidate.exists():
            foundation_path = candidate

    if foundation_path:
        foundation_model = joblib.load(foundation_path)
        print(f"Using existing foundation model for {panel_name}: {foundation_path}", flush=True)
    else:
        panel_id_set = set(panel_ids.astype(str))
        master_reference = master.loc[~master[ID_COL].astype(str).isin(panel_id_set)].copy()
        foundation_X, foundation_y, _ = split_xy(master_reference)
        foundation_model = make_model_pipeline(
            foundation_X,
            seed=seed,
            fast=fast,
            include_optional=True,
        )
        foundation_model.fit(foundation_X, foundation_y)
        joblib.dump(foundation_model, output_model_dir / f"{panel_name}_foundation_model.joblib")
        print(
            f"Trained foundation model for {panel_name} on {len(master_reference)} MASTER rows.",
            flush=True,
        )

    enriched = X.copy()
    enriched["GLOBAL_FOUNDATION_PROBA"] = foundation_model.predict_proba(X)[:, 1]
    return enriched


def save_study_outputs(study: optuna.Study, dataset_name: str, dirs: dict[str, Path]) -> None:
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials.to_csv(dirs["trials"] / f"{dataset_name}_optuna_trials.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    complete = trials.loc[trials["state"] == "COMPLETE"].copy()
    if not complete.empty:
        ax.plot(complete["number"], complete["value"], marker="o", linewidth=1.8)
        best_so_far = complete["value"].cummax()
        ax.plot(complete["number"], best_so_far, linestyle="--", linewidth=1.8, label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("OOF Macro F1")
    ax.set_title(f"{dataset_name} Optuna optimization history")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / f"{dataset_name}_optuna_history.png", dpi=180)
    plt.close(fig)


def save_shap_reports(
    model,
    X: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    feature_names = transformed_feature_names(preprocessor)
    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        sample_idx = np.sort(rng.choice(len(X), size=sample_size, replace=False))
        X_sample = X.iloc[sample_idx]
    else:
        X_sample = X
    transformed = ensure_dense(preprocessor.transform(X_sample))

    for model_name in ("lightgbm", "catboost"):
        if model_name not in classifier.named_estimators_:
            continue
        estimator = classifier.named_estimators_[model_name]
        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(transformed)
            values = select_binary_shap_values(shap_values)
            expected_value = select_binary_expected_value(explainer.expected_value)
        except Exception as exc:
            write_json(
                output_dir / f"{dataset_name}_{model_name}_shap_error.json",
                {"dataset": dataset_name, "model": model_name, "error": repr(exc)},
            )
            continue

        np.save(output_dir / f"{dataset_name}_{model_name}_shap_values.npy", values)
        summary = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        summary.to_csv(output_dir / f"{dataset_name}_{model_name}_shap_summary.csv", index=False)
        write_json(
            output_dir / f"{dataset_name}_{model_name}_shap_metadata.json",
            {
                "dataset": dataset_name,
                "model": model_name,
                "rows_explained": int(len(X_sample)),
                "features": int(len(feature_names)),
                "expected_value": expected_value,
            },
        )

        save_shap_bar(summary, dataset_name, model_name, output_dir)
        save_shap_beeswarm(values, transformed, feature_names, dataset_name, model_name, output_dir)


def select_binary_shap_values(shap_values):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def select_binary_expected_value(expected_value):
    if isinstance(expected_value, list):
        return float(expected_value[-1])
    values = np.asarray(expected_value)
    if values.ndim > 0:
        return float(values.reshape(-1)[-1])
    return float(values)


def save_shap_bar(summary: pd.DataFrame, dataset_name: str, model_name: str, output_dir: Path) -> None:
    top = summary.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2166ac" if model_name == "lightgbm" else "#b2182b")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{dataset_name} {model_name} SHAP feature importance")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset_name}_{model_name}_shap_bar.png", dpi=200)
    plt.close(fig)


def save_shap_beeswarm(
    values: np.ndarray,
    transformed: np.ndarray,
    feature_names: list[str],
    dataset_name: str,
    model_name: str,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(9, 7))
    shap.summary_plot(values, transformed, feature_names=feature_names, max_display=25, show=False)
    plt.title(f"{dataset_name} {model_name} SHAP summary", pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_{model_name}_shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()


def prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "metrics": output_dir / "metrics",
        "models": output_dir / "models",
        "predictions": output_dir / "predictions",
        "reports": output_dir / "reports",
        "figures": output_dir / "figures",
        "params": output_dir / "params",
        "trials": output_dir / "trials",
        "shap": output_dir / "shap",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
