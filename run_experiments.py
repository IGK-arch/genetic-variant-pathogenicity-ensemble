#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.context")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ruvego.data import DATASET_FILES, ID_COL, PANEL_NAMES, dataset_profile, discover_data_dir, load_datasets, overlap_profile, split_xy
from ruvego.diagnostics import (
    adversarial_panel_shift,
    save_calibration_curve,
    save_feature_importance,
    save_low_confidence,
    save_near_duplicate_report,
    save_pr_curve,
)
from ruvego.evaluation import choose_cv_config, cross_validate_oof, summarize_oof
from ruvego.models import make_model_pipeline, optional_model_availability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RUVEGO pathogenicity prediction experiments.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing YARISMA_TRAIN_*.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated artifacts.")
    parser.add_argument("--folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated CV repeats.")
    parser.add_argument("--seed", type=int, default=995380, help="Random seed.")
    parser.add_argument("--fast", action="store_true", help="Use smaller ensembles for quick iteration.")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip plots, feature importance, and duplicate scans.")
    parser.add_argument("--skip-adversarial", action="store_true", help="Skip panel-vs-master adversarial validation.")
    parser.add_argument(
        "--no-optional-models",
        action="store_true",
        help="Do not use LightGBM/CatBoost even if installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    dirs = prepare_output_dirs(output_dir)

    data_dir = args.data_dir.resolve() if args.data_dir else discover_data_dir(ROOT)
    datasets = load_datasets(data_dir)

    profile = dataset_profile(datasets)
    overlaps = overlap_profile(datasets)
    profile.to_csv(dirs["reports"] / "data_profile.csv", index=False)
    overlaps.to_csv(dirs["reports"] / "variant_id_overlaps.csv", index=False)

    availability = optional_model_availability()
    include_optional = not args.no_optional_models
    run_config = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "fast": args.fast,
        "include_optional_models": include_optional,
        "optional_model_availability": availability.__dict__,
        "dataset_files": DATASET_FILES,
    }
    write_json(dirs["reports"] / "run_config.json", run_config)

    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Optional models: {availability}")
    print(profile.to_string(index=False))

    summary_rows = []
    all_fold_metrics = []
    all_threshold_rows = []

    master = datasets["MASTER"]
    for dataset_idx, dataset_name in enumerate(DATASET_FILES):
        print(f"\n=== {dataset_name} ===", flush=True)
        frame = datasets[dataset_name]
        X, y, ids = split_xy(frame)

        foundation_model = None
        if dataset_name in PANEL_NAMES:
            X, foundation_model = add_foundation_feature(
                X=X,
                panel_ids=ids,
                master=master,
                panel_name=dataset_name,
                seed=args.seed + dataset_idx * 100,
                fast=args.fast,
                include_optional=include_optional,
                model_dir=dirs["models"],
            )

        cv_config = choose_cv_config(y, requested_folds=args.folds, requested_repeats=args.repeats)
        print(
            f"Rows={len(frame)}, pathogenic={int(y.sum())}, benign={int((1 - y).sum())}, "
            f"CV={cv_config.folds}x{cv_config.repeats}",
            flush=True,
        )

        def estimator_factory(seed: int):
            return make_model_pipeline(
                X,
                seed=seed,
                fast=args.fast,
                include_optional=include_optional,
            )

        oof, fold_metrics, threshold_report = cross_validate_oof(
            X=X,
            y=y,
            ids=ids,
            estimator_factory=estimator_factory,
            dataset_name=dataset_name,
            folds=cv_config.folds,
            repeats=cv_config.repeats,
            seed=args.seed + dataset_idx * 1000,
        )

        summary = summarize_oof(dataset_name, oof, threshold_report)
        summary["cv_folds"] = cv_config.folds
        summary["cv_repeats"] = cv_config.repeats
        summary["used_foundation_feature"] = dataset_name in PANEL_NAMES
        summary_rows.append(summary)
        all_fold_metrics.append(fold_metrics)
        threshold_rows = pd.DataFrame(threshold_report["threshold_grid"])
        threshold_rows.insert(0, "dataset", dataset_name)
        all_threshold_rows.append(threshold_rows)

        oof.to_csv(dirs["predictions"] / f"{dataset_name}_oof_predictions.csv", index=False)
        fold_metrics.to_csv(dirs["metrics"] / f"{dataset_name}_fold_metrics.csv", index=False)
        write_json(
            dirs["metrics"] / f"{dataset_name}_threshold_report.json",
            {key: value for key, value in threshold_report.items() if key != "threshold_grid"},
        )
        threshold_rows.to_csv(dirs["metrics"] / f"{dataset_name}_threshold_grid.csv", index=False)

        final_model = make_model_pipeline(
            X,
            seed=args.seed + dataset_idx * 2000 + 777,
            fast=args.fast,
            include_optional=include_optional,
        )
        final_model.fit(X, y)
        joblib.dump(final_model, dirs["models"] / f"{dataset_name}_final_model.joblib")

        if not args.skip_diagnostics:
            save_pr_curve(oof, dirs["figures"] / f"{dataset_name}_pr_curve.png")
            save_calibration_curve(oof, dirs["figures"] / f"{dataset_name}_calibration.png")
            save_low_confidence(oof, dirs["reports"] / f"{dataset_name}_low_confidence.csv")
            save_feature_importance(final_model, dirs["reports"] / f"{dataset_name}_feature_importance.csv")
            save_near_duplicate_report(
                X=X,
                y=y,
                ids=ids,
                fitted_model=final_model,
                out_path=dirs["reports"] / f"{dataset_name}_near_duplicates.csv",
            )

        print(
            f"OOF Macro F1={summary['macro_f1']:.4f}, PR-AUC={summary['pr_auc']:.4f}, "
            f"best threshold={summary['best_threshold']:.2f}",
            flush=True,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(dirs["metrics"] / "cv_summary.csv", index=False)
    pd.concat(all_fold_metrics, ignore_index=True).to_csv(dirs["metrics"] / "fold_metrics_all.csv", index=False)
    pd.concat(all_threshold_rows, ignore_index=True).to_csv(dirs["metrics"] / "threshold_grid_all.csv", index=False)

    if not args.skip_adversarial:
        shift_rows = []
        for panel_name in PANEL_NAMES:
            print(f"Adversarial validation: MASTER vs {panel_name}", flush=True)
            shift_rows.append(
                adversarial_panel_shift(
                    master=master,
                    panel=datasets[panel_name],
                    panel_name=panel_name,
                    seed=args.seed + 5000,
                    folds=args.folds,
                )
            )
        pd.DataFrame(shift_rows).to_csv(dirs["metrics"] / "adversarial_panel_shift.csv", index=False)

    print("\n=== CV summary ===")
    cols = [
        "dataset",
        "rows",
        "macro_f1",
        "pathogenic_f1",
        "pathogenic_precision",
        "pathogenic_recall",
        "pr_auc",
        "roc_auc",
        "best_threshold",
        "uncertain_zone_rows",
    ]
    print(summary_df[cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def add_foundation_feature(
    X: pd.DataFrame,
    panel_ids: pd.Series,
    master: pd.DataFrame,
    panel_name: str,
    seed: int,
    fast: bool,
    include_optional: bool,
    model_dir: Path,
) -> tuple[pd.DataFrame, object]:
    panel_id_set = set(panel_ids.astype(str))
    master_reference = master.loc[~master[ID_COL].astype(str).isin(panel_id_set)].copy()
    foundation_X, foundation_y, _ = split_xy(master_reference)
    foundation_model = make_model_pipeline(
        foundation_X,
        seed=seed,
        fast=fast,
        include_optional=include_optional,
    )
    foundation_model.fit(foundation_X, foundation_y)
    foundation_proba = foundation_model.predict_proba(X)[:, 1]

    enriched = X.copy()
    enriched["GLOBAL_FOUNDATION_PROBA"] = foundation_proba
    joblib.dump(foundation_model, model_dir / f"{panel_name}_foundation_model.joblib")

    print(
        f"Foundation feature for {panel_name}: trained on {len(master_reference)} MASTER rows "
        f"after removing {len(master) - len(master_reference)} overlapping IDs.",
        flush=True,
    )
    return enriched, foundation_model


def prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "metrics": output_dir / "metrics",
        "models": output_dir / "models",
        "predictions": output_dir / "predictions",
        "reports": output_dir / "reports",
        "figures": output_dir / "figures",
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
