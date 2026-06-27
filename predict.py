#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from io import StringIO
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.context")

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ruvego.data import ID_COL, LABEL_COL, PANEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RUVEGO predictions from trained artifacts.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with Variant_ID and feature columns.")
    parser.add_argument("--dataset", choices=["MASTER", "KANSER", "PAH", "CFTR"], required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts_dir.resolve()
    model_path = artifacts / "models" / f"{args.dataset}_final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    frame = pd.read_csv(args.input)
    if ID_COL not in frame.columns:
        raise ValueError(f"Input is missing required column: {ID_COL}")

    ids = frame[ID_COL].astype(str)
    X = frame.drop(columns=[ID_COL, LABEL_COL], errors="ignore")

    if args.dataset in PANEL_NAMES:
        foundation_path = artifacts / "models" / f"{args.dataset}_foundation_model.joblib"
        if not foundation_path.exists():
            raise FileNotFoundError(f"Missing foundation model for panel prediction: {foundation_path}")
        foundation_model = joblib.load(foundation_path)
        X = X.copy()
        X["GLOBAL_FOUNDATION_PROBA"] = predict_positive_proba(foundation_model, X)

    model = joblib.load(model_path)
    proba = predict_positive_proba(model, X)
    threshold = args.threshold if args.threshold is not None else load_threshold(artifacts, args.dataset)
    pred = (proba >= threshold).astype(int)

    output = pd.DataFrame(
        {
            ID_COL: ids,
            "pathogenic_probability": proba,
            "prediction": pred,
            "threshold": threshold,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote predictions: {args.output}")


def load_threshold(artifacts: Path, dataset: str) -> float:
    path = artifacts / "metrics" / f"{dataset}_threshold_report.json"
    if not path.exists():
        return 0.5
    report = json.loads(path.read_text(encoding="utf-8"))
    return float(report.get("best_threshold", 0.5))


def predict_positive_proba(model, X):
    buffer = StringIO()
    with redirect_stderr(buffer):
        return model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    main()
