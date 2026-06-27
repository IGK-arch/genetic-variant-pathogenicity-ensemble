from __future__ import annotations

from pathlib import Path

import pandas as pd


ID_COL = "Variant_ID"
LABEL_COL = "Label"

DATASET_FILES = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}

PANEL_NAMES = ("KANSER", "PAH", "CFTR")


def discover_data_dir(root: Path) -> Path:
    """Find the directory containing all expected train CSV files."""
    root = root.resolve()
    candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
    for candidate in candidates:
        if all((candidate / filename).exists() for filename in DATASET_FILES.values()):
            return candidate
    expected = ", ".join(DATASET_FILES.values())
    raise FileNotFoundError(f"Could not find train CSV files under {root}. Expected: {expected}")


def load_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for name, filename in DATASET_FILES.items():
        path = data_dir / filename
        frame = pd.read_csv(path)
        validate_dataset(frame, name, path)
        datasets[name] = frame
    return datasets


def validate_dataset(frame: pd.DataFrame, name: str, path: Path) -> None:
    missing = [col for col in (ID_COL, LABEL_COL) if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} ({path}) is missing required columns: {missing}")
    labels = set(frame[LABEL_COL].dropna().unique())
    if not labels <= {0, 1}:
        raise ValueError(f"{name} has non-binary labels: {sorted(labels)}")
    if frame[ID_COL].duplicated().any():
        dupes = frame.loc[frame[ID_COL].duplicated(), ID_COL].head(5).tolist()
        raise ValueError(f"{name} has duplicated Variant_ID values, examples: {dupes}")


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y = frame[LABEL_COL].astype(int)
    ids = frame[ID_COL].astype(str)
    X = frame.drop(columns=[ID_COL, LABEL_COL])
    return X, y, ids


def dataset_profile(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in datasets.items():
        label_counts = frame[LABEL_COL].value_counts().to_dict()
        rows.append(
            {
                "dataset": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "features": len(frame.columns) - 2,
                "pathogenic": int(label_counts.get(1, 0)),
                "benign": int(label_counts.get(0, 0)),
                "missing_cells": int(frame.isna().sum().sum()),
                "missing_rate": float(frame.isna().sum().sum() / frame.size),
                "categorical_features": int(
                    frame.drop(columns=[ID_COL, LABEL_COL]).select_dtypes(include="object").shape[1]
                ),
                "numeric_features": int(
                    frame.drop(columns=[ID_COL, LABEL_COL]).select_dtypes(exclude="object").shape[1]
                ),
            }
        )
    return pd.DataFrame(rows)


def overlap_profile(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    names = list(datasets)
    for i, left in enumerate(names):
        left_ids = set(datasets[left][ID_COL].astype(str))
        for right in names[i + 1 :]:
            right_ids = set(datasets[right][ID_COL].astype(str))
            rows.append(
                {
                    "left_dataset": left,
                    "right_dataset": right,
                    "overlap": len(left_ids & right_ids),
                    "left_rows": len(left_ids),
                    "right_rows": len(right_ids),
                }
            )
    return pd.DataFrame(rows)
