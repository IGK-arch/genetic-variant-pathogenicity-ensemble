from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [col for col in X.columns if col not in categorical]
    return numeric, categorical


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_cols, categorical_cols = column_groups(X)
    transformers = []

    if numeric_cols:
        numeric_steps = [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            )
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", RobustScaler()))
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns found.")

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{idx}" for idx in range(_transformed_feature_count(preprocessor))]


def _transformed_feature_count(preprocessor: ColumnTransformer) -> int:
    total = 0
    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if transformer == "passthrough":
            total += len(columns)
            continue
        try:
            total += len(transformer.get_feature_names_out())
        except Exception:
            total += len(columns)
    return int(total)


def ensure_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)
