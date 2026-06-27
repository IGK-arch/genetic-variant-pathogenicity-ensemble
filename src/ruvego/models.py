from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.pipeline import Pipeline

from .features import make_preprocessor


@dataclass(frozen=True)
class ModelAvailability:
    lightgbm: bool
    catboost: bool
    shap: bool
    optuna: bool


def optional_model_availability() -> ModelAvailability:
    return ModelAvailability(
        lightgbm=importlib.util.find_spec("lightgbm") is not None,
        catboost=importlib.util.find_spec("catboost") is not None,
        shap=importlib.util.find_spec("shap") is not None,
        optuna=importlib.util.find_spec("optuna") is not None,
    )


def make_model_pipeline(
    X: pd.DataFrame,
    seed: int,
    fast: bool = False,
    include_optional: bool = True,
) -> Pipeline:
    estimators = make_estimators(seed=seed, fast=fast, include_optional=include_optional)
    classifier = VotingClassifier(estimators=estimators, voting="soft", n_jobs=None)
    return Pipeline(
        [
            ("preprocess", make_preprocessor(X, scale_numeric=True)),
            ("classifier", classifier),
        ]
    )


def make_estimators(seed: int, fast: bool = False, include_optional: bool = True):
    n_tree = 180 if fast else 320
    hgb_iter = 160 if fast else 260
    estimators = []

    availability = optional_model_availability()
    if include_optional and availability.lightgbm:
        from lightgbm import LGBMClassifier

        estimators.append(
            (
                "lightgbm",
                LGBMClassifier(
                    objective="binary",
                    class_weight="balanced",
                    n_estimators=450 if fast else 900,
                    learning_rate=0.035,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_lambda=2.0,
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            )
        )

    if include_optional and availability.catboost:
        from catboost import CatBoostClassifier

        estimators.append(
            (
                "catboost",
                CatBoostClassifier(
                    iterations=450 if fast else 900,
                    learning_rate=0.035,
                    depth=6,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    auto_class_weights="Balanced",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                ),
            )
        )

    estimators.extend(
        [
            (
                "hist_gbdt",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.045,
                    max_iter=hgb_iter,
                    max_leaf_nodes=31,
                    min_samples_leaf=12,
                    l2_regularization=0.1,
                    class_weight="balanced",
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=seed,
                ),
            ),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=n_tree,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    bootstrap=False,
                    n_jobs=1,
                    random_state=seed + 17,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=max(120, n_tree - 80),
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=seed + 31,
                ),
            ),
        ]
    )
    return estimators
