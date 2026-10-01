#!/usr/bin/env python3
"""
03_LLM_OBF_train_catboost.py

Stage 03 of the LLM deobfuscation machine-learning pipeline.

This stage runs AFTER:

    01_LLM_OBF_prepare_dataset.py
    02_LLM_OBF_create_train_test_splits.py


INPUT
=====

LLM_OBF_02_train_test_splits/

    seed_0000/
        train.csv
        test.csv

    seed_0001/
        train.csv
        test.csv

    ...


For EACH outer Stage-02 split:

    Stage-02 train.csv
            |
            v
    stratified internal split
            |
            +----------------------> INTERNAL VALIDATION
            |
            v
       INTERNAL TRAIN
            |
            v
       CatBoost training


The Stage-02 test.csv remains the OUTER TEST SET.

It is NEVER used for:

    - hyperparameter optimization
    - baseline-vs-optimized model selection
    - early stopping
    - selecting boosting iterations
    - choosing the winning model


MODEL MODES
===========

USE_HYPERPARAMETER_OPTIMIZATION = False

    -> train ordinary CatBoost baseline
    -> select baseline automatically
    -> refit baseline on full Stage-02 training dataset
    -> evaluate on outer test


USE_HYPERPARAMETER_OPTIMIZATION = True

    -> train ordinary CatBoost baseline
    -> run Bayesian hyperparameter optimization with scikit-optimize BayesSearchCV
    -> train optimized candidate
    -> compare baseline vs optimized candidate using INTERNAL VALIDATION ONLY
    -> select validation winner
    -> refit winning configuration on full Stage-02 training dataset
    -> evaluate winner on untouched Stage-02 test data


EVALUATION
==========

For relevant datasets the script reports:

    1. natural/full distribution

    2. repeated balanced undersampled evaluation

Balanced evaluation repeatedly samples an equal number of observations
from every class.

No synthetic observations are generated during evaluation.


OUTPUT
======

LLM_OBF_03_catboost_models/

    seed_0000/
        best_model.cbm
        best_model_parameters.json
        model_selection.csv
        metrics.csv
        predictions_test.csv
        predictions_validation.csv
        training_report.txt
        bayes_search_results.csv        # if optimization enabled

    seed_0001/
        ...

    global_training_summary.csv
    GLOBAL_REPORT.txt


DEPENDENCIES
============

    pandas
    numpy
    scikit-learn
    catboost
    scikit-optimize    only when optimization is enabled
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from skopt import BayesSearchCV
from skopt.space import Integer, Real


# =============================================================================
# PATHS
# =============================================================================

STAGE02_ROOT = Path(
    "LLM_OBF_02_train_test_splits"
)

STAGE03_ROOT = Path(
    "LLM_OBF_03_catboost_models"
)


TARGET_COLUMN = "class"


# =============================================================================
# WHICH SPLITS SHOULD BE PROCESSED?
# =============================================================================
#
# None:
#     process EVERY seed_* directory found in Stage 02.
#
# Example:
#
#     MAX_SPLITS = 10
#
# useful for testing the pipeline before launching all 1000 runs.
#

MAX_SPLITS: Optional[int] = None


# =============================================================================
# INTERNAL TRAIN / VALIDATION SPLIT
# =============================================================================

VALIDATION_SIZE = 0.20

STRATIFY_INTERNAL_SPLIT = True


# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

USE_HYPERPARAMETER_OPTIMIZATION = False


# Bayesian optimization through scikit-optimize / BayesSearchCV.
#
# BAYES_N_ITER controls how many parameter combinations are proposed by the
# Bayesian optimizer for EACH outer Stage-02 split.
#
# The uploaded Bayesian CatBoost reference script used 20 iterations.
#

BAYES_N_ITER = 50


# Number of cross-validation folds used INSIDE the internal training portion.
#
# IMPORTANT:
#
# The outer Stage-02 test set is never used here.
#
# The internal validation set is also not used during BayesSearchCV. It remains
# reserved for:
#
#     - early stopping of the final optimized candidate
#     - baseline-vs-optimized model selection
#

BAYES_CV = 3


# Parallel workers used by BayesSearchCV.
#
# CatBoost itself may also use multiple threads, so reduce this value if the
# machine becomes oversubscribed.
#

BAYES_N_JOBS = 3


# Metric used to:
#
#     1. optimize hyperparameters
#     2. compare optimized model against baseline
#
# Recommended for potentially imbalanced multiclass classification:
#
#     "macro_f1"
#

MODEL_SELECTION_METRIC = "macro_f1"


# =============================================================================
# CATBOOST BASELINE
# =============================================================================
#
# The baseline intentionally remains close to ordinary CatBoost.
#
# We set only reproducibility/runtime parameters explicitly.
#

BASELINE_PARAMETERS: Dict[str, Any] = {

    "loss_function": "MultiClass",

    "iterations": 1000,

    "random_seed": 42,

    "verbose": False,

    "allow_writing_files": False,
}


# =============================================================================
# EARLY STOPPING
# =============================================================================

USE_EARLY_STOPPING = True

EARLY_STOPPING_ROUNDS = 100


# =============================================================================
# CATBOOST EXECUTION
# =============================================================================

CATBOOST_THREAD_COUNT = -1

CATBOOST_TASK_TYPE = "CPU"

# Example GPU configuration:
#
# CATBOOST_TASK_TYPE = "GPU"
# CATBOOST_DEVICES = "0"

CATBOOST_DEVICES = "0"


# =============================================================================
# BAYESIAN CATBOOST SEARCH SPACE
# =============================================================================
#
# Search ranges are taken from the uploaded BayesSearchCV CatBoost script.
#
# They are intentionally broad:
#
#     depth
#         4 .. 12
#
#     iterations
#         500 .. 8000
#
#     learning_rate
#         0.01 .. 0.30
#         log-uniform
#
#     l2_leaf_reg
#         1.0 .. 20.0
#         log-uniform
#
#     border_count
#         32 .. 255
#
#
# BayesSearchCV does NOT exhaustively evaluate every possible combination.
# Instead, it sequentially proposes parameter configurations based on the
# performance of previous evaluations.
#
# With many outer Stage-02 splits this can still be computationally expensive.
# Therefore MAX_SPLITS can be used to test the pipeline first.
#

BAYES_SEARCH_SPACE = {

    "depth":
        Integer(
            4,
            12,
        ),

    "iterations":
        Integer(
            500,
            8000,
        ),

    "learning_rate":
        Real(
            0.01,
            0.30,
            prior="log-uniform",
        ),

    "l2_leaf_reg":
        Real(
            1.0,
            20.0,
            prior="log-uniform",
        ),

    "border_count":
        Integer(
            32,
            255,
        ),
}


# =============================================================================
# BALANCED EVALUATION
# =============================================================================
#
# Validation and test are evaluated BOTH:
#
#     - normally
#     - using repeated balanced undersampling
#
# For each balanced iteration:
#
#     smallest class size
#         *
#     BALANCED_EVAL_MINORITY_FRACTION
#
# examples are sampled independently from every class.
#

N_BALANCED_EVAL_ITERATIONS = 100

BALANCED_EVAL_MINORITY_FRACTION = 0.90


# =============================================================================
# OUTPUT
# =============================================================================

SAVE_VALIDATION_PREDICTIONS = True

SAVE_TEST_PREDICTIONS = True

SAVE_FINAL_TRAIN_PREDICTIONS = False

OVERWRITE_EXISTING_SEED_OUTPUT = True


# =============================================================================
# RANDOMNESS
# =============================================================================

INTERNAL_SPLIT_SEED_OFFSET = 100_000

BALANCED_VALIDATION_SEED_OFFSET = 200_000

BALANCED_TEST_SEED_OFFSET = 300_000


# =============================================================================
# JSON HELPERS
# =============================================================================

def write_json(
    path: Path,
    payload: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


# =============================================================================
# FEATURE HANDLING
# =============================================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> List[str]:

    return [
        column
        for column in df.columns
        if column != TARGET_COLUMN
    ]


def detect_categorical_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> List[str]:
    """
    CatBoost directly supports symbolic/string categorical features.

    No arbitrary integer encoding is performed.
    """

    categorical: List[str] = []

    for feature in feature_columns:

        series = df[feature]

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):

            categorical.append(feature)

    return categorical


def prepare_catboost_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """
    Prepare feature matrix without changing the underlying model semantics.

    Numeric:
        +/- infinity -> NaN

    Categorical:
        missing -> special string
        convert to string

    CatBoost handles numeric NaN natively.
    """

    X = df[
        list(feature_columns)
    ].copy()

    categorical_set = set(
        categorical_features
    )

    for column in feature_columns:

        if column in categorical_set:

            X[column] = (
                X[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )

        else:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

            X[column] = X[column].replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )

    return X


# =============================================================================
# TARGET HANDLING
# =============================================================================

def prepare_target(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype("string")
        .str.strip()
        .astype(str)
    )


# =============================================================================
# CATBOOST PARAMETERS
# =============================================================================

def make_catboost_parameters(
    parameters: Dict[str, Any],
    random_seed: int,
) -> Dict[str, Any]:

    result = dict(
        parameters
    )

    result["random_seed"] = int(
        random_seed
    )

    result["verbose"] = False

    result["allow_writing_files"] = False

    result["thread_count"] = (
        CATBOOST_THREAD_COUNT
    )

    result["task_type"] = (
        CATBOOST_TASK_TYPE
    )

    if (
        CATBOOST_TASK_TYPE.upper()
        == "GPU"
    ):

        result["devices"] = (
            CATBOOST_DEVICES
        )

    return result


# =============================================================================
# PROBABILITY HANDLING
# =============================================================================

def predict_probabilities(
    model: CatBoostClassifier,
    X: pd.DataFrame,
) -> np.ndarray:

    probabilities = model.predict_proba(
        X
    )

    return np.asarray(
        probabilities,
        dtype=float,
    )


def predict_classes(
    model: CatBoostClassifier,
    X: pd.DataFrame,
) -> np.ndarray:

    prediction = model.predict(
        X
    )

    prediction = np.asarray(
        prediction
    )

    if prediction.ndim > 1:

        prediction = prediction.ravel()

    return prediction.astype(str)


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    model_classes: Sequence[Any],
) -> Dict[str, float]:

    y_true_array = np.asarray(
        y_true
    ).astype(str)

    y_pred_array = np.asarray(
        y_pred
    ).astype(str)

    classes = [
        str(value)
        for value in model_classes
    ]

    result: Dict[str, float] = {}

    result[
        "accuracy"
    ] = float(
        accuracy_score(
            y_true_array,
            y_pred_array,
        )
    )

    result[
        "balanced_accuracy"
    ] = float(
        balanced_accuracy_score(
            y_true_array,
            y_pred_array,
        )
    )

    result[
        "macro_precision"
    ] = float(
        precision_score(
            y_true_array,
            y_pred_array,
            average="macro",
            zero_division=0,
        )
    )

    result[
        "macro_recall"
    ] = float(
        recall_score(
            y_true_array,
            y_pred_array,
            average="macro",
            zero_division=0,
        )
    )

    result[
        "macro_f1"
    ] = float(
        f1_score(
            y_true_array,
            y_pred_array,
            average="macro",
            zero_division=0,
        )
    )

    result[
        "weighted_f1"
    ] = float(
        f1_score(
            y_true_array,
            y_pred_array,
            average="weighted",
            zero_division=0,
        )
    )

    result[
        "mcc"
    ] = float(
        matthews_corrcoef(
            y_true_array,
            y_pred_array,
        )
    )

    # -------------------------------------------------------------------------
    # Log loss
    # -------------------------------------------------------------------------

    try:

        result[
            "log_loss"
        ] = float(
            log_loss(
                y_true_array,
                probabilities,
                labels=classes,
            )
        )

    except Exception:

        result[
            "log_loss"
        ] = np.nan

    # -------------------------------------------------------------------------
    # ROC AUC
    # -------------------------------------------------------------------------

    try:

        if len(classes) == 2:

            positive_class = classes[1]

            positive_index = classes.index(
                positive_class
            )

            y_binary = (
                y_true_array
                == positive_class
            ).astype(int)

            result[
                "roc_auc"
            ] = float(
                roc_auc_score(
                    y_binary,
                    probabilities[
                        :,
                        positive_index
                    ],
                )
            )

        else:

            result[
                "roc_auc"
            ] = float(
                roc_auc_score(
                    y_true_array,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="weighted",
                )
            )

    except Exception:

        result[
            "roc_auc"
        ] = np.nan

    return result


# =============================================================================
# EVALUATE MODEL
# =============================================================================

def evaluate_model(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[
    Dict[str, float],
    np.ndarray,
    np.ndarray,
]:

    predictions = predict_classes(
        model,
        X,
    )

    probabilities = predict_probabilities(
        model,
        X,
    )

    metrics = calculate_metrics(
        y_true=y,
        y_pred=predictions,
        probabilities=probabilities,
        model_classes=model.classes_,
    )

    return (
        metrics,
        predictions,
        probabilities,
    )


# =============================================================================
# BALANCED UNDERSAMPLED EVALUATION
# =============================================================================

def balanced_indices(
    y: pd.Series,
    random_seed: int,
) -> np.ndarray:

    y_array = np.asarray(
        y
    )

    classes, counts = np.unique(
        y_array,
        return_counts=True,
    )

    if len(classes) < 2:

        raise ValueError(
            "Balanced evaluation requires at least two classes."
        )

    smallest_class = int(
        counts.min()
    )

    sample_size = int(
        math.floor(
            smallest_class
            * BALANCED_EVAL_MINORITY_FRACTION
        )
    )

    sample_size = max(
        1,
        sample_size,
    )

    rng = np.random.default_rng(
        random_seed
    )

    selected: List[np.ndarray] = []

    for class_value in classes:

        class_indices = np.flatnonzero(
            y_array == class_value
        )

        chosen = rng.choice(
            class_indices,
            size=sample_size,
            replace=False,
        )

        selected.append(
            chosen
        )

    combined = np.concatenate(
        selected
    )

    rng.shuffle(
        combined
    )

    return combined


def repeated_balanced_evaluation(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    seed_base: int,
) -> Tuple[
    Dict[str, float],
    pd.DataFrame,
]:

    rows: List[
        Dict[str, float]
    ] = []

    for iteration in range(
        N_BALANCED_EVAL_ITERATIONS
    ):

        indices = balanced_indices(
            y,
            random_seed=(
                seed_base
                + iteration
            ),
        )

        X_sample = (
            X.iloc[
                indices
            ]
            .reset_index(
                drop=True
            )
        )

        y_sample = (
            y.iloc[
                indices
            ]
            .reset_index(
                drop=True
            )
        )

        metrics, _, _ = evaluate_model(
            model,
            X_sample,
            y_sample,
        )

        metrics[
            "iteration"
        ] = iteration

        metrics[
            "n"
        ] = len(
            y_sample
        )

        rows.append(
            metrics
        )

    table = pd.DataFrame(
        rows
    )

    summary: Dict[str, float] = {}

    metric_columns = [
        column
        for column in table.columns
        if column not in {
            "iteration",
            "n",
        }
    ]

    for column in metric_columns:

        summary[
            f"{column}_mean"
        ] = float(
            table[column].mean()
        )

        summary[
            f"{column}_std"
        ] = float(
            table[column].std()
        )

    summary[
        "n_iterations"
    ] = int(
        len(table)
    )

    return (
        summary,
        table,
    )


# =============================================================================
# MODEL SELECTION SCORE
# =============================================================================

def get_selection_score(
    metrics: Dict[str, float],
) -> float:

    if MODEL_SELECTION_METRIC not in metrics:

        raise ValueError(
            f"Unknown MODEL_SELECTION_METRIC: "
            f"{MODEL_SELECTION_METRIC}"
        )

    value = metrics[
        MODEL_SELECTION_METRIC
    ]

    if pd.isna(
        value
    ):

        return -np.inf

    return float(
        value
    )


# =============================================================================
# TRAIN CANDIDATE MODEL
# =============================================================================

def train_candidate_model(
    parameters: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_features: Sequence[str],
    random_seed: int,
) -> Tuple[
    CatBoostClassifier,
    int,
]:

    params = make_catboost_parameters(
        parameters,
        random_seed,
    )

    model = CatBoostClassifier(
        **params
    )

    fit_kwargs: Dict[
        str,
        Any
    ] = {

        "X":
            X_train,

        "y":
            y_train,

        "cat_features":
            list(
                categorical_features
            ),
    }

    if USE_EARLY_STOPPING:

        fit_kwargs[
            "eval_set"
        ] = (
            X_validation,
            y_validation,
        )

        fit_kwargs[
            "early_stopping_rounds"
        ] = (
            EARLY_STOPPING_ROUNDS
        )

        fit_kwargs[
            "use_best_model"
        ] = True

    model.fit(
        **fit_kwargs
    )

    best_iteration = model.get_best_iteration()

    if (
        best_iteration is None
        or best_iteration < 0
    ):

        best_iteration = int(
            params.get(
                "iterations",
                1000,
            )
        ) - 1

    return (
        model,
        int(best_iteration),
    )


# =============================================================================
# BASELINE MODEL
# =============================================================================

def train_baseline_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_features: Sequence[str],
    random_seed: int,
) -> Tuple[
    CatBoostClassifier,
    Dict[str, Any],
    int,
]:

    parameters = dict(
        BASELINE_PARAMETERS
    )

    model, best_iteration = train_candidate_model(
        parameters=parameters,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        categorical_features=categorical_features,
        random_seed=random_seed,
    )

    return (
        model,
        parameters,
        best_iteration,
    )


# =============================================================================
# BAYESIAN OPTIMIZATION - SCIKIT-OPTIMIZE
# =============================================================================

def optimize_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_features: Sequence[str],
    random_seed: int,
    output_dir: Path,
) -> Tuple[
    Dict[str, Any],
    pd.DataFrame,
]:
    """
    Optimize CatBoost hyperparameters with scikit-optimize BayesSearchCV.

    IMPORTANT METHODOLOGICAL RULE
    -----------------------------

    BayesSearchCV operates ONLY on X_train / y_train.

    The separately held-out internal validation set:

        X_validation
        y_validation

    is NOT passed into BayesSearchCV.

    It remains reserved for:

        - early stopping after the Bayesian search
        - comparison of baseline vs optimized candidate

    The outer Stage-02 test set remains completely untouched.
    """

    if BAYES_N_ITER <= 0:

        raise ValueError(
            "BAYES_N_ITER must be >= 1."
        )

    if BAYES_CV < 2:

        raise ValueError(
            "BAYES_CV must be >= 2."
        )

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------
    #
    # The Stage-03 script uses macro-F1 as its default model-selection metric.
    #
    # BayesSearchCV therefore uses the corresponding sklearn scorer.
    # -------------------------------------------------------------------------

    if MODEL_SELECTION_METRIC == "macro_f1":

        bayes_scorer = make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        )

    elif MODEL_SELECTION_METRIC == "balanced_accuracy":

        bayes_scorer = "balanced_accuracy"

    elif MODEL_SELECTION_METRIC == "accuracy":

        bayes_scorer = "accuracy"

    else:

        raise ValueError(
            "\nBayesSearchCV scoring is not configured for "
            f"MODEL_SELECTION_METRIC={MODEL_SELECTION_METRIC!r}.\n\n"
            "Supported values:\n"
            "    macro_f1\n"
            "    balanced_accuracy\n"
            "    accuracy\n"
        )

    # -------------------------------------------------------------------------
    # CatBoost estimator used INSIDE BayesSearchCV.
    #
    # Early stopping is deliberately NOT used during CV because every CV fold
    # has its own validation partition. After the best parameter configuration
    # is found, train_candidate_model() performs the normal Stage-03 early
    # stopping against the dedicated internal validation set.
    # -------------------------------------------------------------------------

    base_parameters = make_catboost_parameters(
        {
            "loss_function":
                "MultiClass",
        },
        random_seed,
    )

    base_estimator = CatBoostClassifier(
        **base_parameters
    )

    # -------------------------------------------------------------------------
    # Deterministic stratified CV inside X_train only.
    # -------------------------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=BAYES_CV,
        shuffle=True,
        random_state=random_seed,
    )

    bayes = BayesSearchCV(
        estimator=base_estimator,
        search_spaces=BAYES_SEARCH_SPACE,
        n_iter=BAYES_N_ITER,
        scoring=bayes_scorer,
        cv=cv,
        n_jobs=BAYES_N_JOBS,
        random_state=random_seed,
        verbose=0,
        refit=True,
        return_train_score=True,
    )

    fit_kwargs: Dict[
        str,
        Any
    ] = {}

    if categorical_features:

        fit_kwargs[
            "cat_features"
        ] = list(
            categorical_features
        )

    bayes.fit(
        X_train,
        y_train,
        **fit_kwargs,
    )

    # -------------------------------------------------------------------------
    # Save complete Bayesian search history.
    # -------------------------------------------------------------------------

    search_results = pd.DataFrame(
        bayes.cv_results_
    )

    search_results.to_csv(
        output_dir
        / "bayes_search_results.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Extract best parameter configuration.
    #
    # Runtime/reproducibility settings are added later by
    # make_catboost_parameters().
    # -------------------------------------------------------------------------

    best_parameters = {
        key:
            (
                value.item()
                if isinstance(
                    value,
                    np.generic,
                )
                else value
            )
        for key, value in dict(
            bayes.best_params_
        ).items()
    }

    best_parameters[
        "loss_function"
    ] = "MultiClass"

    # -------------------------------------------------------------------------
    # Persist a compact search summary as JSON.
    # -------------------------------------------------------------------------

    write_json(
        output_dir
        / "bayes_search_best.json",
        {

            "best_score":
                float(
                    bayes.best_score_
                ),

            "best_parameters":
                best_parameters,

            "n_iterations":
                BAYES_N_ITER,

            "cv_folds":
                BAYES_CV,

            "model_selection_metric":
                MODEL_SELECTION_METRIC,

            "search_space":
                {
                    key:
                        str(
                            value
                        )
                    for key, value in BAYES_SEARCH_SPACE.items()
                },
        },
    )

    return (
        best_parameters,
        search_results,
    )


# =============================================================================
# FINAL REFIT
# =============================================================================

def refit_final_model(
    parameters: Dict[str, Any],
    best_iteration: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: Sequence[str],
    random_seed: int,
) -> Tuple[
    CatBoostClassifier,
    Dict[str, Any],
]:

    final_parameters = dict(
        parameters
    )

    # Freeze boosting length selected using internal validation.

    final_parameters[
        "iterations"
    ] = max(
        1,
        int(best_iteration) + 1,
    )

    final_parameters = make_catboost_parameters(
        final_parameters,
        random_seed,
    )

    model = CatBoostClassifier(
        **final_parameters
    )

    model.fit(
        X_train,
        y_train,
        cat_features=list(
            categorical_features
        ),
    )

    return (
        model,
        final_parameters,
    )


# =============================================================================
# PREDICTION CSV
# =============================================================================

def save_predictions(
    path: Path,
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[Any],
) -> None:

    result = pd.DataFrame(
        {
            "true_class":
                np.asarray(
                    y_true
                ).astype(str),

            "predicted_class":
                np.asarray(
                    predictions
                ).astype(str),
        }
    )

    classes = [
        str(value)
        for value in classes
    ]

    for index, class_name in enumerate(
        classes
    ):

        safe_class = (
            class_name
            .replace("/", "_")
            .replace(" ", "_")
        )

        result[
            f"probability_{safe_class}"
        ] = probabilities[
            :,
            index
        ]

    result.to_csv(
        path,
        index=False,
    )


# =============================================================================
# METRICS TABLE HELPER
# =============================================================================

def add_metrics_row(
    rows: List[Dict[str, Any]],
    dataset_name: str,
    evaluation_type: str,
    metrics: Dict[str, Any],
) -> None:

    row: Dict[
        str,
        Any
    ] = {

        "dataset":
            dataset_name,

        "evaluation_type":
            evaluation_type,
    }

    row.update(
        metrics
    )

    rows.append(
        row
    )


# =============================================================================
# REPORT FORMATTING
# =============================================================================

def metric_text(
    value: Any,
) -> str:

    try:

        if pd.isna(
            value
        ):
            return "NaN"

        return f"{float(value):.6f}"

    except Exception:

        return str(
            value
        )


def append_metric_block(
    lines: List[str],
    name: str,
    metrics: Dict[str, Any],
) -> None:

    lines.append(
        name
    )

    lines.append(
        "-" * 80
    )

    preferred_order = [

        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "roc_auc",
        "log_loss",
    ]

    for key in preferred_order:

        if key in metrics:

            lines.append(
                f"{key:<30}"
                f"{metric_text(metrics[key])}"
            )

    lines.append("")


# =============================================================================
# PROCESS ONE OUTER SPLIT
# =============================================================================

def process_seed_directory(
    seed_directory: Path,
    outer_seed: int,
) -> Dict[str, Any]:

    output_directory = (
        STAGE03_ROOT
        / seed_directory.name
    )

    if (
        output_directory.exists()
        and OVERWRITE_EXISTING_SEED_OUTPUT
    ):

        shutil.rmtree(
            output_directory
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        seed_directory
        / "train.csv"
    )

    test_path = (
        seed_directory
        / "test.csv"
    )

    if not train_path.exists():

        raise FileNotFoundError(
            train_path
        )

    if not test_path.exists():

        raise FileNotFoundError(
            test_path
        )

    outer_train_df = pd.read_csv(
        train_path,
        low_memory=False,
    )

    outer_test_df = pd.read_csv(
        test_path,
        low_memory=False,
    )

    if (
        TARGET_COLUMN
        not in outer_train_df.columns
    ):

        raise ValueError(
            f"{TARGET_COLUMN!r} missing from "
            f"{train_path}"
        )

    if (
        TARGET_COLUMN
        not in outer_test_df.columns
    ):

        raise ValueError(
            f"{TARGET_COLUMN!r} missing from "
            f"{test_path}"
        )

    # =========================================================================
    # FEATURE CONTRACT
    # =========================================================================

    feature_columns = get_feature_columns(
        outer_train_df
    )

    test_features = get_feature_columns(
        outer_test_df
    )

    if feature_columns != test_features:

        raise ValueError(
            f"Train/test feature mismatch in "
            f"{seed_directory}"
        )

    categorical_features = (
        detect_categorical_features(
            outer_train_df,
            feature_columns,
        )
    )

    # =========================================================================
    # PREPARE OUTER DATA
    # =========================================================================

    X_outer_train = prepare_catboost_features(
        outer_train_df,
        feature_columns,
        categorical_features,
    )

    y_outer_train = prepare_target(
        outer_train_df[
            TARGET_COLUMN
        ]
    )

    X_outer_test = prepare_catboost_features(
        outer_test_df,
        feature_columns,
        categorical_features,
    )

    y_outer_test = prepare_target(
        outer_test_df[
            TARGET_COLUMN
        ]
    )

    # =========================================================================
    # INTERNAL TRAIN / VALIDATION
    # =========================================================================

    internal_seed = (
        INTERNAL_SPLIT_SEED_OFFSET
        + outer_seed
    )

    stratify_vector = (
        y_outer_train
        if STRATIFY_INTERNAL_SPLIT
        else None
    )

    (
        X_internal_train,
        X_validation,
        y_internal_train,
        y_validation,
    ) = train_test_split(
        X_outer_train,
        y_outer_train,
        test_size=VALIDATION_SIZE,
        random_state=internal_seed,
        stratify=stratify_vector,
    )

    # =========================================================================
    # BASELINE CANDIDATE
    # =========================================================================

    (
        baseline_model,
        baseline_parameters,
        baseline_best_iteration,
    ) = train_baseline_candidate(
        X_train=X_internal_train,
        y_train=y_internal_train,
        X_validation=X_validation,
        y_validation=y_validation,
        categorical_features=categorical_features,
        random_seed=outer_seed,
    )

    (
        baseline_validation_metrics,
        baseline_validation_predictions,
        baseline_validation_probabilities,
    ) = evaluate_model(
        baseline_model,
        X_validation,
        y_validation,
    )

    (
        baseline_validation_balanced,
        baseline_validation_balanced_iterations,
    ) = repeated_balanced_evaluation(
        baseline_model,
        X_validation,
        y_validation,
        seed_base=(
            BALANCED_VALIDATION_SEED_OFFSET
            + outer_seed * 1000
        ),
    )

    # =========================================================================
    # DEFAULT SELECTION
    # =========================================================================

    winning_model_name = (
        "catboost_baseline"
    )

    winning_parameters = dict(
        baseline_parameters
    )

    winning_best_iteration = (
        baseline_best_iteration
    )

    winning_validation_metrics = dict(
        baseline_validation_metrics
    )

    model_selection_rows: List[
        Dict[str, Any]
    ] = [

        {

            "model":
                "catboost_baseline",

            "selection_metric":
                MODEL_SELECTION_METRIC,

            "selection_score":
                get_selection_score(
                    baseline_validation_metrics
                ),

            "best_iteration":
                baseline_best_iteration,

            **baseline_validation_metrics,
        }

    ]

    optimized_validation_metrics = None
    optimized_validation_balanced = None
    optimized_parameters = None
    optimized_best_iteration = None

    # =========================================================================
    # BAYESIAN OPTIMIZATION
    # =========================================================================

    if USE_HYPERPARAMETER_OPTIMIZATION:

        (
            optimized_parameters,
            _,
        ) = optimize_catboost(
            X_train=X_internal_train,
            y_train=y_internal_train,
            X_validation=X_validation,
            y_validation=y_validation,
            categorical_features=categorical_features,
            random_seed=outer_seed,
            output_dir=output_directory,
        )

        (
            optimized_model,
            optimized_best_iteration,
        ) = train_candidate_model(
            parameters=optimized_parameters,
            X_train=X_internal_train,
            y_train=y_internal_train,
            X_validation=X_validation,
            y_validation=y_validation,
            categorical_features=categorical_features,
            random_seed=outer_seed,
        )

        (
            optimized_validation_metrics,
            _,
            _,
        ) = evaluate_model(
            optimized_model,
            X_validation,
            y_validation,
        )

        (
            optimized_validation_balanced,
            optimized_validation_balanced_iterations,
        ) = repeated_balanced_evaluation(
            optimized_model,
            X_validation,
            y_validation,
            seed_base=(
                BALANCED_VALIDATION_SEED_OFFSET
                + 10_000_000
                + outer_seed * 1000
            ),
        )

        optimized_validation_balanced_iterations.to_csv(
            output_directory
            / "optimized_validation_balanced_iterations.csv",
            index=False,
        )

        optimized_score = get_selection_score(
            optimized_validation_metrics
        )

        baseline_score = get_selection_score(
            baseline_validation_metrics
        )

        model_selection_rows.append(
            {

                "model":
                    "catboost_optimized",

                "selection_metric":
                    MODEL_SELECTION_METRIC,

                "selection_score":
                    optimized_score,

                "best_iteration":
                    optimized_best_iteration,

                **optimized_validation_metrics,
            }
        )

        # ---------------------------------------------------------------------
        # CRITICAL:
        #
        # Winner is determined ONLY from internal validation.
        #
        # The outer Stage-02 test set is NOT examined here.
        # ---------------------------------------------------------------------

        if optimized_score > baseline_score:

            winning_model_name = (
                "catboost_optimized"
            )

            winning_parameters = dict(
                optimized_parameters
            )

            winning_best_iteration = int(
                optimized_best_iteration
            )

            winning_validation_metrics = dict(
                optimized_validation_metrics
            )

    # =========================================================================
    # SAVE MODEL-SELECTION TABLE
    # =========================================================================

    model_selection_df = pd.DataFrame(
        model_selection_rows
    )

    model_selection_df[
        "selected"
    ] = (
        model_selection_df[
            "model"
        ]
        == winning_model_name
    )

    model_selection_df.to_csv(
        output_directory
        / "model_selection.csv",
        index=False,
    )

    # =========================================================================
    # REFIT WINNER ON COMPLETE OUTER TRAINING DATA
    # =========================================================================

    (
        final_model,
        final_parameters,
    ) = refit_final_model(
        parameters=winning_parameters,
        best_iteration=winning_best_iteration,
        X_train=X_outer_train,
        y_train=y_outer_train,
        categorical_features=categorical_features,
        random_seed=outer_seed,
    )

    # =========================================================================
    # SAVE FINAL MODEL
    # =========================================================================

    final_model.save_model(
        str(
            output_directory
            / "best_model.cbm"
        )
    )

    write_json(
        output_directory
        / "best_model_parameters.json",
        {

            "outer_seed":
                outer_seed,

            "selected_model":
                winning_model_name,

            "selection_metric":
                MODEL_SELECTION_METRIC,

            "selected_validation_score":
                get_selection_score(
                    winning_validation_metrics
                ),

            "selected_best_iteration":
                winning_best_iteration,

            "final_parameters":
                final_parameters,

            "categorical_features":
                categorical_features,

            "feature_columns":
                feature_columns,
        },
    )

    # =========================================================================
    # FINAL TRAIN EVALUATION
    # =========================================================================

    (
        final_train_metrics,
        final_train_predictions,
        final_train_probabilities,
    ) = evaluate_model(
        final_model,
        X_outer_train,
        y_outer_train,
    )

    (
        final_train_balanced,
        final_train_balanced_iterations,
    ) = repeated_balanced_evaluation(
        final_model,
        X_outer_train,
        y_outer_train,
        seed_base=(
            BALANCED_VALIDATION_SEED_OFFSET
            + 20_000_000
            + outer_seed * 1000
        ),
    )

    # =========================================================================
    # OUTER TEST EVALUATION
    # =========================================================================

    (
        final_test_metrics,
        final_test_predictions,
        final_test_probabilities,
    ) = evaluate_model(
        final_model,
        X_outer_test,
        y_outer_test,
    )

    (
        final_test_balanced,
        final_test_balanced_iterations,
    ) = repeated_balanced_evaluation(
        final_model,
        X_outer_test,
        y_outer_test,
        seed_base=(
            BALANCED_TEST_SEED_OFFSET
            + outer_seed * 1000
        ),
    )

    # =========================================================================
    # SAVE BALANCED ITERATION TABLES
    # =========================================================================

    baseline_validation_balanced_iterations.to_csv(
        output_directory
        / "baseline_validation_balanced_iterations.csv",
        index=False,
    )

    final_train_balanced_iterations.to_csv(
        output_directory
        / "final_train_balanced_iterations.csv",
        index=False,
    )

    final_test_balanced_iterations.to_csv(
        output_directory
        / "final_test_balanced_iterations.csv",
        index=False,
    )

    # =========================================================================
    # SAVE PREDICTIONS
    # =========================================================================

    if SAVE_VALIDATION_PREDICTIONS:

        save_predictions(
            output_directory
            / "predictions_validation_baseline.csv",
            y_validation,
            baseline_validation_predictions,
            baseline_validation_probabilities,
            baseline_model.classes_,
        )

    if SAVE_TEST_PREDICTIONS:

        save_predictions(
            output_directory
            / "predictions_test.csv",
            y_outer_test,
            final_test_predictions,
            final_test_probabilities,
            final_model.classes_,
        )

    if SAVE_FINAL_TRAIN_PREDICTIONS:

        save_predictions(
            output_directory
            / "predictions_train.csv",
            y_outer_train,
            final_train_predictions,
            final_train_probabilities,
            final_model.classes_,
        )

    # =========================================================================
    # METRICS TABLE
    # =========================================================================

    metrics_rows: List[
        Dict[str, Any]
    ] = []

    add_metrics_row(
        metrics_rows,
        "internal_validation_baseline",
        "natural",
        baseline_validation_metrics,
    )

    add_metrics_row(
        metrics_rows,
        "internal_validation_baseline",
        "balanced_undersampling",
        baseline_validation_balanced,
    )

    if optimized_validation_metrics is not None:

        add_metrics_row(
            metrics_rows,
            "internal_validation_optimized",
            "natural",
            optimized_validation_metrics,
        )

        add_metrics_row(
            metrics_rows,
            "internal_validation_optimized",
            "balanced_undersampling",
            optimized_validation_balanced,
        )

    add_metrics_row(
        metrics_rows,
        "final_outer_training",
        "natural",
        final_train_metrics,
    )

    add_metrics_row(
        metrics_rows,
        "final_outer_training",
        "balanced_undersampling",
        final_train_balanced,
    )

    add_metrics_row(
        metrics_rows,
        "final_outer_test",
        "natural",
        final_test_metrics,
    )

    add_metrics_row(
        metrics_rows,
        "final_outer_test",
        "balanced_undersampling",
        final_test_balanced,
    )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_df.to_csv(
        output_directory
        / "metrics.csv",
        index=False,
    )

    # =========================================================================
    # CLASSIFICATION REPORT
    # =========================================================================

    test_classification_report = (
        classification_report(
            np.asarray(
                y_outer_test
            ).astype(str),
            final_test_predictions,
            zero_division=0,
        )
    )

    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================

    labels = [
        str(value)
        for value in final_model.classes_
    ]

    test_confusion = confusion_matrix(
        np.asarray(
            y_outer_test
        ).astype(str),
        final_test_predictions,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        test_confusion,
        index=[
            f"true_{label}"
            for label in labels
        ],
        columns=[
            f"pred_{label}"
            for label in labels
        ],
    )

    confusion_df.to_csv(
        output_directory
        / "test_confusion_matrix.csv"
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    report: List[str] = []

    separator = "=" * 90

    report.append(
        separator
    )

    report.append(
        "LLM DEOBFUSCATION - CATBOOST TRAINING REPORT"
    )

    report.append(
        separator
    )

    report.append("")

    report.append(
        f"Outer random seed:              {outer_seed}"
    )

    report.append(
        f"Outer training rows:            {len(X_outer_train):,}"
    )

    report.append(
        f"Outer test rows:                {len(X_outer_test):,}"
    )

    report.append(
        f"Internal training rows:         {len(X_internal_train):,}"
    )

    report.append(
        f"Internal validation rows:       {len(X_validation):,}"
    )

    report.append(
        f"Number of features:             {len(feature_columns):,}"
    )

    report.append(
        f"Categorical features:           {len(categorical_features):,}"
    )

    report.append(
        f"Optimization enabled:           {USE_HYPERPARAMETER_OPTIMIZATION}"
    )

    report.append(
        f"Model-selection metric:         {MODEL_SELECTION_METRIC}"
    )

    report.append(
        f"Selected model:                 {winning_model_name}"
    )

    report.append(
        f"Selected best iteration:        {winning_best_iteration}"
    )

    report.append("")

    report.append(
        separator
    )

    report.append(
        "MODEL SELECTION"
    )

    report.append(
        separator
    )

    report.append("")

    report.append(
        "The baseline and optimized models are compared using INTERNAL VALIDATION "
        "only. The outer Stage-02 test set is not used for model selection."
    )

    report.append("")

    append_metric_block(
        report,
        "BASELINE - INTERNAL VALIDATION",
        baseline_validation_metrics,
    )

    if optimized_validation_metrics is not None:

        append_metric_block(
            report,
            "OPTIMIZED - INTERNAL VALIDATION",
            optimized_validation_metrics,
        )

    report.append(
        f"WINNER: {winning_model_name}"
    )

    report.append("")

    report.append(
        separator
    )

    report.append(
        "FINAL REFITTED MODEL"
    )

    report.append(
        separator
    )

    report.append("")

    append_metric_block(
        report,
        "OUTER TRAINING - NATURAL/FULL",
        final_train_metrics,
    )

    append_metric_block(
        report,
        "OUTER TRAINING - BALANCED UNDERSAMPLED",
        final_train_balanced,
    )

    append_metric_block(
        report,
        "OUTER TEST - NATURAL/FULL",
        final_test_metrics,
    )

    append_metric_block(
        report,
        "OUTER TEST - BALANCED UNDERSAMPLED",
        final_test_balanced,
    )

    report.append(
        separator
    )

    report.append(
        "OUTER TEST CLASSIFICATION REPORT"
    )

    report.append(
        separator
    )

    report.append("")

    report.append(
        test_classification_report
    )

    report.append("")

    report.append(
        separator
    )

    report.append(
        "FINAL PARAMETERS"
    )

    report.append(
        separator
    )

    report.append("")

    report.append(
        json.dumps(
            final_parameters,
            indent=2,
            default=str,
        )
    )

    report.append("")

    (
        output_directory
        / "training_report.txt"
    ).write_text(
        "\n".join(
            report
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # GLOBAL SUMMARY ROW
    # =========================================================================

    summary: Dict[
        str,
        Any
    ] = {

        "outer_seed":
            outer_seed,

        "selected_model":
            winning_model_name,

        "optimization_enabled":
            USE_HYPERPARAMETER_OPTIMIZATION,

        "baseline_validation_macro_f1":
            baseline_validation_metrics[
                "macro_f1"
            ],

        "baseline_validation_balanced_accuracy":
            baseline_validation_metrics[
                "balanced_accuracy"
            ],

        "selected_validation_macro_f1":
            winning_validation_metrics[
                "macro_f1"
            ],

        "selected_validation_balanced_accuracy":
            winning_validation_metrics[
                "balanced_accuracy"
            ],

        "train_accuracy":
            final_train_metrics[
                "accuracy"
            ],

        "train_balanced_accuracy":
            final_train_metrics[
                "balanced_accuracy"
            ],

        "train_macro_f1":
            final_train_metrics[
                "macro_f1"
            ],

        "test_accuracy":
            final_test_metrics[
                "accuracy"
            ],

        "test_balanced_accuracy":
            final_test_metrics[
                "balanced_accuracy"
            ],

        "test_macro_f1":
            final_test_metrics[
                "macro_f1"
            ],

        "test_weighted_f1":
            final_test_metrics[
                "weighted_f1"
            ],

        "test_mcc":
            final_test_metrics[
                "mcc"
            ],

        "test_roc_auc":
            final_test_metrics[
                "roc_auc"
            ],

        "test_log_loss":
            final_test_metrics[
                "log_loss"
            ],

        "balanced_test_accuracy_mean":
            final_test_balanced[
                "accuracy_mean"
            ],

        "balanced_test_accuracy_std":
            final_test_balanced[
                "accuracy_std"
            ],

        "balanced_test_macro_f1_mean":
            final_test_balanced[
                "macro_f1_mean"
            ],

        "balanced_test_macro_f1_std":
            final_test_balanced[
                "macro_f1_std"
            ],

        "balanced_test_balanced_accuracy_mean":
            final_test_balanced[
                "balanced_accuracy_mean"
            ],

        "balanced_test_balanced_accuracy_std":
            final_test_balanced[
                "balanced_accuracy_std"
            ],

        "best_iteration":
            winning_best_iteration,

        "n_outer_train":
            len(
                X_outer_train
            ),

        "n_outer_test":
            len(
                X_outer_test
            ),

        "n_internal_train":
            len(
                X_internal_train
            ),

        "n_validation":
            len(
                X_validation
            ),
    }

    if optimized_validation_metrics is not None:

        summary[
            "optimized_validation_macro_f1"
        ] = optimized_validation_metrics[
            "macro_f1"
        ]

        summary[
            "optimized_validation_balanced_accuracy"
        ] = optimized_validation_metrics[
            "balanced_accuracy"
        ]

    return summary


# =============================================================================
# DISCOVER STAGE-02 SPLITS
# =============================================================================

def discover_seed_directories() -> List[
    Tuple[int, Path]
]:

    results: List[
        Tuple[int, Path]
    ] = []

    for path in STAGE02_ROOT.glob(
        "seed_*"
    ):

        if not path.is_dir():

            continue

        try:

            seed = int(
                path.name.split(
                    "seed_",
                    1,
                )[1]
            )

        except Exception:

            continue

        if (
            (path / "train.csv").exists()
            and
            (path / "test.csv").exists()
        ):

            results.append(
                (
                    seed,
                    path,
                )
            )

    results.sort(
        key=lambda item: item[0]
    )

    if MAX_SPLITS is not None:

        results = results[
            :MAX_SPLITS
        ]

    return results


# =============================================================================
# GLOBAL REPORT
# =============================================================================

def create_global_report(
    summary: pd.DataFrame,
) -> str:

    lines: List[str] = []

    separator = "=" * 90

    lines.append(
        separator
    )

    lines.append(
        "LLM DEOBFUSCATION - GLOBAL CATBOOST REPORT"
    )

    lines.append(
        separator
    )

    lines.append("")

    lines.append(
        f"Completed outer splits: "
        f"{len(summary):,}"
    )

    lines.append(
        f"Hyperparameter optimization: "
        f"{USE_HYPERPARAMETER_OPTIMIZATION}"
    )

    lines.append(
        f"Bayesian search iterations per split: "
        f"{BAYES_N_ITER if USE_HYPERPARAMETER_OPTIMIZATION else 0}"
    )

    lines.append(
        f"Bayesian CV folds: "
        f"{BAYES_CV if USE_HYPERPARAMETER_OPTIMIZATION else 0}"
    )

    lines.append(
        f"Selection metric: "
        f"{MODEL_SELECTION_METRIC}"
    )

    lines.append("")

    lines.append(
        separator
    )

    lines.append(
        "AGGREGATED OUTER TEST PERFORMANCE"
    )

    lines.append(
        separator
    )

    lines.append("")

    metrics = [

        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_mcc",
        "test_roc_auc",
        "test_log_loss",

        "balanced_test_accuracy_mean",
        "balanced_test_macro_f1_mean",
        "balanced_test_balanced_accuracy_mean",
    ]

    for metric in metrics:

        if metric not in summary.columns:

            continue

        values = pd.to_numeric(
            summary[
                metric
            ],
            errors="coerce",
        )

        lines.append(
            f"{metric}"
        )

        lines.append(
            f"    mean:   "
            f"{values.mean():.6f}"
        )

        lines.append(
            f"    std:    "
            f"{values.std():.6f}"
        )

        lines.append(
            f"    median: "
            f"{values.median():.6f}"
        )

        lines.append(
            f"    min:    "
            f"{values.min():.6f}"
        )

        lines.append(
            f"    max:    "
            f"{values.max():.6f}"
        )

        lines.append("")

    if (
        "selected_model"
        in summary.columns
    ):

        lines.append(
            separator
        )

        lines.append(
            "SELECTED MODEL COUNTS"
        )

        lines.append(
            separator
        )

        lines.append("")

        counts = summary[
            "selected_model"
        ].value_counts()

        for model_name, count in counts.items():

            fraction = (
                100.0
                * count
                / len(summary)
            )

            lines.append(
                f"{model_name:<35}"
                f"{count:>10,}"
                f"{fraction:>12.2f}%"
            )

    lines.append("")

    return "\n".join(
        lines
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    if not STAGE02_ROOT.exists():

        raise SystemExit(
            f"\nStage-02 directory does not exist:\n"
            f"    {STAGE02_ROOT.resolve()}\n"
        )

    if not 0.0 < VALIDATION_SIZE < 1.0:

        raise ValueError(
            "VALIDATION_SIZE must be between 0 and 1."
        )

    if not (
        0.0
        < BALANCED_EVAL_MINORITY_FRACTION
        <= 1.0
    ):

        raise ValueError(
            "BALANCED_EVAL_MINORITY_FRACTION "
            "must be in (0,1]."
        )

    STAGE03_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_directories = (
        discover_seed_directories()
    )

    if not seed_directories:

        raise SystemExit(
            "\nNo valid seed_* directories containing "
            "train.csv and test.csv were found."
        )

    print(
        "=" * 90
    )

    print(
        "LLM OBFUSCATION ML - STAGE 03"
    )

    print(
        "CATBOOST TRAINING"
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"Outer splits:              "
        f"{len(seed_directories):,}"
    )

    print(
        f"Validation fraction:       "
        f"{VALIDATION_SIZE:.2%}"
    )

    print(
        f"Hyperparameter optimization: "
        f"{USE_HYPERPARAMETER_OPTIMIZATION}"
    )

    print(
        f"Selection metric:          "
        f"{MODEL_SELECTION_METRIC}"
    )

    if USE_HYPERPARAMETER_OPTIMIZATION:

        print(
            f"Bayes iterations/split:    "
            f"{BAYES_N_ITER}"
        )

        print(
            f"Bayes CV folds:            "
            f"{BAYES_CV}"
        )

    print()

    summary_rows: List[
        Dict[str, Any]
    ] = []

    failed_rows: List[
        Dict[str, Any]
    ] = []

    for iteration, (
        outer_seed,
        seed_directory,
    ) in enumerate(
        seed_directories,
        start=1,
    ):

        print(
            f"[{iteration:>5,}/"
            f"{len(seed_directories):,}] "
            f"seed={outer_seed}"
        )

        try:

            summary = (
                process_seed_directory(
                    seed_directory,
                    outer_seed,
                )
            )

            summary_rows.append(
                summary
            )

            print(
                f"         "
                f"winner={summary['selected_model']} | "
                f"test macro-F1="
                f"{summary['test_macro_f1']:.4f} | "
                f"balanced accuracy="
                f"{summary['test_balanced_accuracy']:.4f}"
            )

        except Exception as exc:

            print(
                f"         FAILED: {exc}"
            )

            failed_rows.append(
                {
                    "outer_seed":
                        outer_seed,

                    "error":
                        str(exc),
                }
            )

    # =========================================================================
    # GLOBAL SUMMARY
    # =========================================================================

    if not summary_rows:

        raise SystemExit(
            "\nNo CatBoost runs completed successfully."
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        STAGE03_ROOT
        / "global_training_summary.csv",
        index=False,
    )

    if failed_rows:

        pd.DataFrame(
            failed_rows
        ).to_csv(
            STAGE03_ROOT
            / "failed_splits.csv",
            index=False,
        )

    global_report = create_global_report(
        summary_df
    )

    (
        STAGE03_ROOT
        / "GLOBAL_REPORT.txt"
    ).write_text(
        global_report,
        encoding="utf-8",
    )

    print()
    print(
        "=" * 90
    )

    print(
        "STAGE 03 COMPLETE"
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"Successful runs: "
        f"{len(summary_rows):,}"
    )

    print(
        f"Failed runs:     "
        f"{len(failed_rows):,}"
    )

    print(
        f"Output:          "
        f"{STAGE03_ROOT}"
    )

    print(
        f"Summary:         "
        f"{STAGE03_ROOT / 'global_training_summary.csv'}"
    )

    print(
        f"Report:          "
        f"{STAGE03_ROOT / 'GLOBAL_REPORT.txt'}"
    )


if __name__ == "__main__":
    main()