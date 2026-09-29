#!/usr/bin/env python3
"""
04_LLM_OBF_final_catboost_analysis.py

Final post-training analysis and explainability stage for the
LLM deobfuscation CatBoost pipeline.

Expected preceding stages
=========================

01_LLM_OBF_prepare_dataset.py

02_LLM_OBF_create_train_test_splits.py

03_LLM_OBF_train_catboost.py


Expected input structure
========================

LLM_OBF_01_prepared_dataset/
    LLM_OBF_dataset.csv

LLM_OBF_02_train_test_splits/
    seed_0000/
        train.csv
        test.csv
    seed_0001/
        train.csv
        test.csv
    ...

LLM_OBF_03_catboost_models/
    seed_0000/
        best_model.cbm
        best_model_parameters.json
        ...
    seed_0001/
        best_model.cbm
        best_model_parameters.json
        ...
    ...


Analyses
========

1. PERFORMANCE
   - accuracy
   - balanced accuracy
   - macro precision
   - macro recall
   - macro F1
   - weighted F1
   - MCC
   - ROC AUC
   - log loss

   calculated separately for:
       - training data
       - testing data

   and summarized across ALL random splits as:
       mean
       standard deviation
       median
       minimum
       maximum


2. CONFUSION MATRICES

   For every split:
       - ordinary absolute confusion matrix
       - row-normalized confusion matrix

   Globally:
       - mean confusion matrix
       - standard deviation
       - mean row-normalized confusion matrix
       - standard deviation

   Row normalization corresponds to normalization with respect to the
   TRUE class.


3. CATBOOST FEATURE IMPORTANCE

   Native CatBoost PredictionValuesChange importance.

   For every model:
       - absolute/native importance
       - normalized relative importance

   Across all models:
       - mean
       - standard deviation
       - median
       - min
       - max
       - mean normalized importance


4. SHAP

   Native CatBoost SHAP values.

   Calculated separately for:
       - training data
       - testing data

   For each split:
       - mean absolute SHAP per feature

   Across splits:
       - mean mean-|SHAP|
       - standard deviation
       - normalized relative importance
       - ranking

   Also produces a global SHAP beeswarm-style plot from a controlled
   sample of observations from the individual splits.


5. ICE / PDP

   For each selected numeric feature:

       for every sampled observation:
           change feature over a common global grid
           keep all other features fixed
           calculate model probability

   Per split:
       - mean ICE = PDP
       - standard deviation across observations

   Across ALL splits:
       - mean PDP
       - standard deviation between splits

   This means the global ICE/PDP figures explicitly show variation
   between the independently trained models/splits.


6. FEATURE DISTRIBUTIONS BY PREDICTED CLASS

   For every numeric feature:
       - boxplot grouped by predicted class
       - train and test separately
       - per-split mean
       - per-split median
       - per-split standard deviation
       - global mean +/- between-split standard deviation


7. HUMAN-READABLE REPORTS

   Per split:
       seed_xxxx/analysis_report.txt

   Global:
       GLOBAL_ANALYSIS_REPORT.txt


Dependencies
============

Required:

    numpy
    pandas
    matplotlib
    scikit-learn
    catboost

No external SHAP package is required because this script uses
CatBoost's native SHAP implementation.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from catboost import (
    CatBoostClassifier,
    Pool,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# COLOR SCHEME
# =============================================================================
#
# MAXIMUM FIVE COLORS.
#
# Change ONLY these hex codes if you want another visual style.
#
# Every plot in this script uses this palette.
#
# =============================================================================

COLORS: List[str] = [
    "#183153",   # dark blue
    "#2878B5",   # blue
    "#5BA3C6",   # light blue
    "#D9A441",   # ochre
    "#B44C43",   # muted red
]


if len(COLORS) > 5:
    raise ValueError(
        "COLORS may contain a maximum of five colors."
    )


CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "custom_confusion",
    [
        "#FFFFFF",
        COLORS[2],
        COLORS[1],
        COLORS[0],
    ],
)


SHAP_CMAP = LinearSegmentedColormap.from_list(
    "custom_shap",
    [
        COLORS[1],
        COLORS[2],
        COLORS[3],
        COLORS[4],
    ],
)


# =============================================================================
# GLOBAL MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update(
    {
        "figure.figsize": (10.0, 6.0),
        "figure.dpi": 120,

        "savefig.dpi": 250,
        "savefig.bbox": "tight",

        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,

        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "legend.fontsize": 9,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": False,
    }
)


# =============================================================================
# PATHS
# =============================================================================

STAGE01_CSV = Path(
    "LLM_OBF_01_prepared_dataset"
) / "LLM_OBF_dataset.csv"


STAGE02_ROOT = Path(
    "LLM_OBF_02_train_test_splits"
)


STAGE03_ROOT = Path(
    "LLM_OBF_03_catboost_models"
)


OUTPUT_ROOT = Path(
    "LLM_OBF_04_final_analysis"
)


TARGET_COLUMN = "class"


# =============================================================================
# WHICH MODELS TO ANALYZE?
# =============================================================================
#
# None:
#     analyze every model found.
#
# Example:
#
# MAX_SEEDS = 10
#
# useful for testing the script before analyzing all 1000 models.
#

MAX_SEEDS: Optional[int] = None


# =============================================================================
# DATASETS
# =============================================================================

ANALYZE_TRAIN = True

ANALYZE_TEST = True


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

ENABLE_FEATURE_IMPORTANCE = True

FEATURE_IMPORTANCE_TOP_N = 30


# =============================================================================
# SHAP SETTINGS
# =============================================================================

ENABLE_SHAP = True


# Number of rows PER SPLIT used to calculate SHAP.
#
# None:
#     all rows
#
# For 1000 models this can become extremely expensive.
#

SHAP_MAX_ROWS_PER_DATASET: Optional[int] = 1000


# Small subset retained from each split for the global beeswarm plot.

SHAP_BEESWARM_ROWS_PER_SPLIT = 50


SHAP_BEESWARM_TOP_N = 20


# Class whose SHAP values should be visualized.
#
# None:
#     - if class "1" exists, use "1"
#     - otherwise use the final model class
#
# For multiclass problems this can explicitly be changed, e.g.
#
# SHAP_EXPLAIN_CLASS = "Virtualize"
#

SHAP_EXPLAIN_CLASS: Optional[str] = None


# =============================================================================
# ICE SETTINGS
# =============================================================================

ENABLE_ICE = True


# None:
#     analyze EVERY numeric feature.
#
# Or explicitly:
#
# ICE_FEATURES = [
#     "similarity.codebleu",
#     "simplification.decrease_nloc",
# ]
#

ICE_FEATURES: Optional[List[str]] = None


ICE_DATASETS = [
    "train",
    "test",
]


# Number of rows per split contributing ICE curves.

ICE_SAMPLE_ROWS = 150


# Number of values along each ICE feature grid.

ICE_GRID_POINTS = 25


# Avoid extreme tails dominating ICE.

ICE_GRID_LOWER_QUANTILE = 0.05

ICE_GRID_UPPER_QUANTILE = 0.95


# Probability class analyzed by ICE.
#
# None follows the same automatic logic as SHAP.

ICE_EXPLAIN_CLASS: Optional[str] = None


# =============================================================================
# FEATURE-DISTRIBUTION SETTINGS
# =============================================================================

ENABLE_FEATURE_DISTRIBUTIONS = True


# To avoid creating gigantic repeated datasets when 1000 splits exist,
# only this many rows per split are retained for GLOBAL pooled boxplots.

DISTRIBUTION_ROWS_PER_SPLIT = 500


SHOW_BOXPLOT_OUTLIERS = False


# =============================================================================
# OUTPUT
# =============================================================================

SAVE_PNG = True

SAVE_EPS = True

PLOT_DPI = 250


SAVE_PER_SEED_CSV = True

SAVE_PER_SEED_REPORT = True


# Per-seed plots can create enormous numbers of files across 1000 seeds.
#
# Global plots are ALWAYS produced.
#
# Recommended:
#     False
#

SAVE_PER_SEED_PLOTS = False


OVERWRITE_OUTPUT = True


# =============================================================================
# RANDOMNESS
# =============================================================================

RANDOM_STATE = 42


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_dir(
    path: Path,
) -> None:

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_name(
    value: Any,
) -> str:

    result = str(
        value
    )

    for old, new in [
        ("/", "_"),
        ("\\", "_"),
        (" ", "_"),
        (".", "_"),
        (":", "_"),
        (";", "_"),
        ("|", "_"),
    ]:

        result = result.replace(
            old,
            new,
        )

    return result


def write_json(
    path: Path,
    payload: Any,
) -> None:

    ensure_dir(
        path.parent
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


def save_figure(
    fig: plt.Figure,
    base_path: Path,
) -> None:

    ensure_dir(
        base_path.parent
    )

    if SAVE_PNG:

        fig.savefig(
            base_path.with_suffix(
                ".png"
            ),
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

    if SAVE_EPS:

        fig.savefig(
            base_path.with_suffix(
                ".eps"
            ),
            bbox_inches="tight",
        )

    plt.close(
        fig
    )


# =============================================================================
# DISCOVER MODELS
# =============================================================================

def discover_seed_directories() -> List[
    Tuple[int, Path, Path]
]:

    results: List[
        Tuple[int, Path, Path]
    ] = []

    for model_dir in STAGE03_ROOT.glob(
        "seed_*"
    ):

        if not model_dir.is_dir():

            continue

        try:

            seed = int(
                model_dir.name.split(
                    "seed_",
                    1,
                )[1]
            )

        except Exception:

            continue

        model_path = (
            model_dir
            / "best_model.cbm"
        )

        stage02_dir = (
            STAGE02_ROOT
            / model_dir.name
        )

        train_path = (
            stage02_dir
            / "train.csv"
        )

        test_path = (
            stage02_dir
            / "test.csv"
        )

        if (
            model_path.exists()
            and train_path.exists()
            and test_path.exists()
        ):

            results.append(
                (
                    seed,
                    model_dir,
                    stage02_dir,
                )
            )

    results.sort(
        key=lambda item: item[0]
    )

    if MAX_SEEDS is not None:

        results = results[
            :MAX_SEEDS
        ]

    return results


# =============================================================================
# MODEL CONTRACT
# =============================================================================

def load_model_contract(
    model_dir: Path,
    model: CatBoostClassifier,
    train_df: pd.DataFrame,
) -> Tuple[
    List[str],
    List[str],
]:

    parameter_file = (
        model_dir
        / "best_model_parameters.json"
    )

    features: List[str] = []

    categorical: List[str] = []

    if parameter_file.exists():

        try:

            payload = json.loads(
                parameter_file.read_text(
                    encoding="utf-8"
                )
            )

            features = list(
                payload.get(
                    "feature_columns",
                    [],
                )
            )

            categorical = list(
                payload.get(
                    "categorical_features",
                    [],
                )
            )

        except Exception:

            pass

    if not features:

        model_features = list(
            getattr(
                model,
                "feature_names_",
                [],
            )
        )

        if model_features:

            features = model_features

        else:

            features = [
                column
                for column in train_df.columns
                if column != TARGET_COLUMN
            ]

    if not categorical:

        for feature in features:

            if feature not in train_df:

                continue

            series = train_df[
                feature
            ]

            if (
                pd.api.types.is_object_dtype(
                    series
                )
                or
                pd.api.types.is_string_dtype(
                    series
                )
                or isinstance(
                    series.dtype,
                    pd.CategoricalDtype,
                )
            ):

                categorical.append(
                    feature
                )

    return (
        features,
        categorical,
    )


# =============================================================================
# PREPARE DATA
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:

    X = df[
        list(features)
    ].copy()

    categorical_set = set(
        categorical_features
    )

    for feature in features:

        if feature in categorical_set:

            X[feature] = (
                X[feature]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )

        else:

            X[feature] = pd.to_numeric(
                X[feature],
                errors="coerce",
            )

            X[feature] = X[
                feature
            ].replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )

    return X


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
# CLASS SELECTION
# =============================================================================

def choose_explain_class(
    model_classes: Sequence[Any],
    requested: Optional[str],
) -> Tuple[str, int]:

    classes = [
        str(value)
        for value in model_classes
    ]

    if requested is not None:

        requested = str(
            requested
        )

        if requested not in classes:

            raise ValueError(
                f"Requested explanation class "
                f"{requested!r} is not present. "
                f"Available classes: {classes}"
            )

        return (
            requested,
            classes.index(
                requested
            ),
        )

    if "1" in classes:

        return (
            "1",
            classes.index(
                "1"
            ),
        )

    return (
        classes[-1],
        len(classes) - 1,
    )


# =============================================================================
# PREDICTIONS
# =============================================================================

def predict(
    model: CatBoostClassifier,
    X: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    predicted = np.asarray(
        model.predict(
            X
        )
    ).reshape(
        -1
    ).astype(str)

    probabilities = np.asarray(
        model.predict_proba(
            X
        ),
        dtype=float,
    )

    return (
        predicted,
        probabilities,
    )


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[Any],
) -> Dict[str, float]:

    y_true_array = np.asarray(
        y_true
    ).astype(str)

    y_pred_array = np.asarray(
        y_pred
    ).astype(str)

    class_names = [
        str(value)
        for value in classes
    ]

    result: Dict[str, float] = {}

    result["accuracy"] = float(
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

    result["mcc"] = float(
        matthews_corrcoef(
            y_true_array,
            y_pred_array,
        )
    )

    try:

        result[
            "log_loss"
        ] = float(
            log_loss(
                y_true_array,
                probabilities,
                labels=class_names,
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

        if len(class_names) == 2:

            positive_class = (
                "1"
                if "1" in class_names
                else class_names[-1]
            )

            positive_index = (
                class_names.index(
                    positive_class
                )
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
                    labels=class_names,
                    multi_class="ovr",
                    average="macro",
                )
            )

    except Exception:

        result[
            "roc_auc"
        ] = np.nan

    return result


# =============================================================================
# CONFUSION MATRICES
# =============================================================================

def calculate_confusions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    classes: Sequence[Any],
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    labels = [
        str(value)
        for value in classes
    ]

    absolute = confusion_matrix(
        np.asarray(
            y_true
        ).astype(str),
        np.asarray(
            y_pred
        ).astype(str),
        labels=labels,
    )

    normalized = confusion_matrix(
        np.asarray(
            y_true
        ).astype(str),
        np.asarray(
            y_pred
        ).astype(str),
        labels=labels,
        normalize="true",
    )

    return (
        np.asarray(
            absolute,
            dtype=float,
        ),
        np.asarray(
            normalized,
            dtype=float,
        ),
    )


# =============================================================================
# CONFUSION MATRIX PLOT
# =============================================================================

def plot_confusion_with_std(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    classes: Sequence[str],
    title: str,
    normalized: bool,
    output_path: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            6.5,
        )
    )

    image = ax.imshow(
        mean_matrix,
        cmap=CONFUSION_CMAP,
        aspect="equal",
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    ax.set_xticks(
        np.arange(
            len(classes)
        )
    )

    ax.set_yticks(
        np.arange(
            len(classes)
        )
    )

    ax.set_xticklabels(
        classes,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        classes
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_title(
        title
    )

    for row in range(
        mean_matrix.shape[0]
    ):

        for column in range(
            mean_matrix.shape[1]
        ):

            mean_value = (
                mean_matrix[
                    row,
                    column
                ]
            )

            std_value = (
                std_matrix[
                    row,
                    column
                ]
            )

            if normalized:

                text = (
                    f"{mean_value:.3f}\n"
                    f"± {std_value:.3f}"
                )

            else:

                text = (
                    f"{mean_value:.1f}\n"
                    f"± {std_value:.1f}"
                )

            threshold = (
                np.nanmax(
                    mean_matrix
                ) * 0.55
                if mean_matrix.size
                else 0.0
            )

            color = (
                "white"
                if mean_value > threshold
                else "black"
            )

            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


# =============================================================================
# NATIVE CATBOOST FEATURE IMPORTANCE
# =============================================================================

def get_native_feature_importance(
    model: CatBoostClassifier,
    features: Sequence[str],
) -> pd.DataFrame:

    importance = np.asarray(
        model.get_feature_importance(
            type="PredictionValuesChange"
        ),
        dtype=float,
    )

    total = float(
        np.sum(
            importance
        )
    )

    if total > 0:

        normalized = (
            importance
            / total
        )

    else:

        normalized = np.zeros_like(
            importance
        )

    result = pd.DataFrame(
        {
            "feature":
                list(
                    features
                ),

            "importance":
                importance,

            "normalized_importance":
                normalized,
        }
    )

    result = result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result[
        "rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )

    return result


# =============================================================================
# SHAP
# =============================================================================

def sample_dataframe(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: Optional[int],
    random_seed: int,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
]:

    if (
        max_rows is None
        or len(X) <= max_rows
    ):

        return (
            X.reset_index(
                drop=True
            ),
            y.reset_index(
                drop=True
            ),
        )

    rng = np.random.default_rng(
        random_seed
    )

    indices = rng.choice(
        len(X),
        size=max_rows,
        replace=False,
    )

    return (
        X.iloc[
            indices
        ].reset_index(
            drop=True
        ),

        y.iloc[
            indices
        ].reset_index(
            drop=True
        ),
    )


def calculate_native_shap(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    categorical_features: Sequence[str],
    class_index: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    pool = Pool(
        X,
        label=y,
        cat_features=list(
            categorical_features
        ),
        feature_names=list(
            features
        ),
    )

    shap_raw = np.asarray(
        model.get_feature_importance(
            pool,
            type="ShapValues",
        )
    )

    # -------------------------------------------------------------------------
    # Binary / single-output
    #
    # shape:
    #
    #     n_objects x (n_features + 1)
    # -------------------------------------------------------------------------

    if shap_raw.ndim == 2:

        shap_values = (
            shap_raw[
                :,
                :-1
            ]
        )

        expected = (
            shap_raw[
                :,
                -1
            ]
        )

        return (
            shap_values,
            expected,
        )

    # -------------------------------------------------------------------------
    # Multiclass
    #
    # Typical shape:
    #
    #     n_objects x n_classes x (n_features + 1)
    # -------------------------------------------------------------------------

    if shap_raw.ndim == 3:

        shap_values = (
            shap_raw[
                :,
                class_index,
                :-1
            ]
        )

        expected = (
            shap_raw[
                :,
                class_index,
                -1
            ]
        )

        return (
            shap_values,
            expected,
        )

    raise ValueError(
        f"Unexpected CatBoost SHAP shape: "
        f"{shap_raw.shape}"
    )


def shap_feature_summary(
    shap_values: np.ndarray,
    features: Sequence[str],
) -> pd.DataFrame:

    mean_abs = np.mean(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    mean_signed = np.mean(
        shap_values,
        axis=0,
    )

    std_signed = np.std(
        shap_values,
        axis=0,
        ddof=1,
    )

    total = float(
        np.sum(
            mean_abs
        )
    )

    if total > 0:

        relative = (
            mean_abs
            / total
        )

    else:

        relative = np.zeros_like(
            mean_abs
        )

    result = pd.DataFrame(
        {
            "feature":
                list(
                    features
                ),

            "mean_abs_shap":
                mean_abs,

            "mean_signed_shap":
                mean_signed,

            "std_signed_shap":
                std_signed,

            "relative_mean_abs_shap":
                relative,
        }
    )

    result = result.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result[
        "rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )

    return result


# =============================================================================
# ICE
# =============================================================================

def identify_numeric_features(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str],
) -> List[str]:

    categorical_set = set(
        categorical_features
    )

    numeric: List[str] = []

    for feature in features:

        if feature in categorical_set:

            continue

        converted = pd.to_numeric(
            df[
                feature
            ],
            errors="coerce",
        )

        if converted.notna().any():

            numeric.append(
                feature
            )

    return numeric


def build_global_ice_grids(
    master_df: pd.DataFrame,
    numeric_features: Sequence[str],
) -> Dict[
    str,
    np.ndarray,
]:

    grids: Dict[
        str,
        np.ndarray
    ] = {}

    for feature in numeric_features:

        values = pd.to_numeric(
            master_df[
                feature
            ],
            errors="coerce",
        ).dropna()

        if values.empty:

            continue

        low = float(
            values.quantile(
                ICE_GRID_LOWER_QUANTILE
            )
        )

        high = float(
            values.quantile(
                ICE_GRID_UPPER_QUANTILE
            )
        )

        if not np.isfinite(
            low
        ):

            continue

        if not np.isfinite(
            high
        ):

            continue

        if math.isclose(
            low,
            high,
        ):

            grids[
                feature
            ] = np.asarray(
                [
                    low
                ],
                dtype=float,
            )

        else:

            grids[
                feature
            ] = np.linspace(
                low,
                high,
                ICE_GRID_POINTS,
            )

    return grids


def calculate_ice_for_feature(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    feature: str,
    grid: np.ndarray,
    class_index: int,
    random_seed: int,
) -> pd.DataFrame:

    if len(X) > ICE_SAMPLE_ROWS:

        sample = X.sample(
            n=ICE_SAMPLE_ROWS,
            random_state=random_seed,
        ).reset_index(
            drop=True
        )

    else:

        sample = X.reset_index(
            drop=True
        )

    prediction_matrix = np.empty(
        (
            len(sample),
            len(grid),
        ),
        dtype=float,
    )

    for grid_index, value in enumerate(
        grid
    ):

        modified = sample.copy()

        modified[
            feature
        ] = value

        probabilities = np.asarray(
            model.predict_proba(
                modified
            ),
            dtype=float,
        )

        prediction_matrix[
            :,
            grid_index
        ] = probabilities[
            :,
            class_index
        ]

    result = pd.DataFrame(
        {
            "grid_value":
                grid,

            "mean_probability":
                np.mean(
                    prediction_matrix,
                    axis=0,
                ),

            "std_probability":
                np.std(
                    prediction_matrix,
                    axis=0,
                    ddof=1,
                ),

            "median_probability":
                np.median(
                    prediction_matrix,
                    axis=0,
                ),

            "q25_probability":
                np.quantile(
                    prediction_matrix,
                    0.25,
                    axis=0,
                ),

            "q75_probability":
                np.quantile(
                    prediction_matrix,
                    0.75,
                    axis=0,
                ),
        }
    )

    return result


# =============================================================================
# FEATURE DISTRIBUTIONS
# =============================================================================

def calculate_feature_distribution_summary(
    X: pd.DataFrame,
    predicted_class: np.ndarray,
    numeric_features: Sequence[str],
    seed: int,
    dataset_name: str,
) -> pd.DataFrame:

    rows: List[
        Dict[str, Any]
    ] = []

    predicted = pd.Series(
        predicted_class,
        index=X.index,
    )

    classes = sorted(
        predicted.unique()
    )

    for feature in numeric_features:

        values = pd.to_numeric(
            X[
                feature
            ],
            errors="coerce",
        )

        for class_value in classes:

            subset = values[
                predicted
                == class_value
            ].dropna()

            if subset.empty:

                continue

            rows.append(
                {
                    "seed":
                        seed,

                    "dataset":
                        dataset_name,

                    "feature":
                        feature,

                    "predicted_class":
                        str(
                            class_value
                        ),

                    "n":
                        len(
                            subset
                        ),

                    "mean":
                        float(
                            subset.mean()
                        ),

                    "std":
                        float(
                            subset.std()
                        ),

                    "q25":
                        float(
                            subset.quantile(
                                0.25
                            )
                        ),

                    "median":
                        float(
                            subset.median()
                        ),

                    "q75":
                        float(
                            subset.quantile(
                                0.75
                            )
                        ),

                    "min":
                        float(
                            subset.min()
                        ),

                    "max":
                        float(
                            subset.max()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PLOT: PERFORMANCE
# =============================================================================

def plot_metric_summary(
    metrics_summary: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:

    subset = metrics_summary[
        metrics_summary[
            "dataset"
        ]
        == dataset_name
    ].copy()

    desirable_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "roc_auc",
    ]

    subset = subset[
        subset[
            "metric"
        ].isin(
            desirable_metrics
        )
    ]

    if subset.empty:

        return

    fig, ax = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    x = np.arange(
        len(subset)
    )

    ax.errorbar(
        x,
        subset[
            "mean"
        ],
        yerr=subset[
            "std"
        ],
        fmt="o",
        markersize=7,
        capsize=4,
        linewidth=1.8,
        color=COLORS[0],
        ecolor=COLORS[2],
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        subset[
            "metric"
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        f"{dataset_name.capitalize()} performance across splits\n"
        "mean ± standard deviation"
    )

    ax.axhline(
        0.0,
        linewidth=0.8,
        color=COLORS[0],
        alpha=0.4,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


# =============================================================================
# PLOT: FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:

    plot_df = (
        summary
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        .head(
            FEATURE_IMPORTANCE_TOP_N
        )
        .sort_values(
            "mean_importance",
            ascending=True,
        )
    )

    if plot_df.empty:

        return

    fig_height = max(
        5.5,
        0.34 * len(
            plot_df
        ) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            fig_height,
        )
    )

    y = np.arange(
        len(plot_df)
    )

    ax.barh(
        y,
        plot_df[
            "mean_importance"
        ],
        xerr=plot_df[
            "std_importance"
        ],
        color=COLORS[1],
        alpha=0.88,
        error_kw={
            "ecolor":
                COLORS[0],

            "capsize":
                3,
        },
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        plot_df[
            "feature"
        ]
    )

    ax.set_xlabel(
        "CatBoost PredictionValuesChange importance"
    )

    ax.set_title(
        "CatBoost feature importance across models\n"
        "mean ± standard deviation"
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


# =============================================================================
# PLOT: SHAP IMPORTANCE
# =============================================================================

def plot_shap_importance(
    summary: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:

    subset = summary[
        summary[
            "dataset"
        ]
        == dataset_name
    ].copy()

    subset = (
        subset
        .sort_values(
            "mean_mean_abs_shap",
            ascending=False,
        )
        .head(
            FEATURE_IMPORTANCE_TOP_N
        )
        .sort_values(
            "mean_mean_abs_shap",
            ascending=True,
        )
    )

    if subset.empty:

        return

    fig_height = max(
        5.5,
        0.34 * len(
            subset
        ) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            fig_height,
        )
    )

    y = np.arange(
        len(subset)
    )

    ax.barh(
        y,
        subset[
            "mean_mean_abs_shap"
        ],
        xerr=subset[
            "std_mean_abs_shap"
        ],
        color=COLORS[3],
        alpha=0.88,
        error_kw={
            "ecolor":
                COLORS[0],

            "capsize":
                3,
        },
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        subset[
            "feature"
        ]
    )

    ax.set_xlabel(
        "Mean absolute SHAP value"
    )

    ax.set_title(
        f"{dataset_name.capitalize()} SHAP importance across models\n"
        "mean ± standard deviation"
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


# =============================================================================
# SHAP BEESWARM
# =============================================================================

def plot_shap_beeswarm(
    shap_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:

    if shap_df.empty:

        return

    ranked_features = (
        feature_summary[
            feature_summary[
                "dataset"
            ]
            == dataset_name
        ]
        .sort_values(
            "mean_mean_abs_shap",
            ascending=False,
        )[
            "feature"
        ]
        .head(
            SHAP_BEESWARM_TOP_N
        )
        .tolist()
    )

    if not ranked_features:

        return

    fig_height = max(
        6,
        0.45 * len(
            ranked_features
        ) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            fig_height,
        )
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    for position, feature in enumerate(
        reversed(
            ranked_features
        )
    ):

        subset = shap_df[
            shap_df[
                "feature"
            ]
            == feature
        ]

        if subset.empty:

            continue

        feature_values = pd.to_numeric(
            subset[
                "feature_value"
            ],
            errors="coerce",
        )

        finite = feature_values.notna()

        if finite.any():

            values = feature_values.copy()

            low = values.quantile(
                0.05
            )

            high = values.quantile(
                0.95
            )

            if (
                np.isfinite(
                    low
                )
                and np.isfinite(
                    high
                )
                and high > low
            ):

                normalized = (
                    (
                        values
                        - low
                    )
                    / (
                        high
                        - low
                    )
                ).clip(
                    0,
                    1,
                )

            else:

                normalized = pd.Series(
                    0.5,
                    index=subset.index,
                )

        else:

            normalized = pd.Series(
                0.5,
                index=subset.index,
            )

        jitter = rng.normal(
            0,
            0.08,
            size=len(
                subset
            ),
        )

        ax.scatter(
            subset[
                "shap_value"
            ],
            position + jitter,
            c=normalized,
            cmap=SHAP_CMAP,
            vmin=0,
            vmax=1,
            s=13,
            alpha=0.60,
            linewidths=0,
        )

    ax.axvline(
        0,
        color=COLORS[0],
        linewidth=1,
        alpha=0.60,
    )

    ax.set_yticks(
        np.arange(
            len(
                ranked_features
            )
        )
    )

    ax.set_yticklabels(
        list(
            reversed(
                ranked_features
            )
        )
    )

    ax.set_xlabel(
        "SHAP value"
    )

    ax.set_title(
        f"{dataset_name.capitalize()} SHAP distribution across splits"
    )

    ax.grid(
        axis="x",
        alpha=0.15,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


# =============================================================================
# ICE PLOT
# =============================================================================

def plot_global_ice(
    aggregate: pd.DataFrame,
    feature: str,
    dataset_name: str,
    explain_class: str,
    output_path: Path,
) -> None:

    subset = aggregate[
        (
            aggregate[
                "feature"
            ]
            == feature
        )
        &
        (
            aggregate[
                "dataset"
            ]
            == dataset_name
        )
    ].sort_values(
        "grid_value"
    )

    if subset.empty:

        return

    x = subset[
        "grid_value"
    ].to_numpy(
        dtype=float
    )

    mean = subset[
        "mean_probability_across_splits"
    ].to_numpy(
        dtype=float
    )

    std = subset[
        "std_probability_across_splits"
    ].to_numpy(
        dtype=float
    )

    lower = np.clip(
        mean - std,
        0,
        1,
    )

    upper = np.clip(
        mean + std,
        0,
        1,
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5.5,
        )
    )

    ax.plot(
        x,
        mean,
        linewidth=2.2,
        color=COLORS[0],
        label="Mean PDP across splits",
    )

    ax.fill_between(
        x,
        lower,
        upper,
        color=COLORS[2],
        alpha=0.30,
        label="± 1 SD across splits",
    )

    ax.set_xlabel(
        feature
    )

    ax.set_ylabel(
        f"P(class = {explain_class})"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_title(
        f"ICE/PDP: {feature}\n"
        f"{dataset_name}, mean across independently trained models"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


# =============================================================================
# DISTRIBUTION BOXPLOT
# =============================================================================

def plot_feature_boxplot(
    pooled: pd.DataFrame,
    feature: str,
    dataset_name: str,
    output_path: Path,
) -> None:

    subset = pooled[
        (
            pooled[
                "feature"
            ]
            == feature
        )
        &
        (
            pooled[
                "dataset"
            ]
            == dataset_name
        )
    ].copy()

    if subset.empty:

        return

    classes = sorted(
        subset[
            "predicted_class"
        ].unique()
    )

    data = []

    for class_value in classes:

        values = pd.to_numeric(
            subset.loc[
                subset[
                    "predicted_class"
                ]
                == class_value,
                "value",
            ],
            errors="coerce",
        ).dropna()

        data.append(
            values.to_numpy()
        )

    if not any(
        len(values) > 0
        for values in data
    ):

        return

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.5,
        )
    )

    box = ax.boxplot(
        data,
        labels=classes,
        showfliers=SHOW_BOXPLOT_OUTLIERS,
        patch_artist=True,
        medianprops={
            "color":
                COLORS[0],

            "linewidth":
                1.8,
        },
    )

    for index, patch in enumerate(
        box[
            "boxes"
        ]
    ):

        patch.set_facecolor(
            COLORS[
                index
                % len(
                    COLORS
                )
            ]
        )

        patch.set_alpha(
            0.65
        )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        feature
    )

    ax.set_title(
        f"{feature}\n"
        f"{dataset_name} distribution by predicted class"
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


# =============================================================================
# DISTRIBUTION MEAN ± SD PLOT
# =============================================================================

def plot_feature_class_mean_std(
    aggregate: pd.DataFrame,
    feature: str,
    dataset_name: str,
    output_path: Path,
) -> None:

    subset = aggregate[
        (
            aggregate[
                "feature"
            ]
            == feature
        )
        &
        (
            aggregate[
                "dataset"
            ]
            == dataset_name
        )
    ].copy()

    if subset.empty:

        return

    subset = subset.sort_values(
        "predicted_class"
    )

    x = np.arange(
        len(
            subset
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.5,
        )
    )

    ax.errorbar(
        x,
        subset[
            "mean_of_split_means"
        ],
        yerr=subset[
            "std_of_split_means"
        ],
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=1.8,
        color=COLORS[0],
        ecolor=COLORS[2],
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        subset[
            "predicted_class"
        ]
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        feature
    )

    ax.set_title(
        f"{feature}\n"
        f"{dataset_name}: mean feature value ± SD across splits"
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


# =============================================================================
# PROCESS ONE MODEL
# =============================================================================

def process_seed(
    seed: int,
    model_dir: Path,
    split_dir: Path,
    ice_grids: Dict[str, np.ndarray],
) -> Dict[str, Any]:

    seed_output = (
        OUTPUT_ROOT
        / model_dir.name
    )

    ensure_dir(
        seed_output
    )

    # =========================================================================
    # LOAD MODEL
    # =========================================================================

    model = CatBoostClassifier()

    model.load_model(
        str(
            model_dir
            / "best_model.cbm"
        )
    )

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    train_df = pd.read_csv(
        split_dir
        / "train.csv",
        low_memory=False,
    )

    test_df = pd.read_csv(
        split_dir
        / "test.csv",
        low_memory=False,
    )

    (
        features,
        categorical_features,
    ) = load_model_contract(
        model_dir,
        model,
        train_df,
    )

    numeric_features = identify_numeric_features(
        train_df,
        features,
        categorical_features,
    )

    # =========================================================================
    # PREPARE DATA
    # =========================================================================

    X_train = prepare_features(
        train_df,
        features,
        categorical_features,
    )

    y_train = prepare_target(
        train_df[
            TARGET_COLUMN
        ]
    )

    X_test = prepare_features(
        test_df,
        features,
        categorical_features,
    )

    y_test = prepare_target(
        test_df[
            TARGET_COLUMN
        ]
    )

    classes = [
        str(value)
        for value in model.classes_
    ]

    (
        shap_class,
        shap_class_index,
    ) = choose_explain_class(
        classes,
        SHAP_EXPLAIN_CLASS,
    )

    (
        ice_class,
        ice_class_index,
    ) = choose_explain_class(
        classes,
        ICE_EXPLAIN_CLASS,
    )

    # =========================================================================
    # NATIVE FEATURE IMPORTANCE
    # =========================================================================

    feature_importance_df = (
        get_native_feature_importance(
            model,
            features,
        )
    )

    feature_importance_df[
        "seed"
    ] = seed

    if SAVE_PER_SEED_CSV:

        feature_importance_df.to_csv(
            seed_output
            / "feature_importance.csv",
            index=False,
        )

    # =========================================================================
    # CONTAINERS
    # =========================================================================

    metrics_rows: List[
        Dict[str, Any]
    ] = []

    confusion_data: Dict[
        str,
        Dict[str, np.ndarray]
    ] = {}

    shap_summaries: List[
        pd.DataFrame
    ] = []

    shap_plot_samples: List[
        pd.DataFrame
    ] = []

    ice_rows: List[
        pd.DataFrame
    ] = []

    distribution_summaries: List[
        pd.DataFrame
    ] = []

    distribution_samples: List[
        pd.DataFrame
    ] = []

    dataset_definitions = []

    if ANALYZE_TRAIN:

        dataset_definitions.append(
            (
                "train",
                X_train,
                y_train,
            )
        )

    if ANALYZE_TEST:

        dataset_definitions.append(
            (
                "test",
                X_test,
                y_test,
            )
        )

    # =========================================================================
    # DATASET LOOP
    # =========================================================================

    for (
        dataset_name,
        X,
        y,
    ) in dataset_definitions:

        # ---------------------------------------------------------------------
        # Predictions
        # ---------------------------------------------------------------------

        predictions, probabilities = predict(
            model,
            X,
        )

        # ---------------------------------------------------------------------
        # Performance
        # ---------------------------------------------------------------------

        metrics = calculate_metrics(
            y,
            predictions,
            probabilities,
            classes,
        )

        metrics_rows.append(
            {
                "seed":
                    seed,

                "dataset":
                    dataset_name,

                "n":
                    len(
                        y
                    ),

                **metrics,
            }
        )

        # ---------------------------------------------------------------------
        # Confusion matrices
        # ---------------------------------------------------------------------

        absolute, normalized = (
            calculate_confusions(
                y,
                predictions,
                classes,
            )
        )

        confusion_data[
            dataset_name
        ] = {
            "absolute":
                absolute,

            "normalized":
                normalized,
        }

        # ---------------------------------------------------------------------
        # SHAP
        # ---------------------------------------------------------------------

        if ENABLE_SHAP:

            X_shap, y_shap = sample_dataframe(
                X,
                y,
                SHAP_MAX_ROWS_PER_DATASET,
                random_seed=(
                    RANDOM_STATE
                    + seed
                    + (
                        0
                        if dataset_name
                        == "train"
                        else 10_000
                    )
                ),
            )

            (
                shap_values,
                expected_values,
            ) = calculate_native_shap(
                model,
                X_shap,
                y_shap,
                features,
                categorical_features,
                shap_class_index,
            )

            shap_summary = shap_feature_summary(
                shap_values,
                features,
            )

            shap_summary[
                "seed"
            ] = seed

            shap_summary[
                "dataset"
            ] = dataset_name

            shap_summary[
                "explained_class"
            ] = shap_class

            shap_summaries.append(
                shap_summary
            )

            # -----------------------------------------------------------------
            # Small controlled sample for global beeswarm
            # -----------------------------------------------------------------

            n_beeswarm = min(
                SHAP_BEESWARM_ROWS_PER_SPLIT,
                len(
                    X_shap
                ),
            )

            rng = np.random.default_rng(
                RANDOM_STATE
                + seed
                + 90_000
            )

            selected_indices = rng.choice(
                len(
                    X_shap
                ),
                size=n_beeswarm,
                replace=False,
            )

            for row_index in selected_indices:

                for feature_index, feature in enumerate(
                    features
                ):

                    shap_plot_samples.append(
                        pd.DataFrame(
                            {
                                "seed":
                                    [
                                        seed
                                    ],

                                "dataset":
                                    [
                                        dataset_name
                                    ],

                                "feature":
                                    [
                                        feature
                                    ],

                                "feature_value":
                                    [
                                        X_shap.iloc[
                                            row_index
                                        ][
                                            feature
                                        ]
                                    ],

                                "shap_value":
                                    [
                                        shap_values[
                                            row_index,
                                            feature_index
                                        ]
                                    ],

                                "explained_class":
                                    [
                                        shap_class
                                    ],
                            }
                        )
                    )

        # ---------------------------------------------------------------------
        # ICE
        # ---------------------------------------------------------------------

        if (
            ENABLE_ICE
            and dataset_name
            in ICE_DATASETS
        ):

            selected_ice_features = (
                numeric_features
                if ICE_FEATURES is None
                else [
                    feature
                    for feature in ICE_FEATURES
                    if feature
                    in numeric_features
                ]
            )

            for feature in selected_ice_features:

                if feature not in ice_grids:

                    continue

                ice_result = (
                    calculate_ice_for_feature(
                        model=model,
                        X=X,
                        feature=feature,
                        grid=ice_grids[
                            feature
                        ],
                        class_index=ice_class_index,
                        random_seed=(
                            RANDOM_STATE
                            + seed
                        ),
                    )
                )

                ice_result[
                    "seed"
                ] = seed

                ice_result[
                    "dataset"
                ] = dataset_name

                ice_result[
                    "feature"
                ] = feature

                ice_result[
                    "explained_class"
                ] = ice_class

                ice_rows.append(
                    ice_result
                )

        # ---------------------------------------------------------------------
        # Feature distributions by predicted class
        # ---------------------------------------------------------------------

        if ENABLE_FEATURE_DISTRIBUTIONS:

            summary = (
                calculate_feature_distribution_summary(
                    X,
                    predictions,
                    numeric_features,
                    seed,
                    dataset_name,
                )
            )

            distribution_summaries.append(
                summary
            )

            # -------------------------------------------------------------
            # Controlled global pool for boxplots
            # -------------------------------------------------------------

            rng = np.random.default_rng(
                RANDOM_STATE
                + seed
                + (
                    0
                    if dataset_name
                    == "train"
                    else 50_000
                )
            )

            sample_size = min(
                DISTRIBUTION_ROWS_PER_SPLIT,
                len(
                    X
                ),
            )

            indices = rng.choice(
                len(
                    X
                ),
                size=sample_size,
                replace=False,
            )

            sampled_X = X.iloc[
                indices
            ]

            sampled_predictions = predictions[
                indices
            ]

            for feature in numeric_features:

                values = pd.to_numeric(
                    sampled_X[
                        feature
                    ],
                    errors="coerce",
                )

                current = pd.DataFrame(
                    {
                        "seed":
                            seed,

                        "dataset":
                            dataset_name,

                        "feature":
                            feature,

                        "predicted_class":
                            sampled_predictions,

                        "value":
                            values.to_numpy(),
                    }
                )

                distribution_samples.append(
                    current
                )

    # =========================================================================
    # SAVE PER-SEED ANALYSIS
    # =========================================================================

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    shap_summary_df = (
        pd.concat(
            shap_summaries,
            ignore_index=True,
        )
        if shap_summaries
        else pd.DataFrame()
    )

    ice_df = (
        pd.concat(
            ice_rows,
            ignore_index=True,
        )
        if ice_rows
        else pd.DataFrame()
    )

    distribution_summary_df = (
        pd.concat(
            distribution_summaries,
            ignore_index=True,
        )
        if distribution_summaries
        else pd.DataFrame()
    )

    distribution_sample_df = (
        pd.concat(
            distribution_samples,
            ignore_index=True,
        )
        if distribution_samples
        else pd.DataFrame()
    )

    shap_plot_df = (
        pd.concat(
            shap_plot_samples,
            ignore_index=True,
        )
        if shap_plot_samples
        else pd.DataFrame()
    )

    if SAVE_PER_SEED_CSV:

        metrics_df.to_csv(
            seed_output
            / "metrics.csv",
            index=False,
        )

        if not shap_summary_df.empty:

            shap_summary_df.to_csv(
                seed_output
                / "shap_summary.csv",
                index=False,
            )

        if not ice_df.empty:

            ice_df.to_csv(
                seed_output
                / "ice_summary.csv",
                index=False,
            )

        if not distribution_summary_df.empty:

            distribution_summary_df.to_csv(
                seed_output
                / "feature_distribution_summary.csv",
                index=False,
            )

    # =========================================================================
    # REPORT
    # =========================================================================

    if SAVE_PER_SEED_REPORT:

        report: List[str] = []

        separator = "=" * 90

        report.append(
            separator
        )

        report.append(
            "CATBOOST POST-TRAINING ANALYSIS"
        )

        report.append(
            separator
        )

        report.append("")

        report.append(
            f"Seed:                   {seed}"
        )

        report.append(
            f"Features:               {len(features)}"
        )

        report.append(
            f"Categorical features:   {len(categorical_features)}"
        )

        report.append(
            f"Numeric features:       {len(numeric_features)}"
        )

        report.append(
            f"Classes:                {classes}"
        )

        report.append(
            f"SHAP explained class:   {shap_class}"
        )

        report.append(
            f"ICE explained class:    {ice_class}"
        )

        report.append("")

        for _, row in metrics_df.iterrows():

            report.append(
                separator
            )

            report.append(
                f"{str(row['dataset']).upper()} PERFORMANCE"
            )

            report.append(
                separator
            )

            report.append("")

            for metric in [
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "mcc",
                "roc_auc",
                "log_loss",
            ]:

                report.append(
                    f"{metric:<30}"
                    f"{row[metric]:.6f}"
                )

            report.append("")

        (
            seed_output
            / "analysis_report.txt"
        ).write_text(
            "\n".join(
                report
            ),
            encoding="utf-8",
        )

    return {

        "seed":
            seed,

        "classes":
            classes,

        "features":
            features,

        "categorical_features":
            categorical_features,

        "numeric_features":
            numeric_features,

        "metrics":
            metrics_df,

        "confusion":
            confusion_data,

        "feature_importance":
            feature_importance_df,

        "shap_summary":
            shap_summary_df,

        "shap_plot":
            shap_plot_df,

        "ice":
            ice_df,

        "distribution_summary":
            distribution_summary_df,

        "distribution_samples":
            distribution_sample_df,

        "shap_class":
            shap_class,

        "ice_class":
            ice_class,
    }


# =============================================================================
# AGGREGATE PERFORMANCE
# =============================================================================

def aggregate_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:

    metric_columns = [
        column
        for column in metrics.columns
        if column not in {
            "seed",
            "dataset",
            "n",
        }
    ]

    rows = []

    for dataset_name in sorted(
        metrics[
            "dataset"
        ].unique()
    ):

        subset = metrics[
            metrics[
                "dataset"
            ]
            == dataset_name
        ]

        for metric in metric_columns:

            values = pd.to_numeric(
                subset[
                    metric
                ],
                errors="coerce",
            )

            rows.append(
                {
                    "dataset":
                        dataset_name,

                    "metric":
                        metric,

                    "mean":
                        values.mean(),

                    "std":
                        values.std(),

                    "median":
                        values.median(),

                    "min":
                        values.min(),

                    "max":
                        values.max(),

                    "n_splits":
                        values.notna().sum(),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# AGGREGATE FEATURE IMPORTANCE
# =============================================================================

def aggregate_feature_importance(
    importance: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        importance
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_importance=(
                "importance",
                "mean",
            ),

            std_importance=(
                "importance",
                "std",
            ),

            median_importance=(
                "importance",
                "median",
            ),

            min_importance=(
                "importance",
                "min",
            ),

            max_importance=(
                "importance",
                "max",
            ),

            mean_normalized_importance=(
                "normalized_importance",
                "mean",
            ),

            std_normalized_importance=(
                "normalized_importance",
                "std",
            ),

            n_models=(
                "seed",
                "nunique",
            ),
        )
    )

    grouped = grouped.sort_values(
        "mean_importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    grouped[
        "rank"
    ] = np.arange(
        1,
        len(grouped) + 1,
    )

    return grouped


# =============================================================================
# AGGREGATE SHAP
# =============================================================================

def aggregate_shap(
    shap_summary: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        shap_summary
        .groupby(
            [
                "dataset",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            mean_mean_abs_shap=(
                "mean_abs_shap",
                "mean",
            ),

            std_mean_abs_shap=(
                "mean_abs_shap",
                "std",
            ),

            median_mean_abs_shap=(
                "mean_abs_shap",
                "median",
            ),

            mean_signed_shap=(
                "mean_signed_shap",
                "mean",
            ),

            mean_relative_shap=(
                "relative_mean_abs_shap",
                "mean",
            ),

            std_relative_shap=(
                "relative_mean_abs_shap",
                "std",
            ),

            n_models=(
                "seed",
                "nunique",
            ),
        )
    )

    grouped[
        "rank"
    ] = (
        grouped
        .groupby(
            "dataset"
        )[
            "mean_mean_abs_shap"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    return grouped.sort_values(
        [
            "dataset",
            "rank",
        ]
    ).reset_index(
        drop=True
    )


# =============================================================================
# AGGREGATE ICE
# =============================================================================

def aggregate_ice(
    ice: pd.DataFrame,
) -> pd.DataFrame:

    return (
        ice
        .groupby(
            [
                "dataset",
                "feature",
                "explained_class",
                "grid_value",
            ],
            as_index=False,
        )
        .agg(
            mean_probability_across_splits=(
                "mean_probability",
                "mean",
            ),

            std_probability_across_splits=(
                "mean_probability",
                "std",
            ),

            mean_within_split_std=(
                "std_probability",
                "mean",
            ),

            n_models=(
                "seed",
                "nunique",
            ),
        )
    )


# =============================================================================
# AGGREGATE FEATURE DISTRIBUTIONS
# =============================================================================

def aggregate_feature_distributions(
    distributions: pd.DataFrame,
) -> pd.DataFrame:

    return (
        distributions
        .groupby(
            [
                "dataset",
                "feature",
                "predicted_class",
            ],
            as_index=False,
        )
        .agg(
            mean_of_split_means=(
                "mean",
                "mean",
            ),

            std_of_split_means=(
                "mean",
                "std",
            ),

            mean_of_split_medians=(
                "median",
                "mean",
            ),

            std_of_split_medians=(
                "median",
                "std",
            ),

            mean_within_split_std=(
                "std",
                "mean",
            ),

            n_models=(
                "seed",
                "nunique",
            ),
        )
    )


# =============================================================================
# GLOBAL TEXT REPORT
# =============================================================================

def create_global_report(
    metrics_summary: pd.DataFrame,
    importance_summary: pd.DataFrame,
    shap_summary: pd.DataFrame,
    n_models: int,
    classes: Sequence[str],
) -> str:

    lines: List[str] = []

    separator = "=" * 90

    lines.append(
        separator
    )

    lines.append(
        "LLM DEOBFUSCATION - GLOBAL CATBOOST ANALYSIS"
    )

    lines.append(
        separator
    )

    lines.append("")

    lines.append(
        f"Analyzed CatBoost models: {n_models:,}"
    )

    lines.append(
        f"Classes: {list(classes)}"
    )

    lines.append("")

    # =========================================================================
    # PERFORMANCE
    # =========================================================================

    lines.append(
        separator
    )

    lines.append(
        "PERFORMANCE ACROSS RANDOM SPLITS"
    )

    lines.append(
        separator
    )

    lines.append("")

    for dataset_name in [
        "train",
        "test",
    ]:

        subset = metrics_summary[
            metrics_summary[
                "dataset"
            ]
            == dataset_name
        ]

        if subset.empty:

            continue

        lines.append(
            f"{dataset_name.upper()}"
        )

        lines.append(
            "-" * 90
        )

        lines.append(
            f"{'Metric':<28}"
            f"{'Mean':>12}"
            f"{'SD':>12}"
            f"{'Median':>12}"
            f"{'Min':>12}"
            f"{'Max':>12}"
        )

        for _, row in subset.iterrows():

            lines.append(
                f"{row['metric']:<28}"
                f"{row['mean']:>12.6f}"
                f"{row['std']:>12.6f}"
                f"{row['median']:>12.6f}"
                f"{row['min']:>12.6f}"
                f"{row['max']:>12.6f}"
            )

        lines.append("")

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    if not importance_summary.empty:

        lines.append(
            separator
        )

        lines.append(
            "CATBOOST FEATURE IMPORTANCE"
        )

        lines.append(
            separator
        )

        lines.append("")

        lines.append(
            f"{'Rank':<8}"
            f"{'Feature':<50}"
            f"{'Mean':>12}"
            f"{'SD':>12}"
            f"{'Relative':>12}"
        )

        for _, row in importance_summary.iterrows():

            lines.append(
                f"{int(row['rank']):<8}"
                f"{str(row['feature']):<50}"
                f"{row['mean_importance']:>12.6f}"
                f"{row['std_importance']:>12.6f}"
                f"{row['mean_normalized_importance']:>12.6f}"
            )

        lines.append("")

    # =========================================================================
    # SHAP
    # =========================================================================

    if not shap_summary.empty:

        lines.append(
            separator
        )

        lines.append(
            "SHAP FEATURE IMPORTANCE"
        )

        lines.append(
            separator
        )

        lines.append("")

        for dataset_name in [
            "train",
            "test",
        ]:

            subset = (
                shap_summary[
                    shap_summary[
                        "dataset"
                    ]
                    == dataset_name
                ]
                .sort_values(
                    "rank"
                )
            )

            if subset.empty:

                continue

            lines.append(
                dataset_name.upper()
            )

            lines.append(
                "-" * 90
            )

            lines.append(
                f"{'Rank':<8}"
                f"{'Feature':<50}"
                f"{'Mean |SHAP|':>15}"
                f"{'SD':>12}"
            )

            for _, row in subset.iterrows():

                lines.append(
                    f"{int(row['rank']):<8}"
                    f"{str(row['feature']):<50}"
                    f"{row['mean_mean_abs_shap']:>15.6f}"
                    f"{row['std_mean_abs_shap']:>12.6f}"
                )

            lines.append("")

    lines.append(
        separator
    )

    lines.append(
        "INTERPRETATION NOTES"
    )

    lines.append(
        separator
    )

    lines.append("")

    lines.append(
        "1. Test-set results are the primary generalization results."
    )

    lines.append(
        "2. Training results are diagnostic and should not be interpreted as "
        "independent generalization estimates."
    )

    lines.append(
        "3. CatBoost feature importance describes model reliance, not causality."
    )

    lines.append(
        "4. SHAP values describe contributions to the model output for the "
        "analyzed observations; they do not establish causal effects."
    )

    lines.append(
        "5. ICE/PDP curves show model response when one feature is varied while "
        "the other observed feature values are held fixed. Strong feature "
        "dependence/correlation can make such counterfactual combinations less "
        "representative of the empirical data distribution."
    )

    lines.append(
        "6. Error bands in global plots represent variation across independently "
        "trained random-split models unless stated otherwise."
    )

    lines.append(
        "7. Feature distributions grouped by predicted class are descriptive "
        "properties of the model predictions, not evidence that the feature "
        "causes class membership."
    )

    lines.append("")

    return "\n".join(
        lines
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    # =========================================================================
    # VALIDATION
    # =========================================================================

    if not STAGE02_ROOT.exists():

        raise SystemExit(
            f"Stage-02 directory not found:\n"
            f"{STAGE02_ROOT.resolve()}"
        )

    if not STAGE03_ROOT.exists():

        raise SystemExit(
            f"Stage-03 directory not found:\n"
            f"{STAGE03_ROOT.resolve()}"
        )

    if not STAGE01_CSV.exists():

        raise SystemExit(
            f"Stage-01 dataset not found:\n"
            f"{STAGE01_CSV.resolve()}"
        )

    if OUTPUT_ROOT.exists():

        if OVERWRITE_OUTPUT:

            shutil.rmtree(
                OUTPUT_ROOT
            )

        else:

            raise SystemExit(
                f"Output directory already exists:\n"
                f"{OUTPUT_ROOT.resolve()}"
            )

    ensure_dir(
        OUTPUT_ROOT
    )

    # =========================================================================
    # DISCOVER MODELS
    # =========================================================================

    seed_directories = (
        discover_seed_directories()
    )

    if not seed_directories:

        raise SystemExit(
            "No valid Stage-03 CatBoost models were found."
        )

    print(
        "=" * 90
    )

    print(
        "LLM DEOBFUSCATION - FINAL CATBOOST ANALYSIS"
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"Models found:             "
        f"{len(seed_directories):,}"
    )

    print(
        f"Feature importance:       "
        f"{ENABLE_FEATURE_IMPORTANCE}"
    )

    print(
        f"SHAP:                     "
        f"{ENABLE_SHAP}"
    )

    print(
        f"ICE:                      "
        f"{ENABLE_ICE}"
    )

    print(
        f"Feature distributions:    "
        f"{ENABLE_FEATURE_DISTRIBUTIONS}"
    )

    print()

    # =========================================================================
    # MASTER DATASET FOR GLOBAL ICE GRID
    # =========================================================================

    master_df = pd.read_csv(
        STAGE01_CSV,
        low_memory=False,
    )

    # Load the first model contract to establish the common feature list.

    first_seed, first_model_dir, first_split_dir = (
        seed_directories[0]
    )

    first_model = CatBoostClassifier()

    first_model.load_model(
        str(
            first_model_dir
            / "best_model.cbm"
        )
    )

    first_train = pd.read_csv(
        first_split_dir
        / "train.csv",
        low_memory=False,
    )

    (
        common_features,
        common_categorical_features,
    ) = load_model_contract(
        first_model_dir,
        first_model,
        first_train,
    )

    common_numeric_features = (
        identify_numeric_features(
            first_train,
            common_features,
            common_categorical_features,
        )
    )

    selected_ice_features = (
        common_numeric_features
        if ICE_FEATURES is None
        else [
            feature
            for feature in ICE_FEATURES
            if feature
            in common_numeric_features
        ]
    )

    ice_grids = build_global_ice_grids(
        master_df,
        selected_ice_features,
    )

    # =========================================================================
    # GLOBAL COLLECTIONS
    # =========================================================================

    all_metrics: List[
        pd.DataFrame
    ] = []

    all_feature_importance: List[
        pd.DataFrame
    ] = []

    all_shap_summary: List[
        pd.DataFrame
    ] = []

    all_shap_plot: List[
        pd.DataFrame
    ] = []

    all_ice: List[
        pd.DataFrame
    ] = []

    all_distribution_summary: List[
        pd.DataFrame
    ] = []

    all_distribution_samples: List[
        pd.DataFrame
    ] = []

    absolute_confusions: Dict[
        str,
        List[np.ndarray]
    ] = {
        "train": [],
        "test": [],
    }

    normalized_confusions: Dict[
        str,
        List[np.ndarray]
    ] = {
        "train": [],
        "test": [],
    }

    successful_seeds: List[int] = []

    failed_rows: List[
        Dict[str, Any]
    ] = []

    global_classes: Optional[
        List[str]
    ] = None

    # =========================================================================
    # MODEL LOOP
    # =========================================================================

    for iteration, (
        seed,
        model_dir,
        split_dir,
    ) in enumerate(
        seed_directories,
        start=1,
    ):

        print(
            f"[{iteration:>5,}/"
            f"{len(seed_directories):,}] "
            f"seed={seed}"
        )

        try:

            result = process_seed(
                seed,
                model_dir,
                split_dir,
                ice_grids,
            )

            successful_seeds.append(
                seed
            )

            if global_classes is None:

                global_classes = result[
                    "classes"
                ]

            all_metrics.append(
                result[
                    "metrics"
                ]
            )

            all_feature_importance.append(
                result[
                    "feature_importance"
                ]
            )

            if not result[
                "shap_summary"
            ].empty:

                all_shap_summary.append(
                    result[
                        "shap_summary"
                    ]
                )

            if not result[
                "shap_plot"
            ].empty:

                all_shap_plot.append(
                    result[
                        "shap_plot"
                    ]
                )

            if not result[
                "ice"
            ].empty:

                all_ice.append(
                    result[
                        "ice"
                    ]
                )

            if not result[
                "distribution_summary"
            ].empty:

                all_distribution_summary.append(
                    result[
                        "distribution_summary"
                    ]
                )

            if not result[
                "distribution_samples"
            ].empty:

                all_distribution_samples.append(
                    result[
                        "distribution_samples"
                    ]
                )

            for dataset_name, matrices in result[
                "confusion"
            ].items():

                absolute_confusions[
                    dataset_name
                ].append(
                    matrices[
                        "absolute"
                    ]
                )

                normalized_confusions[
                    dataset_name
                ].append(
                    matrices[
                        "normalized"
                    ]
                )

        except Exception as exc:

            print(
                f"         FAILED: {exc}"
            )

            failed_rows.append(
                {
                    "seed":
                        seed,

                    "error":
                        str(
                            exc
                        ),
                }
            )

    # =========================================================================
    # CHECK
    # =========================================================================

    if not successful_seeds:

        raise SystemExit(
            "No model analysis completed successfully."
        )

    # =========================================================================
    # CONCATENATE
    # =========================================================================

    metrics_df = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    feature_importance_df = pd.concat(
        all_feature_importance,
        ignore_index=True,
    )

    shap_summary_df = (
        pd.concat(
            all_shap_summary,
            ignore_index=True,
        )
        if all_shap_summary
        else pd.DataFrame()
    )

    shap_plot_df = (
        pd.concat(
            all_shap_plot,
            ignore_index=True,
        )
        if all_shap_plot
        else pd.DataFrame()
    )

    ice_df = (
        pd.concat(
            all_ice,
            ignore_index=True,
        )
        if all_ice
        else pd.DataFrame()
    )

    distribution_summary_df = (
        pd.concat(
            all_distribution_summary,
            ignore_index=True,
        )
        if all_distribution_summary
        else pd.DataFrame()
    )

    distribution_samples_df = (
        pd.concat(
            all_distribution_samples,
            ignore_index=True,
        )
        if all_distribution_samples
        else pd.DataFrame()
    )

    # =========================================================================
    # AGGREGATE METRICS
    # =========================================================================

    metrics_summary = aggregate_metrics(
        metrics_df
    )

    metrics_df.to_csv(
        OUTPUT_ROOT
        / "all_split_metrics.csv",
        index=False,
    )

    metrics_summary.to_csv(
        OUTPUT_ROOT
        / "performance_summary_mean_std.csv",
        index=False,
    )

    # =========================================================================
    # PERFORMANCE PLOTS
    # =========================================================================

    plot_dir = (
        OUTPUT_ROOT
        / "plots"
    )

    ensure_dir(
        plot_dir
    )

    if ANALYZE_TRAIN:

        plot_metric_summary(
            metrics_summary,
            "train",
            plot_dir
            / "performance_train_mean_std",
        )

    if ANALYZE_TEST:

        plot_metric_summary(
            metrics_summary,
            "test",
            plot_dir
            / "performance_test_mean_std",
        )

    # =========================================================================
    # CONFUSION MATRICES
    # =========================================================================

    confusion_output = (
        OUTPUT_ROOT
        / "confusion_matrices"
    )

    ensure_dir(
        confusion_output
    )

    if global_classes is None:

        global_classes = []

    for dataset_name in [
        "train",
        "test",
    ]:

        if not absolute_confusions[
            dataset_name
        ]:

            continue

        absolute_stack = np.stack(
            absolute_confusions[
                dataset_name
            ],
            axis=0,
        )

        normalized_stack = np.stack(
            normalized_confusions[
                dataset_name
            ],
            axis=0,
        )

        absolute_mean = np.mean(
            absolute_stack,
            axis=0,
        )

        absolute_std = np.std(
            absolute_stack,
            axis=0,
            ddof=1,
        )

        normalized_mean = np.mean(
            normalized_stack,
            axis=0,
        )

        normalized_std = np.std(
            normalized_stack,
            axis=0,
            ddof=1,
        )

        pd.DataFrame(
            absolute_mean,
            index=global_classes,
            columns=global_classes,
        ).to_csv(
            confusion_output
            / f"{dataset_name}_absolute_mean.csv"
        )

        pd.DataFrame(
            absolute_std,
            index=global_classes,
            columns=global_classes,
        ).to_csv(
            confusion_output
            / f"{dataset_name}_absolute_std.csv"
        )

        pd.DataFrame(
            normalized_mean,
            index=global_classes,
            columns=global_classes,
        ).to_csv(
            confusion_output
            / f"{dataset_name}_normalized_mean.csv"
        )

        pd.DataFrame(
            normalized_std,
            index=global_classes,
            columns=global_classes,
        ).to_csv(
            confusion_output
            / f"{dataset_name}_normalized_std.csv"
        )

        plot_confusion_with_std(
            absolute_mean,
            absolute_std,
            global_classes,
            (
                f"{dataset_name.capitalize()} confusion matrix\n"
                "mean count ± SD across splits"
            ),
            normalized=False,
            output_path=(
                confusion_output
                / f"{dataset_name}_confusion_absolute_mean_std"
            ),
        )

        plot_confusion_with_std(
            normalized_mean,
            normalized_std,
            global_classes,
            (
                f"{dataset_name.capitalize()} row-normalized confusion matrix\n"
                "mean proportion ± SD across splits"
            ),
            normalized=True,
            output_path=(
                confusion_output
                / f"{dataset_name}_confusion_normalized_mean_std"
            ),
        )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    feature_importance_summary = pd.DataFrame()

    if ENABLE_FEATURE_IMPORTANCE:

        feature_importance_summary = (
            aggregate_feature_importance(
                feature_importance_df
            )
        )

        feature_importance_df.to_csv(
            OUTPUT_ROOT
            / "feature_importance_all_models.csv",
            index=False,
        )

        feature_importance_summary.to_csv(
            OUTPUT_ROOT
            / "feature_importance_summary.csv",
            index=False,
        )

        plot_feature_importance(
            feature_importance_summary,
            plot_dir
            / "catboost_feature_importance_mean_std",
        )

    # =========================================================================
    # SHAP
    # =========================================================================

    global_shap_summary = pd.DataFrame()

    if (
        ENABLE_SHAP
        and not shap_summary_df.empty
    ):

        global_shap_summary = (
            aggregate_shap(
                shap_summary_df
            )
        )

        shap_summary_df.to_csv(
            OUTPUT_ROOT
            / "shap_all_models.csv",
            index=False,
        )

        global_shap_summary.to_csv(
            OUTPUT_ROOT
            / "shap_summary_mean_std.csv",
            index=False,
        )

        for dataset_name in [
            "train",
            "test",
        ]:

            plot_shap_importance(
                global_shap_summary,
                dataset_name,
                plot_dir
                / f"shap_importance_{dataset_name}_mean_std",
            )

            if not shap_plot_df.empty:

                plot_shap_beeswarm(
                    shap_plot_df[
                        shap_plot_df[
                            "dataset"
                        ]
                        == dataset_name
                    ],
                    global_shap_summary,
                    dataset_name,
                    plot_dir
                    / f"shap_beeswarm_{dataset_name}",
                )

    # =========================================================================
    # ICE
    # =========================================================================

    if (
        ENABLE_ICE
        and not ice_df.empty
    ):

        ice_output = (
            OUTPUT_ROOT
            / "ice"
        )

        ensure_dir(
            ice_output
        )

        ice_df.to_csv(
            ice_output
            / "ice_all_models.csv",
            index=False,
        )

        ice_aggregate = aggregate_ice(
            ice_df
        )

        ice_aggregate.to_csv(
            ice_output
            / "ice_summary_mean_std.csv",
            index=False,
        )

        explained_classes = (
            ice_aggregate[
                "explained_class"
            ]
            .unique()
            .tolist()
        )

        for dataset_name in ICE_DATASETS:

            dataset_dir = (
                ice_output
                / dataset_name
            )

            ensure_dir(
                dataset_dir
            )

            for feature in selected_ice_features:

                subset = ice_aggregate[
                    (
                        ice_aggregate[
                            "feature"
                        ]
                        == feature
                    )
                    &
                    (
                        ice_aggregate[
                            "dataset"
                        ]
                        == dataset_name
                    )
                ]

                if subset.empty:

                    continue

                explain_class = str(
                    subset[
                        "explained_class"
                    ].iloc[0]
                )

                plot_global_ice(
                    ice_aggregate,
                    feature,
                    dataset_name,
                    explain_class,
                    dataset_dir
                    / f"{safe_name(feature)}_ice_pdp_mean_std",
                )

    # =========================================================================
    # FEATURE DISTRIBUTIONS
    # =========================================================================

    if (
        ENABLE_FEATURE_DISTRIBUTIONS
        and not distribution_summary_df.empty
    ):

        distribution_output = (
            OUTPUT_ROOT
            / "feature_distributions"
        )

        ensure_dir(
            distribution_output
        )

        distribution_summary_df.to_csv(
            distribution_output
            / "feature_distributions_all_models.csv",
            index=False,
        )

        distribution_aggregate = (
            aggregate_feature_distributions(
                distribution_summary_df
            )
        )

        distribution_aggregate.to_csv(
            distribution_output
            / "feature_distribution_summary_mean_std.csv",
            index=False,
        )

        if not distribution_samples_df.empty:

            for dataset_name in [
                "train",
                "test",
            ]:

                dataset_dir = (
                    distribution_output
                    / dataset_name
                )

                boxplot_dir = (
                    dataset_dir
                    / "boxplots"
                )

                mean_std_dir = (
                    dataset_dir
                    / "mean_std"
                )

                ensure_dir(
                    boxplot_dir
                )

                ensure_dir(
                    mean_std_dir
                )

                for feature in common_numeric_features:

                    plot_feature_boxplot(
                        distribution_samples_df,
                        feature,
                        dataset_name,
                        boxplot_dir
                        / f"{safe_name(feature)}_boxplot",
                    )

                    plot_feature_class_mean_std(
                        distribution_aggregate,
                        feature,
                        dataset_name,
                        mean_std_dir
                        / f"{safe_name(feature)}_mean_std",
                    )

    # =========================================================================
    # FAILED SEEDS
    # =========================================================================

    if failed_rows:

        pd.DataFrame(
            failed_rows
        ).to_csv(
            OUTPUT_ROOT
            / "failed_seeds.csv",
            index=False,
        )

    # =========================================================================
    # GLOBAL REPORT
    # =========================================================================

    report = create_global_report(
        metrics_summary=metrics_summary,
        importance_summary=feature_importance_summary,
        shap_summary=global_shap_summary,
        n_models=len(
            successful_seeds
        ),
        classes=global_classes,
    )

    (
        OUTPUT_ROOT
        / "GLOBAL_ANALYSIS_REPORT.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    write_json(
        OUTPUT_ROOT
        / "analysis_configuration.json",
        {
            "colors":
                COLORS,

            "models_analyzed":
                len(
                    successful_seeds
                ),

            "successful_seeds":
                successful_seeds,

            "failed_seed_count":
                len(
                    failed_rows
                ),

            "feature_importance":
                ENABLE_FEATURE_IMPORTANCE,

            "shap":
                ENABLE_SHAP,

            "shap_max_rows_per_dataset":
                SHAP_MAX_ROWS_PER_DATASET,

            "shap_beeswarm_rows_per_split":
                SHAP_BEESWARM_ROWS_PER_SPLIT,

            "ice":
                ENABLE_ICE,

            "ice_features":
                selected_ice_features,

            "ice_sample_rows":
                ICE_SAMPLE_ROWS,

            "ice_grid_points":
                ICE_GRID_POINTS,

            "feature_distributions":
                ENABLE_FEATURE_DISTRIBUTIONS,

            "distribution_rows_per_split":
                DISTRIBUTION_ROWS_PER_SPLIT,
        },
    )

    # =========================================================================
    # FINISHED
    # =========================================================================

    print()

    print(
        "=" * 90
    )

    print(
        "FINAL ANALYSIS COMPLETE"
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"Successful models:     "
        f"{len(successful_seeds):,}"
    )

    print(
        f"Failed models:         "
        f"{len(failed_rows):,}"
    )

    print(
        f"Output directory:      "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Global report:         "
        f"{OUTPUT_ROOT / 'GLOBAL_ANALYSIS_REPORT.txt'}"
    )

    print(
        f"Performance summary:   "
        f"{OUTPUT_ROOT / 'performance_summary_mean_std.csv'}"
    )

    print(
        f"Feature importance:    "
        f"{OUTPUT_ROOT / 'feature_importance_summary.csv'}"
    )

    if ENABLE_SHAP:

        print(
            f"SHAP summary:          "
            f"{OUTPUT_ROOT / 'shap_summary_mean_std.csv'}"
        )


if __name__ == "__main__":
    main()