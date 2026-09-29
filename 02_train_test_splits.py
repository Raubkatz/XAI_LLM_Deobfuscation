#!/usr/bin/env python3
"""
02_LLM_OBF_create_train_test_splits.py

Stage 02 of the LLM deobfuscation machine-learning pipeline.

Input
-----
LLM_OBF_01_prepared_dataset/
    LLM_OBF_dataset.csv

Expected structure:

    feature_1
    feature_2
    ...
    feature_n
    class


This script creates repeated stratified train/test splits over many random
seeds.

Optional TRAINING-SET balancing methods:

    1. no balancing
    2. random undersampling
    3. ADASYN
    4. random undersampling + ADASYN

IMPORTANT METHODOLOGICAL RULE
-----------------------------

The workflow for EVERY random seed is:

    complete Stage-01 dataset

            |
            v

    stratified train/test split

            |
            +---------------------> TEST
            |                       untouched
            |
            v
          TRAIN

            |
            v

    optional undersampling
            |
            v
    optional ADASYN

            |
            v

    final training dataset


The TEST dataset is NEVER:

    - undersampled
    - oversampled
    - processed with ADASYN
    - balanced

This prevents evaluation leakage and preserves the natural test distribution.


ADASYN NOTE
-----------

ADASYN requires numeric features.

If ADASYN is enabled and non-numeric feature columns are present, this script
stops with an explicit error.

Do NOT silently label-encode arbitrary categorical variables before ADASYN,
because Euclidean interpolation between arbitrary category codes does not have
a meaningful interpretation.

If categorical variables should later be included in synthetic oversampling,
a categorical-aware method such as SMOTENC should be considered separately.


Outputs
-------

LLM_OBF_02_train_test_splits/

    split_summary.csv
    STAGE02_REPORT.txt

    seed_0000/
        train.csv
        test.csv
        split_report.txt

    seed_0001/
        train.csv
        test.csv
        split_report.txt

    ...

The exact folder numbering corresponds to the configured RANDOM_SEEDS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# =============================================================================
# OPTIONAL IMBALANCED-LEARN IMPORTS
# =============================================================================

try:

    from imblearn.over_sampling import ADASYN
    from imblearn.under_sampling import RandomUnderSampler

    IMBLEARN_AVAILABLE = True

except ImportError:

    IMBLEARN_AVAILABLE = False


# =============================================================================
# PATHS
# =============================================================================

INPUT_CSV = Path(
    "LLM_OBF_01_prepared_dataset"
) / "LLM_OBF_dataset.csv"

OUTPUT_ROOT = Path(
    "LLM_OBF_02_train_test_splits"
)

SUMMARY_CSV = OUTPUT_ROOT / "split_summary.csv"

MAIN_REPORT = OUTPUT_ROOT / "STAGE02_REPORT.txt"


# =============================================================================
# TARGET
# =============================================================================

TARGET_COLUMN = "class"


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

TEST_SIZE = 0.20

STRATIFY = True


# =============================================================================
# RANDOM SEEDS
# =============================================================================
#
# Example:
#
#     N_RANDOM_SEEDS = 1000
#
# produces seeds:
#
#     0, 1, 2, ..., 999
#
# You can change the starting value if desired.
#

N_RANDOM_SEEDS = 100

RANDOM_SEED_START = 0

RANDOM_SEEDS = list(
    range(
        RANDOM_SEED_START,
        RANDOM_SEED_START + N_RANDOM_SEEDS,
    )
)


# =============================================================================
# BALANCING SWITCHES
# =============================================================================
#
# These two switches give the four requested modes.
#
#
# USE_UNDERSAMPLING = False
# USE_ADASYN         = False
#
#     -> no balancing
#
#
# USE_UNDERSAMPLING = True
# USE_ADASYN         = False
#
#     -> undersampling only
#
#
# USE_UNDERSAMPLING = False
# USE_ADASYN         = True
#
#     -> ADASYN only
#
#
# USE_UNDERSAMPLING = True
# USE_ADASYN         = True
#
#     -> undersampling first, then ADASYN
#

USE_UNDERSAMPLING = True

USE_ADASYN = True


# =============================================================================
# UNDERSAMPLING SETTINGS
# =============================================================================
#
# Fraction of each class that may be retained relative to the minority class.
#
# Recommended interpretation:
#
#     1.0
#
# means that undersampling produces an exactly balanced training dataset:
#
#     majority = minority
#
# For binary classes.
#
# With multiclass targets, RandomUnderSampler automatically reduces larger
# classes according to UNDERSAMPLING_STRATEGY.
#

UNDERSAMPLING_STRATEGY = "auto"


# =============================================================================
# ADASYN SETTINGS
# =============================================================================
#
# "auto":
#
#     resample all minority classes toward the majority class.
#

ADASYN_SAMPLING_STRATEGY = "auto"


# Number of nearest neighbours used by ADASYN.
#
# imbalanced-learn's conventional default is 5.
#

ADASYN_N_NEIGHBORS = 5


# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

SAVE_UNBALANCED_TRAIN = False

OVERWRITE_EXISTING = True


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_directory(
    path: Path,
) -> None:

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def class_counts(
    values: pd.Series,
) -> Dict[str, int]:
    """
    Return class counts in a JSON/report-friendly dictionary.
    """

    counts = values.value_counts(
        dropna=False
    )

    result: Dict[str, int] = {}

    for key, value in counts.items():

        if pd.isna(key):
            label = "<MISSING>"
        else:
            label = str(key)

        result[label] = int(value)

    return result


def class_percentages(
    values: pd.Series,
) -> Dict[str, float]:
    """
    Return class percentages.
    """

    counts = values.value_counts(
        normalize=True,
        dropna=False,
    )

    result: Dict[str, float] = {}

    for key, value in counts.items():

        if pd.isna(key):
            label = "<MISSING>"
        else:
            label = str(key)

        result[label] = (
            float(value) * 100.0
        )

    return result


def imbalance_ratio(
    values: pd.Series,
) -> float:
    """
    Majority-class count / minority-class count.

    Returns NaN if fewer than two classes exist.
    """

    counts = values.value_counts(
        dropna=True
    )

    if len(counts) < 2:

        return np.nan

    smallest = int(
        counts.min()
    )

    largest = int(
        counts.max()
    )

    if smallest <= 0:

        return np.inf

    return (
        largest
        / smallest
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_configuration() -> None:

    if not 0.0 < TEST_SIZE < 1.0:

        raise ValueError(
            "TEST_SIZE must be strictly between 0 and 1."
        )

    if N_RANDOM_SEEDS <= 0:

        raise ValueError(
            "N_RANDOM_SEEDS must be >= 1."
        )

    if ADASYN_N_NEIGHBORS <= 0:

        raise ValueError(
            "ADASYN_N_NEIGHBORS must be >= 1."
        )

    if (
        USE_UNDERSAMPLING
        or USE_ADASYN
    ):

        if not IMBLEARN_AVAILABLE:

            raise ImportError(
                "\nThe package 'imbalanced-learn' is required because "
                "undersampling and/or ADASYN is enabled.\n\n"
                "Install it with:\n\n"
                "    pip install imbalanced-learn\n"
            )


# =============================================================================
# NUMERIC CHECK FOR ADASYN
# =============================================================================

def validate_adasyn_features(
    X: pd.DataFrame,
) -> None:
    """
    ADASYN requires a numeric feature space.

    Fail explicitly when symbolic/categorical columns remain.
    """

    if not USE_ADASYN:

        return

    non_numeric = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    if non_numeric:

        raise ValueError(
            "\nADASYN is enabled, but the following model features "
            "are non-numeric:\n\n"
            + "\n".join(
                f"    {column}"
                for column in non_numeric
            )
            + "\n\nADASYN operates in a numeric distance space. "
            "Remove these features from Stage 01, convert them using a "
            "methodologically justified representation, or disable ADASYN."
        )


# =============================================================================
# NUMERIC CLEANUP FOR ADASYN
# =============================================================================

def prepare_numeric_for_adasyn(
    X_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    ADASYN cannot work with NaN or infinity.

    Missing numeric values are filled using medians calculated ONLY from the
    current training split.

    This function is called only when ADASYN is enabled.

    The test set remains untouched.
    """

    X = X_train.copy()

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    for column in X.columns:

        numeric = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        median = numeric.median()

        if pd.isna(median):

            raise ValueError(
                f"\nADASYN cannot process feature '{column}' because "
                "the complete training feature column is missing."
            )

        X[column] = numeric.fillna(
            median
        )

    return X


# =============================================================================
# UNDERSAMPLING
# =============================================================================

def apply_undersampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
]:

    sampler = RandomUnderSampler(
        sampling_strategy=UNDERSAMPLING_STRATEGY,
        random_state=random_seed,
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train,
    )

    X_resampled = pd.DataFrame(
        X_resampled,
        columns=X_train.columns,
    )

    y_resampled = pd.Series(
        y_resampled,
        name=TARGET_COLUMN,
    )

    return (
        X_resampled,
        y_resampled,
    )


# =============================================================================
# ADASYN
# =============================================================================

def calculate_safe_adasyn_neighbors(
    y_train: pd.Series,
) -> int:
    """
    ADASYN requires enough observations in minority classes.

    n_neighbors must be smaller than the number of observations available in
    the smallest class involved in oversampling.

    Reduce the configured number automatically where necessary.
    """

    counts = y_train.value_counts()

    if len(counts) < 2:

        raise ValueError(
            "ADASYN requires at least two target classes."
        )

    minimum_class_size = int(
        counts.min()
    )

    safe_neighbors = min(
        ADASYN_N_NEIGHBORS,
        minimum_class_size - 1,
    )

    if safe_neighbors < 1:

        raise ValueError(
            "\nADASYN cannot be performed because at least one training "
            "class contains fewer than 2 observations."
        )

    return safe_neighbors


def apply_adasyn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    int,
]:

    safe_neighbors = calculate_safe_adasyn_neighbors(
        y_train
    )

    sampler = ADASYN(
        sampling_strategy=ADASYN_SAMPLING_STRATEGY,
        random_state=random_seed,
        n_neighbors=safe_neighbors,
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train,
    )

    X_resampled = pd.DataFrame(
        X_resampled,
        columns=X_train.columns,
    )

    y_resampled = pd.Series(
        y_resampled,
        name=TARGET_COLUMN,
    )

    return (
        X_resampled,
        y_resampled,
        safe_neighbors,
    )


# =============================================================================
# BALANCING PIPELINE
# =============================================================================

def balance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    Dict[str, object],
]:
    """
    Apply the configured balancing operations ONLY to training data.

    Order when both methods are enabled:

        original training data
            -> undersampling
            -> ADASYN
    """

    X_current = X_train.copy()

    y_current = y_train.copy()

    report: Dict[str, object] = {

        "before_balancing":
            class_counts(
                y_current
            ),

        "undersampling_enabled":
            USE_UNDERSAMPLING,

        "adasyn_enabled":
            USE_ADASYN,
    }

    # -------------------------------------------------------------------------
    # UNDERSAMPLING
    # -------------------------------------------------------------------------

    if USE_UNDERSAMPLING:

        X_current, y_current = apply_undersampling(
            X_current,
            y_current,
            random_seed,
        )

        report[
            "after_undersampling"
        ] = class_counts(
            y_current
        )

    # -------------------------------------------------------------------------
    # ADASYN
    # -------------------------------------------------------------------------

    if USE_ADASYN:

        validate_adasyn_features(
            X_current
        )

        X_current = prepare_numeric_for_adasyn(
            X_current
        )

        (
            X_current,
            y_current,
            actual_neighbors,
        ) = apply_adasyn(
            X_current,
            y_current,
            random_seed,
        )

        report[
            "adasyn_n_neighbors"
        ] = actual_neighbors

        report[
            "after_adasyn"
        ] = class_counts(
            y_current
        )

    report[
        "final_training_distribution"
    ] = class_counts(
        y_current
    )

    return (
        X_current,
        y_current,
        report,
    )


# =============================================================================
# SPLIT REPORT
# =============================================================================

def create_split_report(
    random_seed: int,
    original_dataset: pd.DataFrame,
    train_before: pd.DataFrame,
    train_after: pd.DataFrame,
    test: pd.DataFrame,
    balancing_report: Dict[str, object],
) -> str:

    lines: List[str] = []

    separator = "=" * 90

    lines.append(separator)
    lines.append(
        "LLM DEOBFUSCATION ML - TRAIN / TEST SPLIT REPORT"
    )
    lines.append(separator)
    lines.append("")

    lines.append(
        f"Random seed:                 {random_seed}"
    )

    lines.append(
        f"Test fraction:               {TEST_SIZE:.4f}"
    )

    lines.append(
        f"Stratification enabled:      {STRATIFY}"
    )

    lines.append(
        f"Undersampling enabled:       {USE_UNDERSAMPLING}"
    )

    lines.append(
        f"ADASYN enabled:              {USE_ADASYN}"
    )

    lines.append("")

    # -------------------------------------------------------------------------
    # Dataset sizes
    # -------------------------------------------------------------------------

    lines.append(separator)
    lines.append("DATASET SIZES")
    lines.append(separator)
    lines.append("")

    lines.append(
        f"Complete Stage-01 dataset:   "
        f"{len(original_dataset):,}"
    )

    lines.append(
        f"Training before balancing:   "
        f"{len(train_before):,}"
    )

    lines.append(
        f"Training after balancing:    "
        f"{len(train_after):,}"
    )

    lines.append(
        f"Untouched testing dataset:   "
        f"{len(test):,}"
    )

    lines.append("")

    # -------------------------------------------------------------------------
    # Original distribution
    # -------------------------------------------------------------------------

    lines.append(separator)
    lines.append("ORIGINAL DATASET CLASS DISTRIBUTION")
    lines.append(separator)
    lines.append("")

    append_distribution(
        lines,
        original_dataset[TARGET_COLUMN],
    )

    # -------------------------------------------------------------------------
    # Training before balancing
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append(separator)
    lines.append("TRAINING DISTRIBUTION BEFORE BALANCING")
    lines.append(separator)
    lines.append("")

    append_distribution(
        lines,
        train_before[TARGET_COLUMN],
    )

    # -------------------------------------------------------------------------
    # Training after balancing
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append(separator)
    lines.append("TRAINING DISTRIBUTION AFTER BALANCING")
    lines.append(separator)
    lines.append("")

    append_distribution(
        lines,
        train_after[TARGET_COLUMN],
    )

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append(separator)
    lines.append("TEST DISTRIBUTION - UNTOUCHED")
    lines.append(separator)
    lines.append("")

    append_distribution(
        lines,
        test[TARGET_COLUMN],
    )

    # -------------------------------------------------------------------------
    # Method details
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append(separator)
    lines.append("BALANCING DETAILS")
    lines.append(separator)
    lines.append("")

    for key, value in balancing_report.items():

        lines.append(
            f"{key}: {value}"
        )

    lines.append("")

    return "\n".join(
        lines
    )


def append_distribution(
    lines: List[str],
    values: pd.Series,
) -> None:

    counts = values.value_counts(
        dropna=False
    )

    total = len(
        values
    )

    for class_value, count in counts.items():

        percentage = (
            100.0 * count / total
            if total
            else 0.0
        )

        lines.append(
            f"{str(class_value):<40}"
            f"{count:>12,}"
            f"{percentage:>12.2f}%"
        )

    ratio = imbalance_ratio(
        values
    )

    lines.append("")

    lines.append(
        f"Majority/minority ratio: "
        f"{ratio:.6f}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    validate_configuration()

    # =========================================================================
    # INPUT
    # =========================================================================

    if not INPUT_CSV.exists():

        raise SystemExit(
            "\nStage-01 dataset not found:\n"
            f"    {INPUT_CSV.resolve()}\n"
        )

    ensure_directory(
        OUTPUT_ROOT
    )

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print("=" * 90)
    print("LLM OBFUSCATION ML - STAGE 02")
    print("REPEATED STRATIFIED TRAIN / TEST SPLITTING")
    print("=" * 90)
    print()

    print(
        f"Input CSV:       "
        f"{INPUT_CSV.resolve()}"
    )

    print(
        f"Number seeds:    "
        f"{len(RANDOM_SEEDS):,}"
    )

    print(
        f"Test size:       "
        f"{TEST_SIZE:.2%}"
    )

    print(
        f"Undersampling:   "
        f"{USE_UNDERSAMPLING}"
    )

    print(
        f"ADASYN:          "
        f"{USE_ADASYN}"
    )

    print()

    dataset = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    # =========================================================================
    # VALIDATE DATASET
    # =========================================================================

    if TARGET_COLUMN not in dataset.columns:

        raise ValueError(
            f"\nTarget column '{TARGET_COLUMN}' "
            "is missing from the Stage-01 CSV."
        )

    # Remove completely empty rows defensively.

    dataset = (
        dataset
        .dropna(
            how="all"
        )
        .reset_index(
            drop=True
        )
    )

    # Missing target values cannot participate in stratification.

    dataset = dataset[
        dataset[
            TARGET_COLUMN
        ].notna()
    ].copy()

    dataset = dataset.reset_index(
        drop=True
    )

    feature_columns = [
        column
        for column in dataset.columns
        if column != TARGET_COLUMN
    ]

    if not feature_columns:

        raise ValueError(
            "No feature columns remain."
        )

    # =========================================================================
    # CLASS VALIDATION
    # =========================================================================

    target_counts = dataset[
        TARGET_COLUMN
    ].value_counts()

    if len(target_counts) < 2:

        raise ValueError(
            "The target contains fewer than two classes."
        )

    minimum_class_count = int(
        target_counts.min()
    )

    if minimum_class_count < 2:

        raise ValueError(
            "\nAt least one class contains fewer than two observations. "
            "A stratified train/test split cannot be produced reliably."
        )

    # ADASYN feature validation can already be performed globally.

    if USE_ADASYN:

        validate_adasyn_features(
            dataset[
                feature_columns
            ]
        )

    # =========================================================================
    # PRINT INITIAL DISTRIBUTION
    # =========================================================================

    print(
        f"Rows:            "
        f"{len(dataset):,}"
    )

    print(
        f"Features:        "
        f"{len(feature_columns):,}"
    )

    print(
        f"Classes:         "
        f"{len(target_counts):,}"
    )

    print()

    print("Complete dataset class distribution:")

    for class_value, count in target_counts.items():

        percentage = (
            100.0
            * count
            / len(dataset)
        )

        print(
            f"    {str(class_value):<35}"
            f"{count:>10,}"
            f" ({percentage:>7.2f}%)"
        )

    print()

    # =========================================================================
    # X / y
    # =========================================================================

    X = dataset[
        feature_columns
    ].copy()

    y = dataset[
        TARGET_COLUMN
    ].copy()

    # =========================================================================
    # REPEATED SPLITTING
    # =========================================================================

    summary_rows: List[
        Dict[str, object]
    ] = []

    for iteration, random_seed in enumerate(
        RANDOM_SEEDS,
        start=1,
    ):

        # ---------------------------------------------------------------------
        # Stratified split
        # ---------------------------------------------------------------------

        stratification_vector = (
            y
            if STRATIFY
            else None
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=random_seed,
            stratify=stratification_vector,
        )

        # ---------------------------------------------------------------------
        # Preserve pre-balancing training data
        # ---------------------------------------------------------------------

        train_before = X_train.copy()

        train_before[
            TARGET_COLUMN
        ] = y_train.values

        # ---------------------------------------------------------------------
        # TEST is constructed immediately and then left untouched.
        # ---------------------------------------------------------------------

        test_final = X_test.copy()

        test_final[
            TARGET_COLUMN
        ] = y_test.values

        # ---------------------------------------------------------------------
        # Training balancing
        # ---------------------------------------------------------------------

        (
            X_train_final,
            y_train_final,
            balancing_report,
        ) = balance_training_data(
            X_train,
            y_train,
            random_seed,
        )

        train_final = X_train_final.copy()

        train_final[
            TARGET_COLUMN
        ] = y_train_final.values

        # ---------------------------------------------------------------------
        # Seed output directory
        # ---------------------------------------------------------------------

        seed_directory = (
            OUTPUT_ROOT
            / f"seed_{random_seed:04d}"
        )

        ensure_directory(
            seed_directory
        )

        # ---------------------------------------------------------------------
        # Save final training set
        # ---------------------------------------------------------------------

        train_final.to_csv(
            seed_directory
            / "train.csv",
            index=False,
        )

        # ---------------------------------------------------------------------
        # Save untouched test set
        # ---------------------------------------------------------------------

        test_final.to_csv(
            seed_directory
            / "test.csv",
            index=False,
        )

        # ---------------------------------------------------------------------
        # Optional original training data
        # ---------------------------------------------------------------------

        if SAVE_UNBALANCED_TRAIN:

            train_before.to_csv(
                seed_directory
                / "train_before_balancing.csv",
                index=False,
            )

        # ---------------------------------------------------------------------
        # Per-seed report
        # ---------------------------------------------------------------------

        split_report = create_split_report(
            random_seed=random_seed,
            original_dataset=dataset,
            train_before=train_before,
            train_after=train_final,
            test=test_final,
            balancing_report=balancing_report,
        )

        (
            seed_directory
            / "split_report.txt"
        ).write_text(
            split_report,
            encoding="utf-8",
        )

        # ---------------------------------------------------------------------
        # Summary row
        # ---------------------------------------------------------------------

        train_before_counts = class_counts(
            y_train
        )

        train_after_counts = class_counts(
            y_train_final
        )

        test_counts = class_counts(
            y_test
        )

        summary_row: Dict[
            str,
            object
        ] = {

            "iteration":
                iteration,

            "random_seed":
                random_seed,

            "n_total":
                len(dataset),

            "n_train_before_balancing":
                len(train_before),

            "n_train_after_balancing":
                len(train_final),

            "n_test":
                len(test_final),

            "train_imbalance_before":
                imbalance_ratio(
                    y_train
                ),

            "train_imbalance_after":
                imbalance_ratio(
                    y_train_final
                ),

            "test_imbalance":
                imbalance_ratio(
                    y_test
                ),
        }

        # Add class-specific counts.

        all_classes = sorted(
            map(
                str,
                target_counts.index
            )
        )

        for class_name in all_classes:

            summary_row[
                f"train_before_class_{class_name}"
            ] = train_before_counts.get(
                class_name,
                0,
            )

            summary_row[
                f"train_after_class_{class_name}"
            ] = train_after_counts.get(
                class_name,
                0,
            )

            summary_row[
                f"test_class_{class_name}"
            ] = test_counts.get(
                class_name,
                0,
            )

        summary_rows.append(
            summary_row
        )

        # ---------------------------------------------------------------------
        # Console progress
        # ---------------------------------------------------------------------

        if (
            iteration == 1
            or iteration % 10 == 0
            or iteration == len(RANDOM_SEEDS)
        ):

            print(
                f"[{iteration:>5,}/"
                f"{len(RANDOM_SEEDS):,}] "
                f"seed={random_seed:<8} "
                f"train={len(train_final):>8,} "
                f"test={len(test_final):>8,}"
            )

    # =========================================================================
    # SAVE GLOBAL SUMMARY
    # =========================================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # =========================================================================
    # MAIN REPORT
    # =========================================================================

    if (
        USE_UNDERSAMPLING
        and USE_ADASYN
    ):

        balancing_mode = (
            "undersampling + ADASYN"
        )

    elif USE_UNDERSAMPLING:

        balancing_mode = (
            "undersampling only"
        )

    elif USE_ADASYN:

        balancing_mode = (
            "ADASYN only"
        )

    else:

        balancing_mode = (
            "none"
        )

    report_lines: List[str] = []

    separator = "=" * 90

    report_lines.append(
        separator
    )

    report_lines.append(
        "LLM DEOBFUSCATION ML - STAGE 02 REPORT"
    )

    report_lines.append(
        separator
    )

    report_lines.append("")

    report_lines.append(
        f"Input CSV:                  "
        f"{INPUT_CSV.resolve()}"
    )

    report_lines.append(
        f"Number of input rows:       "
        f"{len(dataset):,}"
    )

    report_lines.append(
        f"Number of features:         "
        f"{len(feature_columns):,}"
    )

    report_lines.append(
        f"Target:                     "
        f"{TARGET_COLUMN}"
    )

    report_lines.append(
        f"Number of classes:          "
        f"{len(target_counts):,}"
    )

    report_lines.append(
        f"Train fraction:             "
        f"{1.0 - TEST_SIZE:.2%}"
    )

    report_lines.append(
        f"Test fraction:              "
        f"{TEST_SIZE:.2%}"
    )

    report_lines.append(
        f"Stratified splitting:       "
        f"{STRATIFY}"
    )

    report_lines.append(
        f"Number random seeds:        "
        f"{len(RANDOM_SEEDS):,}"
    )

    report_lines.append(
        f"First random seed:          "
        f"{RANDOM_SEEDS[0]}"
    )

    report_lines.append(
        f"Last random seed:           "
        f"{RANDOM_SEEDS[-1]}"
    )

    report_lines.append(
        f"Balancing mode:             "
        f"{balancing_mode}"
    )

    report_lines.append(
        f"Undersampling strategy:     "
        f"{UNDERSAMPLING_STRATEGY}"
    )

    report_lines.append(
        f"ADASYN strategy:            "
        f"{ADASYN_SAMPLING_STRATEGY}"
    )

    report_lines.append(
        f"ADASYN requested neighbors: "
        f"{ADASYN_N_NEIGHBORS}"
    )

    report_lines.append("")

    report_lines.append(
        "The test partition is created before any balancing and remains untouched."
    )

    report_lines.append(
        "Undersampling and ADASYN are applied only to the training partition."
    )

    report_lines.append("")

    report_lines.append(
        separator
    )

    report_lines.append(
        "ORIGINAL CLASS DISTRIBUTION"
    )

    report_lines.append(
        separator
    )

    report_lines.append("")

    append_distribution(
        report_lines,
        dataset[
            TARGET_COLUMN
        ],
    )

    report_lines.append("")

    report_lines.append(
        separator
    )

    report_lines.append(
        "OUTPUT"
    )

    report_lines.append(
        separator
    )

    report_lines.append("")

    report_lines.append(
        f"Split root:                 "
        f"{OUTPUT_ROOT.resolve()}"
    )

    report_lines.append(
        f"Global split summary:       "
        f"{SUMMARY_CSV.resolve()}"
    )

    report_lines.append("")

    MAIN_REPORT.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # DONE
    # =========================================================================

    print()
    print("=" * 90)
    print("STAGE 02 COMPLETE")
    print("=" * 90)
    print()

    print(
        f"Splits created:  "
        f"{len(RANDOM_SEEDS):,}"
    )

    print(
        f"Balancing mode:  "
        f"{balancing_mode}"
    )

    print(
        f"Output root:     "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Summary:         "
        f"{SUMMARY_CSV}"
    )

    print(
        f"Report:          "
        f"{MAIN_REPORT}"
    )


if __name__ == "__main__":
    main()