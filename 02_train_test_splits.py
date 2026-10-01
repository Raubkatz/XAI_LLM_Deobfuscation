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
    2. standard random undersampling
    3. ADASYN
    4. standard random undersampling + ADASYN
    5. custom minority sampling
    6. ADASYN + custom minority sampling


CUSTOM MINORITY SAMPLING
------------------------

The custom method is designed for binary classification.

The custom method:

    1. identifies the minority and majority class labels
    2. uses the CURRENT number of minority observations
    3. retains a configured fraction of the CURRENT minority class
    4. randomly samples exactly the same number of majority observations


Example WITHOUT ADASYN:

    original minority = 1,000

    CUSTOM_MINORITY_FRACTION = 0.90

    retained minority = 900
    retained majority = 900

    final training data = 1,800


Example WITH ADASYN:

    original minority = 1,000

    ADASYN increases minority to:

        3,000

    CUSTOM_MINORITY_FRACTION = 0.90

    retained minority =

        floor(3,000 * 0.90)
        = 2,700

    retained majority = 2,700

    final training data = 5,400


Therefore the custom sampling size is calculated AFTER ADASYN when ADASYN is
enabled.


INDEPENDENT SWITCHES
--------------------

The following four configurations are supported:


1. NO ADASYN, NO CUSTOM UNDERSAMPLING

    USE_ADASYN = False
    USE_CUSTOM_UNDERSAMPLING = False

    training
        ->
    unchanged training


2. ADASYN ONLY

    USE_ADASYN = True
    USE_CUSTOM_UNDERSAMPLING = False

    training
        ->
    ADASYN
        ->
    final training


3. CUSTOM UNDERSAMPLING ONLY

    USE_ADASYN = False
    USE_CUSTOM_UNDERSAMPLING = True

    training
        ->
    custom sampling
        ->
    final training


4. ADASYN + CUSTOM UNDERSAMPLING

    USE_ADASYN = True
    USE_CUSTOM_UNDERSAMPLING = True

    training
        ->
    ADASYN
        ->
    determine CURRENT post-ADASYN minority count
        ->
    retain configured fraction
        ->
    sample same number from majority
        ->
    final training


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

    configured balancing method

            |
            v

    final training dataset


The TEST dataset is NEVER:

    - undersampled
    - oversampled
    - processed with ADASYN
    - processed with custom minority sampling
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
# ALL METHODS CAN BE SWITCHED INDEPENDENTLY.
#
#
# -------------------------------------------------------------------------
# MODE 1
# -------------------------------------------------------------------------
#
# USE_UNDERSAMPLING        = False
# USE_ADASYN               = False
# USE_CUSTOM_UNDERSAMPLING = False
#
#     -> no balancing
#
#
# -------------------------------------------------------------------------
# MODE 2
# -------------------------------------------------------------------------
#
# USE_UNDERSAMPLING        = False
# USE_ADASYN               = True
# USE_CUSTOM_UNDERSAMPLING = False
#
#     -> ADASYN only
#
#
# -------------------------------------------------------------------------
# MODE 3
# -------------------------------------------------------------------------
#
# USE_UNDERSAMPLING        = False
# USE_ADASYN               = False
# USE_CUSTOM_UNDERSAMPLING = True
#
#     -> custom undersampling only
#
#
# -------------------------------------------------------------------------
# MODE 4
# -------------------------------------------------------------------------
#
# USE_UNDERSAMPLING        = False
# USE_ADASYN               = True
# USE_CUSTOM_UNDERSAMPLING = True
#
#     -> ADASYN FIRST
#     -> custom undersampling SECOND
#
#
# -------------------------------------------------------------------------
# ORIGINAL STANDARD UNDERSAMPLING
# -------------------------------------------------------------------------
#
# USE_UNDERSAMPLING        = True
# USE_CUSTOM_UNDERSAMPLING = False
#
# keeps the original RandomUnderSampler implementation.
#
#
# IMPORTANT:
#
# USE_UNDERSAMPLING and USE_CUSTOM_UNDERSAMPLING represent two alternative
# undersampling methods and must NOT both be True.
#
# =============================================================================


# Original standard RandomUnderSampler.

USE_UNDERSAMPLING = False


# Synthetic minority oversampling.

USE_ADASYN = True


# New custom minority/majority sampling.

USE_CUSTOM_UNDERSAMPLING = True


# =============================================================================
# STANDARD UNDERSAMPLING SETTINGS
# =============================================================================

UNDERSAMPLING_STRATEGY = "auto"


# =============================================================================
# CUSTOM MINORITY SAMPLING SETTINGS
# =============================================================================
#
# Fraction of the CURRENT minority population retained.
#
#
# WITHOUT ADASYN:
#
#     current minority
#         =
#     original training minority
#
#
# WITH ADASYN:
#
#     current minority
#         =
#     minority population AFTER ADASYN
#
#
# Example:
#
#     after ADASYN:
#
#         minority = 4,000
#
#     CUSTOM_MINORITY_FRACTION = 0.90
#
#     custom sample:
#
#         minority = 3,600
#         majority = 3,600
#
# =============================================================================

CUSTOM_MINORITY_FRACTION = 0.90


# =============================================================================
# ADASYN SETTINGS
# =============================================================================

ADASYN_SAMPLING_STRATEGY = "auto"

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

    if not 0.0 < CUSTOM_MINORITY_FRACTION <= 1.0:

        raise ValueError(
            "\nCUSTOM_MINORITY_FRACTION must satisfy:\n\n"
            "    0.0 < CUSTOM_MINORITY_FRACTION <= 1.0\n"
        )

    # -------------------------------------------------------------------------
    # The two undersampling implementations are alternative methods.
    # -------------------------------------------------------------------------

    if (
        USE_UNDERSAMPLING
        and USE_CUSTOM_UNDERSAMPLING
    ):

        raise ValueError(
            "\nUSE_UNDERSAMPLING and USE_CUSTOM_UNDERSAMPLING "
            "cannot both be True.\n\n"
            "Choose either:\n\n"
            "    USE_UNDERSAMPLING = True\n\n"
            "or:\n\n"
            "    USE_CUSTOM_UNDERSAMPLING = True\n"
        )

    # -------------------------------------------------------------------------
    # imbalanced-learn is only required for standard RandomUnderSampler
    # and ADASYN.
    #
    # Custom sampling itself only uses pandas/numpy.
    # -------------------------------------------------------------------------

    if (
        USE_UNDERSAMPLING
        or USE_ADASYN
    ):

        if not IMBLEARN_AVAILABLE:

            raise ImportError(
                "\nThe package 'imbalanced-learn' is required because "
                "standard undersampling and/or ADASYN is enabled.\n\n"
                "Install it with:\n\n"
                "    pip install imbalanced-learn\n"
            )


# =============================================================================
# NUMERIC CHECK FOR ADASYN
# =============================================================================

def validate_adasyn_features(
    X: pd.DataFrame,
) -> None:

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
# STANDARD UNDERSAMPLING
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
# CUSTOM MINORITY SAMPLING
# =============================================================================

def determine_binary_minority_majority_classes(
    y_train: pd.Series,
) -> Tuple[
    object,
    object,
]:

    counts = y_train.value_counts(
        dropna=False
    )

    if len(counts) != 2:

        raise ValueError(
            "\nCustom minority sampling requires exactly two target classes.\n\n"
            f"Number of classes found: {len(counts)}\n\n"
            f"{counts.to_string()}\n"
        )

    minority_class = counts.idxmin()

    majority_class = counts.idxmax()

    return (
        minority_class,
        majority_class,
    )


def apply_custom_minority_sampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int,
    minority_class,
    majority_class,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    Dict[str, object],
]:
    """
    Custom binary-class balancing.

    IMPORTANT:

    The number retained from the minority class is calculated from the CURRENT
    data passed into this function.

    Therefore:

        without ADASYN
            -> based on original training minority count

        with ADASYN
            -> based on POST-ADASYN minority count
    """

    X_current = X_train.reset_index(
        drop=True
    ).copy()

    y_current = y_train.reset_index(
        drop=True
    ).copy()

    y_current.name = TARGET_COLUMN

    # -------------------------------------------------------------------------
    # CURRENT population after all previous operations.
    #
    # If ADASYN was enabled, these values therefore include synthetic samples.
    # -------------------------------------------------------------------------

    minority_indices = y_current[
        y_current == minority_class
    ].index

    majority_indices = y_current[
        y_current == majority_class
    ].index

    current_minority_count = len(
        minority_indices
    )

    current_majority_count = len(
        majority_indices
    )

    if current_minority_count <= 0:

        raise ValueError(
            "\nCustom minority sampling found no observations for minority "
            f"class '{minority_class}'."
        )

    if current_majority_count <= 0:

        raise ValueError(
            "\nCustom minority sampling found no observations for majority "
            f"class '{majority_class}'."
        )

    # -------------------------------------------------------------------------
    # KEY CALCULATION
    #
    # The fraction is calculated from the ACTUAL CURRENT minority count.
    #
    # Example after ADASYN:
    #
    #     current minority = 5,000
    #     fraction         = 0.90
    #
    #     requested = floor(5,000 * 0.90)
    #               = 4,500
    # -------------------------------------------------------------------------

    requested_sample_size = int(
        np.floor(
            current_minority_count
            * CUSTOM_MINORITY_FRACTION
        )
    )

    requested_sample_size = max(
        1,
        requested_sample_size,
    )

    # -------------------------------------------------------------------------
    # Need exactly the same number from each class.
    #
    # Normally the majority class has enough observations.
    #
    # The min() is only a safety mechanism in case ADASYN results in a minority
    # population whose requested 90% is larger than the available majority.
    # -------------------------------------------------------------------------

    actual_sample_size = min(
        requested_sample_size,
        current_minority_count,
        current_majority_count,
    )

    # -------------------------------------------------------------------------
    # Sample minority.
    # -------------------------------------------------------------------------

    minority_selected = (
        pd.Series(
            minority_indices,
            dtype=int,
        )
        .sample(
            n=actual_sample_size,
            replace=False,
            random_state=random_seed,
        )
        .tolist()
    )

    # -------------------------------------------------------------------------
    # Sample exactly the same number from majority.
    # -------------------------------------------------------------------------

    majority_selected = (
        pd.Series(
            majority_indices,
            dtype=int,
        )
        .sample(
            n=actual_sample_size,
            replace=False,
            random_state=random_seed + 1,
        )
        .tolist()
    )

    selected_indices = (
        minority_selected
        + majority_selected
    )

    # -------------------------------------------------------------------------
    # Shuffle final balanced training set.
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        random_seed + 2
    )

    rng.shuffle(
        selected_indices
    )

    X_resampled = (
        X_current
        .iloc[
            selected_indices
        ]
        .reset_index(
            drop=True
        )
    )

    y_resampled = (
        y_current
        .iloc[
            selected_indices
        ]
        .reset_index(
            drop=True
        )
    )

    y_resampled.name = TARGET_COLUMN

    details: Dict[
        str,
        object
    ] = {

        "custom_minority_class":
            str(
                minority_class
            ),

        "custom_majority_class":
            str(
                majority_class
            ),

        "custom_minority_fraction":
            CUSTOM_MINORITY_FRACTION,

        "custom_current_minority_before_sampling":
            current_minority_count,

        "custom_current_majority_before_sampling":
            current_majority_count,

        "custom_requested_per_class":
            requested_sample_size,

        "custom_actual_per_class":
            actual_sample_size,

        "custom_final_minority":
            actual_sample_size,

        "custom_final_majority":
            actual_sample_size,

        "custom_final_total":
            2 * actual_sample_size,
    }

    return (
        X_resampled,
        y_resampled,
        details,
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

    X_current = X_train.copy()

    y_current = y_train.copy()

    report: Dict[str, object] = {

        "before_balancing":
            class_counts(
                y_current
            ),

        "standard_undersampling_enabled":
            USE_UNDERSAMPLING,

        "custom_undersampling_enabled":
            USE_CUSTOM_UNDERSAMPLING,

        "custom_minority_fraction":
            CUSTOM_MINORITY_FRACTION,

        "adasyn_enabled":
            USE_ADASYN,
    }

    # =========================================================================
    # CUSTOM PIPELINE
    # =========================================================================
    #
    # Custom undersampling and ADASYN are independent switches.
    #
    # If custom sampling is enabled:
    #
    #     determine class identities
    #
    # then:
    #
    #     optional ADASYN
    #
    # then:
    #
    #     custom sampling using CURRENT post-ADASYN population
    #
    # =========================================================================

    if USE_CUSTOM_UNDERSAMPLING:

        # ---------------------------------------------------------------------
        # Determine minority/majority LABELS from the original training split.
        #
        # The labels remain fixed so that after ADASYN we still know which class
        # was the original minority class.
        # ---------------------------------------------------------------------

        (
            minority_class,
            majority_class,
        ) = determine_binary_minority_majority_classes(
            y_current
        )

        report[
            "original_minority_class"
        ] = str(
            minority_class
        )

        report[
            "original_majority_class"
        ] = str(
            majority_class
        )

        # ---------------------------------------------------------------------
        # OPTIONAL ADASYN FIRST
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # CUSTOM UNDERSAMPLING
        #
        # IMPORTANT:
        #
        # X_current / y_current are passed HERE.
        #
        # Therefore, when ADASYN is enabled, the custom fraction is calculated
        # from the new POST-ADASYN minority count.
        # ---------------------------------------------------------------------

        (
            X_current,
            y_current,
            custom_details,
        ) = apply_custom_minority_sampling(
            X_train=X_current,
            y_train=y_current,
            random_seed=random_seed,
            minority_class=minority_class,
            majority_class=majority_class,
        )

        report[
            "after_custom_undersampling"
        ] = class_counts(
            y_current
        )

        for key, value in custom_details.items():

            report[
                key
            ] = value

    # =========================================================================
    # NON-CUSTOM PIPELINE
    # =========================================================================

    else:

        # ---------------------------------------------------------------------
        # OPTIONAL ORIGINAL STANDARD UNDERSAMPLING
        # ---------------------------------------------------------------------

        if USE_UNDERSAMPLING:

            X_current, y_current = apply_undersampling(
                X_current,
                y_current,
                random_seed,
            )

            report[
                "after_standard_undersampling"
            ] = class_counts(
                y_current
            )

        # ---------------------------------------------------------------------
        # OPTIONAL ADASYN
        #
        # This works even when standard undersampling is disabled.
        # ---------------------------------------------------------------------

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
        f"Standard undersampling:      {USE_UNDERSAMPLING}"
    )

    lines.append(
        f"Custom undersampling:        {USE_CUSTOM_UNDERSAMPLING}"
    )

    lines.append(
        f"Custom minority fraction:    {CUSTOM_MINORITY_FRACTION:.4f}"
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
        original_dataset[
            TARGET_COLUMN
        ],
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
        train_before[
            TARGET_COLUMN
        ],
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
        train_after[
            TARGET_COLUMN
        ],
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
        test[
            TARGET_COLUMN
        ],
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
        f"Standard under:  "
        f"{USE_UNDERSAMPLING}"
    )

    print(
        f"Custom under:    "
        f"{USE_CUSTOM_UNDERSAMPLING}"
    )

    print(
        f"Custom fraction: "
        f"{CUSTOM_MINORITY_FRACTION:.2%}"
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

    dataset = (
        dataset
        .dropna(
            how="all"
        )
        .reset_index(
            drop=True
        )
    )

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

    if (
        USE_CUSTOM_UNDERSAMPLING
        and len(target_counts) != 2
    ):

        raise ValueError(
            "\nUSE_CUSTOM_UNDERSAMPLING=True requires exactly two classes.\n\n"
            f"Classes found: {len(target_counts)}\n\n"
            f"{target_counts.to_string()}\n"
        )

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

    print(
        "Complete dataset class distribution:"
    )

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
        # Preserve pre-balancing training data.
        # ---------------------------------------------------------------------

        train_before = X_train.copy()

        train_before[
            TARGET_COLUMN
        ] = y_train.values

        # ---------------------------------------------------------------------
        # Test remains untouched.
        # ---------------------------------------------------------------------

        test_final = X_test.copy()

        test_final[
            TARGET_COLUMN
        ] = y_test.values

        # ---------------------------------------------------------------------
        # Training balancing.
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
        # Seed output.
        # ---------------------------------------------------------------------

        seed_directory = (
            OUTPUT_ROOT
            / f"seed_{random_seed:04d}"
        )

        ensure_directory(
            seed_directory
        )

        train_final.to_csv(
            seed_directory
            / "train.csv",
            index=False,
        )

        test_final.to_csv(
            seed_directory
            / "test.csv",
            index=False,
        )

        if SAVE_UNBALANCED_TRAIN:

            train_before.to_csv(
                seed_directory
                / "train_before_balancing.csv",
                index=False,
            )

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
    # DETERMINE BALANCING MODE
    # =========================================================================

    if (
        USE_CUSTOM_UNDERSAMPLING
        and USE_ADASYN
    ):

        balancing_mode = (
            "ADASYN + custom minority sampling"
        )

    elif USE_CUSTOM_UNDERSAMPLING:

        balancing_mode = (
            "custom minority sampling only"
        )

    elif (
        USE_UNDERSAMPLING
        and USE_ADASYN
    ):

        balancing_mode = (
            "standard undersampling + ADASYN"
        )

    elif USE_UNDERSAMPLING:

        balancing_mode = (
            "standard undersampling only"
        )

    elif USE_ADASYN:

        balancing_mode = (
            "ADASYN only"
        )

    else:

        balancing_mode = (
            "none"
        )

    # =========================================================================
    # MAIN REPORT
    # =========================================================================

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
        f"Standard undersampling:     "
        f"{USE_UNDERSAMPLING}"
    )

    report_lines.append(
        f"Standard under strategy:    "
        f"{UNDERSAMPLING_STRATEGY}"
    )

    report_lines.append(
        f"Custom undersampling:       "
        f"{USE_CUSTOM_UNDERSAMPLING}"
    )

    report_lines.append(
        f"Custom minority fraction:   "
        f"{CUSTOM_MINORITY_FRACTION:.6f}"
    )

    report_lines.append(
        f"ADASYN enabled:             "
        f"{USE_ADASYN}"
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

    if (
        USE_CUSTOM_UNDERSAMPLING
        and USE_ADASYN
    ):

        report_lines.append(
            "ADASYN is applied first. The custom sampling fraction is then "
            "calculated from the actual post-ADASYN minority population."
        )

    elif USE_CUSTOM_UNDERSAMPLING:

        report_lines.append(
            "Custom sampling is applied directly to the original training partition."
        )

    elif USE_ADASYN:

        report_lines.append(
            "ADASYN is applied to the training partition."
        )

    else:

        report_lines.append(
            "No synthetic/custom balancing is applied to the training partition."
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