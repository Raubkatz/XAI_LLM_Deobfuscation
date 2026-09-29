#!/usr/bin/env python3
"""
01_LLM_OBF_prepare_dataset.py

Stage 01 of the LLM deobfuscation machine-learning pipeline.

This script ONLY:

    1. loads the previously generated combined_metrics.csv
    2. selects the configured FEATURES
    3. selects ONE configured target column
    4. renames the target column to "class"
    5. performs very light initial cleaning
    6. writes one prepared CSV
    7. writes one descriptive TXT report

This script does NOT perform:

    - train/test splitting
    - ADASYN
    - undersampling
    - oversampling
    - class balancing
    - feature scaling
    - normalization
    - categorical encoding
    - feature selection
    - model training
    - hyperparameter optimization

The resulting CSV is the input for Stage 02.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV = Path(
    "/home/sebastian/PRGRMS/LLM_deobfuscation_analysis/llm_obf/"
    "combined_metrics.csv"
)

OUTPUT_DIR = Path(
    "LLM_OBF_01_prepared_dataset"
)

OUTPUT_CSV = OUTPUT_DIR / "LLM_OBF_dataset.csv"

OUTPUT_REPORT = OUTPUT_DIR / "LLM_OBF_dataset_report.txt"


# =============================================================================
# TARGET
# =============================================================================
#
# Select ONE source column that should become the prediction target.
#
# It will be renamed to:
#
#     class
#
# in the output CSV.
#
# Examples:
#
# TARGET_SOURCE_COLUMN = "obfuscation_type"
# TARGET_SOURCE_COLUMN = "llm"
# TARGET_SOURCE_COLUMN = "decompiler"
# TARGET_SOURCE_COLUMN = "optimization"
# TARGET_SOURCE_COLUMN = "category"
#
# =============================================================================

#TARGET_SOURCE_COLUMN = "obfuscation_type"

TARGET_COLUMN = "class"
TARGET_SOURCE_COLUMN = "exe_pass"


# =============================================================================
# FEATURES
# =============================================================================
#
# This is intentionally a manually editable list.
#
# Initially all potentially useful columns are listed.
# Comment out anything you do not want to use later.
#
# The target source column is automatically removed from FEATURES if it is
# accidentally still listed here.
#
# =============================================================================

FEATURES: List[str] = [

    # -------------------------------------------------------------------------
    # Dataset / experiment information
    # -------------------------------------------------------------------------

    #"category",
    #"filename",
    #"obfuscation_type",
    #"optimization",
    #"decompiler",
    #"llm",

    # -------------------------------------------------------------------------
    # Compilation / execution metrics
    # -------------------------------------------------------------------------

    #"syntax_pass_rate",
    #"exe_pass",
    #"compile_and_link_rate",

    # -------------------------------------------------------------------------
    # Deobfuscated-code metrics
    # -------------------------------------------------------------------------

    "simplification.deobf_nloc",
    "simplification.deobf_func_num",
    "simplification.deobf_cyclomatic",
    "simplification.deobf_halstead_length",
    "simplification.deobf_avg_ccn",

    # -------------------------------------------------------------------------
    # Simplification metrics
    # -------------------------------------------------------------------------

    "simplification.decrease_nloc",
    "simplification.decrease_cyclomatic",
    "simplification.decrease_halstead_length",
    "simplification.nloc_difference_score",

    # -------------------------------------------------------------------------
    # Similarity metrics
    # -------------------------------------------------------------------------

    "similarity.codebleu",
    "similarity.ngram_match_score",
    "similarity.weighted_ngram_match_score",
    "similarity.syntax_match_score",
    "similarity.dataflow_match_score",

    # -------------------------------------------------------------------------
    # Combined metrics
    # -------------------------------------------------------------------------

    #"combined.pass",
    #"combined.simplification",
    #"combined.similarity",
    #"combined.total",
]


# =============================================================================
# LIGHT CLEANING SETTINGS
# =============================================================================

DROP_COMPLETELY_EMPTY_ROWS = True

REMOVE_UNNAMED_COLUMNS = True

NORMALIZE_EMPTY_STRINGS = True

REPLACE_INFINITY_WITH_NAN = True

DROP_ROWS_WITH_MISSING_CLASS = True


# =============================================================================
# HELPERS
# =============================================================================

def clean_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform only minimal cleaning needed to create the Stage-01 CSV.
    """

    df = df.copy()

    # -------------------------------------------------------------------------
    # Clean column names
    # -------------------------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # -------------------------------------------------------------------------
    # Remove accidental pandas index columns
    # -------------------------------------------------------------------------

    if REMOVE_UNNAMED_COLUMNS:

        unnamed_columns = [
            column
            for column in df.columns
            if str(column).lower().startswith("unnamed:")
        ]

        if unnamed_columns:

            df = df.drop(
                columns=unnamed_columns
            )

    # -------------------------------------------------------------------------
    # Normalize blank strings
    # -------------------------------------------------------------------------

    if NORMALIZE_EMPTY_STRINGS:

        df = df.replace(
            r"^\s*$",
            np.nan,
            regex=True,
        )

    # -------------------------------------------------------------------------
    # Drop completely empty rows
    # -------------------------------------------------------------------------

    if DROP_COMPLETELY_EMPTY_ROWS:

        df = df.dropna(
            how="all"
        )

    # -------------------------------------------------------------------------
    # Replace positive / negative infinity
    # -------------------------------------------------------------------------

    if REPLACE_INFINITY_WITH_NAN:

        df = df.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

    return df


def format_number(
    value,
) -> str:
    """
    Format numeric values for the TXT report.
    """

    if pd.isna(value):
        return "NaN"

    try:
        return f"{float(value):.6f}"

    except Exception:
        return str(value)


# =============================================================================
# REPORT
# =============================================================================

def create_report(
    df: pd.DataFrame,
    features: List[str],
) -> str:
    """
    Create a descriptive TXT report for the final Stage-01 CSV.
    """

    lines: List[str] = []

    separator = "=" * 90
    sub_separator = "-" * 90

    # =========================================================================
    # GENERAL
    # =========================================================================

    lines.append(separator)
    lines.append("LLM DEOBFUSCATION MACHINE-LEARNING DATASET REPORT")
    lines.append(separator)
    lines.append("")

    lines.append(f"Dataset rows:       {len(df):,}")
    lines.append(f"Feature columns:    {len(features):,}")
    lines.append(f"Target column:      {TARGET_COLUMN}")
    lines.append(f"Total columns:      {len(df.columns):,}")

    lines.append("")

    # =========================================================================
    # TARGET / CLASS DISTRIBUTION
    # =========================================================================

    lines.append(separator)
    lines.append("CLASS DISTRIBUTION")
    lines.append(separator)
    lines.append("")

    class_counts = df[TARGET_COLUMN].value_counts(
        dropna=False
    )

    total = len(df)

    for class_value, count in class_counts.items():

        percentage = (
            100.0 * count / total
            if total > 0
            else 0.0
        )

        lines.append(
            f"{str(class_value):<40}"
            f"{count:>12,}"
            f"{percentage:>12.2f}%"
        )

    lines.append("")

    # -------------------------------------------------------------------------
    # Imbalance statistics
    # -------------------------------------------------------------------------

    non_missing_counts = df[
        TARGET_COLUMN
    ].value_counts(
        dropna=True
    )

    if len(non_missing_counts) >= 2:

        majority_class = non_missing_counts.idxmax()
        minority_class = non_missing_counts.idxmin()

        majority_count = int(
            non_missing_counts.max()
        )

        minority_count = int(
            non_missing_counts.min()
        )

        imbalance_ratio = (
            majority_count / minority_count
            if minority_count > 0
            else np.inf
        )

        lines.append(
            f"Number of classes:       "
            f"{len(non_missing_counts)}"
        )

        lines.append(
            f"Majority class:           "
            f"{majority_class}"
        )

        lines.append(
            f"Majority class count:     "
            f"{majority_count:,}"
        )

        lines.append(
            f"Minority class:           "
            f"{minority_class}"
        )

        lines.append(
            f"Minority class count:     "
            f"{minority_count:,}"
        )

        lines.append(
            f"Majority/minority ratio:  "
            f"{imbalance_ratio:.4f}"
        )

    else:

        lines.append(
            "Class-imbalance statistics unavailable: "
            "fewer than two non-missing classes."
        )

    lines.append("")

    # =========================================================================
    # MISSING VALUES
    # =========================================================================

    lines.append(separator)
    lines.append("MISSING VALUES")
    lines.append(separator)
    lines.append("")

    for column in df.columns:

        missing = int(
            df[column].isna().sum()
        )

        missing_fraction = (
            100.0 * missing / len(df)
            if len(df) > 0
            else 0.0
        )

        lines.append(
            f"{column:<55}"
            f"{missing:>12,}"
            f"{missing_fraction:>12.2f}%"
        )

    lines.append("")

    # =========================================================================
    # FEATURE STATISTICS
    # =========================================================================

    lines.append(separator)
    lines.append("FEATURE STATISTICS")
    lines.append(separator)
    lines.append("")

    for feature in features:

        series = df[feature]

        lines.append(sub_separator)
        lines.append(f"FEATURE: {feature}")
        lines.append(sub_separator)

        lines.append(
            f"Pandas dtype:       {series.dtype}"
        )

        lines.append(
            f"Non-missing:        {series.notna().sum():,}"
        )

        lines.append(
            f"Missing:            {series.isna().sum():,}"
        )

        lines.append(
            f"Unique non-missing: {series.nunique(dropna=True):,}"
        )

        lines.append("")

        # ---------------------------------------------------------------------
        # Determine whether the feature is genuinely numeric
        # ---------------------------------------------------------------------

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        original_non_missing = int(
            series.notna().sum()
        )

        numeric_non_missing = int(
            numeric_series.notna().sum()
        )

        # Treat as numeric only when all non-missing values can be interpreted
        # numerically.
        is_numeric = (
            original_non_missing > 0
            and numeric_non_missing == original_non_missing
        )

        # ---------------------------------------------------------------------
        # Numeric statistics
        # ---------------------------------------------------------------------

        if is_numeric:

            lines.append("TYPE: NUMERIC")
            lines.append("")

            lines.append(
                f"Mean:               "
                f"{format_number(numeric_series.mean())}"
            )

            lines.append(
                f"Median:             "
                f"{format_number(numeric_series.median())}"
            )

            lines.append(
                f"Standard deviation: "
                f"{format_number(numeric_series.std())}"
            )

            lines.append(
                f"Minimum:            "
                f"{format_number(numeric_series.min())}"
            )

            lines.append(
                f"Maximum:            "
                f"{format_number(numeric_series.max())}"
            )

            lines.append(
                f"25th percentile:    "
                f"{format_number(numeric_series.quantile(0.25))}"
            )

            lines.append(
                f"75th percentile:    "
                f"{format_number(numeric_series.quantile(0.75))}"
            )

        # ---------------------------------------------------------------------
        # Categorical / symbolic statistics
        # ---------------------------------------------------------------------

        else:

            lines.append("TYPE: CATEGORICAL / SYMBOLIC")
            lines.append("")

            counts = series.value_counts(
                dropna=False
            )

            lines.append("Value counts:")
            lines.append("")

            for value, count in counts.items():

                label = (
                    "<MISSING>"
                    if pd.isna(value)
                    else str(value)
                )

                percentage = (
                    100.0 * count / len(series)
                    if len(series) > 0
                    else 0.0
                )

                lines.append(
                    f"    {label:<45}"
                    f"{count:>10,}"
                    f"{percentage:>10.2f}%"
                )

        lines.append("")

    # =========================================================================
    # COMPLETE COLUMN LIST
    # =========================================================================

    lines.append(separator)
    lines.append("FINAL CSV COLUMN ORDER")
    lines.append(separator)
    lines.append("")

    for index, column in enumerate(
        df.columns,
        start=1,
    ):

        if column == TARGET_COLUMN:
            role = "TARGET"
        else:
            role = "FEATURE"

        lines.append(
            f"{index:>3}. "
            f"{column:<60} "
            f"[{role}]"
        )

    lines.append("")
    lines.append(separator)

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    # =========================================================================
    # INPUT
    # =========================================================================

    if not INPUT_CSV.exists():

        raise SystemExit(
            f"Input CSV not found:\n"
            f"{INPUT_CSV.resolve()}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("LLM OBFUSCATION ML - STAGE 01")
    print("CREATE INITIAL MACHINE-LEARNING CSV")
    print("=" * 80)

    print()
    print(f"Input:  {INPUT_CSV.resolve()}")
    print(f"Target: {TARGET_SOURCE_COLUMN}")
    print()

    # =========================================================================
    # LOAD
    # =========================================================================

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    print(
        f"Loaded dataset: "
        f"{len(df):,} rows x "
        f"{len(df.columns):,} columns"
    )

    # =========================================================================
    # LIGHT CLEANING
    # =========================================================================

    df = clean_dataframe(
        df
    )

    # =========================================================================
    # CHECK TARGET
    # =========================================================================

    if TARGET_SOURCE_COLUMN not in df.columns:

        raise ValueError(
            f"\nTarget column does not exist:\n"
            f"    {TARGET_SOURCE_COLUMN}\n\n"
            f"Available columns:\n"
            + "\n".join(
                f"    {column}"
                for column in df.columns
            )
        )

    # =========================================================================
    # ACTIVE FEATURES
    # =========================================================================

    # Automatically prevent direct target leakage.

    active_features = [
        feature
        for feature in FEATURES
        if feature != TARGET_SOURCE_COLUMN
    ]

    missing_features = [
        feature
        for feature in active_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "\nThe following configured FEATURES are missing from the CSV:\n"
            + "\n".join(
                f"    {feature}"
                for feature in missing_features
            )
        )

    # =========================================================================
    # BUILD FINAL DATASET
    # =========================================================================

    selected_columns = (
        active_features
        + [TARGET_SOURCE_COLUMN]
    )

    prepared = df[
        selected_columns
    ].copy()

    # -------------------------------------------------------------------------
    # Rename target to "class"
    # -------------------------------------------------------------------------

    prepared = prepared.rename(
        columns={
            TARGET_SOURCE_COLUMN:
                TARGET_COLUMN
        }
    )

    # -------------------------------------------------------------------------
    # Drop rows without target
    # -------------------------------------------------------------------------

    rows_before_missing_class = len(
        prepared
    )

    if DROP_ROWS_WITH_MISSING_CLASS:

        prepared = prepared[
            prepared[TARGET_COLUMN].notna()
        ].copy()

    rows_removed_missing_class = (
        rows_before_missing_class
        - len(prepared)
    )

    # -------------------------------------------------------------------------
    # Reset row numbers
    # -------------------------------------------------------------------------

    prepared = prepared.reset_index(
        drop=True
    )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    prepared.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =========================================================================
    # CREATE REPORT
    # =========================================================================

    report = create_report(
        prepared,
        active_features,
    )

    report_header = (
        f"Input CSV:\n"
        f"{INPUT_CSV.resolve()}\n\n"
        f"Output CSV:\n"
        f"{OUTPUT_CSV.resolve()}\n\n"
        f"Original target column: {TARGET_SOURCE_COLUMN}\n"
        f"Output target column:   {TARGET_COLUMN}\n"
        f"Rows removed because class was missing: "
        f"{rows_removed_missing_class:,}\n\n"
    )

    OUTPUT_REPORT.write_text(
        report_header + report,
        encoding="utf-8",
    )

    # =========================================================================
    # CONSOLE OUTPUT
    # =========================================================================

    print()
    print("=" * 80)
    print("STAGE 01 COMPLETE")
    print("=" * 80)
    print()

    print(
        f"Final rows:      "
        f"{len(prepared):,}"
    )

    print(
        f"Features:        "
        f"{len(active_features):,}"
    )

    print(
        f"Target:          "
        f"{TARGET_COLUMN}"
    )

    print(
        f"Number classes:  "
        f"{prepared[TARGET_COLUMN].nunique(dropna=True):,}"
    )

    print()

    print("Class distribution:")

    counts = prepared[
        TARGET_COLUMN
    ].value_counts(
        dropna=False
    )

    for value, count in counts.items():

        percentage = (
            100.0 * count / len(prepared)
        )

        print(
            f"    {str(value):<35}"
            f"{count:>10,}"
            f"  ({percentage:>7.2f}%)"
        )

    print()
    print(
        f"CSV report:      "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Statistics:      "
        f"{OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()