#!/usr/bin/env python3
"""
01_create_dataset.py

Stage 01 of the LLM deobfuscation machine-learning pipeline.

INPUT
=====

combined_metrics_with_origin.csv


This CSV contains both:

    - LLM evaluation/result metrics
    - origin/source-code complexity metrics


This script ONLY:

    1. loads the merged CSV
    2. optionally selects one decompiler scenario
    3. selects the configured FEATURES
    4. selects ONE configured target column
    5. renames the target column to "class"
    6. performs very light initial cleaning
    7. writes one prepared CSV
    8. writes one descriptive TXT report


This script does NOT perform:

    - train/test splitting
    - ADASYN
    - undersampling
    - oversampling
    - class balancing
    - feature scaling
    - normalization
    - categorical encoding
    - feature selection beyond the explicit FEATURES list
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
    "./"
    "combined_metrics_with_origin.csv"
)


OUTPUT_DIR = Path(
    "LLM_OBF_01_prepared_dataset"
)


OUTPUT_CSV = (
    OUTPUT_DIR
    / "LLM_OBF_dataset.csv"
)


OUTPUT_REPORT = (
    OUTPUT_DIR
    / "LLM_OBF_dataset_report.txt"
)


# =============================================================================
# TARGET
# =============================================================================
#
# Select exactly ONE source column that should become the prediction target.
#
# The selected column will be renamed:
#
#     class
#
# in the final Stage-01 CSV.
#
#
# Examples:
#
# TARGET_SOURCE_COLUMN = "obfuscation_type"
# TARGET_SOURCE_COLUMN = "llm"
# TARGET_SOURCE_COLUMN = "decompiler"
# TARGET_SOURCE_COLUMN = "optimization"
# TARGET_SOURCE_COLUMN = "category"
#
# TARGET_SOURCE_COLUMN = "exe_pass"
# TARGET_SOURCE_COLUMN = "syntax_pass_rate"
#
#
# Origin metrics may also technically be selected as targets if required:
#
# TARGET_SOURCE_COLUMN = "origin.cyclomatic"
#
# =============================================================================

TARGET_SOURCE_COLUMN = "exe_pass"


TARGET_COLUMN = "class"


# =============================================================================
# OPTIONAL SINGLE-DECOMPILER SELECTION
# =============================================================================
#
# False:
#
#     use all decompiler scenarios exactly as before.
#
# True:
#
#     retain only rows belonging to SELECTED_DECOMPILER before the final
#     Stage-01 feature/target dataset is created.
#
#
# Expected values:
#
#     "ghidra"
#     "binja"
#
#
# Example:
#
#     USE_SINGLE_DECOMPILER_SELECTION = True
#     SELECTED_DECOMPILER = "ghidra"
#
# results in:
#
#     only ghidra rows
#
#
# Example:
#
#     USE_SINGLE_DECOMPILER_SELECTION = True
#     SELECTED_DECOMPILER = "binja"
#
# results in:
#
#     only binja rows
#
#
# Example:
#
#     USE_SINGLE_DECOMPILER_SELECTION = False
#
# results in:
#
#     ghidra + binja
#
#
# Selection is case-insensitive.
#
# Therefore:
#
#     "GHIDRA"
#
# and:
#
#     "ghidra"
#
# are treated identically.
#
# =============================================================================

USE_SINGLE_DECOMPILER_SELECTION = True


SELECTED_DECOMPILER = "ghidra"


# =============================================================================
# FEATURES
# =============================================================================
#
# This is intentionally a MANUALLY EDITABLE feature list.
#
# Comment out any features that should NOT be used by the machine-learning
# model.
#
# The selected TARGET_SOURCE_COLUMN is automatically removed from the feature
# list even if it remains listed below.
#
#
# IMPORTANT
# =========
#
# Identifiers and path-like metadata are shown below for completeness but are
# commented out by default because they can allow trivial sample memorization.
#
# =============================================================================

FEATURES: List[str] = [

    # =========================================================================
    # DATASET / EXPERIMENT INFORMATION
    # =========================================================================
    #
    # Enable only deliberately.
    #
    # These can contain experiment identity information.
    # =========================================================================

    # "category",
    # "filename",
    # "obfuscation_type",
    # "optimization",
    # "experiment_suffix",
    # "decompiler",
    # "llm",


    # =========================================================================
    # EXECUTION / COMPILATION METRICS
    # =========================================================================

    # "syntax_pass_rate",

    # "exe_pass",

    # "compile_and_link_rate",


    # =========================================================================
    # DEOBFUSCATED-CODE COMPLEXITY
    # =========================================================================

    # "simplification.deobf_nloc",

    # "simplification.deobf_func_num",

    # "simplification.deobf_cyclomatic",

    # "simplification.deobf_halstead_length",

    # "simplification.deobf_avg_ccn",


    # =========================================================================
    # DEOBFUSCATION / SIMPLIFICATION CHANGE METRICS
    # =========================================================================

    # "simplification.decrease_nloc",

    # "simplification.decrease_cyclomatic",

    # "simplification.decrease_halstead_length",

    # "simplification.nloc_difference_score",


    # =========================================================================
    # SIMILARITY METRICS
    # =========================================================================

    # "similarity.codebleu",

    # "similarity.ngram_match_score",

    # "similarity.weighted_ngram_match_score",

    # "similarity.syntax_match_score",

    # "similarity.dataflow_match_score",


    # =========================================================================
    # COMBINED RESULT METRICS
    # =========================================================================

    # "combined.pass",

    # "combined.simplification",

    # "combined.similarity",

    # "combined.total",


    # =========================================================================
    # ORIGIN COMPLEXITY
    # =========================================================================
    #
    # These are the core metrics corresponding to the complexity conventions
    # already used in simplification.py.
    # =========================================================================

    # "origin.nloc",

    # "origin.func_num",

    # "origin.cyclomatic",

    # "origin.tokens",

    # "origin.avg_ccn",


    # =========================================================================
    # ORIGIN CYCLOMATIC-COMPLEXITY DISTRIBUTION
    # =========================================================================

    "origin.ccn_mean",

    "origin.ccn_median",

    "origin.ccn_std",

    "origin.ccn_min",

    "origin.ccn_max",


    # =========================================================================
    # ORIGIN FUNCTION-NLOC DISTRIBUTION
    # =========================================================================

    "origin.function_nloc_mean",

    "origin.function_nloc_median",

    "origin.function_nloc_std",

    "origin.function_nloc_min",

    "origin.function_nloc_max",


    # =========================================================================
    # ORIGIN FUNCTION-TOKEN DISTRIBUTION
    # =========================================================================

    "origin.function_tokens_mean",

    "origin.function_tokens_median",

    "origin.function_tokens_std",

    "origin.function_tokens_min",

    "origin.function_tokens_max",


    # =========================================================================
    # ORIGIN FUNCTION PARAMETER STATISTICS
    # =========================================================================

    "origin.parameter_count_total",

    "origin.parameters_mean",

    "origin.parameters_median",

    "origin.parameters_std",

    "origin.parameters_min",

    "origin.parameters_max",


    # =========================================================================
    # ORIGIN FUNCTION-LENGTH STATISTICS
    # =========================================================================

    "origin.function_length_mean",

    "origin.function_length_median",

    "origin.function_length_std",

    "origin.function_length_min",

    "origin.function_length_max",


    # =========================================================================
    # ORIGIN HALSTEAD METRICS
    # =========================================================================

    "origin.halstead_length",

    "origin.halstead_vocab",

    "origin.halstead_volume",

    "origin.halstead_distinct_operators",

    "origin.halstead_distinct_operands",

    "origin.halstead_total_operators",

    "origin.halstead_total_operands",


    # =========================================================================
    # ORIGIN SOURCE-SIZE STATISTICS
    # =========================================================================

    "origin.physical_lines",

    "origin.nonempty_lines",

    "origin.blank_lines",

    "origin.preprocessor_lines",

    "origin.comment_only_lines_approx",

    "origin.characters",

    "origin.bytes_utf8",


    # =========================================================================
    # ORIGIN CONTROL-FLOW COUNTS
    # =========================================================================

    "origin.keyword_if_count",

    "origin.keyword_else_count",

    "origin.keyword_for_count",

    "origin.keyword_while_count",

    "origin.keyword_do_count",

    "origin.keyword_switch_count",

    "origin.keyword_case_count",

    "origin.keyword_goto_count",

    "origin.keyword_return_count",

    "origin.keyword_break_count",

    "origin.keyword_continue_count",

    "origin.logical_and_count",

    "origin.logical_or_count",

    "origin.ternary_question_count",


    # =========================================================================
    # ORIGIN STRUCTURAL NESTING
    # =========================================================================

    "origin.max_brace_nesting",

    "origin.avg_open_brace_nesting",


    # =========================================================================
    # OPTIONAL ORIGIN QUALITY / EXTRACTION INFORMATION
    # =========================================================================
    #
    # Usually I would NOT use these as predictive ML features.
    #
    # They are shown here so they are easy to activate if required.
    # =========================================================================

    # "source_empty",

    # "lizard_parse_success",

    # "gcc_syntax_checked",

    # "gcc_syntax_pass",


    # =========================================================================
    # ORIGIN GENERATION METADATA
    # =========================================================================
    #
    # Usually exclude these from model input because they identify the
    # construction process rather than source complexity itself.
    # =========================================================================

    # "compiler",

    # "gcc_version",

    # "obfuscator",

    # "tigress_version",


    # =========================================================================
    # UNIQUE / PATH-LIKE IDENTIFIERS
    # =========================================================================
    #
    # DO NOT normally use these for prediction.
    #
    # They are retained in the merged source CSV for traceability but should
    # normally remain excluded from FEATURES.
    # =========================================================================

    # "experiment_folder",

    # "sample_id",

    # "merge_key",

    # "c_filename",

    # "c_path",

    # "source_sha256",

    # "json_path",
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
        str(
            column
        ).strip()
        for column in df.columns
    ]

    # -------------------------------------------------------------------------
    # Remove accidental pandas-index columns
    # -------------------------------------------------------------------------

    if REMOVE_UNNAMED_COLUMNS:

        unnamed_columns = [
            column
            for column in df.columns
            if str(
                column
            ).lower().startswith(
                "unnamed:"
            )
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
    # Replace +/- infinity
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

    if pd.isna(
        value
    ):

        return "NaN"

    try:

        return (
            f"{float(value):.6f}"
        )

    except Exception:

        return str(
            value
        )


# =============================================================================
# REPORT
# =============================================================================

def create_report(
    df: pd.DataFrame,
    features: List[str],
) -> str:

    lines: List[str] = []

    separator = "=" * 90

    sub_separator = "-" * 90

    # =========================================================================
    # GENERAL
    # =========================================================================

    lines.append(
        separator
    )

    lines.append(
        "LLM DEOBFUSCATION MACHINE-LEARNING DATASET REPORT"
    )

    lines.append(
        separator
    )

    lines.append("")

    lines.append(
        f"Dataset rows:       "
        f"{len(df):,}"
    )

    lines.append(
        f"Feature columns:    "
        f"{len(features):,}"
    )

    lines.append(
        f"Target column:      "
        f"{TARGET_COLUMN}"
    )

    lines.append(
        f"Total columns:      "
        f"{len(df.columns):,}"
    )

    lines.append("")

    # =========================================================================
    # CLASS DISTRIBUTION
    # =========================================================================

    lines.append(
        separator
    )

    lines.append(
        "CLASS DISTRIBUTION"
    )

    lines.append(
        separator
    )

    lines.append("")

    class_counts = df[
        TARGET_COLUMN
    ].value_counts(
        dropna=False
    )

    total = len(
        df
    )

    for class_value, count in class_counts.items():

        percentage = (
            100.0
            * count
            / total
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
    # Imbalance
    # -------------------------------------------------------------------------

    non_missing_counts = df[
        TARGET_COLUMN
    ].value_counts(
        dropna=True
    )

    if len(
        non_missing_counts
    ) >= 2:

        majority_class = (
            non_missing_counts.idxmax()
        )

        minority_class = (
            non_missing_counts.idxmin()
        )

        majority_count = int(
            non_missing_counts.max()
        )

        minority_count = int(
            non_missing_counts.min()
        )

        imbalance_ratio = (
            majority_count
            / minority_count
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

    lines.append(
        separator
    )

    lines.append(
        "MISSING VALUES"
    )

    lines.append(
        separator
    )

    lines.append("")

    for column in df.columns:

        missing = int(
            df[
                column
            ].isna().sum()
        )

        missing_fraction = (
            100.0
            * missing
            / len(
                df
            )
            if len(
                df
            ) > 0
            else 0.0
        )

        lines.append(
            f"{column:<60}"
            f"{missing:>12,}"
            f"{missing_fraction:>12.2f}%"
        )

    lines.append("")

    # =========================================================================
    # FEATURE STATISTICS
    # =========================================================================

    lines.append(
        separator
    )

    lines.append(
        "FEATURE STATISTICS"
    )

    lines.append(
        separator
    )

    lines.append("")

    for feature in features:

        series = df[
            feature
        ]

        lines.append(
            sub_separator
        )

        lines.append(
            f"FEATURE: {feature}"
        )

        lines.append(
            sub_separator
        )

        lines.append(
            f"Pandas dtype:       "
            f"{series.dtype}"
        )

        lines.append(
            f"Non-missing:        "
            f"{series.notna().sum():,}"
        )

        lines.append(
            f"Missing:            "
            f"{series.isna().sum():,}"
        )

        lines.append(
            f"Unique non-missing: "
            f"{series.nunique(dropna=True):,}"
        )

        lines.append("")

        # ---------------------------------------------------------------------
        # Numeric detection
        # ---------------------------------------------------------------------

        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        )

        original_non_missing = int(
            series.notna().sum()
        )

        numeric_non_missing = int(
            numeric_series.notna().sum()
        )

        is_numeric = (
            original_non_missing > 0
            and
            numeric_non_missing
            == original_non_missing
        )

        if is_numeric:

            lines.append(
                "TYPE: NUMERIC"
            )

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

        else:

            lines.append(
                "TYPE: CATEGORICAL / SYMBOLIC"
            )

            lines.append("")

            counts = series.value_counts(
                dropna=False
            )

            lines.append(
                "Value counts:"
            )

            lines.append("")

            for value, count in counts.items():

                label = (
                    "<MISSING>"
                    if pd.isna(
                        value
                    )
                    else str(
                        value
                    )
                )

                percentage = (
                    100.0
                    * count
                    / len(
                        series
                    )
                    if len(
                        series
                    )
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

    lines.append(
        separator
    )

    lines.append(
        "FINAL CSV COLUMN ORDER"
    )

    lines.append(
        separator
    )

    lines.append("")

    for index, column in enumerate(
        df.columns,
        start=1,
    ):

        role = (
            "TARGET"
            if column
            == TARGET_COLUMN
            else "FEATURE"
        )

        lines.append(
            f"{index:>3}. "
            f"{column:<65} "
            f"[{role}]"
        )

    lines.append("")

    lines.append(
        separator
    )

    return "\n".join(
        lines
    )


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

    print(
        "=" * 90
    )

    print(
        "LLM OBFUSCATION ML - STAGE 01"
    )

    print(
        "CREATE MACHINE-LEARNING DATASET FROM MERGED METRICS"
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"Input:  "
        f"{INPUT_CSV.resolve()}"
    )

    print(
        f"Target: "
        f"{TARGET_SOURCE_COLUMN}"
    )

    print(
        f"Single decompiler selection: "
        f"{USE_SINGLE_DECOMPILER_SELECTION}"
    )

    if USE_SINGLE_DECOMPILER_SELECTION:

        print(
            f"Selected decompiler: "
            f"{SELECTED_DECOMPILER}"
        )

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
    # CLEAN
    # =========================================================================

    df = clean_dataframe(
        df
    )

    # =========================================================================
    # OPTIONAL SINGLE-DECOMPILER SELECTION
    # =========================================================================
    #
    # False:
    #
    #     no filtering
    #
    # True:
    #
    #     keep only rows corresponding to SELECTED_DECOMPILER
    #
    #
    # This happens BEFORE feature selection.
    #
    # Therefore "decompiler" does NOT need to be included in FEATURES in order
    # to select a single decompiler scenario.
    # =========================================================================

    rows_before_decompiler_selection = len(
        df
    )

    available_decompilers: List[str] = []

    if USE_SINGLE_DECOMPILER_SELECTION:

        if "decompiler" not in df.columns:

            raise ValueError(
                "\nSingle-decompiler selection is enabled, but the column "
                "'decompiler' does not exist in the merged CSV."
            )

        available_decompilers = sorted(
            {
                str(
                    value
                )
                .strip()
                .lower()
                for value in df[
                    "decompiler"
                ].dropna().unique()
            }
        )

        selected_decompiler_normalized = (
            str(
                SELECTED_DECOMPILER
            )
            .strip()
            .lower()
        )

        if (
            selected_decompiler_normalized
            not in available_decompilers
        ):

            raise ValueError(
                "\nSelected decompiler does not exist in the merged CSV:\n\n"
                f"    {SELECTED_DECOMPILER}\n\n"
                "Available decompilers:\n\n"
                + "\n".join(
                    f"    {value}"
                    for value in available_decompilers
                )
            )

        decompiler_normalized = (
            df[
                "decompiler"
            ]
            .astype(
                "string"
            )
            .str.strip()
            .str.lower()
        )

        df = df.loc[
            decompiler_normalized
            == selected_decompiler_normalized
        ].copy()

        df = df.reset_index(
            drop=True
        )

    rows_removed_decompiler_selection = (
        rows_before_decompiler_selection
        - len(
            df
        )
    )

    # =========================================================================
    # DECOMPILER-SELECTION OUTPUT
    # =========================================================================

    if USE_SINGLE_DECOMPILER_SELECTION:

        print()

        print(
            "Single-decompiler selection applied:"
        )

        print(
            f"    Selected:      "
            f"{SELECTED_DECOMPILER}"
        )

        print(
            f"    Rows before:   "
            f"{rows_before_decompiler_selection:,}"
        )

        print(
            f"    Rows removed:  "
            f"{rows_removed_decompiler_selection:,}"
        )

        print(
            f"    Rows retained: "
            f"{len(df):,}"
        )

    # =========================================================================
    # TARGET CHECK
    # =========================================================================

    if (
        TARGET_SOURCE_COLUMN
        not in df.columns
    ):

        raise ValueError(
            "\nTarget column does not exist:\n"
            f"    {TARGET_SOURCE_COLUMN}\n\n"
            "Available columns:\n"
            + "\n".join(
                f"    {column}"
                for column in df.columns
            )
        )

    # =========================================================================
    # ACTIVE FEATURES
    # =========================================================================
    #
    # Automatically prevent direct target leakage.
    # =========================================================================

    active_features = [
        feature
        for feature in FEATURES
        if feature
        != TARGET_SOURCE_COLUMN
    ]

    # -------------------------------------------------------------------------
    # Check duplicate entries in FEATURES
    # -------------------------------------------------------------------------

    duplicate_features = sorted(
        {
            feature
            for feature in active_features
            if active_features.count(
                feature
            ) > 1
        }
    )

    if duplicate_features:

        raise ValueError(
            "\nDuplicate entries exist in FEATURES:\n"
            + "\n".join(
                f"    {feature}"
                for feature in duplicate_features
            )
        )

    # -------------------------------------------------------------------------
    # Check missing features
    # -------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in active_features
        if feature
        not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "\nThe following configured FEATURES are missing "
            "from the merged CSV:\n"
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
        + [
            TARGET_SOURCE_COLUMN
        ]
    )

    prepared = df[
        selected_columns
    ].copy()

    # =========================================================================
    # RENAME TARGET
    # =========================================================================

    prepared = prepared.rename(
        columns={
            TARGET_SOURCE_COLUMN:
                TARGET_COLUMN
        }
    )

    # =========================================================================
    # DROP MISSING TARGET
    # =========================================================================

    rows_before_missing_class = len(
        prepared
    )

    if DROP_ROWS_WITH_MISSING_CLASS:

        prepared = prepared[
            prepared[
                TARGET_COLUMN
            ].notna()
        ].copy()

    rows_removed_missing_class = (
        rows_before_missing_class
        - len(
            prepared
        )
    )

    prepared = prepared.reset_index(
        drop=True
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    prepared.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =========================================================================
    # REPORT
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

        f"Original target column: "
        f"{TARGET_SOURCE_COLUMN}\n"

        f"Output target column:   "
        f"{TARGET_COLUMN}\n\n"

        f"Single decompiler selection enabled: "
        f"{USE_SINGLE_DECOMPILER_SELECTION}\n"

        f"Selected decompiler: "
        f"{SELECTED_DECOMPILER if USE_SINGLE_DECOMPILER_SELECTION else '<ALL>'}\n"

        f"Rows before decompiler selection: "
        f"{rows_before_decompiler_selection:,}\n"

        f"Rows removed by decompiler selection: "
        f"{rows_removed_decompiler_selection:,}\n"

        f"Rows after decompiler selection: "
        f"{len(df):,}\n\n"

        f"Rows removed because class was missing: "
        f"{rows_removed_missing_class:,}\n\n"
    )

    OUTPUT_REPORT.write_text(
        report_header
        + report,
        encoding="utf-8",
    )

    # =========================================================================
    # CONSOLE OUTPUT
    # =========================================================================

    print()

    print(
        "=" * 90
    )

    print(
        "STAGE 01 COMPLETE"
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"Single decompiler selection: "
        f"{USE_SINGLE_DECOMPILER_SELECTION}"
    )

    if USE_SINGLE_DECOMPILER_SELECTION:

        print(
            f"Selected decompiler: "
            f"{SELECTED_DECOMPILER}"
        )

        print(
            f"Rows before selection: "
            f"{rows_before_decompiler_selection:,}"
        )

        print(
            f"Rows removed by selection: "
            f"{rows_removed_decompiler_selection:,}"
        )

        print(
            f"Rows retained by selection: "
            f"{len(df):,}"
        )

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
        f"Source target:   "
        f"{TARGET_SOURCE_COLUMN}"
    )

    print(
        f"Number classes:  "
        f"{prepared[TARGET_COLUMN].nunique(dropna=True):,}"
    )

    print()

    print(
        "Class distribution:"
    )

    counts = prepared[
        TARGET_COLUMN
    ].value_counts(
        dropna=False
    )

    for value, count in counts.items():

        percentage = (
            100.0
            * count
            / len(
                prepared
            )
            if len(
                prepared
            )
            else 0.0
        )

        print(
            f"    {str(value):<35}"
            f"{count:>10,}"
            f"  ({percentage:>7.2f}%)"
        )

    print()

    print(
        f"CSV:"
        f"\n    {OUTPUT_CSV}"
    )

    print()

    print(
        f"Report:"
        f"\n    {OUTPUT_REPORT}"
    )


if __name__ == "__main__":

    main()