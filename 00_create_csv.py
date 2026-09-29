#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path
from statistics import mean


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT_DIR = Path(
    "/home/sebastian/PRGRMS/LLM_deobfuscation_analysis/llm_obf"
)

OUTPUT_CSV = ROOT_DIR / "combined_metrics.csv"

TARGET_JSON = "result.metrics.summary.json"


# Metrics to extract from every result.metrics.summary.json.
#
# The names on the left become CSV column names.
# The names on the right describe the expected JSON paths.
#
METRICS = {
    "syntax_pass_rate":
        "syntax_pass_rate",

    "exe_pass":
        "averages_over_syntax_passing.exe_pass",

    "compile_and_link_rate":
        "averages_over_syntax_passing.compile_and_link_rate",

    "simplification.deobf_nloc":
        "averages_over_syntax_passing.simplification.deobf_nloc",

    "simplification.deobf_func_num":
        "averages_over_syntax_passing.simplification.deobf_func_num",

    "simplification.deobf_cyclomatic":
        "averages_over_syntax_passing.simplification.deobf_cyclomatic",

    "simplification.deobf_halstead_length":
        "averages_over_syntax_passing.simplification.deobf_halstead_length",

    "simplification.deobf_avg_ccn":
        "averages_over_syntax_passing.simplification.deobf_avg_ccn",

    "simplification.decrease_nloc":
        "averages_over_syntax_passing.simplification.decrease_nloc",

    "simplification.decrease_cyclomatic":
        "averages_over_syntax_passing.simplification.decrease_cyclomatic",

    "simplification.decrease_halstead_length":
        "averages_over_syntax_passing.simplification.decrease_halstead_length",

    "simplification.nloc_difference_score":
        "averages_over_syntax_passing.simplification.nloc_difference_score",

    "similarity.codebleu":
        "averages_over_syntax_passing.similarity.codebleu",

    "similarity.ngram_match_score":
        "averages_over_syntax_passing.similarity.ngram_match_score",

    "similarity.weighted_ngram_match_score":
        "averages_over_syntax_passing.similarity.weighted_ngram_match_score",

    "similarity.syntax_match_score":
        "averages_over_syntax_passing.similarity.syntax_match_score",

    "similarity.dataflow_match_score":
        "averages_over_syntax_passing.similarity.dataflow_match_score",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_nested(data, path):
    """
    Retrieve a value from a dictionary using a dot-separated path.

    IMPORTANT:
    The JSON contains literal keys with dots, for example:

        {
            "averages_over_syntax_passing": {
                "simplification.deobf_nloc": 36.0,
                "similarity.codebleu": 0.379...
            }
        }

    Therefore this function supports BOTH:

        genuinely nested dictionaries

    AND:

        literal JSON keys containing dots.
    """

    current = data
    parts = path.split(".")

    i = 0

    while i < len(parts):

        if not isinstance(current, dict):
            return None

        # ---------------------------------------------------------------------
        # FIRST:
        # Try the complete remaining path as a literal dictionary key.
        #
        # Example:
        #
        # current =
        # {
        #     "simplification.deobf_nloc": 36.0
        # }
        #
        # remaining =
        # "simplification.deobf_nloc"
        #
        # This is exactly how the supplied JSON is structured.
        # ---------------------------------------------------------------------

        remaining = ".".join(parts[i:])

        if remaining in current:
            return current[remaining]

        # ---------------------------------------------------------------------
        # SECOND:
        # Try normal dictionary traversal.
        #
        # Example:
        #
        # "averages_over_syntax_passing"
        # ---------------------------------------------------------------------

        key = parts[i]

        if key not in current:
            return None

        current = current[key]
        i += 1

    return current


def search_key_recursively(data, target_key):
    """
    Search recursively for a key anywhere inside the JSON.

    This works especially well for this JSON format because keys such as:

        simplification.deobf_nloc
        simplification.decrease_nloc
        similarity.codebleu

    are stored literally as complete keys.
    """

    if isinstance(data, dict):

        # Exact key match first.
        if target_key in data:
            return data[target_key]

        for value in data.values():
            result = search_key_recursively(value, target_key)

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:
            result = search_key_recursively(item, target_key)

            if result is not None:
                return result

    return None


def extract_metric(data, column_name, expected_path):
    """
    First try the exact expected JSON path.

    The get_nested() function understands both normal nested JSON and
    literal dotted keys.

    If that fails, search recursively for the complete metric key.
    """

    value = get_nested(data, expected_path)

    if value is not None:
        return value

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Search for the COMPLETE column name first.
    #
    # Example:
    #
    #     simplification.deobf_nloc
    #
    # NOT only:
    #
    #     deobf_nloc
    #
    # because the actual JSON key literally contains the complete dotted name.
    # -------------------------------------------------------------------------

    value = search_key_recursively(data, column_name)

    if value is not None:
        return value

    # -------------------------------------------------------------------------
    # Final fallback:
    # Search only for the last component.
    # -------------------------------------------------------------------------

    final_key = column_name.split(".")[-1]

    return search_key_recursively(data, final_key)


def numeric(value):
    """
    Convert a value to float where possible.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# =============================================================================
# PATH PARSING
# =============================================================================

def parse_experiment_folder(folder_name):
    """
    Parse directories such as:

        david_tigress_4_0_11_gcc_16_1_0-add_opaque_bug_list-O3

    into:

        obfuscation_type = add_opaque_bug_list
        optimization     = O3

    The important delimiter is:

        gcc_16_1_0-

    followed by the obfuscation name and then:

        -O0
        -O1
        -O2
        -O3
        -Os
        ...

    The GCC version itself is allowed to vary.
    """

    pattern = re.compile(
        r"gcc_[0-9_]+-(?P<obfuscation>.+)-(?P<optimization>O[0-9sgfast]+)$",
        re.IGNORECASE
    )

    match = pattern.search(folder_name)

    if match:
        return (
            match.group("obfuscation"),
            match.group("optimization")
        )

    # More general fallback:
    # everything following the compiler section until the last -O...
    fallback = re.search(
        r"gcc_[^-]+-(.+)-(O[^-]+)$",
        folder_name,
        re.IGNORECASE
    )

    if fallback:
        return fallback.group(1), fallback.group(2)

    return None, None


def parse_path(json_file):
    """
    Expected structure:

    ROOT_DIR/
        category/
        filename/
        experiment/
        decompiler/
        llm/
        result.metrics.summary.json
    """

    try:
        relative = json_file.relative_to(ROOT_DIR)
    except ValueError:
        return None

    parts = relative.parts

    # Need:
    #
    # category
    # filename
    # experiment
    # decompiler
    # llm
    # JSON filename
    #
    if len(parts) < 6:
        print(f"[WARNING] Unexpected path structure: {json_file}")
        return None

    category = parts[-6]
    filename = parts[-5]
    experiment_folder = parts[-4]
    decompiler = parts[-3]
    llm = parts[-2]

    obfuscation_type, optimization = parse_experiment_folder(
        experiment_folder
    )

    return {
        "category": category,
        "filename": filename,
        "obfuscation_type": obfuscation_type,
        "optimization": optimization,
        "decompiler": decompiler,
        "llm": llm,
        "experiment_folder": experiment_folder,
        "json_path": str(json_file),
    }


# =============================================================================
# COMBINED SCORES
# =============================================================================

def safe_mean(values):
    """
    Mean while ignoring missing/non-numeric values.
    """

    numbers = []

    for value in values:
        value = numeric(value)

        if value is not None:
            numbers.append(value)

    if not numbers:
        return None

    return mean(numbers)


def calculate_combined_scores(row):
    """
    Calculate logically grouped combined scores.

    IMPORTANT:

    decrease_nloc,
    decrease_cyclomatic,
    decrease_halstead_length

    are already improvement measures.

    Example:

        decrease_nloc = 0.97

    means approximately 97% reduction.

    Therefore HIGHER is BETTER and these values must NOT be inverted.

    Similarity metrics are also HIGHER = BETTER.

    Pass rates are HIGHER = BETTER.

    nloc_difference_score is kept separately because its precise semantic
    direction may differ from the decrease metrics.
    """

    # -------------------------------------------------------------------------
    # Compilation / executability
    # -------------------------------------------------------------------------

    row["combined.pass"] = safe_mean([
        row.get("syntax_pass_rate"),
        row.get("exe_pass"),
        row.get("compile_and_link_rate"),
    ])

    # -------------------------------------------------------------------------
    # Simplification
    #
    # Higher decrease = stronger simplification.
    # -------------------------------------------------------------------------

    row["combined.simplification"] = safe_mean([
        row.get("simplification.decrease_nloc"),
        row.get("simplification.decrease_cyclomatic"),
        row.get("simplification.decrease_halstead_length"),
    ])

    # -------------------------------------------------------------------------
    # Similarity
    # -------------------------------------------------------------------------

    row["combined.similarity"] = safe_mean([
        row.get("similarity.codebleu"),
        row.get("similarity.ngram_match_score"),
        row.get("similarity.weighted_ngram_match_score"),
        row.get("similarity.syntax_match_score"),
        row.get("similarity.dataflow_match_score"),
    ])

    # -------------------------------------------------------------------------
    # Main combined score
    #
    # Three equally weighted conceptual groups:
    #
    #   1. code validity / executability
    #   2. simplification
    #   3. similarity to the reference
    #
    # This prevents the five similarity metrics from automatically having
    # more influence simply because there are more of them.
    # -------------------------------------------------------------------------

    row["combined.total"] = safe_mean([
        row.get("combined.pass"),
        row.get("combined.simplification"),
        row.get("combined.similarity"),
    ])

    return row


# =============================================================================
# READ ONE JSON
# =============================================================================

def process_json(json_file):
    """
    Turn one result.metrics.summary.json into one CSV row.
    """

    metadata = parse_path(json_file)

    if metadata is None:
        return None

    try:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as exc:
        print(f"[ERROR] Could not read {json_file}")
        print(f"        {exc}")
        return None

    row = metadata.copy()

    for column_name, json_path in METRICS.items():
        row[column_name] = extract_metric(
            data,
            column_name,
            json_path
        )

    calculate_combined_scores(row)

    return row


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(f"Searching below:")
    print(f"  {ROOT_DIR}")
    print()

    json_files = sorted(
        ROOT_DIR.rglob(TARGET_JSON)
    )

    print(
        f"Found {len(json_files)} "
        f"{TARGET_JSON} files."
    )

    rows = []

    for i, json_file in enumerate(json_files, start=1):

        print(
            f"[{i:5d}/{len(json_files):5d}] "
            f"{json_file}"
        )

        row = process_json(json_file)

        if row is not None:
            rows.append(row)

    if not rows:
        print("No valid rows found.")
        return

    # -------------------------------------------------------------------------
    # CSV column order
    # -------------------------------------------------------------------------

    fieldnames = [
        "category",
        "filename",
        "obfuscation_type",
        "optimization",
        "decompiler",
        "llm",

        "syntax_pass_rate",
        "exe_pass",
        "compile_and_link_rate",

        "simplification.deobf_nloc",
        "simplification.deobf_func_num",
        "simplification.deobf_cyclomatic",
        "simplification.deobf_halstead_length",
        "simplification.deobf_avg_ccn",

        "simplification.decrease_nloc",
        "simplification.decrease_cyclomatic",
        "simplification.decrease_halstead_length",
        "simplification.nloc_difference_score",

        "similarity.codebleu",
        "similarity.ngram_match_score",
        "similarity.weighted_ngram_match_score",
        "similarity.syntax_match_score",
        "similarity.dataflow_match_score",

        "combined.pass",
        "combined.simplification",
        "combined.similarity",
        "combined.total",

        "experiment_folder",
        "json_path",
    ]

    # -------------------------------------------------------------------------
    # Sort rows to make the CSV reproducible/readable
    # -------------------------------------------------------------------------

    rows.sort(
        key=lambda x: (
            str(x.get("category", "")),
            str(x.get("filename", "")),
            str(x.get("obfuscation_type", "")),
            str(x.get("optimization", "")),
            str(x.get("decompiler", "")),
            str(x.get("llm", "")),
        )
    )

    # -------------------------------------------------------------------------
    # Write CSV
    # -------------------------------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Finished.")
    print(f"Rows written: {len(rows)}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()