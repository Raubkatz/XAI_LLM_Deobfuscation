# LLM Deobfuscation Machine-Learning Pipeline

This repository contains the machine-learning and explainability pipeline for the LLM deobfuscation experiments. It extracts experiment metrics from nested JSON result files, creates a configurable classification dataset, generates repeated stratified train/test splits with optional class balancing, trains CatBoost models, and performs aggregated post-training explainability analysis.

## Pipeline

| Stage | Script | Purpose |
|---|---|---|
| 00 | `00_create_csv.py` | Recursively extract experiment metadata and metrics from `result.metrics.summary.json` files and create `combined_metrics.csv`. |
| 01 | `01_create_dataset.py` | Select features and one prediction target, rename the target to `class`, perform light cleaning, and create the final ML input CSV and descriptive report. |
| 02 | `02_train_test_splits.py` | Create repeated stratified train/test splits over configurable random seeds. Optional random undersampling and ADASYN are applied **only to the training data**. |
| 03 | `03_train_CatBoost.py` | Train one CatBoost model for every Stage-02 split. Supports ordinary CatBoost or optional Optuna/TPE hyperparameter optimization. |
| 04 | `04_explainability.py` | Load all trained CatBoost models and perform aggregated evaluation, feature importance, native SHAP, ICE/PDP, confusion-matrix, and predicted-class feature-distribution analysis. |

Stage 01 explicitly selects the feature set and converts the selected prediction target to the common column name `class`.

Stage 02 performs the stratified split first and applies undersampling and/or ADASYN only to the training partition; the test partition remains untouched. 

Stage 03 creates an internal training/validation split from each Stage-02 training set. The outer test set is not used for hyperparameter optimization, early stopping, or model selection.

Stage 04 aggregates evaluation and explainability results across the independently trained models, including performance statistics, confusion matrices, CatBoost feature importance, SHAP, ICE/PDP, and predicted-class feature distributions. 

## Basic Workflow

Run the scripts in order:

```bash
python 00_create_csv.py
python 01_create_dataset.py
python 02_train_test_splits.py
python 03_train_CatBoost.py
python 04_explainability.py
```

The main generated directories are:

```text
LLM_OBF_01_prepared_dataset/
LLM_OBF_02_train_test_splits/
LLM_OBF_03_catboost_models/
LLM_OBF_04_final_analysis/
```

A typical branch is:

```text
LLM_OBF_02_train_test_splits/
└── seed_0000/
    ├── train.csv
    ├── test.csv
    └── split_report.txt

LLM_OBF_03_catboost_models/
└── seed_0000/
    ├── best_model.cbm
    ├── best_model_parameters.json
    ├── metrics.csv
    └── training_report.txt
```

## Main Configuration

In Stage 01, select the target and features:

```python
TARGET_SOURCE_COLUMN = "exe_pass"

FEATURES = [
    "simplification.deobf_nloc",
    "simplification.deobf_func_num",
    "simplification.deobf_cyclomatic",
    ...
]
```

The selected source target is renamed to:

```text
class
```

In Stage 02, configure the number of repeated splits and balancing:

```python
N_RANDOM_SEEDS = 100

USE_UNDERSAMPLING = True
USE_ADASYN = True
```

The four balancing modes are:

```python
False, False   # no balancing
True,  False   # undersampling only
False, True    # ADASYN only
True,  True    # undersampling + ADASYN
```

ADASYN requires numeric features and operates only on the training partition.

In Stage 03, enable or disable hyperparameter optimization:

```python
USE_HYPERPARAMETER_OPTIMIZATION = False

# True:
# CatBoost baseline + Optuna/TPE optimization
# followed by validation-based model selection.
```

When optimization is enabled, the optimized candidate is compared against the ordinary CatBoost baseline using the internal validation data only.

## Python Environment

Python 3.11 is recommended. The pipeline was designed around a Python 3.11 environment.

A suitable environment is:

```bash
conda create -n llm_obf_ml python=3.11.9
conda activate llm_obf_ml
```

Recommended pinned package versions:

```text
python==3.11.9
numpy==2.1.3
scipy==1.14.1
pandas==2.2.3
scikit-learn==1.5.2
imbalanced-learn==0.12.4
catboost==1.2.8
optuna==4.1.0
matplotlib==3.9.2
```

Install the required packages with:

```bash
pip install \
    numpy==2.1.3 \
    scipy==1.14.1 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    imbalanced-learn==0.12.4 \
    catboost==1.2.8 \
    optuna==4.1.0 \
    matplotlib==3.9.2
```

`imbalanced-learn` provides `RandomUnderSampler` and `ADASYN` used in Stage 02.

`Optuna` is required only when hyperparameter optimization is enabled in Stage 03. The training script otherwise requires NumPy, pandas, scikit-learn, and CatBoost.

No external `shap` package is required for Stage 04 because the analysis uses CatBoost's native SHAP implementation.

No TensorFlow, PyTorch, LightGBM, LIME, or seaborn installation is required for the current pipeline.

## Main Outputs

The complete workflow produces:

- combined experiment-level CSV data;
- configurable feature/class datasets;
- repeated stratified train/test splits;
- optional undersampled and/or ADASYN-balanced training datasets;
- CatBoost models for every random split;
- optional Optuna/TPE optimized models;
- training, validation, and held-out test metrics;
- absolute and normalized confusion matrices;
- CatBoost native feature importance;
- native CatBoost SHAP analyses;
- ICE/PDP analyses;
- predicted-class feature-distribution plots;
- aggregated means and standard deviations across all trained models;
- CSV, JSON, TXT, PNG, and EPS outputs.

The untouched Stage-02 `test.csv` partitions should be used for final generalization results. Training and internal-validation results are primarily diagnostic.
