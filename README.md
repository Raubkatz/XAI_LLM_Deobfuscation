# LLM Deobfuscation Machine-Learning Pipeline

This repository contains the machine-learning and explainability pipeline for the LLM deobfuscation experiments. It starts from the merged experiment dataset `combined_metrics_with_origin.csv`, creates a configurable classification dataset, generates repeated stratified train/test splits with optional class balancing, trains CatBoost models, and performs aggregated post-training explainability analysis.

## Pipeline

| Stage | Script | Purpose |
|---|---|---|
| 01 | `01_create_dataset.py` | Load `combined_metrics_with_origin.csv`, optionally select one decompiler scenario, select features and one prediction target, rename the target to `class`, perform light cleaning, and create the final ML input CSV and descriptive report. |
| 02 | `02_train_test_splits.py` | Create repeated stratified train/test splits over configurable random seeds. Optional random undersampling, ADASYN, and custom minority sampling are applied **only to the training data**. |
| 03 | `03_train_CatBoost.py` | Train one CatBoost model for every Stage-02 split. Supports ordinary CatBoost or optional Bayesian hyperparameter optimization using scikit-optimize `BayesSearchCV`. |
| 04 | `04_explainability.py` | Load all trained CatBoost models and perform aggregated evaluation, feature importance, native SHAP, ICE/PDP, confusion-matrix, and predicted-class feature-distribution analysis. |

Stage 01 loads the initial dataset:

```text
combined_metrics_with_origin.csv
```

It optionally selects one decompiler scenario, explicitly selects the feature set, and converts the selected prediction target to the common column name `class`.

Stage 02 performs the stratified split first and applies undersampling, ADASYN, and/or custom minority sampling only to the training partition; the test partition remains untouched.

Stage 03 creates an internal training/validation split from each Stage-02 training set. The outer test set is not used for hyperparameter optimization, early stopping, or model selection.

Stage 04 aggregates evaluation and explainability results across the independently trained models, including performance statistics, confusion matrices, CatBoost feature importance, SHAP, ICE/PDP, and predicted-class feature distributions.

## Basic Workflow

The pipeline starts from:

```text
combined_metrics_with_origin.csv
```

Run the scripts in order:

```bash
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
    ├── model_selection.csv
    ├── metrics.csv
    ├── predictions_test.csv
    ├── predictions_validation.csv
    ├── bayes_search_results.csv
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

Stage 01 can optionally restrict the dataset to one decompiler:

```python
USE_SINGLE_DECOMPILER_SELECTION = True
SELECTED_DECOMPILER = "ghidra"
```

or:

```python
USE_SINGLE_DECOMPILER_SELECTION = True
SELECTED_DECOMPILER = "binja"
```

To retain all available decompiler scenarios:

```python
USE_SINGLE_DECOMPILER_SELECTION = False
```

In Stage 02, configure the number of repeated splits and balancing:

```python
N_RANDOM_SEEDS = 100

USE_UNDERSAMPLING = False
USE_ADASYN = True
USE_CUSTOM_UNDERSAMPLING = True

CUSTOM_MINORITY_FRACTION = 0.90
```

The main balancing modes are:

```python
False, False, False   # no balancing
True,  False, False   # standard random undersampling only
False, True,  False   # ADASYN only
False, False, True    # custom minority sampling only
False, True,  True    # ADASYN + custom minority sampling
True,  True,  False   # standard random undersampling + ADASYN
```

where the values correspond to:

```text
USE_UNDERSAMPLING
USE_ADASYN
USE_CUSTOM_UNDERSAMPLING
```

Standard undersampling and custom undersampling are alternative methods and must not both be enabled.

When ADASYN and custom minority sampling are both enabled, ADASYN is applied first. The custom sample size is then calculated from the current post-ADASYN minority population. With:

```python
CUSTOM_MINORITY_FRACTION = 0.90
```

90% of the current minority population is retained and exactly the same number of majority observations is sampled.

ADASYN requires numeric features and operates only on the training partition.

In Stage 03, enable or disable hyperparameter optimization:

```python
USE_HYPERPARAMETER_OPTIMIZATION = False

# True:
# CatBoost baseline + Bayesian hyperparameter optimization
# with scikit-optimize BayesSearchCV,
# followed by validation-based model selection.
```

When optimization is enabled, Stage 03 performs Bayesian optimization using:

```python
BAYES_N_ITER = 50
BAYES_CV = 3
BAYES_N_JOBS = 3

MODEL_SELECTION_METRIC = "macro_f1"
```

The current CatBoost Bayesian search space is:

```python
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
```

`BayesSearchCV` operates only on the internal training portion using stratified cross-validation. The separate internal validation set is not used during Bayesian optimization. It remains reserved for early stopping and baseline-versus-optimized model selection.

When optimization is enabled, the optimized candidate is compared against the ordinary CatBoost baseline using the internal validation data only. The winning configuration is then refitted on the full Stage-02 training dataset and evaluated on the untouched Stage-02 test set.

## Python Environment

Python 3.11 is recommended. The pipeline was designed around a Python 3.11 environment.

A suitable environment is:

```bash
conda create -n llm_obf_ml python=3.11
conda activate llm_obf_ml
```

Recommended pinned package versions compatible with Python 3.11:

```text
python==3.11
numpy==2.4.6
scipy==1.17.1
pandas==3.0.6
scikit-learn==1.9.1
imbalanced-learn==0.14.2
catboost==1.2.10
scikit-optimize==0.10.2
matplotlib==3.11.2
```

Install the required packages with:

```bash
pip install \
    numpy==2.4.6 \
    scipy==1.17.1 \
    pandas==3.0.6 \
    scikit-learn==1.9.1 \
    imbalanced-learn==0.14.2 \
    catboost==1.2.10 \
    scikit-optimize==0.10.2 \
    matplotlib==3.11.2
```

`imbalanced-learn` provides `RandomUnderSampler` and `ADASYN` used in Stage 02. The custom minority-sampling implementation itself uses pandas and NumPy.

`scikit-optimize` is required for Bayesian hyperparameter optimization in Stage 03. The training script uses `BayesSearchCV` together with `Integer` and `Real` search dimensions.

The Stage-03 training script otherwise requires NumPy, pandas, scikit-learn, and CatBoost.

No external `shap` package is required for Stage 04 because the analysis uses CatBoost's native SHAP implementation.

No TensorFlow, PyTorch, LightGBM, LIME, or seaborn installation is required for the current pipeline.

## Main Outputs

The complete workflow produces:

- configurable feature/class datasets;
- repeated stratified train/test splits;
- optional standard undersampling, ADASYN, custom minority sampling, and combined balancing of training datasets;
- CatBoost models for every random split;
- optional Bayesian `BayesSearchCV` optimized CatBoost models;
- Bayesian search-result CSVs and selected hyperparameter configurations;
- training, validation, and held-out test metrics;
- absolute and normalized confusion matrices;
- CatBoost native feature importance;
- native CatBoost SHAP analyses;
- ICE/PDP analyses;
- predicted-class feature-distribution plots;
- aggregated means and standard deviations across all trained models;
- CSV, JSON, TXT, PNG, and EPS outputs.

The untouched Stage-02 `test.csv` partitions should be used for final generalization results. Training and internal-validation results are primarily diagnostic.
