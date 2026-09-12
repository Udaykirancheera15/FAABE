
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAABE – Leakage-Free CUDA Evaluation Across 10 Software-Effort Datasets
=======================================================================

Datasets
--------
1. Albrecht
2. China
3. COCOMO81
4. Desharnais
5. Kemerer
6. Kitchenham
7. Maxwell
8. Miyazaki94
9. Subbiah
10. Valdes-Souto

Protocol
--------
- 10-fold outer cross-validation
- training-only median imputation
- training-only MinMax scaling
- training-only Pearson feature selection
- training-only Firefly optimization
- leave-one-out training MMRE for Firefly fitness
- untouched outer-test prediction
- out-of-fold predictions for every project
- ABE vs FAABE
- MMRE, MAE, MSE, RMSE
- paired Wilcoxon signed-rank test on project-level OOF MRE
- Cliff's Delta
- all selected features and Firefly weights saved
- CUDA/PyTorch support
- deterministic seed where possible
- per-dataset + combined result files

IMPORTANT
---------
No outer-test labels are used for imputation, scaling, feature selection,
Firefly optimization, or model/weight selection.

The implementation intentionally excludes identifier/text/date columns from
the numerical ABE/FAABE similarity space. Dataset-specific exclusions are
listed in DATASETS below.
"""

import os
import re
import csv
import json
import math
import random
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wilcoxon


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Put this script in the same folder as the datasets, or change DATA_DIR.
DATA_DIR = Path(__file__).resolve().parent

RESULTS_DIR = DATA_DIR / "results_FAABE_ABLATION_ALL_10_DATASETS"

PEARSON_THRESHOLD = 0.25
K_ANALOGIES = 3

NUM_FIREFLIES = 10
GAMMA = 1.0
ALPHA = 0.2
MAX_ITERATIONS = 100

N_FOLDS = 10
RANDOM_STATE = 42

EPS = 1e-12
DELTA = 1e-4

# This follows the paper/code definition:
# Sim(p,p') = 1 / sqrt(sum_i w_i * |a_i-a_i'| + delta)
SIMILARITY = "paper_weighted_euclidean_absolute_difference"

# Print Firefly progress every N iterations.
LOG_EVERY = 10


# =============================================================================
# ABLATION CONFIGURATION
# =============================================================================

# Variants are designed to isolate the three architectural choices:
#   1) Pearson feature selection
#   2) Firefly feature-weight optimization
#   3) IWM similarity-weighted effort aggregation
#
# "All features" means no Pearson filtering.
# "Uniform" means no Firefly optimization.
# "Mean" means simple arithmetic mean of top-k analogue efforts (no IWM).
#
# Main variants:
#   ABE_BASE              : all features + uniform weights + IWM
#   ABE_PEARSON           : Pearson + uniform weights + IWM
#   FAABE_NO_PEARSON      : all features + Firefly + IWM
#   FAABE_NO_FIREFLY      : Pearson + uniform weights + IWM
#   FAABE_NO_IWM          : Pearson + Firefly + simple mean
#   FAABE_FULL            : Pearson + Firefly + IWM
#
# ABE_PEARSON and FAABE_NO_FIREFLY are intentionally identical. We keep only
# FAABE_NO_FIREFLY in the paper-facing table to avoid duplicate rows.

ABLATION_VARIANTS = [
    "ABE_BASE",
    "FAABE_NO_PEARSON",
    "FAABE_NO_FIREFLY",
    "FAABE_NO_IWM",
    "FAABE_FULL",
]


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASETS = {
    "Albrecht": {
        "file": "albrecht.csv",
        "target": "Effort",
        "ignore": ["id"],
        "features": None,
    },

    "China": {
        "file": "china.csv",
        "target": "Effort",
        "ignore": ["id", "DevType"],
        "features": None,
    },

    "COCOMO81": {
        "file": "cocomo81.csv",
        "target": "actual",
        "ignore": [],
        "features": None,
    },

    "Desharnais": {
        "file": "desharnais.csv",
        "target": "Effort",
        # Project and id are identifiers, not predictors.
        "ignore": ["id", "Project"],
        "features": None,
    },

    "Kemerer": {
        "file": "kemerer.csv",
        "target": "EffortMM",
        "ignore": [],
        "features": None,
    },

    "Kitchenham": {
        "file": "kitchenham.arff",
        "target": "Actual.effort",
        # String/date/nominal fields are intentionally excluded.
        # The numerical effort-estimation attributes remain.
        "ignore": [
            "Project",
            "Client.code",
            "Project.type",
            "Actual.start.date",
            "Estimated.completion.date",
            "First.estimate.method",
        ],
        "features": None,
    },

    "Maxwell": {
        "file": "maxwell.csv",
        "target": "Effort",
        "ignore": ["id"],
        "features": None,
    },

    "Miyazaki94": {
        "file": "miyazaki94.arff",
        "target": "MM",
        "ignore": ["ID"],
        "features": None,
    },

    "Subbiah": {
        "file": "Subbiah.csv",
        "target": "TotalLaborCost",
        "ignore": ["Project"],
        # Preserve the feature set used in the supplied Subbiah experiment.
        "features": [
            "TotalUFP",
            "ExternalInterface",
            "ExternalQueries",
            "ExternalInputs",
            "InternalLogicalFiles",
            "ExternalOutputs",
        ],
    },

    "Valdes-Souto": {
        "file": "Valdes-Souto.csv",
        "target": "Effort",
        # Project / ID are identifiers; BASE is nominal text.
        "ignore": ["Project", "ID", "BASE"],
        "features": None,
    },
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# FILE / LOG HELPERS
# =============================================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_LOG = RESULTS_DIR / "master_experiment.log"


def master_log(message=""):
    message = str(message)
    print(message)
    with MASTER_LOG.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def make_dataset_logger(dataset_dir):
    log_path = dataset_dir / "experiment.log"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("")

    def log(message=""):
        message = str(message)
        print(message)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    return log


# =============================================================================
# ARFF LOADER
# =============================================================================

def parse_arff_value(value):
    value = value.strip()

    if value == "?":
        return np.nan

    if (
        len(value) >= 2
        and (
            (value[0] == "'" and value[-1] == "'")
            or (value[0] == '"' and value[-1] == '"')
        )
    ):
        value = value[1:-1]

    return value


def load_arff_general(path):
    """
    Small ARFF reader supporting numeric, nominal, string, and date fields.
    It is used because scipy.io.arff.loadarff does not support string fields.
    """

    attributes = []
    data_rows = []
    in_data = False

    # utf-8-sig removes BOM if present.
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("%"):
                continue

            lower = line.lower()

            if not in_data:
                if lower.startswith("@attribute"):
                    # Match:
                    # @attribute name type
                    # @attribute 'name with space' type
                    match = re.match(
                        r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))\s+(.+)$",
                        line,
                        flags=re.IGNORECASE,
                    )

                    if not match:
                        raise ValueError(f"Cannot parse ARFF attribute line: {line}")

                    name = next(
                        group
                        for group in match.groups()[:3]
                        if group is not None
                    )
                    attr_type = match.group(4).strip()

                    attributes.append((name, attr_type))

                elif lower.startswith("@data"):
                    in_data = True

            else:
                # ARFF rows here are ordinary comma-separated rows.
                row = next(csv.reader([line], skipinitialspace=True))
                data_rows.append([parse_arff_value(v) for v in row])

    names = [name for name, _ in attributes]

    if not names:
        raise ValueError(f"No ARFF attributes found in {path}")

    if any(len(row) != len(names) for row in data_rows):
        bad = [
            i
            for i, row in enumerate(data_rows, start=1)
            if len(row) != len(names)
        ]
        raise ValueError(
            f"ARFF row width mismatch in {path}; bad data rows: {bad[:10]}"
        )

    df = pd.DataFrame(data_rows, columns=names)

    # Convert declared numeric attributes to numeric.
    for name, attr_type in attributes:
        attr_lower = attr_type.lower()

        if (
            attr_lower.startswith("numeric")
            or attr_lower.startswith("real")
            or attr_lower.startswith("integer")
        ):
            df[name] = pd.to_numeric(df[name], errors="coerce")

    return df


def load_dataset(path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        # Treat common missing markers as NaN.
        return pd.read_csv(
            path,
            na_values=["NA", "N/A", "?", "null", "NULL", ""],
            keep_default_na=True,
        )

    if suffix == ".arff":
        return load_arff_general(path)

    raise ValueError(f"Unsupported file format: {path}")


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_dataset(dataset_name, config, log):
    path = DATA_DIR / "Datasets" / config["file"]
    if not path.exists():
        path = DATA_DIR / config["file"]

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {config['file']} (checked {DATA_DIR / 'Datasets'} and {DATA_DIR})")

    df = load_dataset(path)

    target = config["target"]

    if target not in df.columns:
        raise ValueError(
            f"{dataset_name}: target '{target}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Target is always forced numeric.
    y = pd.to_numeric(df[target], errors="coerce")

    valid_target = y.notna() & np.isfinite(y.to_numpy(dtype=np.float64))

    removed_target = int((~valid_target).sum())

    if removed_target:
        log(f"Removing {removed_target} rows with invalid target.")

    df = df.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    explicit_features = config.get("features")

    if explicit_features is not None:
        missing = [c for c in explicit_features if c not in df.columns]

        if missing:
            raise ValueError(
                f"{dataset_name}: configured predictors missing: {missing}"
            )

        feature_columns = list(explicit_features)

    else:
        ignored = set(config.get("ignore", [])) | {target}

        candidates = [
            c
            for c in df.columns
            if c not in ignored
        ]

        # Keep genuinely numerical columns or columns that can be converted
        # almost completely to numeric. Text/date/nominal fields stay out.
        feature_columns = []

        for c in candidates:
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_columns.append(c)
                continue

            converted = pd.to_numeric(df[c], errors="coerce")
            non_missing_original = df[c].notna().sum()

            if non_missing_original == 0:
                continue

            numeric_ratio = converted.notna().sum() / non_missing_original

            # Only accept text-looking columns when essentially all
            # non-missing values are actually numeric.
            if numeric_ratio >= 0.999:
                df[c] = converted
                feature_columns.append(c)

    if not feature_columns:
        raise ValueError(f"{dataset_name}: no usable numerical predictors.")

    X = df[feature_columns].copy()

    for c in feature_columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    X_np = X.to_numpy(dtype=np.float64)
    y_np = y.to_numpy(dtype=np.float64)

    if len(y_np) < N_FOLDS:
        raise ValueError(
            f"{dataset_name}: only {len(y_np)} usable projects, "
            f"but N_FOLDS={N_FOLDS}."
        )

    log(f"Loaded: {path.name}")
    log(f"Projects: {len(y_np)}")
    log(f"Predictors ({len(feature_columns)}): {feature_columns}")
    log(f"Target: {target}")

    return df, X_np, y_np, feature_columns, path


# =============================================================================
# METRICS
# =============================================================================

def safe_mre(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (np.abs(y_true) > EPS)
    )

    output = np.full(len(y_true), np.nan, dtype=np.float64)

    output[valid] = np.abs(
        (y_true[valid] - y_pred[valid]) / y_true[valid]
    )

    return output


def MMRE(y_true, y_pred):
    errors = safe_mre(y_true, y_pred)
    valid = np.isfinite(errors)

    if not valid.any():
        return np.nan

    return float(np.mean(errors[valid]))


def MAE(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if not valid.any():
        return np.nan

    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])))


def MSE(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if not valid.any():
        return np.nan

    return float(np.mean((y_true[valid] - y_pred[valid]) ** 2))


def RMSE(y_true, y_pred):
    mse = MSE(y_true, y_pred)
    return float(np.sqrt(mse)) if np.isfinite(mse) else np.nan


def mmre_torch(y_true, y_pred):
    valid = (
        torch.isfinite(y_true)
        & torch.isfinite(y_pred)
        & (torch.abs(y_true) > EPS)
    )

    if not torch.any(valid):
        return torch.tensor(float("inf"), device=DEVICE)

    errors = torch.abs(
        (y_true[valid] - y_pred[valid]) / y_true[valid]
    )

    errors = torch.nan_to_num(
        errors,
        nan=1e6,
        posinf=1e6,
        neginf=1e6,
    )

    return torch.mean(errors)


# =============================================================================
# TRAINING-ONLY IMPUTATION
# =============================================================================

def fit_imputer(X_train):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(X_train, axis=0)

    # Entirely-missing training feature -> deterministic zero.
    medians = np.where(np.isfinite(medians), medians, 0.0)

    return medians


def apply_imputer(X_data, medians):
    X_out = np.array(X_data, dtype=np.float64, copy=True)
    bad = ~np.isfinite(X_out)

    for j in range(X_out.shape[1]):
        X_out[bad[:, j], j] = medians[j]

    return X_out


# =============================================================================
# TRAINING-ONLY PEARSON FEATURE SELECTION
# =============================================================================

def pearson_feature_selection(
    X_train,
    y_train,
    names,
    threshold,
):
    selected = []
    correlations = {}

    for j, name in enumerate(names):
        x = X_train[:, j]

        if (
            len(x) < 2
            or np.std(x) <= EPS
            or np.std(y_train) <= EPS
        ):
            corr = 0.0
        else:
            corr = np.corrcoef(x, y_train)[0, 1]

            if not np.isfinite(corr):
                corr = 0.0

        correlations[name] = float(corr)

        if abs(corr) >= threshold:
            selected.append(name)

    # Never allow an empty feature set.
    if not selected:
        best_feature = max(
            correlations,
            key=lambda k: abs(correlations[k]),
        )
        selected = [best_feature]

    selected_indices = [names.index(name) for name in selected]

    return selected_indices, selected, correlations


# =============================================================================
# WEIGHTS / SIMILARITY
# =============================================================================

def normalize_weights(weights):
    weights = torch.nan_to_num(
        weights,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    weights = torch.clamp(weights, min=0.0, max=1.0)

    total = weights.sum()

    if (not torch.isfinite(total)) or total <= EPS:
        weights = torch.ones_like(weights)
        total = weights.sum()

    return weights / total


@torch.no_grad()
def euclidean_similarity(query, reference, weights):
    """
    Paper-style similarity:

        Sim(p,p') =
            1 / sqrt(sum_i [w_i * |a_i - a'_i|] + DELTA)
    """

    query = torch.nan_to_num(
        query,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    reference = torch.nan_to_num(
        reference,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    weights = normalize_weights(weights)

    differences = torch.abs(
        query[:, None, :] - reference[None, :, :]
    )

    weighted_distance = (
        differences * weights[None, None, :]
    ).sum(dim=2)

    weighted_distance = torch.nan_to_num(
        weighted_distance,
        nan=1e6,
        posinf=1e6,
        neginf=0.0,
    )

    distance = torch.sqrt(
        torch.clamp(
            weighted_distance + DELTA,
            min=DELTA,
        )
    )

    similarity = 1.0 / torch.clamp(distance, min=EPS)

    return torch.nan_to_num(
        similarity,
        nan=0.0,
        posinf=1.0 / math.sqrt(DELTA),
        neginf=0.0,
    )


@torch.no_grad()
def iwm_predict(
    query,
    reference,
    reference_y,
    weights,
    k,
):
    n_reference = reference.shape[0]

    if n_reference == 0:
        raise ValueError("No reference projects.")

    k = min(int(k), int(n_reference))

    similarities = euclidean_similarity(
        query,
        reference,
        weights,
    )

    top_similarity, top_indices = torch.topk(
        similarities,
        k=k,
        dim=1,
    )

    top_y = reference_y[top_indices]

    top_similarity = torch.nan_to_num(
        top_similarity,
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    )

    top_y = torch.nan_to_num(
        top_y,
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )

    denominator = top_similarity.sum(dim=1, keepdim=True)

    normalized = top_similarity / torch.clamp(
        denominator,
        min=EPS,
    )

    zero_mask = denominator <= EPS

    if torch.any(zero_mask):
        equal = (
            torch.ones_like(top_similarity)
            / top_similarity.shape[1]
        )

        normalized = torch.where(
            zero_mask,
            equal,
            normalized,
        )

    prediction = (normalized * top_y).sum(dim=1)

    return torch.nan_to_num(
        prediction,
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )


# =============================================================================
# VECTORIZED LEAVE-ONE-OUT TRAINING FITNESS
# =============================================================================

@torch.no_grad()
def firefly_fitness(weights, X_train, y_train):
    """
    Training-only leave-one-out fitness.

    Every training project is predicted from the OTHER training projects.
    The outer test set is absent from this function.
    """

    n = int(X_train.shape[0])

    if n < 2:
        return torch.tensor(
            1e6,
            dtype=torch.float32,
            device=DEVICE,
        )

    k = min(K_ANALOGIES, n - 1)

    similarities = euclidean_similarity(
        X_train,
        X_train,
        weights,
    )

    # Exclude each project from being its own analogy.
    similarities.fill_diagonal_(-float("inf"))

    top_similarity, top_indices = torch.topk(
        similarities,
        k=k,
        dim=1,
    )

    # After top-k, all retained values should be finite.
    top_similarity = torch.nan_to_num(
        top_similarity,
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    )

    top_y = y_train[top_indices]

    denominator = top_similarity.sum(dim=1, keepdim=True)

    normalized = top_similarity / torch.clamp(
        denominator,
        min=EPS,
    )

    zero_mask = denominator <= EPS

    if torch.any(zero_mask):
        equal = (
            torch.ones_like(top_similarity)
            / top_similarity.shape[1]
        )
        normalized = torch.where(
            zero_mask,
            equal,
            normalized,
        )

    predictions = (normalized * top_y).sum(dim=1)

    fitness = mmre_torch(y_train, predictions)

    if not torch.isfinite(fitness):
        return torch.tensor(
            1e6,
            dtype=torch.float32,
            device=DEVICE,
        )

    return fitness


# =============================================================================
# FIREFLY OPTIMIZATION
# =============================================================================

@torch.no_grad()
def optimize_firefly(
    X_train,
    y_train,
    seed,
    log,
):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_features = X_train.shape[1]

    fireflies = torch.rand(
        NUM_FIREFLIES,
        n_features,
        dtype=torch.float32,
        device=DEVICE,
    )

    for i in range(NUM_FIREFLIES):
        fireflies[i] = normalize_weights(fireflies[i])

    fitness = torch.empty(
        NUM_FIREFLIES,
        dtype=torch.float32,
        device=DEVICE,
    )

    for i in range(NUM_FIREFLIES):
        fitness[i] = firefly_fitness(
            fireflies[i],
            X_train,
            y_train,
        )

    initial_best = float(torch.min(fitness).item())

    history = [{
        "Iteration": 0,
        "Best_Training_MMRE": initial_best,
    }]

    for iteration in range(MAX_ITERATIONS):
        for i in range(NUM_FIREFLIES):
            for j in range(NUM_FIREFLIES):
                if fitness[j] < fitness[i]:
                    distance = torch.linalg.norm(
                        fireflies[i] - fireflies[j]
                    )

                    beta = torch.exp(
                        -GAMMA * distance * distance
                    )

                    random_step = (
                        torch.rand(
                            n_features,
                            dtype=torch.float32,
                            device=DEVICE,
                        )
                        - 0.5
                    )

                    candidate = (
                        fireflies[i]
                        + beta * (fireflies[j] - fireflies[i])
                        + ALPHA * random_step
                    )

                    candidate = normalize_weights(candidate)

                    candidate_fitness = firefly_fitness(
                        candidate,
                        X_train,
                        y_train,
                    )

                    if torch.isfinite(candidate_fitness):
                        # Preserve the movement rule of the supplied FAABE
                        # implementation: move toward a brighter firefly and
                        # then update the moved firefly's fitness.
                        fireflies[i] = candidate
                        fitness[i] = candidate_fitness

        best_fitness = float(torch.min(fitness).item())

        history.append({
            "Iteration": iteration + 1,
            "Best_Training_MMRE": best_fitness,
        })

        if (
            (iteration + 1) % LOG_EVERY == 0
            or iteration == 0
        ):
            log(
                f"      Iteration {iteration + 1:3d}/"
                f"{MAX_ITERATIONS} | "
                f"Best training MMRE = {best_fitness:.8f}"
            )

    best_index = int(torch.argmin(fitness).item())

    best_weights = normalize_weights(
        fireflies[best_index]
    ).detach().clone()

    best_fitness = float(fitness[best_index].item())

    return (
        best_weights,
        best_fitness,
        fitness.detach().cpu().numpy(),
        history,
    )


# =============================================================================
# STATISTICS
# =============================================================================

def cliffs_delta(x, y):
    """
    Standard Cliff's Delta between two distributions.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    greater = 0
    less = 0

    for a in x:
        greater += np.sum(a > y)
        less += np.sum(a < y)

    return float(
        (greater - less)
        / (len(x) * len(y))
    )


def cliffs_effect(delta):
    if not np.isfinite(delta):
        return "Undefined"

    magnitude = abs(delta)

    if magnitude < 0.147:
        return "Negligible"
    if magnitude < 0.330:
        return "Small"
    if magnitude < 0.474:
        return "Medium"

    return "Large"


def relative_improvement(baseline, proposed):
    if (
        not np.isfinite(baseline)
        or abs(baseline) <= EPS
    ):
        return np.nan

    return float(
        ((baseline - proposed) / baseline) * 100.0
    )



# =============================================================================
# ABLATION-SPECIFIC PREDICTION HELPERS
# =============================================================================

@torch.no_grad()
def mean_predict(
    query,
    reference,
    reference_y,
    weights,
    k,
):
    """
    Top-k analogy prediction using a simple arithmetic mean.

    This is the "no IWM" ablation. The SAME weighted similarity function is
    used to retrieve analogues; only the final effort aggregation changes from
    similarity-weighted IWM to a plain mean.
    """
    n_reference = reference.shape[0]

    if n_reference == 0:
        raise ValueError("No reference projects.")

    k = min(int(k), int(n_reference))

    similarities = euclidean_similarity(
        query,
        reference,
        weights,
    )

    _, top_indices = torch.topk(
        similarities,
        k=k,
        dim=1,
    )

    top_y = reference_y[top_indices]

    return torch.mean(
        top_y,
        dim=1,
    )


@torch.no_grad()
def firefly_fitness_mean(
    weights,
    X_train,
    y_train,
):
    """
    Training-only leave-one-out MMRE for the no-IWM ablation.
    """
    n = int(X_train.shape[0])

    if n < 2:
        return torch.tensor(
            1e6,
            dtype=torch.float32,
            device=DEVICE,
        )

    k = min(K_ANALOGIES, n - 1)

    similarities = euclidean_similarity(
        X_train,
        X_train,
        weights,
    )

    similarities.fill_diagonal_(-float("inf"))

    _, top_indices = torch.topk(
        similarities,
        k=k,
        dim=1,
    )

    top_y = y_train[top_indices]
    predictions = torch.mean(top_y, dim=1)

    fitness = mmre_torch(
        y_train,
        predictions,
    )

    if not torch.isfinite(fitness):
        return torch.tensor(
            1e6,
            dtype=torch.float32,
            device=DEVICE,
        )

    return fitness


@torch.no_grad()
def optimize_firefly_with_fitness(
    X_train,
    y_train,
    seed,
    log,
    fitness_fn,
):
    """
    Same Firefly optimizer as the main method, but with a supplied
    training-only fitness function. This lets the no-IWM ablation optimize
    weights fairly without ever using the outer test set.
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_features = X_train.shape[1]

    fireflies = torch.rand(
        NUM_FIREFLIES,
        n_features,
        dtype=torch.float32,
        device=DEVICE,
    )

    for i in range(NUM_FIREFLIES):
        fireflies[i] = normalize_weights(
            fireflies[i]
        )

    fitness = torch.empty(
        NUM_FIREFLIES,
        dtype=torch.float32,
        device=DEVICE,
    )

    for i in range(NUM_FIREFLIES):
        fitness[i] = fitness_fn(
            fireflies[i],
            X_train,
            y_train,
        )

    history = [{
        "Iteration": 0,
        "Best_Training_MMRE": float(torch.min(fitness).item()),
    }]

    for iteration in range(MAX_ITERATIONS):
        for i in range(NUM_FIREFLIES):
            for j in range(NUM_FIREFLIES):
                if fitness[j] < fitness[i]:
                    distance = torch.linalg.norm(
                        fireflies[i] - fireflies[j]
                    )

                    beta = torch.exp(
                        -GAMMA * distance * distance
                    )

                    random_step = (
                        torch.rand(
                            n_features,
                            dtype=torch.float32,
                            device=DEVICE,
                        )
                        - 0.5
                    )

                    candidate = (
                        fireflies[i]
                        + beta * (fireflies[j] - fireflies[i])
                        + ALPHA * random_step
                    )

                    candidate = normalize_weights(
                        candidate
                    )

                    candidate_fitness = fitness_fn(
                        candidate,
                        X_train,
                        y_train,
                    )

                    if torch.isfinite(candidate_fitness):
                        # Preserve the Firefly movement rule used in the main run.
                        fireflies[i] = candidate
                        fitness[i] = candidate_fitness

        best_fitness = float(
            torch.min(fitness).item()
        )

        history.append({
            "Iteration": iteration + 1,
            "Best_Training_MMRE": best_fitness,
        })

        if (
            (iteration + 1) % LOG_EVERY == 0
            or iteration == 0
        ):
            log(
                f"      Iteration {iteration + 1:3d}/"
                f"{MAX_ITERATIONS} | "
                f"Best training MMRE = {best_fitness:.8f}"
            )

    best_index = int(
        torch.argmin(fitness).item()
    )

    best_weights = normalize_weights(
        fireflies[best_index]
    ).detach().clone()

    best_fitness = float(
        fitness[best_index].item()
    )

    return (
        best_weights,
        best_fitness,
        history,
    )


# =============================================================================
# STATISTICS HELPERS
# =============================================================================

def paired_wilcoxon(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.nan, np.nan

    if np.allclose(x, y, equal_nan=True):
        return 0.0, 1.0

    try:
        result = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
        )
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full_like(pvalues, np.nan)

    valid_idx = np.where(np.isfinite(pvalues))[0]

    if len(valid_idx) == 0:
        return adjusted

    p = pvalues[valid_idx]
    order = np.argsort(p)
    m = len(p)

    running = 0.0
    temp = np.empty(m, dtype=np.float64)

    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        temp[idx] = min(running, 1.0)

    adjusted[valid_idx] = temp

    return adjusted


# =============================================================================
# ONE DATASET – ABLATION
# =============================================================================

def run_ablation_dataset(
    dataset_name,
    config,
):
    dataset_dir = RESULTS_DIR / dataset_name
    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log = make_dataset_logger(
        dataset_dir
    )

    log("=" * 110)
    log(f"ABLATION DATASET: {dataset_name}")
    log("=" * 110)
    log(f"Timestamp: {datetime.now().isoformat()}")
    log(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        log(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        log(
            f"CUDA version: {torch.version.cuda}"
        )

    (
        df,
        X_np,
        y_np,
        feature_columns,
        data_path,
    ) = prepare_dataset(
        dataset_name,
        config,
        log,
    )

    n_projects = len(y_np)

    outer_cv = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_actual = np.full(
        n_projects,
        np.nan,
        dtype=np.float64,
    )

    oof_fold = np.full(
        n_projects,
        -1,
        dtype=int,
    )

    oof = {
        variant: np.full(
            n_projects,
            np.nan,
            dtype=np.float64,
        )
        for variant in ABLATION_VARIANTS
    }

    fold_rows = []
    prediction_rows = []
    feature_rows = []
    weight_rows = []
    history_rows = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        outer_cv.split(X_np),
        start=1,
    ):
        log("")
        log("-" * 110)
        log(f"OUTER FOLD {fold}/{N_FOLDS}")
        log("-" * 110)

        X_train_raw = X_np[train_idx]
        X_test_raw = X_np[test_idx]

        y_train = y_np[train_idx]
        y_test = y_np[test_idx]

        # ---------------------------------------------------------------------
        # Training-only imputation
        # ---------------------------------------------------------------------

        imputer = fit_imputer(
            X_train_raw
        )

        X_train_imp = apply_imputer(
            X_train_raw,
            imputer,
        )

        X_test_imp = apply_imputer(
            X_test_raw,
            imputer,
        )

        # ---------------------------------------------------------------------
        # Training-only scaling
        # ---------------------------------------------------------------------

        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(
            X_train_imp
        )

        X_test_scaled = scaler.transform(
            X_test_imp
        )

        X_train_scaled = np.nan_to_num(
            X_train_scaled,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        X_test_scaled = np.nan_to_num(
            X_test_scaled,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        # ---------------------------------------------------------------------
        # Training-only Pearson selection for variants that use it
        # ---------------------------------------------------------------------

        (
            selected_indices,
            selected_names,
            correlations,
        ) = pearson_feature_selection(
            X_train_scaled,
            y_train,
            feature_columns,
            PEARSON_THRESHOLD,
        )

        all_indices = list(
            range(len(feature_columns))
        )

        all_names = list(
            feature_columns
        )

        for feature in feature_columns:
            feature_rows.append({
                "Dataset": dataset_name,
                "Fold": fold,
                "Feature": feature,
                "Pearson_r": correlations[feature],
                "Abs_Pearson_r": abs(correlations[feature]),
                "Selected": feature in selected_names,
            })

        log(
            f"Pearson selected "
            f"{len(selected_names)}/{len(feature_columns)}: "
            f"{selected_names}"
        )

        # ---------------------------------------------------------------------
        # Torch representations: all-features and Pearson-selected
        # ---------------------------------------------------------------------

        X_train_all = torch.tensor(
            X_train_scaled[:, all_indices],
            dtype=torch.float32,
            device=DEVICE,
        )

        X_test_all = torch.tensor(
            X_test_scaled[:, all_indices],
            dtype=torch.float32,
            device=DEVICE,
        )

        X_train_sel = torch.tensor(
            X_train_scaled[:, selected_indices],
            dtype=torch.float32,
            device=DEVICE,
        )

        X_test_sel = torch.tensor(
            X_test_scaled[:, selected_indices],
            dtype=torch.float32,
            device=DEVICE,
        )

        y_train_gpu = torch.tensor(
            y_train,
            dtype=torch.float32,
            device=DEVICE,
        )

        variant_predictions = {}
        variant_training_mmre = {}

        # =====================================================================
        # 1. ABE_BASE: all features + uniform weights + IWM
        # =====================================================================

        uniform_all = normalize_weights(
            torch.ones(
                len(all_indices),
                dtype=torch.float32,
                device=DEVICE,
            )
        )

        pred = iwm_predict(
            X_test_all,
            X_train_all,
            y_train_gpu,
            uniform_all,
            K_ANALOGIES,
        )

        variant_predictions["ABE_BASE"] = (
            pred.detach().cpu().numpy().astype(np.float64)
        )

        variant_training_mmre["ABE_BASE"] = np.nan

        row = {
            "Dataset": dataset_name,
            "Fold": fold,
            "Variant": "ABE_BASE",
            "Training_MMRE": np.nan,
        }

        for name, weight in zip(
            all_names,
            uniform_all.detach().cpu().numpy(),
        ):
            row[name] = float(weight)

        weight_rows.append(row)

        # =====================================================================
        # 2. FAABE_NO_FIREFLY: Pearson + uniform weights + IWM
        # =====================================================================

        uniform_sel = normalize_weights(
            torch.ones(
                len(selected_indices),
                dtype=torch.float32,
                device=DEVICE,
            )
        )

        pred = iwm_predict(
            X_test_sel,
            X_train_sel,
            y_train_gpu,
            uniform_sel,
            K_ANALOGIES,
        )

        variant_predictions["FAABE_NO_FIREFLY"] = (
            pred.detach().cpu().numpy().astype(np.float64)
        )

        variant_training_mmre["FAABE_NO_FIREFLY"] = np.nan

        row = {
            "Dataset": dataset_name,
            "Fold": fold,
            "Variant": "FAABE_NO_FIREFLY",
            "Training_MMRE": np.nan,
        }

        for name, weight in zip(
            selected_names,
            uniform_sel.detach().cpu().numpy(),
        ):
            row[name] = float(weight)

        weight_rows.append(row)

        # =====================================================================
        # 3. FAABE_NO_PEARSON: all features + Firefly + IWM
        # =====================================================================

        log("Running FAABE_NO_PEARSON...")

        (
            weights_no_pearson,
            fitness_no_pearson,
            population_fitness,
            history,
        ) = optimize_firefly(
            X_train_all,
            y_train_gpu,
            seed=RANDOM_STATE + fold + 1000,
            log=log,
        )

        pred = iwm_predict(
            X_test_all,
            X_train_all,
            y_train_gpu,
            weights_no_pearson,
            K_ANALOGIES,
        )

        variant_predictions["FAABE_NO_PEARSON"] = (
            pred.detach().cpu().numpy().astype(np.float64)
        )

        variant_training_mmre["FAABE_NO_PEARSON"] = (
            fitness_no_pearson
        )

        for hist in history:
            history_rows.append({
                "Dataset": dataset_name,
                "Fold": fold,
                "Variant": "FAABE_NO_PEARSON",
                **hist,
            })

        row = {
            "Dataset": dataset_name,
            "Fold": fold,
            "Variant": "FAABE_NO_PEARSON",
            "Training_MMRE": fitness_no_pearson,
        }

        for name, weight in zip(
            all_names,
            weights_no_pearson.detach().cpu().numpy(),
        ):
            row[name] = float(weight)

        weight_rows.append(row)

        # =====================================================================
        # 4. FAABE_NO_IWM: Pearson + Firefly + simple mean
        # =====================================================================

        log("Running FAABE_NO_IWM...")

        (
            weights_no_iwm,
            fitness_no_iwm,
            history_no_iwm,
        ) = optimize_firefly_with_fitness(
            X_train_sel,
            y_train_gpu,
            seed=RANDOM_STATE + fold + 2000,
            log=log,
            fitness_fn=firefly_fitness_mean,
        )

        pred = mean_predict(
            X_test_sel,
            X_train_sel,
            y_train_gpu,
            weights_no_iwm,
            K_ANALOGIES,
        )

        variant_predictions["FAABE_NO_IWM"] = (
            pred.detach().cpu().numpy().astype(np.float64)
        )

        variant_training_mmre["FAABE_NO_IWM"] = (
            fitness_no_iwm
        )

        for hist in history_no_iwm:
            history_rows.append({
                "Dataset": dataset_name,
                "Fold": fold,
                "Variant": "FAABE_NO_IWM",
                **hist,
            })

        row = {
            "Dataset": dataset_name,
            "Fold": fold,
            "Variant": "FAABE_NO_IWM",
            "Training_MMRE": fitness_no_iwm,
        }

        for name, weight in zip(
            selected_names,
            weights_no_iwm.detach().cpu().numpy(),
        ):
            row[name] = float(weight)

        weight_rows.append(row)

        # =====================================================================
        # 5. FAABE_FULL: Pearson + Firefly + IWM
        # =====================================================================

        log("Running FAABE_FULL...")

        (
            weights_full,
            fitness_full,
            population_fitness,
            history_full,
        ) = optimize_firefly(
            X_train_sel,
            y_train_gpu,
            seed=RANDOM_STATE + fold + 3000,
            log=log,
        )

        pred = iwm_predict(
            X_test_sel,
            X_train_sel,
            y_train_gpu,
            weights_full,
            K_ANALOGIES,
        )

        variant_predictions["FAABE_FULL"] = (
            pred.detach().cpu().numpy().astype(np.float64)
        )

        variant_training_mmre["FAABE_FULL"] = (
            fitness_full
        )

        for hist in history_full:
            history_rows.append({
                "Dataset": dataset_name,
                "Fold": fold,
                "Variant": "FAABE_FULL",
                **hist,
            })

        row = {
            "Dataset": dataset_name,
            "Fold": fold,
            "Variant": "FAABE_FULL",
            "Training_MMRE": fitness_full,
        }

        for name, weight in zip(
            selected_names,
            weights_full.detach().cpu().numpy(),
        ):
            row[name] = float(weight)

        weight_rows.append(row)

        # ---------------------------------------------------------------------
        # Clean predictions and fold metrics
        # ---------------------------------------------------------------------

        fold_row = {
            "Dataset": dataset_name,
            "Fold": fold,
            "Train_N": len(train_idx),
            "Test_N": len(test_idx),
            "Selected_Feature_Count": len(selected_names),
            "Selected_Features": ",".join(selected_names),
        }

        for variant in ABLATION_VARIANTS:
            pred = variant_predictions[variant]
            pred[~np.isfinite(pred)] = np.nan

            oof[variant][test_idx] = pred

            fold_row[f"{variant}_MMRE"] = MMRE(
                y_test,
                pred,
            )

            fold_row[f"{variant}_MAE"] = MAE(
                y_test,
                pred,
            )

            fold_row[f"{variant}_MSE"] = MSE(
                y_test,
                pred,
            )

            fold_row[f"{variant}_RMSE"] = RMSE(
                y_test,
                pred,
            )

            fold_row[
                f"{variant}_Training_MMRE"
            ] = variant_training_mmre[variant]

        fold_rows.append(fold_row)

        # ---------------------------------------------------------------------
        # Project-level OOF rows
        # ---------------------------------------------------------------------

        for local_idx, project_idx in enumerate(test_idx):
            actual = float(y_test[local_idx])

            oof_actual[project_idx] = actual
            oof_fold[project_idx] = fold

            row = {
                "Dataset": dataset_name,
                "Fold": fold,
                "Project_Index": int(project_idx),
                "Actual": actual,
            }

            for variant in ABLATION_VARIANTS:
                value = float(
                    variant_predictions[
                        variant
                    ][local_idx]
                )

                row[f"{variant}_Prediction"] = value
                row[f"{variant}_Absolute_Error"] = abs(
                    actual - value
                )
                row[f"{variant}_MRE"] = safe_mre(
                    [actual],
                    [value],
                )[0]

            prediction_rows.append(row)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # =========================================================================
    # OOF COVERAGE CHECK
    # =========================================================================

    if np.any(np.isnan(oof_actual)):
        raise RuntimeError(
            f"{dataset_name}: missing actual OOF values."
        )

    for variant in ABLATION_VARIANTS:
        missing = np.isnan(oof[variant])

        if np.any(missing):
            raise RuntimeError(
                f"{dataset_name}: {variant} missing "
                f"{int(np.sum(missing))} OOF predictions."
            )

    # =========================================================================
    # OVERALL ABLATION RESULTS
    # =========================================================================

    overall_rows = []

    for variant in ABLATION_VARIANTS:
        pred = oof[variant]

        overall_rows.append({
            "Dataset": dataset_name,
            "Variant": variant,
            "N_Projects": n_projects,
            "MMRE": MMRE(oof_actual, pred),
            "MAE": MAE(oof_actual, pred),
            "MSE": MSE(oof_actual, pred),
            "RMSE": RMSE(oof_actual, pred),
        })

    overall_df = pd.DataFrame(overall_rows)

    # =========================================================================
    # PAIRWISE STATISTICS: EACH ABLATION VS FULL FAABE
    # =========================================================================

    full_mre = safe_mre(
        oof_actual,
        oof["FAABE_FULL"],
    )

    comparison_rows = []

    for variant in ABLATION_VARIANTS:
        if variant == "FAABE_FULL":
            continue

        variant_mre = safe_mre(
            oof_actual,
            oof[variant],
        )

        valid = (
            np.isfinite(variant_mre)
            & np.isfinite(full_mre)
        )

        x = variant_mre[valid]
        y = full_mre[valid]

        stat, p = paired_wilcoxon(
            x,
            y,
        )

        delta = cliffs_delta(
            x,
            y,
        )

        variant_mmre = MMRE(
            oof_actual,
            oof[variant],
        )

        full_mmre = MMRE(
            oof_actual,
            oof["FAABE_FULL"],
        )

        comparison_rows.append({
            "Dataset": dataset_name,
            "Ablation": variant,
            "Proposed": "FAABE_FULL",
            "Valid_Pairs": int(np.sum(valid)),
            "Ablation_MMRE": variant_mmre,
            "FAABE_FULL_MMRE": full_mmre,
            "Full_Improvement_%": relative_improvement(
                variant_mmre,
                full_mmre,
            ),
            "Wilcoxon_Statistic": stat,
            "Wilcoxon_p": p,
            "Cliffs_Delta": delta,
            "Cliffs_Effect": cliffs_effect(delta),
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    if not comparison_df.empty:
        comparison_df[
            "Wilcoxon_p_Holm"
        ] = holm_adjust(
            comparison_df[
                "Wilcoxon_p"
            ].values
        )

        comparison_df[
            "Significant_Holm_0.05"
        ] = (
            comparison_df[
                "Wilcoxon_p_Holm"
            ] < 0.05
        )

    # =========================================================================
    # OOF DATAFRAME
    # =========================================================================

    oof_data = {
        "Dataset": dataset_name,
        "Project_Index": np.arange(n_projects),
        "Fold": oof_fold,
        "Actual": oof_actual,
    }

    for variant in ABLATION_VARIANTS:
        oof_data[
            f"{variant}_Prediction"
        ] = oof[variant]

        oof_data[
            f"{variant}_Absolute_Error"
        ] = np.abs(
            oof_actual
            - oof[variant]
        )

        oof_data[
            f"{variant}_MRE"
        ] = safe_mre(
            oof_actual,
            oof[variant],
        )

    oof_df = pd.DataFrame(
        oof_data
    )

    # =========================================================================
    # RANKING
    # =========================================================================

    ranking_df = (
        overall_df
        .sort_values(
            "MMRE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    ranking_df.insert(
        0,
        "MMRE_Rank",
        np.arange(
            1,
            len(ranking_df) + 1,
        ),
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    fold_df = pd.DataFrame(
        fold_rows
    )

    pred_df = pd.DataFrame(
        prediction_rows
    )

    feature_df = pd.DataFrame(
        feature_rows
    )

    weight_df = pd.DataFrame(
        weight_rows
    )

    history_df = pd.DataFrame(
        history_rows
    )

    overall_df.to_csv(
        dataset_dir
        / f"{dataset_name}_ABLATION_overall.csv",
        index=False,
    )

    comparison_df.to_csv(
        dataset_dir
        / f"{dataset_name}_ABLATION_statistics.csv",
        index=False,
    )

    ranking_df.to_csv(
        dataset_dir
        / f"{dataset_name}_ABLATION_ranking.csv",
        index=False,
    )

    fold_df.to_csv(
        dataset_dir
        / f"{dataset_name}_ABLATION_fold_results.csv",
        index=False,
    )

    pred_df.to_csv(
        dataset_dir
        / f"{dataset_name}_ABLATION_out_of_fold_predictions.csv",
        index=False,
    )

    oof_df.to_csv(
        dataset_dir
        / f"{dataset_name}_ABLATION_OOF_predictions.csv",
        index=False,
    )

    feature_df.to_csv(
        dataset_dir
        / f"{dataset_name}_ABLATION_feature_selection.csv",
        index=False,
    )

    weight_df.to_csv(
        dataset_dir
        / f"{dataset_name}_ABLATION_weights.csv",
        index=False,
    )

    history_df.to_csv(
        dataset_dir
        / f"{dataset_name}_ABLATION_firefly_history.csv",
        index=False,
    )

    run_config = {
        "dataset": dataset_name,
        "file": str(data_path),
        "target": config["target"],
        "features": feature_columns,
        "variants": ABLATION_VARIANTS,
        "pearson_threshold": PEARSON_THRESHOLD,
        "k_analogies": K_ANALOGIES,
        "num_fireflies": NUM_FIREFLIES,
        "gamma": GAMMA,
        "alpha": ALPHA,
        "max_iterations": MAX_ITERATIONS,
        "outer_folds": N_FOLDS,
        "random_seed": RANDOM_STATE,
        "similarity": SIMILARITY,
        "delta": DELTA,
        "training_fitness":
            "leave-one-out MMRE on outer-training fold only",
        "test_labels_used_during_training": False,
        "imputer_fit_on_test": False,
        "scaler_fit_on_test": False,
        "pearson_fit_on_test": False,
        "statistics":
            "paired project-level OOF MRE; Wilcoxon + Cliff's Delta + Holm",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }

    with open(
        dataset_dir
        / "ablation_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_config,
            f,
            indent=2,
        )

    log("")
    log("=" * 110)
    log("FINAL ABLATION RANKING")
    log("=" * 110)

    for _, row in ranking_df.iterrows():
        log(
            f"{int(row['MMRE_Rank']):2d}. "
            f"{row['Variant']:<20} | "
            f"MMRE={row['MMRE']:.8f} | "
            f"MAE={row['MAE']:.6f} | "
            f"RMSE={row['RMSE']:.6f}"
        )

    return {
        "overall": overall_df,
        "comparisons": comparison_df,
        "ranking": ranking_df,
        "folds": fold_df,
        "predictions": pred_df,
        "oof": oof_df,
        "features": feature_df,
        "weights": weight_df,
        "history": history_df,
    }


# =============================================================================
# MAIN – ALL 10 DATASETS
# =============================================================================

def main():
    master_log("=" * 120)
    master_log(
        "FAABE ABLATION – ALL 10 DATASETS"
    )
    master_log("=" * 120)

    master_log(
        f"Start: {datetime.now().isoformat()}"
    )

    master_log(
        f"Data dir: {DATA_DIR}"
    )

    master_log(
        f"Results dir: {RESULTS_DIR}"
    )

    master_log(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":
        master_log(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    master_log(
        f"Variants: {ABLATION_VARIANTS}"
    )

    collections = {
        "overall": [],
        "comparisons": [],
        "ranking": [],
        "folds": [],
        "predictions": [],
        "oof": [],
        "features": [],
        "weights": [],
        "history": [],
    }

    status_rows = []

    for number, (
        dataset_name,
        config,
    ) in enumerate(
        DATASETS.items(),
        start=1,
    ):
        master_log("")
        master_log("#" * 120)
        master_log(
            f"[{number}/{len(DATASETS)}] "
            f"{dataset_name}"
        )
        master_log("#" * 120)

        try:
            result = run_ablation_dataset(
                dataset_name,
                config,
            )

            for key in collections:
                collections[key].append(
                    result[key]
                )

            status_rows.append({
                "Dataset": dataset_name,
                "Status": "SUCCESS",
                "Error": "",
            })

            master_log(
                f"{dataset_name}: SUCCESS"
            )

        except Exception as exc:
            error_text = (
                f"{type(exc).__name__}: {exc}"
            )

            status_rows.append({
                "Dataset": dataset_name,
                "Status": "FAILED",
                "Error": error_text,
            })

            master_log(
                f"{dataset_name}: FAILED -> "
                f"{error_text}"
            )

            error_dir = (
                RESULTS_DIR
                / dataset_name
            )

            error_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                error_dir / "ERROR.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    traceback.format_exc()
                )

        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    def save_combined(
        frames,
        filename,
    ):
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )

        combined.to_csv(
            RESULTS_DIR / filename,
            index=False,
        )

        return combined

    combined_overall = save_combined(
        collections["overall"],
        "ALL_DATASETS_ABLATION_overall.csv",
    )

    combined_comparisons = save_combined(
        collections["comparisons"],
        "ALL_DATASETS_ABLATION_statistics.csv",
    )

    save_combined(
        collections["ranking"],
        "ALL_DATASETS_ABLATION_rankings.csv",
    )

    save_combined(
        collections["folds"],
        "ALL_DATASETS_ABLATION_fold_results.csv",
    )

    save_combined(
        collections["predictions"],
        "ALL_DATASETS_ABLATION_out_of_fold_predictions.csv",
    )

    save_combined(
        collections["oof"],
        "ALL_DATASETS_ABLATION_OOF_predictions.csv",
    )

    save_combined(
        collections["features"],
        "ALL_DATASETS_ABLATION_feature_selection.csv",
    )

    save_combined(
        collections["weights"],
        "ALL_DATASETS_ABLATION_weights.csv",
    )

    save_combined(
        collections["history"],
        "ALL_DATASETS_ABLATION_firefly_history.csv",
    )

    status_df = pd.DataFrame(
        status_rows
    )

    status_df.to_csv(
        RESULTS_DIR
        / "ALL_DATASETS_ABLATION_run_status.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Paper table
    # -------------------------------------------------------------------------

    if not combined_overall.empty:
        paper_table = combined_overall[
            [
                "Dataset",
                "Variant",
                "MMRE",
                "MAE",
                "MSE",
                "RMSE",
            ]
        ].copy()

        paper_table.to_csv(
            RESULTS_DIR
            / "PAPER_TABLE_ABLATION_ALL_DATASETS.csv",
            index=False,
        )

        mmre_wide = (
            combined_overall
            .pivot(
                index="Dataset",
                columns="Variant",
                values="MMRE",
            )
            .reset_index()
        )

        mmre_wide.to_csv(
            RESULTS_DIR
            / "PAPER_TABLE_ABLATION_MMRE_WIDE.csv",
            index=False,
        )

        rank_rows = []

        for dataset_name, group in (
            combined_overall.groupby(
                "Dataset"
            )
        ):
            temp = group.copy()

            temp["MMRE_Rank"] = (
                temp["MMRE"]
                .rank(
                    method="average",
                    ascending=True,
                )
            )

            rank_rows.append(
                temp[
                    [
                        "Dataset",
                        "Variant",
                        "MMRE",
                        "MMRE_Rank",
                    ]
                ]
            )

        if rank_rows:
            ranks = pd.concat(
                rank_rows,
                ignore_index=True,
            )

            average_ranks = (
                ranks
                .groupby(
                    "Variant",
                    as_index=False,
                )[
                    "MMRE_Rank"
                ]
                .mean()
                .rename(
                    columns={
                        "MMRE_Rank":
                            "Average_MMRE_Rank"
                    }
                )
                .sort_values(
                    "Average_MMRE_Rank"
                )
            )

            average_ranks.to_csv(
                RESULTS_DIR
                / "PAPER_TABLE_ABLATION_AVERAGE_RANKS.csv",
                index=False,
            )

    successes = int(
        (
            status_df["Status"]
            == "SUCCESS"
        ).sum()
    )

    failures = int(
        (
            status_df["Status"]
            == "FAILED"
        ).sum()
    )

    master_log("")
    master_log("=" * 120)
    master_log("ABLATION COMPLETE")
    master_log("=" * 120)
    master_log(
        f"Successful datasets: "
        f"{successes}/{len(DATASETS)}"
    )
    master_log(
        f"Failed datasets: {failures}"
    )
    master_log(
        f"Results: {RESULTS_DIR}"
    )

    print("")
    print("RUN STATUS")
    print(
        status_df.to_string(
            index=False
        )
    )

    if not combined_overall.empty:
        print("")
        print("ABLATION RESULTS")
        print(
            combined_overall[
                [
                    "Dataset",
                    "Variant",
                    "MMRE",
                    "MAE",
                    "RMSE",
                ]
            ].to_string(
                index=False
            )
        )

    if not combined_comparisons.empty:
        print("")
        print("ABLATIONS VS FULL FAABE")
        print(
            combined_comparisons[
                [
                    "Dataset",
                    "Ablation",
                    "Ablation_MMRE",
                    "FAABE_FULL_MMRE",
                    "Full_Improvement_%",
                    "Wilcoxon_p",
                    "Wilcoxon_p_Holm",
                    "Cliffs_Delta",
                    "Cliffs_Effect",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
