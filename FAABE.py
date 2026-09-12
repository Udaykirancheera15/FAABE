"""
SINGLE-FILE FAABE + BASELINE CUDA EVALUATION
============================================

Methods: ABE, GA-ABE, PSO-ABE, DE-ABE, GWO-ABE, WOA-ABE, FAABE.
All methods use the same leakage-free 10-fold protocol and training-only optimization.
Place this one file in the same folder as the 10 datasets and run:
    CUDA_VISIBLE_DEVICES=0 python3 FAABE.py
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

RESULTS_DIR = DATA_DIR / "results_ALL_BASELINES_CUDA"

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

def master_log(message=""):
    message = str(message)
    print(message)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "master_experiment.log").open("a", encoding="utf-8") as f:
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
# ONE DATASET
# =============================================================================

def run_one_dataset(dataset_name, config):
    dataset_dir = RESULTS_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    log = make_dataset_logger(dataset_dir)

    log("=" * 100)
    log(f"DATASET: {dataset_name}")
    log("=" * 100)
    log(f"Timestamp: {datetime.now().isoformat()}")
    log(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"CUDA version: {torch.version.cuda}")

    log(f"PyTorch version: {torch.__version__}")

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

    fold_results = []
    prediction_rows = []
    weight_rows = []
    feature_rows = []
    history_rows = []

    oof_actual = np.full(
        n_projects,
        np.nan,
        dtype=np.float64,
    )

    oof_abe = np.full(
        n_projects,
        np.nan,
        dtype=np.float64,
    )

    oof_faabe = np.full(
        n_projects,
        np.nan,
        dtype=np.float64,
    )

    oof_fold = np.full(
        n_projects,
        -1,
        dtype=int,
    )

    project_ids = np.arange(n_projects)

    # =========================================================================
    # OUTER FOLDS
    # =========================================================================

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_np),
        start=1,
    ):
        log("")
        log("-" * 100)
        log(f"OUTER FOLD {fold}/{N_FOLDS}")
        log("-" * 100)
        log(f"Training projects: {len(train_idx)}")
        log(f"Test projects: {len(test_idx)}")

        X_train_raw = X_np[train_idx]
        X_test_raw = X_np[test_idx]

        y_train = y_np[train_idx]
        y_test = y_np[test_idx]

        # ---------------------------------------------------------------------
        # TRAINING-ONLY IMPUTATION
        # ---------------------------------------------------------------------

        imputer = fit_imputer(X_train_raw)

        X_train_imp = apply_imputer(
            X_train_raw,
            imputer,
        )

        X_test_imp = apply_imputer(
            X_test_raw,
            imputer,
        )

        # ---------------------------------------------------------------------
        # TRAINING-ONLY SCALING
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
        # TRAINING-ONLY PEARSON FEATURE SELECTION
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

        log(
            f"Selected {len(selected_names)}/"
            f"{len(feature_columns)} features: "
            f"{selected_names}"
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

        X_train_selected = X_train_scaled[:, selected_indices]
        X_test_selected = X_test_scaled[:, selected_indices]

        # ---------------------------------------------------------------------
        # CUDA / TORCH
        # ---------------------------------------------------------------------

        X_train_gpu = torch.tensor(
            X_train_selected,
            dtype=torch.float32,
            device=DEVICE,
        )

        X_test_gpu = torch.tensor(
            X_test_selected,
            dtype=torch.float32,
            device=DEVICE,
        )

        y_train_gpu = torch.tensor(
            y_train,
            dtype=torch.float32,
            device=DEVICE,
        )

        # ---------------------------------------------------------------------
        # ABE: UNIFORM WEIGHTS
        # ---------------------------------------------------------------------

        uniform_weights = torch.ones(
            len(selected_indices),
            dtype=torch.float32,
            device=DEVICE,
        )

        uniform_weights = normalize_weights(
            uniform_weights
        )

        abe_test_predictions = iwm_predict(
            X_test_gpu,
            X_train_gpu,
            y_train_gpu,
            uniform_weights,
            K_ANALOGIES,
        )

        # ---------------------------------------------------------------------
        # FAABE: TRAINING-ONLY OPTIMIZATION
        # ---------------------------------------------------------------------

        log("Running training-only Firefly optimization...")

        (
            best_weights,
            training_mmre,
            population_fitness,
            history,
        ) = optimize_firefly(
            X_train_gpu,
            y_train_gpu,
            seed=RANDOM_STATE + fold,
            log=log,
        )

        for row in history:
            history_rows.append({
                "Dataset": dataset_name,
                "Fold": fold,
                **row,
            })

        weights_cpu = (
            best_weights
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        weight_row = {
            "Dataset": dataset_name,
            "Fold": fold,
            "Training_MMRE": training_mmre,
        }

        for name, weight in zip(
            selected_names,
            weights_cpu,
        ):
            weight_row[name] = float(weight)

        weight_rows.append(weight_row)

        # ---------------------------------------------------------------------
        # UNTOUCHED OUTER TEST PREDICTION
        # ---------------------------------------------------------------------

        faabe_test_predictions = iwm_predict(
            X_test_gpu,
            X_train_gpu,
            y_train_gpu,
            best_weights,
            K_ANALOGIES,
        )

        abe_pred = (
            abe_test_predictions
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        faabe_pred = (
            faabe_test_predictions
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        abe_pred[~np.isfinite(abe_pred)] = np.nan
        faabe_pred[~np.isfinite(faabe_pred)] = np.nan

        # ---------------------------------------------------------------------
        # FOLD METRICS
        # ---------------------------------------------------------------------

        abe_mmre = MMRE(y_test, abe_pred)
        faabe_mmre = MMRE(y_test, faabe_pred)

        abe_mae = MAE(y_test, abe_pred)
        faabe_mae = MAE(y_test, faabe_pred)

        abe_mse = MSE(y_test, abe_pred)
        faabe_mse = MSE(y_test, faabe_pred)

        abe_rmse = RMSE(y_test, abe_pred)
        faabe_rmse = RMSE(y_test, faabe_pred)

        log(
            f"Fold {fold}: "
            f"ABE MMRE={abe_mmre:.8f} | "
            f"FAABE MMRE={faabe_mmre:.8f}"
        )

        fold_results.append({
            "Dataset": dataset_name,
            "Fold": fold,
            "Train_N": len(train_idx),
            "Test_N": len(test_idx),
            "Selected_Feature_Count": len(selected_names),
            "Selected_Features": ",".join(selected_names),
            "Training_MMRE": training_mmre,

            "ABE_MMRE": abe_mmre,
            "FAABE_MMRE": faabe_mmre,
            "MMRE_Improvement_%":
                relative_improvement(abe_mmre, faabe_mmre),

            "ABE_MAE": abe_mae,
            "FAABE_MAE": faabe_mae,
            "MAE_Improvement_%":
                relative_improvement(abe_mae, faabe_mae),

            "ABE_MSE": abe_mse,
            "FAABE_MSE": faabe_mse,
            "MSE_Improvement_%":
                relative_improvement(abe_mse, faabe_mse),

            "ABE_RMSE": abe_rmse,
            "FAABE_RMSE": faabe_rmse,
            "RMSE_Improvement_%":
                relative_improvement(abe_rmse, faabe_rmse),
        })

        # ---------------------------------------------------------------------
        # OOF STORAGE
        # ---------------------------------------------------------------------

        for local_idx, project_idx in enumerate(test_idx):
            actual = float(y_test[local_idx])
            abe_value = float(abe_pred[local_idx])
            faabe_value = float(faabe_pred[local_idx])

            abe_mre = safe_mre(
                [actual],
                [abe_value],
            )[0]

            faabe_mre = safe_mre(
                [actual],
                [faabe_value],
            )[0]

            oof_actual[project_idx] = actual
            oof_abe[project_idx] = abe_value
            oof_faabe[project_idx] = faabe_value
            oof_fold[project_idx] = fold

            prediction_rows.append({
                "Dataset": dataset_name,
                "Fold": fold,
                "Project_Index": int(project_idx),
                "Actual": actual,
                "ABE_Prediction": abe_value,
                "FAABE_Prediction": faabe_value,
                "ABE_Absolute_Error":
                    abs(actual - abe_value),
                "FAABE_Absolute_Error":
                    abs(actual - faabe_value),
                "ABE_MRE": abe_mre,
                "FAABE_MRE": faabe_mre,
            })

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # =========================================================================
    # OOF COVERAGE
    # =========================================================================

    missing_oof = (
        np.isnan(oof_actual)
        | np.isnan(oof_abe)
        | np.isnan(oof_faabe)
    )

    if np.any(missing_oof):
        raise RuntimeError(
            f"{dataset_name}: missing OOF predictions for "
            f"{int(np.sum(missing_oof))} projects."
        )

    oof_abe_mre = safe_mre(
        oof_actual,
        oof_abe,
    )

    oof_faabe_mre = safe_mre(
        oof_actual,
        oof_faabe,
    )

    valid_oof = (
        np.isfinite(oof_abe_mre)
        & np.isfinite(oof_faabe_mre)
    )

    abe_errors = oof_abe_mre[valid_oof]
    faabe_errors = oof_faabe_mre[valid_oof]

    if len(abe_errors) == 0:
        raise RuntimeError(
            f"{dataset_name}: no valid paired OOF MRE values."
        )

    # =========================================================================
    # PROJECT-LEVEL STATISTICS
    # =========================================================================

    try:
        if np.allclose(
            abe_errors,
            faabe_errors,
            equal_nan=True,
        ):
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
        else:
            result = wilcoxon(
                abe_errors,
                faabe_errors,
                alternative="two-sided",
                zero_method="wilcox",
            )
            wilcoxon_stat = float(result.statistic)
            wilcoxon_p = float(result.pvalue)

    except ValueError:
        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan

    delta = cliffs_delta(
        abe_errors,
        faabe_errors,
    )

    effect = cliffs_effect(delta)

    # =========================================================================
    # OVERALL OOF METRICS
    # =========================================================================

    abe_mmre = MMRE(oof_actual, oof_abe)
    faabe_mmre = MMRE(oof_actual, oof_faabe)

    abe_mae = MAE(oof_actual, oof_abe)
    faabe_mae = MAE(oof_actual, oof_faabe)

    abe_mse = MSE(oof_actual, oof_abe)
    faabe_mse = MSE(oof_actual, oof_faabe)

    abe_rmse = RMSE(oof_actual, oof_abe)
    faabe_rmse = RMSE(oof_actual, oof_faabe)

    overall_metrics = {
        "Dataset": dataset_name,
        "File": data_path.name,
        "N_Projects": n_projects,
        "N_Input_Features": len(feature_columns),
        "Valid_OOF_Pairs": len(abe_errors),

        "ABE_MMRE": abe_mmre,
        "FAABE_MMRE": faabe_mmre,
        "MMRE_Improvement_%":
            relative_improvement(abe_mmre, faabe_mmre),

        "ABE_MAE": abe_mae,
        "FAABE_MAE": faabe_mae,
        "MAE_Improvement_%":
            relative_improvement(abe_mae, faabe_mae),

        "ABE_MSE": abe_mse,
        "FAABE_MSE": faabe_mse,
        "MSE_Improvement_%":
            relative_improvement(abe_mse, faabe_mse),

        "ABE_RMSE": abe_rmse,
        "FAABE_RMSE": faabe_rmse,
        "RMSE_Improvement_%":
            relative_improvement(abe_rmse, faabe_rmse),

        "Wilcoxon_Statistic": wilcoxon_stat,
        "Wilcoxon_p": wilcoxon_p,
        "Wilcoxon_Significant_0.05":
            bool(wilcoxon_p < 0.05)
            if np.isfinite(wilcoxon_p)
            else False,

        "Cliffs_Delta": delta,
        "Cliffs_Effect": effect,
    }

    # =========================================================================
    # SAVE PER-DATASET FILES
    # =========================================================================

    fold_df = pd.DataFrame(fold_results)
    pred_df = pd.DataFrame(prediction_rows)
    weights_df = pd.DataFrame(weight_rows)
    features_df = pd.DataFrame(feature_rows)
    history_df = pd.DataFrame(history_rows)
    overall_df = pd.DataFrame([overall_metrics])

    oof_df = pd.DataFrame({
        "Dataset": dataset_name,
        "Project_Index": project_ids,
        "Fold": oof_fold,
        "Actual": oof_actual,
        "ABE_Prediction": oof_abe,
        "FAABE_Prediction": oof_faabe,
        "ABE_Absolute_Error": np.abs(
            oof_actual - oof_abe
        ),
        "FAABE_Absolute_Error": np.abs(
            oof_actual - oof_faabe
        ),
        "ABE_MRE": oof_abe_mre,
        # Corrected: this is MRE, not the prediction.
        "FAABE_MRE": oof_faabe_mre,
    })

    fold_df.to_csv(
        dataset_dir / f"{dataset_name}_fold_results.csv",
        index=False,
    )

    overall_df.to_csv(
        dataset_dir / f"{dataset_name}_overall_statistics.csv",
        index=False,
    )

    pred_df.to_csv(
        dataset_dir / f"{dataset_name}_out_of_fold_predictions.csv",
        index=False,
    )

    oof_df.to_csv(
        dataset_dir / f"{dataset_name}_OOF_predictions.csv",
        index=False,
    )

    weights_df.to_csv(
        dataset_dir / f"{dataset_name}_firefly_weights.csv",
        index=False,
    )

    features_df.to_csv(
        dataset_dir / f"{dataset_name}_feature_selection.csv",
        index=False,
    )

    history_df.to_csv(
        dataset_dir / f"{dataset_name}_firefly_history.csv",
        index=False,
    )

    # Dataset description.
    description = {
        "Dataset": dataset_name,
        "Projects": n_projects,
        "Input_Features": len(feature_columns),
        "Target": config["target"],
        "Target_Min": float(np.min(y_np)),
        "Target_Max": float(np.max(y_np)),
        "Target_Mean": float(np.mean(y_np)),
        "Target_Median": float(np.median(y_np)),
        "Features": ",".join(feature_columns),
    }

    pd.DataFrame([description]).to_csv(
        dataset_dir / f"{dataset_name}_dataset_description.csv",
        index=False,
    )

    run_config = {
        "dataset": dataset_name,
        "file": str(data_path),
        "target": config["target"],
        "features": feature_columns,
        "ignored_columns": config.get("ignore", []),

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

        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    with open(
        dataset_dir / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_config, f, indent=2)

    log("")
    log("=" * 100)
    log("FINAL OOF RESULTS")
    log("=" * 100)
    log(f"ABE   MMRE : {abe_mmre:.8f}")
    log(f"FAABE MMRE : {faabe_mmre:.8f}")
    log(
        f"MMRE improvement: "
        f"{overall_metrics['MMRE_Improvement_%']:.4f}%"
    )
    log(f"ABE   MAE  : {abe_mae:.8f}")
    log(f"FAABE MAE  : {faabe_mae:.8f}")
    log(f"ABE   MSE  : {abe_mse:.8f}")
    log(f"FAABE MSE  : {faabe_mse:.8f}")
    log(f"ABE   RMSE : {abe_rmse:.8f}")
    log(f"FAABE RMSE : {faabe_rmse:.8f}")
    log(f"Wilcoxon p : {wilcoxon_p}")
    log(f"Cliff Delta: {delta} ({effect})")
    log(f"Saved to   : {dataset_dir}")

    return {
        "overall": overall_df,
        "folds": fold_df,
        "predictions": pred_df,
        "oof": oof_df,
        "features": features_df,
        "weights": weights_df,
        "history": history_df,
        "description": pd.DataFrame([description]),
    }


# =============================================================================
# MAIN: RUN ALL 10 DATASETS
# =============================================================================

def main():
    master_log("=" * 110)
    master_log("FAABE – ALL 10 DATASETS")
    master_log("=" * 110)
    master_log(f"Start time: {datetime.now().isoformat()}")
    master_log(f"Data dir: {DATA_DIR}")
    master_log(f"Results dir: {RESULTS_DIR}")
    master_log(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        master_log(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        master_log(
            f"CUDA version: {torch.version.cuda}"
        )
    else:
        master_log(
            "WARNING: CUDA is not available. "
            "The script will run on CPU."
        )

    all_overall = []
    all_folds = []
    all_predictions = []
    all_oof = []
    all_features = []
    all_weights = []
    all_history = []
    all_descriptions = []

    run_status = []

    for number, (dataset_name, config) in enumerate(
        DATASETS.items(),
        start=1,
    ):
        master_log("")
        master_log("#" * 110)
        master_log(
            f"[{number}/{len(DATASETS)}] RUNNING {dataset_name}"
        )
        master_log("#" * 110)

        try:
            result = run_one_dataset(
                dataset_name,
                config,
            )

            all_overall.append(result["overall"])
            all_folds.append(result["folds"])
            all_predictions.append(result["predictions"])
            all_oof.append(result["oof"])
            all_features.append(result["features"])
            all_weights.append(result["weights"])
            all_history.append(result["history"])
            all_descriptions.append(result["description"])

            run_status.append({
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

            run_status.append({
                "Dataset": dataset_name,
                "Status": "FAILED",
                "Error": error_text,
            })

            master_log(
                f"{dataset_name}: FAILED -> {error_text}"
            )

            error_dir = RESULTS_DIR / dataset_name
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

    # =========================================================================
    # COMBINED FILES
    # =========================================================================

    def save_combined(frames, filename):
        if frames:
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

        return pd.DataFrame()

    combined_overall = save_combined(
        all_overall,
        "ALL_DATASETS_overall_results.csv",
    )

    save_combined(
        all_folds,
        "ALL_DATASETS_fold_results.csv",
    )

    save_combined(
        all_predictions,
        "ALL_DATASETS_out_of_fold_predictions.csv",
    )

    save_combined(
        all_oof,
        "ALL_DATASETS_OOF_predictions.csv",
    )

    save_combined(
        all_features,
        "ALL_DATASETS_feature_selection.csv",
    )

    save_combined(
        all_weights,
        "ALL_DATASETS_firefly_weights.csv",
    )

    save_combined(
        all_history,
        "ALL_DATASETS_firefly_history.csv",
    )

    save_combined(
        all_descriptions,
        "ALL_DATASETS_description.csv",
    )

    status_df = pd.DataFrame(run_status)

    status_df.to_csv(
        RESULTS_DIR / "ALL_DATASETS_run_status.csv",
        index=False,
    )

    # =========================================================================
    # PAPER-FRIENDLY SUMMARY
    # =========================================================================

    if not combined_overall.empty:
        paper_columns = [
            "Dataset",
            "N_Projects",

            "ABE_MMRE",
            "FAABE_MMRE",
            "MMRE_Improvement_%",

            "ABE_MAE",
            "FAABE_MAE",

            "ABE_RMSE",
            "FAABE_RMSE",

            "Wilcoxon_p",
            "Wilcoxon_Significant_0.05",

            "Cliffs_Delta",
            "Cliffs_Effect",
        ]

        available_columns = [
            c
            for c in paper_columns
            if c in combined_overall.columns
        ]

        paper_df = combined_overall[
            available_columns
        ].copy()

        paper_df.to_csv(
            RESULTS_DIR / "PAPER_TABLE_ALL_DATASETS.csv",
            index=False,
        )

        # Ranking by proposed MMRE, lower is better.
        ranking = combined_overall[
            [
                "Dataset",
                "FAABE_MMRE",
                "MMRE_Improvement_%",
                "Wilcoxon_p",
                "Cliffs_Delta",
            ]
        ].copy()

        ranking = ranking.sort_values(
            "FAABE_MMRE",
            ascending=True,
        )

        ranking.insert(
            0,
            "FAABE_MMRE_Rank",
            np.arange(
                1,
                len(ranking) + 1,
            ),
        )

        ranking.to_csv(
            RESULTS_DIR / "ALL_DATASETS_FAABE_MMRE_ranking.csv",
            index=False,
        )

    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    successes = int(
        (status_df["Status"] == "SUCCESS").sum()
    )

    failures = int(
        (status_df["Status"] == "FAILED").sum()
    )

    master_log("")
    master_log("=" * 110)
    master_log("ALL-DATASET RUN COMPLETE")
    master_log("=" * 110)
    master_log(
        f"Successful datasets: {successes}/{len(DATASETS)}"
    )
    master_log(
        f"Failed datasets: {failures}/{len(DATASETS)}"
    )
    master_log(
        f"Results saved in: {RESULTS_DIR}"
    )
    master_log(
        f"End time: {datetime.now().isoformat()}"
    )
    master_log("=" * 110)

    print("")
    print("RUN STATUS")
    print(status_df.to_string(index=False))

    if not combined_overall.empty:
        print("")
        print("OVERALL RESULTS")
        display_cols = [
            "Dataset",
            "ABE_MMRE",
            "FAABE_MMRE",
            "MMRE_Improvement_%",
            "Wilcoxon_p",
            "Cliffs_Delta",
            "Cliffs_Effect",
        ]

        print(
            combined_overall[
                [
                    c
                    for c in display_cols
                    if c in combined_overall.columns
                ]
            ].to_string(index=False)
        )




# ============================================================================
# BASELINE COMPARISON RUNNER
# ============================================================================

"""
Leakage-Free CUDA Baseline Comparison for FAABE
================================================

Methods:
  ABE, GA-ABE, PSO-ABE, DE-ABE, GWO-ABE, WOA-ABE, FAABE

All methods share the same:
  - 10-fold outer CV
  - training-only median imputation
  - training-only MinMax scaling
  - training-only Pearson feature selection (|r| >= 0.25)
  - paper-style weighted Euclidean similarity
  - IWM with k=3
  - training-only leave-one-out MMRE objective
  - untouched outer test set
  - out-of-fold predictions for every project
  - project-level paired Wilcoxon + Cliff's Delta
  - common population size and common maximum fitness-evaluation budget

Place this file and faabe_core.py in the same directory as the 10 datasets.
"""

from pathlib import Path
import json, math, time, traceback
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wilcoxon


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent
RESULTS_DIR = DATA_DIR / "results_ALL_BASELINES_CUDA"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = DEVICE
SEED = RANDOM_STATE
N_FOLDS = N_FOLDS
K_ANALOGIES = K_ANALOGIES
PEARSON_THRESHOLD = PEARSON_THRESHOLD
EPS = EPS

POP_SIZE = 10
MAX_ITER = 100
MAX_EVALS = 1000
LOG_EVERY = 10

# FA
FA_ALPHA = 0.2
FA_BETA0 = 1.0
FA_GAMMA = 1.0

# GA
GA_CROSSOVER = 0.90
GA_MUTATION = 0.15
GA_MUTATION_STD = 0.10
GA_TOURNAMENT = 3
GA_ELITE = 1

# PSO
PSO_W = 0.729
PSO_C1 = 1.49445
PSO_C2 = 1.49445

# DE
DE_F = 0.5
DE_CR = 0.9

# WOA
WOA_B = 1.0

METHODS = ["ABE", "GA-ABE", "PSO-ABE", "DE-ABE", "GWO-ABE", "WOA-ABE", "FAABE"]
META_METHODS = METHODS[1:]


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def make_logger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg=""):
        msg = str(msg)
        print(msg)
        with path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    return log

MASTER_LOG = make_logger(RESULTS_DIR / "master.log")


# -----------------------------------------------------------------------------
# Shared optimizer helpers
# -----------------------------------------------------------------------------

class Budget:
    def __init__(self, X_train, y_train, max_evals=MAX_EVALS):
        self.X_train = X_train
        self.y_train = y_train
        self.max_evals = int(max_evals)
        self.count = 0

    def can(self):
        return self.count < self.max_evals

    @torch.no_grad()
    def evaluate(self, w):
        if not self.can():
            return None
        self.count += 1
        return firefly_fitness(w, self.X_train, self.y_train)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_population(n_features, seed):
    set_seed(seed)
    pop = torch.rand(POP_SIZE, n_features, device=DEVICE, dtype=torch.float32)
    for i in range(POP_SIZE):
        pop[i] = normalize_weights(pop[i])
    return pop


def evaluate_population(pop, budget):
    fit = torch.full((len(pop),), float("inf"), device=DEVICE)
    for i in range(len(pop)):
        score = budget.evaluate(pop[i])
        if score is None:
            break
        fit[i] = score
    return fit


def hist(iteration, best, evals):
    return {
        "Iteration": int(iteration),
        "Best_Training_MMRE": float(best),
        "Fitness_Evaluations": int(evals),
    }


def seed_for(method, fold):
    offsets = {
        "GA-ABE": 1000,
        "PSO-ABE": 2000,
        "DE-ABE": 3000,
        "GWO-ABE": 4000,
        "WOA-ABE": 5000,
        "FAABE": 6000,
    }
    return SEED + offsets[method] + fold


# -----------------------------------------------------------------------------
# GA
# -----------------------------------------------------------------------------

@torch.no_grad()
def optimize_ga(X, y, seed, log):
    n_features = X.shape[1]
    budget = Budget(X, y)
    pop = init_population(n_features, seed)
    fit = evaluate_population(pop, budget)
    rng = np.random.default_rng(seed)
    history = [hist(0, fit.min().item(), budget.count)]

    def tournament():
        ids = rng.choice(POP_SIZE, size=min(GA_TOURNAMENT, POP_SIZE), replace=False)
        ids_t = torch.tensor(ids, dtype=torch.long, device=DEVICE)
        return int(ids[int(torch.argmin(fit[ids_t]).item())])

    for it in range(1, MAX_ITER + 1):
        if not budget.can(): break
        order = torch.argsort(fit)
        children = [pop[order[e]].clone() for e in range(min(GA_ELITE, POP_SIZE))]

        while len(children) < POP_SIZE:
            p1, p2 = pop[tournament()].clone(), pop[tournament()].clone()
            child = p1.clone()
            if rng.random() < GA_CROSSOVER:
                lam = torch.rand(n_features, device=DEVICE)
                child = lam * p1 + (1.0 - lam) * p2
            mask = torch.rand(n_features, device=DEVICE) < GA_MUTATION
            child = child + mask.float() * GA_MUTATION_STD * torch.randn(n_features, device=DEVICE)
            children.append(normalize_weights(child))

        pop = torch.stack(children[:POP_SIZE])
        fit = torch.full((POP_SIZE,), float("inf"), device=DEVICE)
        for i in range(POP_SIZE):
            score = budget.evaluate(pop[i])
            if score is None: break
            fit[i] = score

        best = fit.min().item()
        history.append(hist(it, best, budget.count))
        if it == 1 or it % LOG_EVERY == 0:
            log(f"      GA  iter {it:3d} | best={best:.8f} | evals={budget.count}")

    idx = int(torch.argmin(fit).item())
    return normalize_weights(pop[idx]).clone(), float(fit[idx].item()), history, budget.count


# -----------------------------------------------------------------------------
# PSO
# -----------------------------------------------------------------------------

@torch.no_grad()
def optimize_pso(X, y, seed, log):
    n_features = X.shape[1]
    budget = Budget(X, y)
    pos = init_population(n_features, seed)
    set_seed(seed + 17)
    vel = 0.1 * torch.randn_like(pos)
    fit = evaluate_population(pos, budget)
    pbest, pbest_fit = pos.clone(), fit.clone()
    gi = int(torch.argmin(pbest_fit).item())
    gbest, gbest_fit = pbest[gi].clone(), float(pbest_fit[gi].item())
    history = [hist(0, gbest_fit, budget.count)]

    for it in range(1, MAX_ITER + 1):
        if not budget.can(): break
        r1, r2 = torch.rand_like(pos), torch.rand_like(pos)
        vel = PSO_W * vel + PSO_C1 * r1 * (pbest - pos) + PSO_C2 * r2 * (gbest[None, :] - pos)
        pos = pos + vel
        for i in range(POP_SIZE): pos[i] = normalize_weights(pos[i])

        new_fit = torch.full((POP_SIZE,), float("inf"), device=DEVICE)
        for i in range(POP_SIZE):
            score = budget.evaluate(pos[i])
            if score is None: break
            new_fit[i] = score

        improved = new_fit < pbest_fit
        pbest[improved] = pos[improved]
        pbest_fit[improved] = new_fit[improved]
        gi = int(torch.argmin(pbest_fit).item())
        if float(pbest_fit[gi].item()) < gbest_fit:
            gbest_fit = float(pbest_fit[gi].item())
            gbest = pbest[gi].clone()

        history.append(hist(it, gbest_fit, budget.count))
        if it == 1 or it % LOG_EVERY == 0:
            log(f"      PSO iter {it:3d} | best={gbest_fit:.8f} | evals={budget.count}")

    return normalize_weights(gbest).clone(), gbest_fit, history, budget.count


# -----------------------------------------------------------------------------
# Differential Evolution
# -----------------------------------------------------------------------------

@torch.no_grad()
def optimize_de(X, y, seed, log):
    if POP_SIZE < 4:
        raise ValueError("DE requires POP_SIZE >= 4")
    n_features = X.shape[1]
    budget = Budget(X, y)
    pop = init_population(n_features, seed)
    fit = evaluate_population(pop, budget)
    rng = np.random.default_rng(seed)
    history = [hist(0, fit.min().item(), budget.count)]

    for it in range(1, MAX_ITER + 1):
        if not budget.can(): break
        for i in range(POP_SIZE):
            if not budget.can(): break
            choices = [j for j in range(POP_SIZE) if j != i]
            a, b, c = rng.choice(choices, 3, replace=False)
            mutant = normalize_weights(pop[a] + DE_F * (pop[b] - pop[c]))
            mask = torch.rand(n_features, device=DEVICE) < DE_CR
            mask[int(rng.integers(0, n_features))] = True
            trial = normalize_weights(torch.where(mask, mutant, pop[i]))
            tf = budget.evaluate(trial)
            if tf is not None and tf < fit[i]:
                pop[i], fit[i] = trial, tf

        best = fit.min().item()
        history.append(hist(it, best, budget.count))
        if it == 1 or it % LOG_EVERY == 0:
            log(f"      DE  iter {it:3d} | best={best:.8f} | evals={budget.count}")

    idx = int(torch.argmin(fit).item())
    return normalize_weights(pop[idx]).clone(), float(fit[idx].item()), history, budget.count


# -----------------------------------------------------------------------------
# GWO
# -----------------------------------------------------------------------------

@torch.no_grad()
def optimize_gwo(X, y, seed, log):
    n_features = X.shape[1]
    budget = Budget(X, y)
    wolves = init_population(n_features, seed)
    fit = evaluate_population(wolves, budget)
    history = [hist(0, fit.min().item(), budget.count)]

    for it in range(1, MAX_ITER + 1):
        if not budget.can(): break
        order = torch.argsort(fit)
        alpha = wolves[order[0]].clone()
        beta = wolves[order[min(1, POP_SIZE - 1)]].clone()
        delta = wolves[order[min(2, POP_SIZE - 1)]].clone()
        a = 2.0 - 2.0 * (it / MAX_ITER)
        new_wolves = []

        for i in range(POP_SIZE):
            parts = []
            for leader in (alpha, beta, delta):
                r1 = torch.rand(n_features, device=DEVICE)
                r2 = torch.rand(n_features, device=DEVICE)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                D = torch.abs(C * leader - wolves[i])
                parts.append(leader - A * D)
            new_wolves.append(normalize_weights(sum(parts) / 3.0))

        wolves = torch.stack(new_wolves)
        fit = torch.full((POP_SIZE,), float("inf"), device=DEVICE)
        for i in range(POP_SIZE):
            score = budget.evaluate(wolves[i])
            if score is None: break
            fit[i] = score

        best = fit.min().item()
        history.append(hist(it, best, budget.count))
        if it == 1 or it % LOG_EVERY == 0:
            log(f"      GWO iter {it:3d} | best={best:.8f} | evals={budget.count}")

    idx = int(torch.argmin(fit).item())
    return normalize_weights(wolves[idx]).clone(), float(fit[idx].item()), history, budget.count


# -----------------------------------------------------------------------------
# WOA
# -----------------------------------------------------------------------------

@torch.no_grad()
def optimize_woa(X, y, seed, log):
    n_features = X.shape[1]
    budget = Budget(X, y)
    whales = init_population(n_features, seed)
    fit = evaluate_population(whales, budget)
    idx = int(torch.argmin(fit).item())
    best, best_fit = whales[idx].clone(), float(fit[idx].item())
    rng = np.random.default_rng(seed)
    history = [hist(0, best_fit, budget.count)]

    for it in range(1, MAX_ITER + 1):
        if not budget.can(): break
        a = 2.0 - 2.0 * (it / MAX_ITER)
        new_whales = []

        for i in range(POP_SIZE):
            r1, r2 = torch.rand(n_features, device=DEVICE), torch.rand(n_features, device=DEVICE)
            A, C = 2.0 * a * r1 - a, 2.0 * r2
            p = rng.random()
            if p < 0.5:
                if torch.mean(torch.abs(A)).item() < 1.0:
                    D = torch.abs(C * best - whales[i])
                    candidate = best - A * D
                else:
                    rw = whales[int(rng.integers(0, POP_SIZE))]
                    D = torch.abs(C * rw - whales[i])
                    candidate = rw - A * D
            else:
                l = rng.uniform(-1.0, 1.0)
                candidate = (
                    torch.abs(best - whales[i])
                    * math.exp(WOA_B * l)
                    * math.cos(2.0 * math.pi * l)
                    + best
                )
            new_whales.append(normalize_weights(candidate))

        whales = torch.stack(new_whales)
        fit = torch.full((POP_SIZE,), float("inf"), device=DEVICE)
        for i in range(POP_SIZE):
            score = budget.evaluate(whales[i])
            if score is None: break
            fit[i] = score
        ci = int(torch.argmin(fit).item())
        if float(fit[ci].item()) < best_fit:
            best_fit = float(fit[ci].item())
            best = whales[ci].clone()

        history.append(hist(it, best_fit, budget.count))
        if it == 1 or it % LOG_EVERY == 0:
            log(f"      WOA iter {it:3d} | best={best_fit:.8f} | evals={budget.count}")

    return normalize_weights(best).clone(), best_fit, history, budget.count


# -----------------------------------------------------------------------------
# Firefly with the same evaluation budget
# -----------------------------------------------------------------------------

@torch.no_grad()
def optimize_fa(X, y, seed, log):
    n_features = X.shape[1]
    budget = Budget(X, y)
    fireflies = init_population(n_features, seed)
    fit = evaluate_population(fireflies, budget)
    bi = int(torch.argmin(fit).item())
    best, best_fit = fireflies[bi].clone(), float(fit[bi].item())
    history = [hist(0, best_fit, budget.count)]

    for it in range(1, MAX_ITER + 1):
        if not budget.can(): break
        old_pos, old_fit = fireflies.clone(), fit.clone()

        for i in range(POP_SIZE):
            if not budget.can(): break
            candidate = old_pos[i].clone()
            for j in range(POP_SIZE):
                if old_fit[j] < old_fit[i]:
                    r = torch.linalg.norm(candidate - old_pos[j])
                    beta = FA_BETA0 * torch.exp(-FA_GAMMA * r * r)
                    candidate = (
                        candidate
                        + beta * (old_pos[j] - candidate)
                        + FA_ALPHA * (torch.rand(n_features, device=DEVICE) - 0.5)
                    )
                    candidate = normalize_weights(candidate)

            cf = budget.evaluate(candidate)
            if cf is not None and cf <= fit[i]:
                fireflies[i], fit[i] = candidate, cf

        ci = int(torch.argmin(fit).item())
        if float(fit[ci].item()) < best_fit:
            best_fit = float(fit[ci].item())
            best = fireflies[ci].clone()

        history.append(hist(it, best_fit, budget.count))
        if it == 1 or it % LOG_EVERY == 0:
            log(f"      FA  iter {it:3d} | best={best_fit:.8f} | evals={budget.count}")

    return normalize_weights(best).clone(), best_fit, history, budget.count


OPTIMIZERS = {
    "GA-ABE": optimize_ga,
    "PSO-ABE": optimize_pso,
    "DE-ABE": optimize_de,
    "GWO-ABE": optimize_gwo,
    "WOA-ABE": optimize_woa,
    "FAABE": optimize_fa,
}


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum(np.sum(a > y) for a in x)
    lt = sum(np.sum(a < y) for a in x)
    return float((gt - lt) / (len(x) * len(y)))


def cliffs_effect(d):
    if not np.isfinite(d): return "Undefined"
    d = abs(d)
    if d < 0.147: return "Negligible"
    if d < 0.330: return "Small"
    if d < 0.474: return "Medium"
    return "Large"


def paired_wilcoxon(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0: return np.nan, np.nan
    if np.allclose(x, y): return 0.0, 1.0
    try:
        r = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(r.statistic), float(r.pvalue)
    except ValueError:
        return np.nan, np.nan


def holm_adjust(pvalues):
    p = np.asarray(pvalues, float)
    out = np.full_like(p, np.nan)
    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0: return out
    vals = p[valid_idx]
    order = np.argsort(vals)
    m = len(vals)
    running = 0.0
    tmp = np.empty(m)
    for rank, oi in enumerate(order):
        running = max(running, vals[oi] * (m - rank))
        tmp[oi] = min(running, 1.0)
    out[valid_idx] = tmp
    return out


# -----------------------------------------------------------------------------
# Run one dataset
# -----------------------------------------------------------------------------

def run_dataset(dataset_name, config):
    ddir = RESULTS_DIR / dataset_name
    ddir.mkdir(parents=True, exist_ok=True)
    log = make_logger(ddir / "experiment.log")

    log("=" * 100)
    log(f"DATASET: {dataset_name}")
    log("=" * 100)
    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda": log(f"GPU: {torch.cuda.get_device_name(0)}")

    _, X_np, y_np, feature_names, data_path = prepare_dataset(dataset_name, config, log)
    n = len(y_np)
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof = {m: np.full(n, np.nan) for m in METHODS}
    actual = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)

    fold_rows, pred_rows, feat_rows, weight_rows, history_rows, runtime_rows = [], [], [], [], [], []

    for fold, (tr, te) in enumerate(cv.split(X_np), start=1):
        log("\n" + "-" * 100)
        log(f"OUTER FOLD {fold}/{N_FOLDS} | train={len(tr)} test={len(te)}")

        Xtr_raw, Xte_raw = X_np[tr], X_np[te]
        ytr, yte = y_np[tr], y_np[te]

        med = fit_imputer(Xtr_raw)
        Xtr = apply_imputer(Xtr_raw, med)
        Xte = apply_imputer(Xte_raw, med)

        scaler = MinMaxScaler()
        Xtr = np.nan_to_num(scaler.fit_transform(Xtr), nan=0.0, posinf=1.0, neginf=0.0)
        Xte = np.nan_to_num(scaler.transform(Xte), nan=0.0, posinf=1.0, neginf=0.0)

        sel_idx, sel_names, corrs = pearson_feature_selection(
            Xtr, ytr, feature_names, PEARSON_THRESHOLD
        )
        log(f"Selected {len(sel_names)}/{len(feature_names)}: {sel_names}")

        for name in feature_names:
            feat_rows.append({
                "Dataset": dataset_name, "Fold": fold, "Feature": name,
                "Pearson_r": corrs[name], "Abs_Pearson_r": abs(corrs[name]),
                "Selected": name in sel_names,
            })

        Xtr_t = torch.tensor(Xtr[:, sel_idx], dtype=torch.float32, device=DEVICE)
        Xte_t = torch.tensor(Xte[:, sel_idx], dtype=torch.float32, device=DEVICE)
        ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)

        # ABE
        uniform = normalize_weights(torch.ones(len(sel_idx), device=DEVICE))
        t0 = time.perf_counter()
        pred = iwm_predict(Xte_t, Xtr_t, ytr_t, uniform, K_ANALOGIES)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        runtime_rows.append({"Dataset": dataset_name, "Fold": fold, "Method": "ABE", "Runtime_Seconds": time.perf_counter()-t0, "Fitness_Evaluations": 0})
        pred_np = pred.detach().cpu().numpy().astype(float)
        oof["ABE"][te] = pred_np
        fold_predictions = {"ABE": pred_np}

        wrow = {"Dataset": dataset_name, "Fold": fold, "Method": "ABE", "Training_MMRE": np.nan, "Fitness_Evaluations": 0}
        for name, w in zip(sel_names, uniform.cpu().numpy()): wrow[name] = float(w)
        weight_rows.append(wrow)

        # Optimized methods
        train_fit = {"ABE": np.nan}
        for method in META_METHODS:
            log(f"\nRunning {method}...")
            opt = OPTIMIZERS[method]
            s = seed_for(method, fold)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            best_w, best_fit, history, evals = opt(Xtr_t, ytr_t, s, log)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            runtime = time.perf_counter() - t0

            pred = iwm_predict(Xte_t, Xtr_t, ytr_t, best_w, K_ANALOGIES)
            pred_np = pred.detach().cpu().numpy().astype(float)
            oof[method][te] = pred_np
            fold_predictions[method] = pred_np
            train_fit[method] = best_fit

            runtime_rows.append({"Dataset": dataset_name, "Fold": fold, "Method": method, "Runtime_Seconds": runtime, "Fitness_Evaluations": evals})
            wrow = {"Dataset": dataset_name, "Fold": fold, "Method": method, "Training_MMRE": best_fit, "Fitness_Evaluations": evals}
            for name, w in zip(sel_names, best_w.cpu().numpy()): wrow[name] = float(w)
            weight_rows.append(wrow)
            for h in history:
                history_rows.append({"Dataset": dataset_name, "Fold": fold, "Method": method, **h})

        # Fold metrics
        frow = {"Dataset": dataset_name, "Fold": fold, "Train_N": len(tr), "Test_N": len(te), "Selected_Feature_Count": len(sel_names), "Selected_Features": ",".join(sel_names)}
        for method in METHODS:
            p = fold_predictions[method]
            frow[f"{method}_MMRE"] = MMRE(yte, p)
            frow[f"{method}_MAE"] = MAE(yte, p)
            frow[f"{method}_MSE"] = MSE(yte, p)
            frow[f"{method}_RMSE"] = RMSE(yte, p)
            frow[f"{method}_Training_MMRE"] = train_fit[method]
        fold_rows.append(frow)

        # Project-level OOF rows
        for li, pi in enumerate(te):
            actual[pi], fold_id[pi] = float(yte[li]), fold
            row = {"Dataset": dataset_name, "Fold": fold, "Project_Index": int(pi), "Actual": float(yte[li])}
            for method in METHODS:
                pv = float(fold_predictions[method][li])
                row[f"{method}_Prediction"] = pv
                row[f"{method}_Absolute_Error"] = abs(float(yte[li]) - pv)
                row[f"{method}_MRE"] = safe_mre([float(yte[li])], [pv])[0]
            pred_rows.append(row)

        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    # Coverage check
    for method in METHODS:
        if np.isnan(oof[method]).any():
            raise RuntimeError(f"{dataset_name}: missing OOF predictions for {method}")

    # Overall metrics
    overall_rows = []
    for method in METHODS:
        p = oof[method]
        overall_rows.append({
            "Dataset": dataset_name, "Method": method, "N_Projects": n,
            "MMRE": MMRE(actual, p), "MAE": MAE(actual, p),
            "MSE": MSE(actual, p), "RMSE": RMSE(actual, p),
        })
    overall_df = pd.DataFrame(overall_rows)

    # Pairwise FAABE vs every baseline, on paired project-level OOF MRE
    fa_mre = safe_mre(actual, oof["FAABE"])
    comp_rows = []
    for baseline in METHODS:
        if baseline == "FAABE": continue
        b_mre = safe_mre(actual, oof[baseline])
        valid = np.isfinite(b_mre) & np.isfinite(fa_mre)
        stat, pval = paired_wilcoxon(b_mre[valid], fa_mre[valid])
        delta = cliffs_delta(b_mre[valid], fa_mre[valid])
        bmmre = MMRE(actual, oof[baseline])
        fmmre = MMRE(actual, oof["FAABE"])
        comp_rows.append({
            "Dataset": dataset_name, "Baseline": baseline, "Proposed": "FAABE",
            "Valid_Pairs": int(valid.sum()), "Baseline_MMRE": bmmre, "FAABE_MMRE": fmmre,
            "FAABE_MMRE_Improvement_%": relative_improvement(bmmre, fmmre),
            "Wilcoxon_Statistic": stat, "Wilcoxon_p": pval,
            "Cliffs_Delta": delta, "Cliffs_Effect": cliffs_effect(delta),
        })
    comp_df = pd.DataFrame(comp_rows)
    if not comp_df.empty:
        comp_df["Wilcoxon_p_Holm"] = holm_adjust(comp_df["Wilcoxon_p"].values)
        comp_df["Significant_Holm_0.05"] = comp_df["Wilcoxon_p_Holm"] < 0.05

    # OOF table
    oof_dict = {"Dataset": dataset_name, "Project_Index": np.arange(n), "Fold": fold_id, "Actual": actual}
    for method in METHODS:
        oof_dict[f"{method}_Prediction"] = oof[method]
        oof_dict[f"{method}_Absolute_Error"] = np.abs(actual - oof[method])
        oof_dict[f"{method}_MRE"] = safe_mre(actual, oof[method])
    oof_df = pd.DataFrame(oof_dict)

    # Ranking
    ranking_df = overall_df.sort_values("MMRE").reset_index(drop=True)
    ranking_df.insert(0, "MMRE_Rank", np.arange(1, len(ranking_df)+1))

    # Save
    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    feat_df = pd.DataFrame(feat_rows)
    weights_df = pd.DataFrame(weight_rows)
    hist_df = pd.DataFrame(history_rows)
    runtime_df = pd.DataFrame(runtime_rows)

    overall_df.to_csv(ddir / f"{dataset_name}_ALL_METHODS_overall.csv", index=False)
    comp_df.to_csv(ddir / f"{dataset_name}_FAABE_pairwise_statistics.csv", index=False)
    ranking_df.to_csv(ddir / f"{dataset_name}_method_ranking.csv", index=False)
    fold_df.to_csv(ddir / f"{dataset_name}_fold_results.csv", index=False)
    pred_df.to_csv(ddir / f"{dataset_name}_out_of_fold_predictions.csv", index=False)
    oof_df.to_csv(ddir / f"{dataset_name}_OOF_predictions.csv", index=False)
    feat_df.to_csv(ddir / f"{dataset_name}_feature_selection.csv", index=False)
    weights_df.to_csv(ddir / f"{dataset_name}_all_optimizer_weights.csv", index=False)
    hist_df.to_csv(ddir / f"{dataset_name}_optimizer_history.csv", index=False)
    runtime_df.to_csv(ddir / f"{dataset_name}_runtime_and_budget.csv", index=False)

    log("\nFINAL RANKING")
    log(ranking_df[["MMRE_Rank", "Method", "MMRE", "MAE", "RMSE"]].to_string(index=False))
    log("\nFAABE VS BASELINES")
    log(comp_df.to_string(index=False))

    return {
        "overall": overall_df,
        "comparisons": comp_df,
        "ranking": ranking_df,
        "folds": fold_df,
        "predictions": pred_df,
        "oof": oof_df,
        "features": feat_df,
        "weights": weights_df,
        "history": hist_df,
        "runtime": runtime_df,
    }


# -----------------------------------------------------------------------------
# Main: all 10 datasets
# -----------------------------------------------------------------------------

def main():
    MASTER_LOG("=" * 120)
    MASTER_LOG("LEAKAGE-FREE CUDA: ABE + GA + PSO + DE + GWO + WOA + FAABE")
    MASTER_LOG("=" * 120)
    MASTER_LOG(f"Device: {DEVICE}")
    if DEVICE.type == "cuda": MASTER_LOG(f"GPU: {torch.cuda.get_device_name(0)}")
    MASTER_LOG(f"Population={POP_SIZE} | max fitness evaluations={MAX_EVALS} | outer folds={N_FOLDS}")

    buckets = {k: [] for k in ["overall", "comparisons", "ranking", "folds", "predictions", "oof", "features", "weights", "history", "runtime"]}
    status = []

    for i, (name, cfg) in enumerate(DATASETS.items(), start=1):
        MASTER_LOG(f"\n[{i}/{len(DATASETS)}] {name}")
        try:
            result = run_dataset(name, cfg)
            for k in buckets: buckets[k].append(result[k])
            status.append({"Dataset": name, "Status": "SUCCESS", "Error": ""})
            MASTER_LOG(f"{name}: SUCCESS")
        except Exception as e:
            status.append({"Dataset": name, "Status": "FAILED", "Error": f"{type(e).__name__}: {e}"})
            MASTER_LOG(f"{name}: FAILED -> {type(e).__name__}: {e}")
            ddir = RESULTS_DIR / name
            ddir.mkdir(parents=True, exist_ok=True)
            (ddir / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            if DEVICE.type == "cuda": torch.cuda.empty_cache()

    def save(key, filename):
        if not buckets[key]: return pd.DataFrame()
        df = pd.concat(buckets[key], ignore_index=True, sort=False)
        df.to_csv(RESULTS_DIR / filename, index=False)
        return df

    overall = save("overall", "ALL_DATASETS_ALL_METHODS_overall.csv")
    comparisons = save("comparisons", "ALL_DATASETS_FAABE_pairwise_statistics.csv")
    save("ranking", "ALL_DATASETS_method_rankings.csv")
    save("folds", "ALL_DATASETS_fold_results.csv")
    save("predictions", "ALL_DATASETS_out_of_fold_predictions.csv")
    save("oof", "ALL_DATASETS_OOF_predictions.csv")
    save("features", "ALL_DATASETS_feature_selection.csv")
    save("weights", "ALL_DATASETS_all_optimizer_weights.csv")
    save("history", "ALL_DATASETS_optimizer_history.csv")
    save("runtime", "ALL_DATASETS_runtime_and_budget.csv")

    status_df = pd.DataFrame(status)
    status_df.to_csv(RESULTS_DIR / "ALL_DATASETS_run_status.csv", index=False)

    if not overall.empty:
        overall.to_csv(RESULTS_DIR / "PAPER_TABLE_ALL_METHODS.csv", index=False)
        mmre_wide = overall.pivot(index="Dataset", columns="Method", values="MMRE").reset_index()
        mmre_wide.to_csv(RESULTS_DIR / "PAPER_TABLE_MMRE_WIDE.csv", index=False)

        ranks = []
        for dataset, g in overall.groupby("Dataset"):
            g = g.copy()
            g["MMRE_Rank"] = g["MMRE"].rank(method="average", ascending=True)
            ranks.append(g[["Dataset", "Method", "MMRE", "MMRE_Rank"]])
        ranks = pd.concat(ranks, ignore_index=True)
        avg_rank = ranks.groupby("Method", as_index=False)["MMRE_Rank"].mean().rename(columns={"MMRE_Rank": "Average_MMRE_Rank"}).sort_values("Average_MMRE_Rank")
        avg_rank.to_csv(RESULTS_DIR / "PAPER_TABLE_AVERAGE_METHOD_RANKS.csv", index=False)

    config = {
        "methods": METHODS,
        "outer_folds": N_FOLDS,
        "pearson_threshold": PEARSON_THRESHOLD,
        "k_analogies": K_ANALOGIES,
        "population_size": POP_SIZE,
        "max_iterations": MAX_ITER,
        "max_fitness_evaluations_per_metaheuristic_per_fold": MAX_EVALS,
        "fitness": "training-only leave-one-out MMRE",
        "statistics": "paired project-level out-of-fold MRE",
        "multiple_testing": "Holm correction across FAABE-vs-baseline comparisons within each dataset",
        "test_labels_used_during_training": False,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (RESULTS_DIR / "GLOBAL_experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    MASTER_LOG("\nCOMPLETE")
    MASTER_LOG(status_df.to_string(index=False))
    if not overall.empty:
        print("\nOVERALL RESULTS\n", overall[["Dataset", "Method", "MMRE", "MAE", "RMSE"]].to_string(index=False))
    if not comparisons.empty:
        print("\nFAABE VS BASELINES\n", comparisons[["Dataset", "Baseline", "Baseline_MMRE", "FAABE_MMRE", "FAABE_MMRE_Improvement_%", "Wilcoxon_p", "Wilcoxon_p_Holm", "Cliffs_Delta", "Cliffs_Effect"]].to_string(index=False))


if __name__ == "__main__":
    main()
