"""
============================================================
WEEK 04 — OPTUNA HYPERPARAMETER TUNING FOR XGBoost RANKER
============================================================

This script runs an Optuna hyperparameter search to find the
best XGBoost ranking parameters for the hybrid recommender.

Key design choices:
  - Study is PERSISTED to experiments/optuna_study.db (SQLite),
    so interrupted runs automatically resume from where they left off.
  - Objective: maximise NDCG@10 on a held-out validation split.
  - Best params are saved to experiments/best_params.json for use
    by training_pipeline.py.
  - A plot of the optimization history is saved if matplotlib is available.

Search space:
  n_estimators        [50 – 500]
  max_depth           [3 – 8]
  learning_rate       [0.01 – 0.3]    (log scale)
  subsample           [0.6 – 1.0]
  colsample_bytree    [0.6 – 1.0]
  min_child_weight    [1 – 20]
  reg_alpha           [1e-4 – 10.0]   (log scale)
  reg_lambda          [1e-4 – 10.0]   (log scale)

Usage:
    python tune_xgboost.py              # runs 50 trials
    python tune_xgboost.py --n-trials 100
============================================================
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd

# ============================================================
# PATH SETUP — allow imports from WEEK_04/Models/
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # experiments/
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../Models"))
sys.path.insert(0, MODELS_DIR)

# ============================================================
# LOCAL IMPORTS
# ============================================================

# pyrefly: ignore [missing-import]
from itr_model import (
    load_user_item_matrix,
    load_product_metadata,
    load_product_embeddings,
    load_furniture_csv,
    build_feature_matrix,
    FEATURE_COLS
)

# ============================================================
# XGBOOST
# ============================================================

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("xgboost not installed. Run: pip install xgboost")

# ============================================================
# OPTUNA
# ============================================================

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("optuna not installed. Run: pip install optuna")

# ============================================================
# SKLEARN SPLIT
# ============================================================

from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

# SQLite DB path (already exists in experiments/)
STUDY_DB_PATH = os.path.join(CURRENT_DIR, "optuna_study.db")
STUDY_NAME = "xgb_ranker_study"

# Best params output
BEST_PARAMS_PATH = os.path.join(CURRENT_DIR, "best_params.json")

# Optimization plot
PLOT_PATH = os.path.join(CURRENT_DIR, "optuna_history.png")


# ============================================================
# NDCG@K HELPER (independent of metrics.py for speed)
# ============================================================

def _ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@K given predicted scores and binary labels.

    This standalone version is used inside the Optuna objective
    to avoid loading the full metrics module.

    Args:
        scores : Array of predicted relevance scores.
        labels : Array of ground-truth binary labels (1 = relevant).
        k      : Cut-off rank.

    Returns:
        NDCG@K value in [0, 1].
    """

    import math

    # Sort by predicted score (descending)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order[:k]]

    # DCG
    dcg = sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(sorted_labels)
    )

    # IDCG (ideal: all positives first)
    n_pos = int(labels.sum())
    ideal_labels = [1] * min(n_pos, k) + [0] * (k - min(n_pos, k))
    idcg = sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(ideal_labels)
    )

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# DATA LOADING (cached at module level to avoid re-loading)
# ============================================================

_data_cache = {}


def _get_data():
    """
    Load and cache all required datasets.
    Cached on first call so Optuna trials don't re-read from disk.
    """

    global _data_cache

    if not _data_cache:
        logger.info("Loading datasets for Optuna study...")
        _data_cache["uim"] = load_user_item_matrix()
        _data_cache["meta"] = load_product_metadata()
        _data_cache["embeddings"] = load_product_embeddings()
        _data_cache["furn"] = load_furniture_csv()
        logger.info("✅ Datasets loaded and cached.")

    return (
        _data_cache["uim"],
        _data_cache["meta"],
        _data_cache["embeddings"],
        _data_cache["furn"]
    )


# ============================================================
# FEATURE MATRIX (cached once, split inside objective)
# ============================================================

_feature_df_cache = None


def _get_feature_df(max_users: int = 300):
    """Build or return cached feature matrix."""

    global _feature_df_cache

    if _feature_df_cache is None:
        uim, meta, embeddings, furn = _get_data()
        logger.info(f"Building feature matrix (max_users={max_users})...")
        _feature_df_cache = build_feature_matrix(
            user_item_matrix=uim,
            furniture_df=furn,
            product_meta=meta,
            product_embeddings=embeddings,
            max_users=max_users
        )
        logger.info(f"Feature matrix ready: {_feature_df_cache.shape}")

    return _feature_df_cache


# ============================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================

def objective(trial: optuna.Trial) -> float:
    """
    Optuna trial objective function.

    Samples a set of XGBoost hyperparameters, trains a ranker on
    the training split, evaluates NDCG@10 on the validation split.

    Args:
        trial : Optuna Trial object.

    Returns:
        NDCG@10 on the validation split (higher = better).
    """

    # --------------------------------------------------------
    # Sample hyperparameters
    # --------------------------------------------------------
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    # --------------------------------------------------------
    # Load feature matrix
    # --------------------------------------------------------
    feature_df = _get_feature_df(max_users=300)

    # --------------------------------------------------------
    # Train / validation split (group-aware: by user_id)
    # --------------------------------------------------------
    # GroupShuffleSplit ensures no user appears in both splits
    groups = feature_df["user_id"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(feature_df, groups=groups))

    train_df = feature_df.iloc[train_idx].copy()
    val_df = feature_df.iloc[val_idx].copy()

    # --------------------------------------------------------
    # Prepare XGBRanker training data
    # --------------------------------------------------------
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    group_train = train_df.groupby("user_id").size().values

    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["label"].values

    # --------------------------------------------------------
    # Build and train model
    # --------------------------------------------------------
    model = xgb.XGBRanker(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        objective="rank:pairwise",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train, y_train, group=group_train)

    # --------------------------------------------------------
    # Evaluate NDCG@10 per user on validation set
    # --------------------------------------------------------
    val_scores = model.predict(X_val)
    val_df = val_df.copy()
    val_df["pred_score"] = val_scores

    ndcg_scores = []
    for user_id, group in val_df.groupby("user_id"):
        scores = group["pred_score"].values
        labels = group["label"].values

        # Only evaluate users that have at least one positive
        if labels.sum() == 0:
            continue

        ndcg_scores.append(_ndcg_at_k(scores, labels, k=10))

    if not ndcg_scores:
        return 0.0

    mean_ndcg = float(np.mean(ndcg_scores))

    logger.debug(f"Trial {trial.number}: NDCG@10 = {mean_ndcg:.4f} | params = {params}")

    return mean_ndcg


# ============================================================
# RUN OPTUNA STUDY
# ============================================================

def run_study(n_trials: int = 50) -> dict:
    """
    Create (or resume) an Optuna study and run hyperparameter search.

    The study persists to experiments/optuna_study.db, so if you
    stop and restart, Optuna automatically picks up where it left off.

    Args:
        n_trials : Number of trials to run in this session.

    Returns:
        Dict of best hyperparameters found.
    """

    storage_url = f"sqlite:///{STUDY_DB_PATH}"

    logger.info(f"Optuna storage: {storage_url}")
    logger.info(f"Study name: {STUDY_NAME}")
    logger.info(f"Running {n_trials} trials...")

    # --------------------------------------------------------
    # Create or load existing study
    # --------------------------------------------------------
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        direction="maximize",       # we want to maximise NDCG@10
        sampler=TPESampler(seed=42),
        load_if_exists=True         # resume if study already exists
    )

    # --------------------------------------------------------
    # Run trials with a progress callback
    # --------------------------------------------------------
    def _callback(study, trial):
        logger.info(
            f"  Trial {trial.number:>4} | "
            f"NDCG@10 = {trial.value:.4f} | "
            f"Best so far = {study.best_value:.4f}"
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_callback],
        show_progress_bar=False
    )

    # --------------------------------------------------------
    # Report best result
    # --------------------------------------------------------
    best_params = study.best_params
    best_value = study.best_value

    print("\n" + "="*50)
    print("  OPTUNA STUDY COMPLETE")
    print("="*50)
    print(f"  Best NDCG@10     : {best_value:.4f}")
    print(f"  Best trial number: {study.best_trial.number}")
    print("  Best parameters  :")
    for k, v in best_params.items():
        print(f"    {k:<25} = {v}")
    print("="*50 + "\n")

    # --------------------------------------------------------
    # Save best params to JSON
    # --------------------------------------------------------
    output = {
        "best_ndcg_at_10": best_value,
        "best_trial_number": study.best_trial.number,
        "n_trials_total": len(study.trials),
        "params": best_params
    }

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"✅ Best params saved → {BEST_PARAMS_PATH}")

    # --------------------------------------------------------
    # Optional: plot optimization history
    # --------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import optuna.visualization.matplotlib as ov_mpl

        fig = ov_mpl.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=120)
        plt.close()
        logger.info(f"✅ Optimization plot saved → {PLOT_PATH}")

    except Exception as e:
        logger.debug(f"Could not save plot: {e}")

    return best_params


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter tuning for XGBoost ranker."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run (default: 50)."
    )
    args = parser.parse_args()

    run_study(n_trials=args.n_trials)
