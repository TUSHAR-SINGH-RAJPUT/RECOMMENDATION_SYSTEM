"""
============================================================
WEEK 04 — FULL TRAINING PIPELINE ORCHESTRATOR
============================================================

This is the master script that ties together all Week 04 components:

  Step 1: Load data from Week 02 / Week 03 artefacts
  Step 2: Build the XGBoost training feature matrix
  Step 3: Load Optuna best hyperparameters (if available)
  Step 4: Train XGBoost ranker
  Step 5: Save model checkpoint
  Step 6: Run evaluation and print metrics

This script is designed to be run after:
  - WEEK_02/evaluation/matrix.py   (produces user_item_matrix.csv)
  - WEEK_03/src/embedder.py        (produces product_embeddings.npy)
  - WEEK_03/src/vector_store.py    (produces chroma_db)

Optionally, run experiments/tune_xgboost.py BEFORE this script
to benefit from optimal hyperparameters.

Usage:
    python training_pipeline.py
    python training_pipeline.py --max-users 500 --eval-users 100

============================================================
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Models/
WEEK04_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
EVALUTION_DIR = os.path.join(WEEK04_DIR, "Evalution")
EXPERIMENTS_DIR = os.path.join(WEEK04_DIR, "experiments")

# Add Evalution/ to path for metrics module
sys.path.insert(0, EVALUTION_DIR)

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
    train_xgb_ranker,
    save_model,
    load_model,
    predict_ranking,
    FEATURE_COLS,
    CHECKPOINT_DIR
)

# pyrefly: ignore [missing-import]
from metrics import print_metrics_table, evaluate_recommender

# ============================================================
# LOGGING — write to both console and log file
# ============================================================

LOG_PATH = os.path.join(CHECKPOINT_DIR, "training.log")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),           # console output
        logging.FileHandler(LOG_PATH, mode="a")  # file output
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# BEST PARAMS PATH
# ============================================================

BEST_PARAMS_PATH = os.path.join(EXPERIMENTS_DIR, "best_params.json")


# ============================================================
# LOAD OPTUNA BEST PARAMS
# ============================================================

def load_best_params() -> dict:
    """
    Load the best XGBoost hyperparameters saved by Optuna.

    Returns:
        Dict of best params, or empty dict if not available
        (training_pipeline will use defaults in that case).
    """

    if not os.path.exists(BEST_PARAMS_PATH):
        logger.info(
            "No Optuna best_params.json found. "
            "Using default XGBoost hyperparameters.\n"
            "  → Run experiments/tune_xgboost.py for better performance."
        )
        return {}

    with open(BEST_PARAMS_PATH, "r") as f:
        data = json.load(f)

    best_params = data.get("params", {})
    best_score = data.get("best_ndcg_at_10", "N/A")
    n_trials = data.get("n_trials_total", "N/A")

    logger.info(
        f"Loaded Optuna best params "
        f"(NDCG@10={best_score}, from {n_trials} trials): {best_params}"
    )

    return best_params


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training_pipeline(
    max_train_users: int = 500,
    max_eval_users: int = 200,
    k_values: list = None
) -> None:
    """
    Execute the full training pipeline end-to-end.

    Steps:
      1. Load data artefacts from previous weeks
      2. Build XGBoost feature matrix
      3. Load Optuna best hyperparameters (if available)
      4. Train XGBoost ranker
      5. Save model checkpoint + metadata
      6. Evaluate on held-out test users

    Args:
        max_train_users : Max users for feature matrix construction.
        max_eval_users  : Max users for evaluation.
        k_values        : K cut-offs for evaluation (default [5, 10, 20]).
    """

    if k_values is None:
        k_values = [5, 10, 20]

    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("WEEK 04 TRAINING PIPELINE STARTED")
    logger.info(f"Timestamp: {start_time.isoformat()}")
    logger.info("=" * 60)


    # --------------------------------------------------------
    # STEP 1: Load data
    # --------------------------------------------------------
    logger.info("\n[STEP 1] Loading data artefacts...")

    uim = load_user_item_matrix()
    logger.info(f"  user_item_matrix : {uim.shape}")

    meta = load_product_metadata()
    logger.info(f"  product metadata : {len(meta)} products")

    embeddings = load_product_embeddings()
    logger.info(f"  embeddings       : {embeddings.shape}")

    furn = load_furniture_csv()
    logger.info(f"  furniture CSV    : {furn.shape}")


    # --------------------------------------------------------
    # STEP 2: Build feature matrix
    # --------------------------------------------------------
    logger.info(f"\n[STEP 2] Building feature matrix (max_users={max_train_users})...")

    feature_df = build_feature_matrix(
        user_item_matrix=uim,
        furniture_df=furn,
        product_meta=meta,
        product_embeddings=embeddings,
        max_users=max_train_users
    )

    logger.info(
        f"  Feature matrix shape : {feature_df.shape}\n"
        f"  Positive samples     : {int(feature_df['label'].sum())}\n"
        f"  Negative samples     : {int((feature_df['label'] == 0).sum())}"
    )

    # Quick feature stats for diagnostics
    logger.info(f"\n  Feature stats:\n{feature_df[FEATURE_COLS].describe().to_string()}")


    # --------------------------------------------------------
    # STEP 3: Load Optuna best hyperparameters
    # --------------------------------------------------------
    logger.info("\n[STEP 3] Loading Optuna hyperparameters...")

    best_params = load_best_params()


    # --------------------------------------------------------
    # STEP 4: Train XGBoost ranker
    # --------------------------------------------------------
    logger.info("\n[STEP 4] Training XGBoost ranker...")

    model = train_xgb_ranker(feature_df, params=best_params if best_params else None)

    logger.info("  ✅ Training complete.")

    # Log feature importances
    try:
        importances = model.feature_importances_
        importance_map = dict(zip(FEATURE_COLS, importances))
        sorted_imp = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)

        logger.info("  Feature importances:")
        for feat, imp in sorted_imp:
            logger.info(f"    {feat:<30} {imp:.4f}")
    except Exception as e:
        logger.debug(f"Could not extract feature importances: {e}")


    # --------------------------------------------------------
    # STEP 5: Save model checkpoint
    # --------------------------------------------------------
    logger.info("\n[STEP 5] Saving model checkpoint...")

    save_model(model, feature_cols=FEATURE_COLS)


    # --------------------------------------------------------
    # STEP 6: Evaluation
    # --------------------------------------------------------
    logger.info(f"\n[STEP 6] Running evaluation (max_users={max_eval_users})...")

    # Import evaluation utilities here to avoid circular imports
    from evaluate_pipeline import build_test_data, xgb_recommender_factory, popularity_recommender_factory

    # Build test ground-truth
    test_data = build_test_data(
        user_item_matrix=uim,
        max_users=max_eval_users,
        min_interactions=5
    )

    # XGBoost recommender
    xgb_rec_fn = xgb_recommender_factory(
        model=model,
        user_item_matrix=uim,
        furniture_df=furn,
        product_meta=meta,
        product_embeddings=embeddings,
        feature_cols=FEATURE_COLS,
        top_k=max(k_values)
    )

    xgb_metrics = evaluate_recommender(
        recommender_fn=xgb_rec_fn,
        test_data=test_data,
        k_values=k_values
    )
    print_metrics_table(xgb_metrics, title="XGBoost Ranker Metrics")

    # Popularity baseline for comparison
    pop_rec_fn = popularity_recommender_factory(
        user_item_matrix=uim,
        top_k=max(k_values)
    )
    baseline_metrics = evaluate_recommender(
        recommender_fn=pop_rec_fn,
        test_data=test_data,
        k_values=k_values
    )
    print_metrics_table(baseline_metrics, title="Popularity Baseline Metrics")

    # --------------------------------------------------------
    # Save training report
    # --------------------------------------------------------
    elapsed = (datetime.now() - start_time).total_seconds()

    report = {
        "timestamp": start_time.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "feature_matrix_shape": list(feature_df.shape),
        "n_test_users": len(test_data),
        "k_values": k_values,
        "used_optuna_params": bool(best_params),
        "xgb_ranker_metrics": xgb_metrics,
        "popularity_baseline_metrics": baseline_metrics
    }

    report_path = os.path.join(CHECKPOINT_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n✅ Training report saved → {report_path}")
    logger.info(f"✅ Total pipeline time: {elapsed:.1f}s")
    logger.info("=" * 60)
    logger.info("WEEK 04 TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Week 04 XGBoost training pipeline."
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=500,
        help="Max number of users for training feature matrix (default: 500)."
    )
    parser.add_argument(
        "--eval-users",
        type=int,
        default=200,
        help="Max number of test users for evaluation (default: 200)."
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for evaluation metrics (default: 5 10 20)."
    )
    args = parser.parse_args()

    run_training_pipeline(
        max_train_users=args.max_users,
        max_eval_users=args.eval_users,
        k_values=args.k
    )
