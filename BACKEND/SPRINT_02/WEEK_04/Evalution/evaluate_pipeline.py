"""
============================================================
WEEK 04 — END-TO-END EVALUATION PIPELINE
============================================================

This script runs the full evaluation of the hybrid recommender
system using ground-truth purchase data from Week 02.

Pipeline:
  1. Load user-item interaction matrix (Week 02 artefact)
  2. Split users into train/test (80/20)
  3. For each test user:
       - Hold out items from the second half of their history
         as ground-truth relevance
       - Run the hybrid_recommender to get a ranked list
       - Compute P@K, R@K, NDCG@K, HR@K, MRR
  4. Print a formatted results table
  5. Save JSON report to Evalution/eval_report.json

Usage:
    python evaluate_pipeline.py
    python evaluate_pipeline.py --k 5 10 20 --max-users 200

============================================================
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Evalution/
WEEK04_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODELS_DIR = os.path.join(WEEK04_DIR, "Models")
WEEK02_DATA = os.path.abspath(
    os.path.join(WEEK04_DIR, "../../../SPRINT_01/WEEK_02/data")
)

# Add Models dir to path for itr_model imports
sys.path.insert(0, MODELS_DIR)

# ============================================================
# LOCAL IMPORTS
# ============================================================

# pyrefly: ignore [missing-import]
from metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    evaluate_recommender,
    print_metrics_table
)

# pyrefly: ignore [missing-import]
from itr_model import (
    load_user_item_matrix,
    load_product_metadata,
    load_product_embeddings,
    load_furniture_csv,
    load_model,
    predict_ranking,
    FEATURE_COLS
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# OUTPUT PATH
# ============================================================

REPORT_PATH = os.path.join(CURRENT_DIR, "eval_report.json")


# ============================================================
# TEST DATA BUILDER
# ============================================================

def build_test_data(
    user_item_matrix: pd.DataFrame,
    max_users: int = 200,
    min_interactions: int = 5
) -> dict:
    """
    Build test ground-truth from the user-item interaction matrix.

    Strategy:
      - For each user, sort their interacted items by column index
        (as a proxy for time ordering since no timestamps in matrix).
      - Use the LAST 20% of items as ground-truth 'relevant' set.
      - Only keep users with >= min_interactions items.

    Args:
        user_item_matrix  : DataFrame (users × items).
        max_users         : Max number of test users.
        min_interactions  : Minimum interactions required for a user.

    Returns:
        Dict: user_id → set of relevant item IDs (strings).
    """

    logger.info(f"Building test data (max_users={max_users})...")

    test_data = {}

    users = user_item_matrix.index.tolist()

    # Shuffle deterministically
    import random
    random.seed(42)
    random.shuffle(users)

    for user_id in users:

        if len(test_data) >= max_users:
            break

        user_row = user_item_matrix.loc[user_id]
        interacted = user_row[user_row > 0].index.astype(str).tolist()

        if len(interacted) < min_interactions:
            continue

        # Last 20% as ground truth (simulate leave-one-out or temporal split)
        cutoff = max(1, int(len(interacted) * 0.2))
        relevant_items = set(interacted[-cutoff:])

        test_data[user_id] = relevant_items

    logger.info(f"Test users built: {len(test_data)}")
    return test_data


# ============================================================
# SIMPLE BASELINE (Popularity)
# ============================================================

def popularity_recommender_factory(user_item_matrix: pd.DataFrame, top_k: int = 20):
    """
    Returns a simple popularity-based recommender function.

    Used as a baseline to compare against the XGBoost ranker.

    Args:
        user_item_matrix : User-item interaction matrix.
        top_k            : Number of items to return.

    Returns:
        Callable(user_id) → List[item_id].
    """

    # Pre-compute global item popularity
    item_popularity = (
        user_item_matrix.sum(axis=0)
        .sort_values(ascending=False)
    )
    top_items = item_popularity.index.astype(str).tolist()[:top_k]

    def recommender(user_id):
        # Return top popular items not already seen
        if user_id in user_item_matrix.index:
            seen = set(
                user_item_matrix.loc[user_id][
                    user_item_matrix.loc[user_id] > 0
                ].index.astype(str)
            )
            return [i for i in top_items if i not in seen]
        return top_items

    return recommender


# ============================================================
# XGBOOST RECOMMENDER FACTORY
# ============================================================

def xgb_recommender_factory(
    model,
    user_item_matrix: pd.DataFrame,
    furniture_df: pd.DataFrame,
    product_meta: pd.DataFrame,
    product_embeddings: np.ndarray,
    feature_cols: list,
    top_k: int = 20
):
    """
    Returns an XGBoost-based recommender function wrapped for evaluation.

    For each user, the recommender:
      1. Collects all items in the matrix as candidates
      2. Filters out already-seen items
      3. Ranks them using the XGBRanker
      4. Returns the top_k ranked item IDs

    Args:
        model             : Trained XGBRanker.
        user_item_matrix  : User-item interaction matrix.
        furniture_df      : Product features DataFrame.
        product_meta      : Product metadata DataFrame.
        product_embeddings: numpy embeddings.
        feature_cols      : Feature column order.
        top_k             : Number of items to recommend.

    Returns:
        Callable(user_id) → List[item_id] (ranked).
    """

    # Pre-collect all item IDs once
    all_items = user_item_matrix.columns.astype(str).tolist()

    def recommender(user_id):
        # Items the user has already seen
        if user_id in user_item_matrix.index:
            user_row = user_item_matrix.loc[user_id]
            seen = set(
                user_row[user_row > 0].index.astype(str)
            )
        else:
            seen = set()

        # Candidate items = all items not seen
        candidates = [i for i in all_items if i not in seen]

        # Limit candidates for speed (random sample of 200)
        if len(candidates) > 200:
            import random
            random.seed(42)
            candidates = random.sample(candidates, 200)

        # Rank using XGBRanker
        ranked = predict_ranking(
            model=model,
            user_id=user_id,
            candidate_items=candidates,
            user_item_matrix=user_item_matrix,
            furniture_df=furniture_df,
            product_meta=product_meta,
            product_embeddings=product_embeddings,
            feature_cols=feature_cols
        )

        return [str(i) for i in ranked[:top_k]]

    return recommender


# ============================================================
# MAIN EVALUATION RUNNER
# ============================================================

def run_evaluation(k_values: list = None, max_users: int = 200) -> dict:
    """
    Run the full evaluation pipeline and return the results report.

    Args:
        k_values  : List of K cut-offs (default [5, 10, 20]).
        max_users : Max number of test users to evaluate.

    Returns:
        Dict with keys: 'xgb_metrics', 'baseline_metrics', 'meta'.
    """

    if k_values is None:
        k_values = [5, 10, 20]

    logger.info("="*60)
    logger.info("STARTING EVALUATION PIPELINE")
    logger.info("="*60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    uim = load_user_item_matrix()
    meta = load_product_metadata()
    embeddings = load_product_embeddings()
    furn = load_furniture_csv()

    # --------------------------------------------------------
    # Build test data (ground-truth)
    # --------------------------------------------------------
    test_data = build_test_data(
        user_item_matrix=uim,
        max_users=max_users,
        min_interactions=5
    )

    # --------------------------------------------------------
    # Evaluate XGBoost Recommender
    # --------------------------------------------------------
    xgb_metrics = {}
    try:
        model, feature_cols = load_model()

        xgb_rec_fn = xgb_recommender_factory(
            model=model,
            user_item_matrix=uim,
            furniture_df=furn,
            product_meta=meta,
            product_embeddings=embeddings,
            feature_cols=feature_cols,
            top_k=max(k_values)
        )

        logger.info("Evaluating XGBoost Ranker...")
        xgb_metrics = evaluate_recommender(
            recommender_fn=xgb_rec_fn,
            test_data=test_data,
            k_values=k_values
        )
        print_metrics_table(xgb_metrics, title="XGBoost Ranker Metrics")

    except FileNotFoundError:
        logger.warning(
            "XGBoost model checkpoint not found. "
            "Please run training_pipeline.py first."
        )
        xgb_metrics = {"error": "Model not trained yet."}

    # --------------------------------------------------------
    # Evaluate Popularity Baseline
    # --------------------------------------------------------
    logger.info("Evaluating Popularity Baseline...")
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
    # Save report
    # --------------------------------------------------------
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_test_users": len(test_data),
        "k_values": k_values,
        "xgb_ranker": xgb_metrics,
        "popularity_baseline": baseline_metrics
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"✅ Evaluation report saved → {REPORT_PATH}")

    return report


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run hybrid recommender evaluation pipeline."
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for evaluation (default: 5 10 20)."
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=200,
        help="Max number of test users to evaluate (default: 200)."
    )
    args = parser.parse_args()

    run_evaluation(k_values=args.k, max_users=args.max_users)
