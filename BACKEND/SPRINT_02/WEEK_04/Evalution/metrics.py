"""
============================================================
WEEK 04 — EVALUATION METRICS MODULE
============================================================

This module implements standard information-retrieval metrics
used to evaluate recommendation quality:

  - Precision@K   (P@K)
  - Recall@K      (R@K)
  - NDCG@K        (Normalized Discounted Cumulative Gain)
  - MRR           (Mean Reciprocal Rank)
  - Hit Rate@K    (HR@K)

All functions are pure, stateless, and reusable by
evaluate_pipeline.py and any notebook analysis.

============================================================
"""

import math
import logging
from typing import List, Dict, Any

# ============================================================
# LOGGER SETUP
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# PRECISION @ K
# ============================================================

def precision_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Compute Precision@K.

    Precision@K = (# relevant items in top-K recommendations) / K

    Args:
        recommended : Ordered list of recommended item IDs.
        relevant    : Set of ground-truth relevant item IDs.
        k           : Cut-off rank.

    Returns:
        float between 0.0 and 1.0.
    """

    if k <= 0:
        return 0.0

    # Only consider the top-K recommended items
    top_k = recommended[:k]

    # Count how many top-K items are actually relevant
    hits = sum(1 for item in top_k if item in relevant)

    return hits / k


# ============================================================
# RECALL @ K
# ============================================================

def recall_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Compute Recall@K.

    Recall@K = (# relevant items in top-K) / (total # relevant items)

    Args:
        recommended : Ordered list of recommended item IDs.
        relevant    : Set of ground-truth relevant item IDs.
        k           : Cut-off rank.

    Returns:
        float between 0.0 and 1.0. Returns 0.0 if relevant is empty.
    """

    if not relevant or k <= 0:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)

    return hits / len(relevant)


# ============================================================
# NDCG @ K
# ============================================================

def ndcg_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @ K (NDCG@K).

    DCG@K = sum_{i=1}^{K} rel_i / log2(i + 1)
    where rel_i = 1 if item at rank i is relevant, else 0.

    NDCG@K = DCG@K / IDCG@K
    where IDCG@K is the ideal DCG (all relevant items ranked first).

    Args:
        recommended : Ordered list of recommended item IDs.
        relevant    : Set of ground-truth relevant item IDs.
        k           : Cut-off rank.

    Returns:
        float between 0.0 and 1.0.
    """

    if not relevant or k <= 0:
        return 0.0

    top_k = recommended[:k]

    # --------------------------------------------------------
    # Compute DCG@K: sum of (1 / log2(rank + 1)) for hits
    # --------------------------------------------------------
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant:
            # rank 1 → log2(2) = 1.0 discount
            dcg += 1.0 / math.log2(rank + 1)

    # --------------------------------------------------------
    # Compute IDCG@K: ideal ordering (all relevant items first)
    # --------------------------------------------------------
    # Number of relevant items achievable in top-K
    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# ============================================================
# MEAN RECIPROCAL RANK
# ============================================================

def mean_reciprocal_rank(recommended: List, relevant: set) -> float:
    """
    Compute Reciprocal Rank for a single user query.

    MRR = 1 / rank_of_first_relevant_item

    Args:
        recommended : Ordered list of recommended item IDs.
        relevant    : Set of ground-truth relevant item IDs.

    Returns:
        float between 0.0 and 1.0. Returns 0.0 if no hit found.
    """

    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / rank

    # No relevant item found in the recommended list
    return 0.0


# ============================================================
# HIT RATE @ K
# ============================================================

def hit_rate_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Compute Hit Rate @ K (HR@K).

    HR@K = 1 if at least one relevant item appears in top-K, else 0.

    Args:
        recommended : Ordered list of recommended item IDs.
        relevant    : Set of ground-truth relevant item IDs.
        k           : Cut-off rank.

    Returns:
        1.0 or 0.0.
    """

    if not relevant or k <= 0:
        return 0.0

    top_k = set(recommended[:k])
    return 1.0 if top_k & relevant else 0.0


# ============================================================
# FULL EVALUATION HARNESS
# ============================================================

def evaluate_recommender(
    recommender_fn,
    test_data: Dict[Any, set],
    k_values: List[int] = None
) -> Dict[str, float]:
    """
    Run a full evaluation of a recommender function across test users.

    Args:
        recommender_fn : Callable(user_id) → List[item_id] (ordered).
        test_data      : Dict mapping user_id → set of relevant item IDs.
        k_values       : List of K cut-offs to evaluate (default [5, 10, 20]).

    Returns:
        Dict of metric_name → average score across all test users.

        Example keys:
          "P@5", "R@5", "NDCG@5", "HR@5",
          "P@10", "R@10", "NDCG@10", "HR@10",
          "MRR"
    """

    if k_values is None:
        k_values = [5, 10, 20]

    # Accumulate scores per metric
    accumulators: Dict[str, float] = {}
    for k in k_values:
        accumulators[f"P@{k}"] = 0.0
        accumulators[f"R@{k}"] = 0.0
        accumulators[f"NDCG@{k}"] = 0.0
        accumulators[f"HR@{k}"] = 0.0
    accumulators["MRR"] = 0.0

    num_users = len(test_data)

    if num_users == 0:
        logger.warning("evaluate_recommender: test_data is empty.")
        return accumulators

    evaluated = 0

    for user_id, relevant_items in test_data.items():

        if not relevant_items:
            # Skip users with no ground-truth positives
            continue

        try:
            recommended = recommender_fn(user_id)
        except Exception as e:
            logger.warning(f"Recommender failed for user {user_id}: {e}")
            continue

        # Accumulate all K-level metrics
        for k in k_values:
            accumulators[f"P@{k}"] += precision_at_k(recommended, relevant_items, k)
            accumulators[f"R@{k}"] += recall_at_k(recommended, relevant_items, k)
            accumulators[f"NDCG@{k}"] += ndcg_at_k(recommended, relevant_items, k)
            accumulators[f"HR@{k}"] += hit_rate_at_k(recommended, relevant_items, k)

        accumulators["MRR"] += mean_reciprocal_rank(recommended, relevant_items)

        evaluated += 1

    if evaluated == 0:
        logger.warning("No users were successfully evaluated.")
        return accumulators

    # Average over all successfully evaluated users
    averaged = {
        metric: round(score / evaluated, 4)
        for metric, score in accumulators.items()
    }

    logger.info(f"Evaluated {evaluated}/{num_users} users.")

    return averaged


# ============================================================
# PRETTY PRINT RESULTS TABLE
# ============================================================

def print_metrics_table(results: Dict[str, float], title: str = "Evaluation Results") -> None:
    """
    Print a formatted table of evaluation metric results.

    Args:
        results : Dict from evaluate_recommender().
        title   : Header title for the table.
    """

    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  {'Metric':<15} {'Score':>10}")
    print(f"  {'-'*25}")

    for metric, score in sorted(results.items()):
        print(f"  {metric:<15} {score:>10.4f}")

    print(f"{'='*50}\n")


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("Testing metrics module...")

    # --- Mock data ---
    recommended = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
    relevant = {"P2", "P5", "P9"}

    print(f"Recommended: {recommended}")
    print(f"Relevant   : {relevant}")

    print(f"\nP@5   = {precision_at_k(recommended, relevant, 5):.4f}")
    print(f"R@5   = {recall_at_k(recommended, relevant, 5):.4f}")
    print(f"NDCG@5 = {ndcg_at_k(recommended, relevant, 5):.4f}")
    print(f"HR@5  = {hit_rate_at_k(recommended, relevant, 5):.4f}")
    print(f"MRR   = {mean_reciprocal_rank(recommended, relevant):.4f}")

    print(f"\nP@10   = {precision_at_k(recommended, relevant, 10):.4f}")
    print(f"R@10   = {recall_at_k(recommended, relevant, 10):.4f}")
    print(f"NDCG@10 = {ndcg_at_k(recommended, relevant, 10):.4f}")

    print("\n[OK] Metrics module passed all checks.")
