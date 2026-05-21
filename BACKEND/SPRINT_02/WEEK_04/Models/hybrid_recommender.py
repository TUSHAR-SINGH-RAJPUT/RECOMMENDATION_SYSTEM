"""
============================================================
WEEK 04 — HYBRID RECOMMENDER (XGBoost-Powered)
============================================================

This is the main entry point for the hybrid recommendation
engine. It integrates all signals from previous weeks:

  Week 02 → Collaborative Filtering (user-item matrix)
  Week 03 → Content-Based (ChromaDB vector search)
  Week 04 → XGBoost ranker (re-ranks combined candidates)

Pipeline:
  1. Content-based candidates  ← search_products() [Week 03]
  2. CF-based candidates       ← recommend_cf()    [Week 02]
  3. Combine unique candidates (deduplication)
  4. Build feature vector per candidate (6 signals)
  5. XGBoost ranker re-ranks candidates
  6. Fallback to weighted scoring if model not trained yet

Data Sources (real, not simulated):
  - user_item_matrix.csv   (Week 02 ETL output)
  - product_embeddings.npy (Week 03 sentence-transformer output)
  - embedding_metadata.json (Week 03 product catalogue)
  - Furniture.csv           (Week 03 raw product data)
  - xgb_ranker.pkl          (Week 04 trained model)

============================================================
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# PATH RESOLUTION
# ============================================================

# Absolute directory of this file: WEEK_04/Models/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Week 02 models dir (for collaborative_filter.py)
WEEK02_MODELS = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../../SPRINT_01/WEEK_02/models")
)

# Week 03 src dir (for content_recommender.py)
WEEK03_SRC = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../WEEK_03/src")
)

# Week 03 embeddings dir
WEEK03_EMBEDDINGS = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../WEEK_03/embeddings")
)

# Week 02 data dir
WEEK02_DATA = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../../SPRINT_01/WEEK_02/data")
)

# Add both to sys.path so imports resolve
sys.path.insert(0, WEEK02_MODELS)
sys.path.insert(0, WEEK03_SRC)


# ============================================================
# IMPORT PREVIOUS WEEK MODULES
# ============================================================

# --- Week 03: Content-Based Recommender ---
try:
    # pyrefly: ignore [missing-import]
    from content_recommender import search_products
    CONTENT_RECOMMENDER_AVAILABLE = True
    logger.info("✅ Content recommender loaded (Week 03)")
except Exception as e:
    logger.warning(f"⚠ Content recommender unavailable: {e}")
    CONTENT_RECOMMENDER_AVAILABLE = False

    def search_products(query, top_k=5, category=None, threshold=0.35):
        """Fallback stub when Week 03 ChromaDB is unavailable."""
        return []


# --- Week 02: Collaborative Filter ---
try:
    # pyrefly: ignore [missing-import]
    from collaborative_filter import recommend_cf
    CF_RECOMMENDER_AVAILABLE = True
    logger.info("✅ Collaborative filter loaded (Week 02)")
except Exception as e:
    logger.warning(f"⚠ Collaborative filter unavailable: {e}")
    CF_RECOMMENDER_AVAILABLE = False

    def recommend_cf(user_id, n=10):
        """Fallback stub when Week 02 CF model is unavailable."""
        return []


# ============================================================
# IMPORT WEEK 04 ITR MODEL (XGBoost)
# ============================================================

try:
    from itr_model import (
        load_model as load_xgb_model,
        predict_ranking,
        load_user_item_matrix,
        load_product_metadata,
        load_product_embeddings,
        load_furniture_csv,
        FEATURE_COLS
    )
    XGB_MODEL_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠ itr_model import failed: {e}")
    XGB_MODEL_AVAILABLE = False


# ============================================================
# GLOBAL DATA CACHE
# ============================================================
# Loaded once at module level — shared across all function calls
# to avoid re-reading large files on every recommendation request.

_cache: Dict[str, Any] = {}


def _load_data_cache() -> bool:
    """
    Load all required data into the module-level cache.

    Returns:
        True if data loaded successfully, False on failure.
    """

    global _cache

    if _cache.get("loaded"):
        return True

    try:
        _cache["uim"] = load_user_item_matrix()
        _cache["meta"] = load_product_metadata()
        _cache["embeddings"] = load_product_embeddings()
        _cache["furn"] = load_furniture_csv()
        _cache["loaded"] = True
        logger.info("✅ Data cache loaded.")
        return True

    except Exception as e:
        logger.warning(f"⚠ Could not load data cache: {e}")
        _cache["loaded"] = False
        return False


def _load_xgb_cache() -> bool:
    """
    Load the trained XGBoost model into cache.

    Returns:
        True if model loaded successfully, False on failure.
    """

    global _cache

    if _cache.get("xgb_model") is not None:
        return True

    if not XGB_MODEL_AVAILABLE:
        return False

    try:
        model, feature_cols = load_xgb_model()
        _cache["xgb_model"] = model
        _cache["feature_cols"] = feature_cols
        logger.info("✅ XGBoost model loaded from checkpoint.")
        return True

    except FileNotFoundError:
        logger.info(
            "ℹ XGBoost model not yet trained. "
            "Will use weighted-score fallback. "
            "Run Models/training_pipeline.py to train."
        )
        return False

    except Exception as e:
        logger.warning(f"⚠ Could not load XGBoost model: {e}")
        return False


# ============================================================
# COSINE SIMILARITY UTILITY
# ============================================================

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1, vec2 : numpy arrays of the same dimension.

    Returns:
        Float in [-1, 1]. Returns 0.0 if either vector has zero norm.
    """

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# ============================================================
# USER PROFILE BUILDER
# ============================================================

def _build_user_profile(user_id) -> Dict[str, Any]:
    """
    Build a user preference profile from their interaction history.

    Uses the real user-item matrix from Week 02 to determine:
      - purchased_items  : set of item IDs the user interacted with
      - avg_price        : average price of interacted items
      - preferred_styles : set of furniture styles
      - preferred_categories: set of furniture categories
      - embedding_vector : mean embedding of their items

    Args:
        user_id : User ID (must match index in user_item_matrix).

    Returns:
        Dict with profile fields, or fallback defaults.
    """

    profile = {
        "purchased_items": set(),
        "avg_price": 15000.0,     # market-wide fallback
        "preferred_styles": set(),
        "preferred_categories": set(),
        "embedding_vector": None
    }

    if not _cache.get("loaded"):
        return profile

    uim = _cache["uim"]
    furn = _cache["furn"]
    embeddings = _cache["embeddings"]
    meta = _cache["meta"]

    # User exists in interaction matrix?
    if user_id not in uim.index:
        logger.debug(f"User {user_id} not found in matrix — using defaults.")
        return profile

    user_row = uim.loc[user_id]
    interacted = user_row[user_row > 0].index.astype(str)
    profile["purchased_items"] = set(interacted)

    if not profile["purchased_items"]:
        return profile

    # --- Price and style from furniture catalogue ---
    pos_furn = furn[furn["product_id"].isin(profile["purchased_items"])]

    if not pos_furn.empty:
        profile["avg_price"] = float(pos_furn["price"].mean())

        if "style" in pos_furn.columns:
            profile["preferred_styles"] = set(pos_furn["style"].dropna().tolist())

        if "category" in pos_furn.columns:
            profile["preferred_categories"] = set(pos_furn["category"].dropna().tolist())

    # --- Embedding profile (mean of interacted item embeddings) ---
    pid_to_idx = {
        pid: idx for idx, pid in enumerate(meta["product_id"])
    }

    embed_vecs = [
        embeddings[pid_to_idx[pid]]
        for pid in profile["purchased_items"]
        if pid in pid_to_idx and pid_to_idx[pid] < len(embeddings)
    ]

    if embed_vecs:
        profile["embedding_vector"] = np.mean(embed_vecs, axis=0)

    return profile


# ============================================================
# FEATURE ENGINEERING FOR SCORING
# ============================================================

def _compute_features(
    product_id: str,
    cf_recs: List,
    content_recs: List[Dict],
    user_profile: Dict
) -> Dict[str, float]:
    """
    Compute the 6 feature scores for one candidate product.

    Features:
      cf_score           : 1 if item appears in CF recommendations
      content_score      : relevance score from ChromaDB (0–1)
      embedding_similarity: cosine similarity between user and item embeddings
      price_score        : 1 − normalised price distance from user preference
      style_match        : 1 if item style matches user's preferred styles
      item_popularity    : normalised total interaction count for this item

    Args:
        product_id    : String product ID (e.g. "P123").
        cf_recs       : List of product IDs from collaborative filter.
        content_recs  : List of dicts from search_products().
        user_profile  : Dict from _build_user_profile().

    Returns:
        Dict of feature_name → float score.
    """

    furn = _cache.get("furn", pd.DataFrame())
    uim = _cache.get("uim", pd.DataFrame())
    embeddings = _cache.get("embeddings", np.array([]))
    meta = _cache.get("meta", pd.DataFrame())

    # ---- CF score ----
    cf_score = 1.0 if str(product_id) in [str(r) for r in cf_recs] else 0.0

    # ---- Content score ----
    # Use the similarity score from ChromaDB if available
    content_score = 0.0
    for rec in content_recs:
        rec_pid = rec.get("product_id", "")
        if str(rec_pid) == str(product_id):
            content_score = float(rec.get("score", 0.0))
            break
    # Fallback: binary presence if no score available
    if content_score == 0.0:
        content_names = [r.get("product_name", "") for r in content_recs]
        if str(product_id) in content_names:
            content_score = 0.5

    # ---- Embedding similarity ----
    embedding_similarity = 0.0
    user_vec = user_profile.get("embedding_vector")
    if user_vec is not None and not meta.empty:
        pid_to_idx = {
            pid: idx for idx, pid in enumerate(meta["product_id"])
        }
        if str(product_id) in pid_to_idx:
            item_idx = pid_to_idx[str(product_id)]
            if item_idx < len(embeddings):
                embedding_similarity = _cosine_similarity(
                    user_vec, embeddings[item_idx]
                )

    # ---- Price score ----
    price_score = 0.5  # neutral default
    if not furn.empty:
        item_row = furn[furn["product_id"] == str(product_id)]
        if not item_row.empty:
            item_price = float(item_row.iloc[0]["price"])
            avg_price = user_profile.get("avg_price", 15000.0)
            max_price = float(furn["price"].max())
            if max_price > 0:
                price_score = float(
                    1.0 - abs(item_price - avg_price) / max_price
                )
                price_score = max(0.0, min(1.0, price_score))

    # ---- Style match ----
    style_score = 0.0
    if not furn.empty:
        item_row = furn[furn["product_id"] == str(product_id)]
        if not item_row.empty and "style" in item_row.columns:
            item_style = item_row.iloc[0].get("style", "")
            if item_style in user_profile.get("preferred_styles", set()):
                style_score = 1.0

    # ---- Item popularity ----
    item_popularity = 0.0
    if not uim.empty and str(product_id) in uim.columns:
        max_pop = float(uim.sum(axis=0).max())
        if max_pop > 0:
            item_popularity = float(uim[str(product_id)].sum()) / max_pop

    return {
        "cf_score": cf_score,
        "content_score": content_score,
        "embedding_similarity": embedding_similarity,
        "price_score": price_score,
        "style_match": style_score,
        "item_popularity": item_popularity
    }


# ============================================================
# WEIGHTED SCORE FALLBACK
# ============================================================
#
# Used when the XGBoost model is not yet trained.
# Weights tuned to match the spirit of the hybrid approach.
#
FALLBACK_WEIGHTS = {
    "cf_score":            0.30,
    "content_score":       0.25,
    "embedding_similarity": 0.20,
    "price_score":         0.10,
    "style_match":         0.10,
    "item_popularity":     0.05
}


def _weighted_score(features: Dict[str, float]) -> float:
    """
    Compute a simple weighted linear combination of features.

    Used as fallback when XGBoost model is not available.

    Args:
        features : Dict from _compute_features().

    Returns:
        Float score.
    """

    return sum(
        FALLBACK_WEIGHTS.get(feat, 0.0) * value
        for feat, value in features.items()
    )


# ============================================================
# MAIN HYBRID RECOMMENDER
# ============================================================

def hybrid_recommender(
    user_id,
    query: str,
    top_k: int = 10,
    category: Optional[str] = None,
    use_xgboost: bool = True
) -> List[Dict[str, Any]]:
    """
    Hybrid recommender combining CF + Content + XGBoost re-ranking.

    Pipeline:
      1. Get content-based candidates from Week 03 (ChromaDB)
      2. Get CF candidates from Week 02 (user-user similarity)
      3. Combine and deduplicate candidates
      4. Build feature vectors for each candidate
      5. Re-rank using XGBoost (or fallback weighted scoring)
      6. Return top_k results with full score breakdown

    Args:
        user_id     : User ID (int or str, must match interaction matrix).
        query       : Free-text product query (e.g. "modern sofa").
        top_k       : Number of final recommendations to return.
        category    : Optional product category filter for content search.
        use_xgboost : Whether to use XGBoost ranker (True by default).
                      If False, or if model not trained, falls back to
                      weighted scoring.

    Returns:
        List of dicts, each with:
          product_id, product_name, category, description (if available),
          cf_score, content_score, embedding_similarity,
          price_score, style_match, item_popularity, final_score.
        Sorted by final_score descending.
    """

    logger.info("\n" + "="*60)
    logger.info("HYBRID RECOMMENDER — START")
    logger.info(f"  user_id  : {user_id}")
    logger.info(f"  query    : '{query}'")
    logger.info(f"  top_k    : {top_k}")
    logger.info(f"  category : {category}")
    logger.info("="*60)

    # --------------------------------------------------------
    # Ensure data cache is loaded
    # --------------------------------------------------------
    _load_data_cache()

    # --------------------------------------------------------
    # STEP 1: Content-based candidates (Week 03)
    # --------------------------------------------------------
    logger.info("\n[1] Content-based search...")

    content_recs = []
    if CONTENT_RECOMMENDER_AVAILABLE:
        try:
            content_recs = search_products(
                query=query,
                top_k=top_k * 2,        # fetch more than needed for merging
                category=category
            )
            logger.info(f"    → {len(content_recs)} content candidates")
        except Exception as e:
            logger.warning(f"    ⚠ Content search failed: {e}")
            content_recs = []
    else:
        logger.info("    → Content recommender unavailable (skipped)")

    # Extract product IDs from content results
    content_pids = [str(r.get("product_id", "")) for r in content_recs]

    # --------------------------------------------------------
    # STEP 2: Collaborative filter candidates (Week 02)
    # --------------------------------------------------------
    logger.info("\n[2] Collaborative filtering...")

    cf_recs = []
    if CF_RECOMMENDER_AVAILABLE:
        try:
            cf_recs = recommend_cf(user_id=user_id, n=top_k * 2)
            cf_recs = [str(r) for r in cf_recs]
            logger.info(f"    → {len(cf_recs)} CF candidates")
        except Exception as e:
            logger.warning(f"    ⚠ CF failed: {e}")
            cf_recs = []
    else:
        logger.info("    → CF recommender unavailable (skipped)")

    # --------------------------------------------------------
    # STEP 3: Combine and deduplicate candidates
    # --------------------------------------------------------
    logger.info("\n[3] Merging candidates...")

    # Preserve order: content first, then CF-only items
    seen = set()
    combined_pids = []

    for pid in content_pids + cf_recs:
        if pid and pid not in seen:
            seen.add(pid)
            combined_pids.append(pid)

    logger.info(f"    → {len(combined_pids)} unique candidates after merge")

    if not combined_pids:
        logger.warning("    ⚠ No candidates from any source. Returning empty list.")
        return []

    # --------------------------------------------------------
    # STEP 4: Build user profile
    # --------------------------------------------------------
    logger.info("\n[4] Building user profile...")

    user_profile = _build_user_profile(user_id)

    logger.info(
        f"    → {len(user_profile['purchased_items'])} past interactions | "
        f"avg_price={user_profile['avg_price']:.0f} | "
        f"styles={user_profile['preferred_styles']}"
    )

    # --------------------------------------------------------
    # STEP 5: Re-rank using XGBoost or weighted fallback
    # --------------------------------------------------------
    logger.info("\n[5] Ranking candidates...")

    xgb_available = use_xgboost and _load_xgb_cache()

    if xgb_available:
        # ---- XGBoost path ----
        logger.info("    → Using XGBoost ranker")

        try:
            model = _cache["xgb_model"]
            feature_cols = _cache.get("feature_cols", FEATURE_COLS)

            ranked_pids = predict_ranking(
                model=model,
                user_id=user_id,
                candidate_items=combined_pids,
                user_item_matrix=_cache["uim"],
                furniture_df=_cache["furn"],
                product_meta=_cache["meta"],
                product_embeddings=_cache["embeddings"],
                feature_cols=feature_cols
            )

        except Exception as e:
            logger.warning(f"    ⚠ XGBoost ranking failed: {e}. Falling back.")
            xgb_available = False
            ranked_pids = combined_pids  # preserve original order as fallback

    if not xgb_available:
        # ---- Weighted score fallback ----
        logger.info("    → Using weighted-score fallback")

        scored = []
        for pid in combined_pids:
            features = _compute_features(
                product_id=pid,
                cf_recs=cf_recs,
                content_recs=content_recs,
                user_profile=user_profile
            )
            score = _weighted_score(features)
            scored.append((pid, score, features))

        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_pids = [pid for pid, _, _ in scored]

    # --------------------------------------------------------
    # STEP 6: Build final result list with score breakdowns
    # --------------------------------------------------------
    logger.info("\n[6] Building final result set...")

    final_results = []

    # Pre-build product name lookup from content_recs and metadata
    pid_to_content_info: Dict[str, Dict] = {
        str(r.get("product_id", "")): r for r in content_recs
    }

    furn = _cache.get("furn", pd.DataFrame())
    meta = _cache.get("meta", pd.DataFrame())

    for pid in ranked_pids[:top_k]:

        # ---- Compute features for score breakdown ----
        features = _compute_features(
            product_id=pid,
            cf_recs=cf_recs,
            content_recs=content_recs,
            user_profile=user_profile
        )

        # ---- Compute final score ----
        if xgb_available:
            # XGBoost score is implicit from ranking position;
            # compute weighted score for display purposes
            final_score = _weighted_score(features)
        else:
            final_score = _weighted_score(features)

        # ---- Look up product info ----
        product_name = pid        # fallback = product_id
        category_name = "N/A"
        description = ""

        # Try content_recs first (has product_name and category)
        if pid in pid_to_content_info:
            info = pid_to_content_info[pid]
            product_name = info.get("product_name", pid)
            category_name = info.get("category", "N/A")
            description = info.get("description", "")

        # Try furniture catalogue
        elif not furn.empty:
            row = furn[furn["product_id"] == str(pid)]
            if not row.empty:
                r = row.iloc[0]
                product_name = r.get("product_name", pid) if "product_name" in r else pid
                category_name = r.get("category", "N/A")

        # Try metadata JSON
        if product_name == pid and not meta.empty:
            m_row = meta[meta["product_id"] == str(pid)]
            if not m_row.empty:
                product_name = m_row.iloc[0].get("product_name", pid)
                category_name = m_row.iloc[0].get("category", "N/A")
                description = m_row.iloc[0].get("description", "")

        final_results.append({
            "product_id": pid,
            "product_name": product_name,
            "category": category_name,
            "description": description,
            "cf_score":             round(features["cf_score"], 4),
            "content_score":        round(features["content_score"], 4),
            "embedding_similarity": round(features["embedding_similarity"], 4),
            "price_score":          round(features["price_score"], 4),
            "style_match":          round(features["style_match"], 4),
            "item_popularity":      round(features["item_popularity"], 4),
            "final_score":          round(final_score, 4),
            "ranked_by":            "XGBoost" if xgb_available else "WeightedScore"
        })

    logger.info(f"\n✅ Returning {len(final_results)} recommendations")

    return final_results


# ============================================================
# PRETTY PRINT HELPER
# ============================================================

def display_recommendations(
    recommendations: List[Dict],
    title: str = "HYBRID RECOMMENDATIONS"
) -> None:
    """
    Print a formatted display of recommendation results.

    Args:
        recommendations : Output from hybrid_recommender().
        title           : Section header.
    """

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    if not recommendations:
        print("  ❌ No recommendations found.")
        return

    ranked_by = recommendations[0].get("ranked_by", "WeightedScore")
    print(f"  Ranked by : {ranked_by}")
    print(f"  Count     : {len(recommendations)}")
    print()

    for idx, item in enumerate(recommendations, start=1):

        print(f"  {idx:>2}. {item['product_name']}")
        print(f"      Product ID    : {item['product_id']}")
        print(f"      Category      : {item['category']}")

        if item.get("description"):
            # Truncate long descriptions
            desc = item["description"][:80] + "..." if len(item["description"]) > 80 else item["description"]
            print(f"      Description   : {desc}")

        print(f"      ─ CF Score            : {item['cf_score']}")
        print(f"      ─ Content Score       : {item['content_score']}")
        print(f"      ─ Embedding Sim.      : {item['embedding_similarity']}")
        print(f"      ─ Price Score         : {item['price_score']}")
        print(f"      ─ Style Match         : {item['style_match']}")
        print(f"      ─ Item Popularity     : {item['item_popularity']}")
        print(f"      ★ Final Score         : {item['final_score']}")
        print()

    print(f"{'='*60}\n")


# ============================================================
# STANDALONE TEST / DEMO
# ============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  HYBRID RECOMMENDER — STANDALONE TEST")
    print("="*60)

    # Try loading data cache first (needed for real-data mode)
    data_ok = _load_data_cache()

    if data_ok:
        # Use first real user ID from the interaction matrix
        uim = _cache.get("uim")
        test_user_id = uim.index[0] if uim is not None else 1
    else:
        # Fallback to a mock user ID
        test_user_id = 1

    print(f"\nTest user ID : {test_user_id}")

    # --- Run hybrid recommender ---
    recs = hybrid_recommender(
        user_id=test_user_id,
        query="modern wooden sofa",
        top_k=10,
        use_xgboost=True
    )

    display_recommendations(recs, title="Test Recommendations")

    # --- Also test with category filter ---
    print("\n--- Category filter test: 'chair' ---")
    recs_cat = hybrid_recommender(
        user_id=test_user_id,
        query="comfortable chair",
        top_k=5,
        category="chair",
        use_xgboost=True
    )
    display_recommendations(recs_cat, title="Chair Recommendations")
