"""
============================================================
WEEK 04 — XGBoost ITR (Item-To-Ranking) MODEL
============================================================

This module implements the core XGBoost-based ranking model
that powers the hybrid recommender system.

Responsibilities:
  1. Feature engineering — builds a training DataFrame with
     per-user, per-item features derived from:
       - Week 02: user-item interaction matrix (CF signals)
       - Week 03: product embeddings + metadata (content signals)
       - Derived: price sensitivity, style match, CTR proxy

  2. Training — trains an XGBoost ranker on labelled data
     (items the user interacted with = positive labels).

  3. Save / Load — persists model checkpoints to
     Models/Model_checkpoints/ so they survive between runs.

  4. Inference — given a user_id and a list of candidate
     item IDs, returns items ranked by predicted relevance.

============================================================
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import List

# ============================================================
# XGBOOST IMPORT
# ============================================================

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠ xgboost not installed. Run: pip install xgboost")

# ============================================================
# LOGGER
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# PATH RESOLUTION
# ============================================================

# Absolute path to this file's directory  (WEEK_04/Models/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Week 02 data directory
# Week 02 lives in SPRINT_01, not SPRINT_02
WEEK02_DATA = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../../SPRINT_01/WEEK_02/data")
)

# Week 03 embeddings directory
WEEK03_EMBEDDINGS = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../WEEK_03/embeddings")
)

# Model checkpoint directory
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, "Model_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Checkpoint file paths
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "xgb_ranker.pkl")
FEATURE_META_PATH = os.path.join(CHECKPOINT_DIR, "feature_meta.json")


# ============================================================
# DATA LOADING UTILITIES
# ============================================================

def load_user_item_matrix() -> pd.DataFrame:
    """
    Load the user-item interaction matrix built by Week 02 ETL pipeline.

    Returns:
        DataFrame with users as index, product_ids as columns,
        interaction scores as values.
    """

    path = os.path.join(WEEK02_DATA, "user_item_matrix.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"user_item_matrix.csv not found at: {path}\n"
            "Please run WEEK_02/evaluation/matrix.py first."
        )

    df = pd.read_csv(path, index_col=0)
    logger.info(f"Loaded user-item matrix: {df.shape}")
    return df


def load_product_metadata() -> pd.DataFrame:
    """
    Load the product metadata JSON created by Week 03 embedder.

    Returns:
        DataFrame with columns: product_id, product_name, category, description.
    """

    path = os.path.join(WEEK03_EMBEDDINGS, "embedding_metadata.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"embedding_metadata.json not found at: {path}\n"
            "Please run WEEK_03/src/embedder.py first."
        )

    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logger.info(f"Loaded product metadata: {len(df)} products")
    return df


def load_product_embeddings() -> np.ndarray:
    """
    Load the 384-dim sentence transformer embeddings from Week 03.

    Returns:
        numpy array of shape (n_products, 384).
    """

    path = os.path.join(WEEK03_EMBEDDINGS, "product_embeddings.npy")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"product_embeddings.npy not found at: {path}\n"
            "Please run WEEK_03/src/embedder.py first."
        )

    embeddings = np.load(path)
    logger.info(f"Loaded product embeddings: {embeddings.shape}")
    return embeddings


def load_furniture_csv() -> pd.DataFrame:
    """
    Load the raw Furniture.csv for price / category / material features.

    Returns:
        Cleaned DataFrame with columns: price, category, material, color, style.
    """

    path = os.path.join(WEEK03_EMBEDDINGS, "Furniture.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Furniture.csv not found at: {path}")

    df = pd.read_csv(path)

    # Keep only the columns we need
    keep = [c for c in ["price", "category", "material", "color"] if c in df.columns]
    df = df[keep].copy()

    # Clean
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"].fillna(df["price"].median(), inplace=True)

    for col in ["category", "material", "color"]:
        if col in df:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Derive style from price quantiles (consistent with embedder.py)
    q1, q2 = df["price"].quantile([0.33, 0.66])

    def _style(price):
        if price >= q2:
            return "luxury"
        elif price >= q1:
            return "modern"
        return "minimalist"

    df["style"] = df["price"].apply(_style)

    # Synthetic product_id aligned with embedding index
    df["product_id"] = ["P" + str(i) for i in df.index]

    logger.info(f"Loaded Furniture.csv: {df.shape}")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_feature_matrix(
    user_item_matrix: pd.DataFrame,
    furniture_df: pd.DataFrame,
    product_meta: pd.DataFrame,
    product_embeddings: np.ndarray,
    max_users: int = 500
) -> pd.DataFrame:
    """
    Build the feature matrix for XGBoost training.

    For each (user, item) pair in the interaction matrix we compute:

      Feature               Description
      ─────────────────     ────────────────────────────────────────
      cf_score              CF interaction score (raw from matrix)
      content_score         1 if product appears in user's category pref
      embedding_similarity  Cosine similarity between user/item embeddings
      price_score           1 - norm(|item_price - user_avg_price|)
      style_match           1 if item style matches user's preferred style
      item_popularity       Column sum of item across all users (normalised)
      label                 1 if user actually interacted (positive), 0 otherwise

    Args:
        user_item_matrix   : DataFrame (users × items).
        furniture_df       : Product features from Furniture.csv.
        product_meta       : Metadata JSON as DataFrame.
        product_embeddings : numpy (n_products, dim).
        max_users          : Cap on number of training users (speed).

    Returns:
        DataFrame with feature columns + 'user_id' + 'item_id' + 'label'.
    """

    logger.info("Building feature matrix for XGBoost training...")

    rows = []

    # --------------------------------------------------------
    # Pre-compute item-level popularity scores (normalised)
    # --------------------------------------------------------
    item_popularity = user_item_matrix.sum(axis=0)  # Series: item → total score
    max_pop = item_popularity.max()
    if max_pop > 0:
        item_popularity = item_popularity / max_pop

    # --------------------------------------------------------
    # Pre-compute user embedding profiles (mean of item embeddings)
    # --------------------------------------------------------
    # Map product_meta product_id → embedding index (positional)
    pid_to_idx = {
        pid: idx for idx, pid in enumerate(product_meta["product_id"])
    }

    # --------------------------------------------------------
    # Iterate over a sample of users
    # --------------------------------------------------------
    users = user_item_matrix.index.tolist()
    if len(users) > max_users:
        import random
        random.seed(42)
        users = random.sample(users, max_users)

    for user_id in users:

        user_row = user_item_matrix.loc[user_id]

        # Positive items (actually interacted)
        positive_items = set(user_row[user_row > 0].index.astype(str))

        if not positive_items:
            continue

        # --------------------------------------------------------
        # Build user embedding profile from their positive items
        # --------------------------------------------------------
        user_embed_vecs = []
        for item_id in positive_items:
            if item_id in pid_to_idx:
                idx = pid_to_idx[item_id]
                if idx < len(product_embeddings):
                    user_embed_vecs.append(product_embeddings[idx])

        if user_embed_vecs:
            user_profile_vec = np.mean(user_embed_vecs, axis=0)
        else:
            user_profile_vec = np.zeros(product_embeddings.shape[1])

        # --------------------------------------------------------
        # User average price (from furniture_df matching positive items)
        # --------------------------------------------------------
        # furniture_df product_id aligns with product_meta product_id
        pos_meta = furniture_df[furniture_df["product_id"].isin(positive_items)]
        avg_price = pos_meta["price"].mean() if not pos_meta.empty else furniture_df["price"].mean()

        # User preferred styles (most common in positive items)
        if not pos_meta.empty and "style" in pos_meta.columns:
            user_styles = set(pos_meta["style"].tolist())
        else:
            user_styles = set()

        # User preferred categories
        if not pos_meta.empty and "category" in pos_meta.columns:
            user_categories = set(pos_meta["category"].tolist())
        else:
            user_categories = set()

        # --------------------------------------------------------
        # Sample negative items (items the user did NOT interact with)
        # --------------------------------------------------------
        all_items = user_item_matrix.columns.astype(str).tolist()
        negative_items = [i for i in all_items if i not in positive_items]

        # Balance: sample negatives equal to 3× positives (max 30)
        n_neg = min(len(negative_items), max(3 * len(positive_items), 30))
        import random as _rand
        _rand.seed(42)
        sampled_negatives = _rand.sample(negative_items, n_neg)

        # --------------------------------------------------------
        # Build feature rows for all (user, item) pairs
        # --------------------------------------------------------
        all_candidate_items = list(positive_items) + sampled_negatives

        for item_id in all_candidate_items:

            label = 1 if item_id in positive_items else 0

            # --- CF score ---
            # Raw interaction score from the matrix (0 if not interacted)
            if item_id in user_item_matrix.columns:
                cf_raw = float(user_row.get(item_id, 0))
            else:
                cf_raw = 0.0

            # --- Content score ---
            # 1 if item is in a category the user has shown preference for
            item_meta_row = furniture_df[furniture_df["product_id"] == item_id]
            if not item_meta_row.empty:
                item_category = item_meta_row.iloc[0].get("category", "")
                item_price = float(item_meta_row.iloc[0].get("price", avg_price))
                item_style = item_meta_row.iloc[0].get("style", "")
            else:
                item_category = ""
                item_price = avg_price
                item_style = ""

            content_score = 1.0 if item_category in user_categories else 0.0

            # --- Embedding similarity ---
            if item_id in pid_to_idx:
                item_idx = pid_to_idx[item_id]
                if item_idx < len(product_embeddings):
                    item_vec = product_embeddings[item_idx]
                    norm_u = np.linalg.norm(user_profile_vec)
                    norm_i = np.linalg.norm(item_vec)
                    if norm_u > 0 and norm_i > 0:
                        embed_sim = float(np.dot(user_profile_vec, item_vec) / (norm_u * norm_i))
                    else:
                        embed_sim = 0.0
                else:
                    embed_sim = 0.0
            else:
                embed_sim = 0.0

            # --- Price score ---
            max_price = furniture_df["price"].max()
            price_diff = abs(item_price - avg_price)
            price_score = float(1.0 - (price_diff / max_price)) if max_price > 0 else 0.0
            price_score = max(0.0, min(1.0, price_score))

            # --- Style match ---
            style_score = 1.0 if item_style in user_styles else 0.0

            # --- Item popularity ---
            pop_score = float(item_popularity.get(item_id, 0.0))

            rows.append({
                "user_id": user_id,
                "item_id": item_id,
                "cf_score": cf_raw,
                "content_score": content_score,
                "embedding_similarity": embed_sim,
                "price_score": price_score,
                "style_match": style_score,
                "item_popularity": pop_score,
                "label": label
            })

    feature_df = pd.DataFrame(rows)
    logger.info(
        f"Feature matrix built: {len(feature_df)} rows "
        f"({feature_df['label'].sum()} positives, "
        f"{(feature_df['label'] == 0).sum()} negatives)"
    )
    return feature_df


# ============================================================
# FEATURE COLUMNS (CONSTANT)
# ============================================================

FEATURE_COLS = [
    "cf_score",
    "content_score",
    "embedding_similarity",
    "price_score",
    "style_match",
    "item_popularity"
]


# ============================================================
# XGBOOST TRAINING
# ============================================================

def train_xgb_ranker(
    feature_df: pd.DataFrame,
    params: dict = None
) -> "xgb.XGBRanker":
    """
    Train an XGBoost pairwise ranker on the feature matrix.

    The XGBRanker with 'rank:pairwise' objective learns to rank
    positive items above negative items for each user query group.

    Args:
        feature_df : DataFrame from build_feature_matrix().
        params     : Optional dict of XGBoost hyperparameters.
                     If None, defaults are used.

    Returns:
        Trained XGBRanker model.
    """

    if not XGB_AVAILABLE:
        raise ImportError("xgboost is required. Install with: pip install xgboost")

    # Default hyperparameters (can be overridden by Optuna best_params)
    default_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@10",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0
    }

    if params:
        # Merge user-supplied params (Optuna best) into defaults
        default_params.update(params)
        logger.info(f"Using custom XGBoost params: {params}")
    else:
        logger.info("Using default XGBoost params.")

    # --------------------------------------------------------
    # Prepare training data
    # --------------------------------------------------------
    X = feature_df[FEATURE_COLS].values
    y = feature_df["label"].values

    # XGBRanker requires group sizes:
    # each 'group' = all items associated with one user
    group_sizes = (
        feature_df.groupby("user_id").size().values
    )

    logger.info(f"Training XGBRanker with {len(feature_df)} samples, "
                f"{len(group_sizes)} user groups...")

    # --------------------------------------------------------
    # Build and train the model
    # --------------------------------------------------------
    model = xgb.XGBRanker(
        n_estimators=default_params.pop("n_estimators", 100),
        max_depth=default_params.pop("max_depth", 4),
        learning_rate=default_params.pop("learning_rate", 0.1),
        subsample=default_params.pop("subsample", 0.8),
        colsample_bytree=default_params.pop("colsample_bytree", 0.8),
        min_child_weight=default_params.pop("min_child_weight", 5),
        reg_alpha=default_params.pop("reg_alpha", 0.1),
        reg_lambda=default_params.pop("reg_lambda", 1.0),
        objective=default_params.pop("objective", "rank:pairwise"),
        eval_metric=default_params.pop("eval_metric", "ndcg@10"),
        random_state=default_params.pop("random_state", 42),
        n_jobs=default_params.pop("n_jobs", -1),
        verbosity=default_params.pop("verbosity", 0),
        **default_params
    )

    model.fit(X, y, group=group_sizes)

    logger.info("✅ XGBRanker training complete.")
    return model


# ============================================================
# MODEL SAVE / LOAD
# ============================================================

def save_model(model: "xgb.XGBRanker", feature_cols: List = None) -> None:
    """
    Persist the trained XGBRanker to disk.

    Saves:
      - xgb_ranker.pkl          — pickled model
      - feature_meta.json       — feature column names + version info

    Args:
        model        : Trained XGBRanker instance.
        feature_cols : List of feature column names used during training.
    """

    if feature_cols is None:
        feature_cols = FEATURE_COLS

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Pickle the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save feature metadata
    meta = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "model_type": "XGBRanker",
        "objective": "rank:pairwise"
    }
    with open(FEATURE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"✅ Model saved → {MODEL_PATH}")
    logger.info(f"✅ Feature meta saved → {FEATURE_META_PATH}")


def load_model() -> tuple:
    """
    Load the persisted XGBRanker from disk.

    Returns:
        Tuple of (model, feature_cols).

    Raises:
        FileNotFoundError if checkpoint does not exist.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model checkpoint found at: {MODEL_PATH}\n"
            "Please run training_pipeline.py first."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    feature_cols = FEATURE_COLS  # default
    if os.path.exists(FEATURE_META_PATH):
        with open(FEATURE_META_PATH, "r") as f:
            meta = json.load(f)
        feature_cols = meta.get("feature_cols", FEATURE_COLS)

    logger.info(f"✅ Model loaded from {MODEL_PATH}")
    return model, feature_cols


# ============================================================
# INFERENCE — RANK CANDIDATES FOR A USER
# ============================================================

def predict_ranking(
    model: "xgb.XGBRanker",
    user_id,
    candidate_items: list,
    user_item_matrix: pd.DataFrame,
    furniture_df: pd.DataFrame,
    product_meta: pd.DataFrame,
    product_embeddings: np.ndarray,
    feature_cols: list = None
) -> list:
    """
    Rank a list of candidate items for a given user using the XGBRanker.

    Args:
        model             : Loaded XGBRanker.
        user_id           : Target user ID (must exist in user_item_matrix).
        candidate_items   : List of item IDs to rank.
        user_item_matrix  : Full interaction matrix (for CF signals).
        furniture_df      : Product features DataFrame.
        product_meta      : Product metadata DataFrame.
        product_embeddings: numpy embeddings array.
        feature_cols      : Feature column order (must match training).

    Returns:
        List of candidate item IDs sorted by predicted relevance score
        (most relevant first).
    """

    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if not candidate_items:
        return []

    # --------------------------------------------------------
    # Build feature rows for each candidate
    # --------------------------------------------------------
    pid_to_idx = {
        pid: idx for idx, pid in enumerate(product_meta["product_id"])
    }

    # User interaction row
    if user_id in user_item_matrix.index:
        user_row = user_item_matrix.loc[user_id]
        positive_items = set(user_row[user_row > 0].index.astype(str))
    else:
        user_row = None
        positive_items = set()

    # User embedding profile
    user_embed_vecs = []
    for item_id in positive_items:
        if item_id in pid_to_idx:
            idx = pid_to_idx[item_id]
            if idx < len(product_embeddings):
                user_embed_vecs.append(product_embeddings[idx])

    user_profile_vec = (
        np.mean(user_embed_vecs, axis=0)
        if user_embed_vecs
        else np.zeros(product_embeddings.shape[1])
    )

    # User average price & styles
    pos_meta = furniture_df[furniture_df["product_id"].isin(positive_items)]
    avg_price = pos_meta["price"].mean() if not pos_meta.empty else furniture_df["price"].mean()
    user_styles = set(pos_meta["style"].tolist()) if not pos_meta.empty else set()
    user_categories = set(pos_meta["category"].tolist()) if not pos_meta.empty else set()

    item_popularity = user_item_matrix.sum(axis=0)
    max_pop = item_popularity.max()
    if max_pop > 0:
        item_popularity = item_popularity / max_pop

    max_price = furniture_df["price"].max()

    rows = []
    for item_id in candidate_items:
        # CF score
        cf_raw = float(user_row.get(item_id, 0)) if user_row is not None else 0.0

        # Content score
        item_meta_row = furniture_df[furniture_df["product_id"] == str(item_id)]
        if not item_meta_row.empty:
            item_category = item_meta_row.iloc[0].get("category", "")
            item_price = float(item_meta_row.iloc[0].get("price", avg_price))
            item_style = item_meta_row.iloc[0].get("style", "")
        else:
            item_category = ""
            item_price = avg_price
            item_style = ""

        content_score = 1.0 if item_category in user_categories else 0.0

        # Embedding similarity
        if str(item_id) in pid_to_idx:
            item_idx = pid_to_idx[str(item_id)]
            if item_idx < len(product_embeddings):
                item_vec = product_embeddings[item_idx]
                norm_u = np.linalg.norm(user_profile_vec)
                norm_i = np.linalg.norm(item_vec)
                embed_sim = float(
                    np.dot(user_profile_vec, item_vec) / (norm_u * norm_i)
                ) if norm_u > 0 and norm_i > 0 else 0.0
            else:
                embed_sim = 0.0
        else:
            embed_sim = 0.0

        # Price score
        price_diff = abs(item_price - avg_price)
        price_score = float(1.0 - price_diff / max_price) if max_price > 0 else 0.0
        price_score = max(0.0, min(1.0, price_score))

        # Style match
        style_score = 1.0 if item_style in user_styles else 0.0

        # Item popularity
        pop_score = float(item_popularity.get(str(item_id), 0.0))

        rows.append({
            "item_id": item_id,
            "cf_score": cf_raw,
            "content_score": content_score,
            "embedding_similarity": embed_sim,
            "price_score": price_score,
            "style_match": style_score,
            "item_popularity": pop_score
        })

    df_candidates = pd.DataFrame(rows)
    X_pred = df_candidates[feature_cols].values

    # Get predicted scores
    scores = model.predict(X_pred)

    # Sort candidates by score (descending)
    df_candidates["pred_score"] = scores
    df_candidates = df_candidates.sort_values("pred_score", ascending=False)

    return df_candidates["item_id"].tolist()


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    logger.info("Running itr_model.py standalone test...")

    # --- Load data ---
    uim = load_user_item_matrix()
    meta = load_product_metadata()
    embeddings = load_product_embeddings()
    furn = load_furniture_csv()

    # --- Build features (use small sample) ---
    feature_df = build_feature_matrix(
        user_item_matrix=uim,
        furniture_df=furn,
        product_meta=meta,
        product_embeddings=embeddings,
        max_users=100
    )

    print(f"\nFeature matrix shape: {feature_df.shape}")
    print(feature_df[FEATURE_COLS + ["label"]].describe())

    # --- Train model ---
    model = train_xgb_ranker(feature_df)

    # --- Save model ---
    save_model(model)

    # --- Test inference ---
    test_user = uim.index[0]
    test_candidates = uim.columns[:10].tolist()

    ranked = predict_ranking(
        model=model,
        user_id=test_user,
        candidate_items=test_candidates,
        user_item_matrix=uim,
        furniture_df=furn,
        product_meta=meta,
        product_embeddings=embeddings
    )

    print(f"\n🎯 Ranked items for user {test_user}:")
    for rank, item in enumerate(ranked, 1):
        print(f"  {rank}. {item}")

    print("\n✅ itr_model.py standalone test complete.")
