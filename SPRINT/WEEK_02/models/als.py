import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "user_item_matrix.csv")
)

FACTORS = 20
REGULARIZATION = 0.1
ITERATIONS = 20


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)
    return df


# -----------------------------
# Prepare Sparse Matrix
# -----------------------------
def prepare_matrix(df):
    # Convert to sparse
    user_item = csr_matrix(df.values)

    # ALS needs item-user matrix
    item_user = user_item.T

    return user_item, item_user


# -----------------------------
# Train ALS Model
# -----------------------------
def train_als(item_user_matrix):
    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        random_state=42
    )

    # implicit expects confidence, not raw values
    alpha = 15
    confidence = item_user_matrix * alpha

    model.fit(confidence)

    return model


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_als(user_id, df, model, user_item_matrix, n=10):
    if user_id not in df.index:
        return []

    user_idx = df.index.get_loc(user_id)

    # get recommendations
    item_ids, scores = model.recommend(
        userid=user_idx,
        user_items=user_item_matrix,
        N=n,
        filter_already_liked_items=True
    )

    # map indices → product_ids
    product_ids = df.columns[item_ids].tolist()

    return product_ids


# -----------------------------
# Popularity fallback
# -----------------------------
def popularity_fallback(df, n=10):
    return (
        df.sum(axis=0)
        .sort_values(ascending=False)
        .head(n)
        .index
        .tolist()
    )


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = load_data()

    user_item_matrix, item_user_matrix = prepare_matrix(df)

    model = train_als(item_user_matrix)

    # Test
    user_id = df.index[0]

    recs = recommend_als(user_id, df, model, user_item_matrix, 5)

    if not recs:
        recs = popularity_fallback(df, 5)

    print("ALS Recommendations:", recs)