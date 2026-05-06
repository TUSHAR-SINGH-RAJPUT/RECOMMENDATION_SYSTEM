# import numpy as np
# import pandas as pd
# import os
# from sklearn.decomposition import TruncatedSVD

# # -----------------------------
# # Load Data

# BASE_DIR = os.path.dirname(__file__)
# input_path =  os.path.abspath(os.path.join(BASE_DIR,"..","data","user_item_matrix.csv"))

# df = pd.read_csv(input_path, index_col=0)


# nparray = np.array(df)
# # print("Original Matrix Shape:", nparray.shape)

# # -----------------------------
# # Matrix Factorization using SVD
# k = 20  # Number of latent factors
# svd = TruncatedSVD(n_components=k, random_state=42)
# user_features = svd.fit_transform(nparray)
# item_features = svd.components_
# # print("User Features Shape:", user_features.shape)
# # print("Item Features Shape:", item_features.shape)

# # reconstruct the matrix
# reconstructed_matrix = np.dot(user_features, item_features)
# # print("Reconstructed Matrix Shape:", reconstructed_matrix.shape)

# # building recommendation function
# def recommend_mf(user_id, n=10):
#     # Step 0: check if user exists
#     if user_id not in df.index:
#         return []

#     # Step 1: get row index of user
#     user_idx = df.index.get_loc(user_id)

#     # Step 2: get predicted scores for that user
#     predicted_scores = reconstructed_matrix[user_idx]

#     # Step 3: sort items by score (descending)
#     item_indices = np.argsort(predicted_scores)[::-1]

#     # Step 4: get items already seen by user
#     user_row = df.loc[user_id]
#     seen_items = set(user_row[user_row > 0].index)

#     # Step 5: build recommendations
#     recommendations = []

#     for idx in item_indices:
#         product_id = df.columns[idx]  # map index → product_id

#         if product_id not in seen_items:
#             recommendations.append(product_id)

#         if len(recommendations) == n:
#             break

#     return recommendations

# print("MF Recommendations:", recommend_mf(2, 5))








import numpy as np
import pandas as pd
import os
from sklearn.decomposition import TruncatedSVD

# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "user_item_matrix.csv")
)
MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data")
)

N_COMPONENTS = 20  # latent factors


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)
    return df


# -----------------------------
# Train Model
# -----------------------------
def train_svd(matrix):
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    user_features = svd.fit_transform(matrix)
    item_features = svd.components_
    return user_features, item_features


# -----------------------------
# Reconstruct Matrix
# -----------------------------
def reconstruct_matrix(user_features, item_features):
    return np.dot(user_features, item_features)


# -----------------------------
# Save Model Artifacts
# -----------------------------
def save_model(user_features, item_features):
    os.makedirs(MODEL_DIR, exist_ok=True)

    np.save(os.path.join(MODEL_DIR, "user_features.npy"), user_features)
    np.save(os.path.join(MODEL_DIR, "item_features.npy"), item_features)

    print("✅ Model saved")


# -----------------------------
# Load Model Artifacts
# -----------------------------
def load_model():
    user_features = np.load(os.path.join(MODEL_DIR, "user_features.npy"))
    item_features = np.load(os.path.join(MODEL_DIR, "item_features.npy"))
    return user_features, item_features


# -----------------------------
# Popularity fallback
# -----------------------------
def popularity_fallback(df, n=10):
    popularity = (
        df.sum(axis=0)
        .sort_values(ascending=False)
        .head(n)
        .index
        .tolist()
    )
    return popularity


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_mf(user_id, df, reconstructed_matrix, n=10):
    # Cold start fallback
    if user_id not in df.index:
        return popularity_fallback(df, n)

    user_idx = df.index.get_loc(user_id)
    predicted_scores = reconstructed_matrix[user_idx]

    item_indices = np.argsort(predicted_scores)[::-1]

    # Already seen items
    user_row = df.loc[user_id]
    seen_items = set(user_row[user_row > 0].index)

    recommendations = []

    for idx in item_indices:
        product_id = df.columns[idx]

        if product_id not in seen_items:
            recommendations.append(product_id)

        if len(recommendations) == n:
            break

    # If nothing found → fallback
    if not recommendations:
        return popularity_fallback(df, n)

    return recommendations


# -----------------------------
# Main (Training + Testing)
# -----------------------------
if __name__ == "__main__":
    df = load_data()
    matrix = df.values

    # Train
    user_features, item_features = train_svd(matrix)

    # Save model
    save_model(user_features, item_features)

    # Reconstruct
    reconstructed = reconstruct_matrix(user_features, item_features)

    # Test
    print("MF Recommendations:", recommend_mf(2, df, reconstructed, 5))