import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# -----------------------------
# Load Data
# -----------------------------

BASE_DIR = os.path.dirname(__file__)
input_path = os.path.abspath(
    os.path.join(BASE_DIR,"..","data","user_item_matrix.csv")
)
df= pd.read_csv(input_path, index_col=0)

user_item_matrix =df
 
# print(user_item_matrix.shape)
# print(user_item_matrix.head())

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
# print("User Similarity Matrix Shape:", user_similarity.shape)
np.fill_diagonal(user_similarity, 0)  # This line is ONLY to remove self-comparison
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)
# print(user_similarity_df.head())


def get_similar_users(user_id, top_k=5):
    if user_id not in user_similarity_df.index:
        return []

    #  sort by similarity score
    similar_users = user_similarity_df.loc[user_id] \
        .sort_values(ascending=False) 

    # filter only meaningful similarities
    similar_users = similar_users[similar_users > 0]

    # return top K users as list
    return similar_users.head(top_k).index.tolist()


def recommend_cf(user_id, n=10):
    """Recommend unseen items from the nearest users in the interaction matrix."""

    if user_id not in user_item_matrix.index:
        return []

    similar_users = user_similarity_df.loc[user_id] \
        .sort_values(ascending=False)

    similar_users = similar_users[similar_users > 0]

    user_items = set(
        user_item_matrix.loc[user_id][
            user_item_matrix.loc[user_id] > 0
        ].index
    )

    scores = {}

    for sim_user, sim_score in similar_users.items():

        sim_user_items = user_item_matrix.loc[sim_user]

        for item, value in sim_user_items.items():

            if value > 0 and item not in user_items:

                scores[item] = scores.get(item, 0) + sim_score

    ranked_items = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for item, score in ranked_items[:n]:

        recommendations.append({
            "product_id": str(item),
            "product_name": str(item),
            "category": "Collaborative Recommendation",
            "score": float(score)
        })

    return recommendations
