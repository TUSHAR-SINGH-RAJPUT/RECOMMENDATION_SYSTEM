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

    similar_users = user_similarity_df.loc[user_id] \
        .sort_values(ascending=False)

    # filter only meaningful similarities
    similar_users = similar_users[similar_users > 0]

    # return top K users
    return similar_users.head(top_k).index.tolist()


# print(get_similar_users(2))
def recommend_cf(user_id, n=10):
    # Step 0: check if user exists
    if user_id not in user_item_matrix.index:
        return []

    # Step 1: get similar users with scores
    similar_users = user_similarity_df.loc[user_id] \
        .sort_values(ascending=False)

    # keep only positive similarity
    similar_users = similar_users[similar_users > 0]

    # Step 2: get items already seen by user
    user_items = set(
        user_item_matrix.loc[user_id][
            user_item_matrix.loc[user_id] > 0
        ].index
    )

    # Step 3: collect recommendation scores
    scores = {}

    for sim_user, sim_score in similar_users.items():
        sim_user_items = user_item_matrix.loc[sim_user]

        for item, value in sim_user_items.items():
            # only consider items the similar user interacted with
            if value > 0 and item not in user_items:
                # weighted scoring
                scores[item] = scores.get(item, 0) + sim_score

    # Step 4: sort items by score
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Step 5: return top N product_ids
    return [item for item, _ in ranked_items[:n]]

