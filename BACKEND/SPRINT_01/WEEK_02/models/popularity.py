import pandas as pd
import os

# -----------------------------
# Load Data
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

input_path = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "processed_user_behavior.csv")
)

df = pd.read_csv(input_path)

# -----------------------------
# Build Popularity Model
# -----------------------------
popularity_df = (
    df.groupby("product_id")["interaction_score"]
    .sum()
    .reset_index()
    .sort_values("interaction_score", ascending=False)
)

# -----------------------------
# Recommendation Functions
# -----------------------------
def recommend_top_n(n=10):
    return popularity_df.head(n)["product_id"].tolist()


def recommend_for_user(user_id, n=10):
    seen_items = df[df["user_id"] == user_id]["product_id"].unique()

    recommendations = popularity_df[
        ~popularity_df["product_id"].isin(seen_items)
    ]

    return recommendations.head(n)["product_id"].tolist()


# -----------------------------
# Save Model
# -----------------------------
def save_model():
    output_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "data", "popularity_model.csv")
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    popularity_df.to_csv(output_path, index=False)

    print("✅ Model saved at:", output_path)


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("Top 5 products:", recommend_top_n(5))
    print("User 123 recommendations:", recommend_for_user(123, 5))

    save_model()