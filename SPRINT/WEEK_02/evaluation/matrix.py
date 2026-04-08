import pandas as pd
import os
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.abspath(
    os.path.join(BASE_DIR,"..","data","processed_user_behavior.csv")
)
OUTPUT_FILE = os.path.abspath(
    os.path.join(BASE_DIR,"..","data","user_item_matrix.csv")
)
df = pd.read_csv(INPUT_FILE)

# Clean column names
df.columns = df.columns.str.strip()

print("Columns in dataset:", df.columns)

# Rename if needed
df = df.rename(columns={
    "item_id": "product_id",
    "interaction": "interaction_type"
})

# Drop missing
df = df.dropna(subset=["user_id", "product_id", "interaction_type"])

# Map interactions
interaction_weights = {
    "view": 1,
    "cart": 2,
    "purchase": 3
}

df["interaction_score"] = df["interaction_type"].map(interaction_weights)

# Create matrix
user_item_matrix = df.pivot_table(
    index="user_id",
    columns="product_id",
    values="interaction_score",
    aggfunc="sum",
    fill_value=0
)

print("Matrix Shape:", user_item_matrix.shape)

# Sparse
sparse_matrix = csr_matrix(user_item_matrix.values)

# Save
user_item_matrix.to_csv(OUTPUT_FILE)
save_npz("user_item_sparse.npz", sparse_matrix)
print(df.to_string(index=False))

print("Done ✅")