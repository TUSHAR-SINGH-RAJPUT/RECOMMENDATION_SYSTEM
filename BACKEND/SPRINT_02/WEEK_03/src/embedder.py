import pandas as pd
import os
import json

DATA_PATH = "../embeddings/Furniture.csv"
OUTPUT_PATH = "../embeddings/embedding_metadata.json"

df = pd.read_csv(DATA_PATH)

# Use available columns (adjust if your CSV has name column)
cols = ['price', 'category', 'material', 'color']
df = df[cols]

# Clean
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['price'].fillna(df['price'].median(), inplace=True)

df.fillna({
    "material": "unknown",
    "color": "unknown",
    "category": "furniture"
}, inplace=True)

for col in ['category', 'material', 'color']:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Style
q1, q2 = df['price'].quantile([0.33, 0.66])

def get_style(price):
    if price >= q2:
        return "luxury"
    elif price >= q1:
        return "modern"
    return "minimalist"

df['style'] = df['price'].apply(get_style)

# Product name (constructed)
df['product_name'] = df['material'] + " " + df['category']

# Description
def create_description(row):
    return (
        f"{row['style']} {row['material']} {row['category']} "
        f"in {row['color']} color. price: {int(row['price'])}"
    )

df['description'] = df.apply(create_description, axis=1)

# ID
df['product_id'] = ["P" + str(i) for i in df.index]

df = df.drop_duplicates(subset=['description']).reset_index(drop=True)

# Save JSON
records = df[['product_id', 'product_name', 'category', 'description']].to_dict(orient="records")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(records, f, indent=4)

print("✅ Task 1 completed")







# TASK 02
from sentence_transformers import SentenceTransformer
import numpy as np

print("\n🔹 TASK 2: Embeddings")

with open(OUTPUT_PATH, "r") as f:
    data = json.load(f)

texts = [item["description"] for item in data]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

np.save("../embeddings/product_embeddings.npy", embeddings)

print("✅ Embeddings saved")