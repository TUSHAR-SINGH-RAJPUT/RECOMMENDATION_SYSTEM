import chromadb
import json
import numpy as np

EMBEDDING_PATH = "../embeddings/product_embeddings.npy"
METADATA_PATH = "../embeddings/embedding_metadata.json"

client = chromadb.PersistentClient(path="../vector_db/chroma_db")

try:
    client.delete_collection("products")
except:
    pass

collection = client.create_collection(name="products")

embeddings = np.load(EMBEDDING_PATH)

with open(METADATA_PATH) as f:
    data = json.load(f)

ids = [item["product_id"] for item in data]
documents = [item["description"] for item in data]

# 🔥 IMPORTANT: structured metadata
metadatas = [
    {
        "category": item["category"],
        "product_name": item["product_name"]
    }
    for item in data
]

collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=documents,
    metadatas=metadatas
)

print("✅ Chroma DB setup done")