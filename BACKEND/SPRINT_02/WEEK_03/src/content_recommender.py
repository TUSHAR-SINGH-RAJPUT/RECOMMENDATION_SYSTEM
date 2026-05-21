import chromadb

# =========================================================
# CONFIG
# =========================================================
import os

BASE_DIR = os.path.dirname(__file__)

CHROMA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../vector_db/chroma_db")
)
COLLECTION_NAME = "products"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.35

# =========================================================
# LOAD DB
# =========================================================
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

print("✅ Recommender Ready")

# =========================================================
# CORE SEARCH FUNCTION
# =========================================================
def search_products(query, top_k=TOP_K, category=None, threshold=SIMILARITY_THRESHOLD):
    """
    Query → Embed → Search → Filter → Return results
    """

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"category": category} if category else None
    )

    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]

    final_results = []

    for i in range(len(ids)):

        # Convert distance → similarity
        score = 1 / (1 + distances[i])

        if score < threshold:
            continue

        final_results.append({
            "product_id": ids[i],
            "product_name": metas[i]["product_name"],
            "category": metas[i]["category"],
            "description": docs[i],
            "score": round(score, 3)
        })

    return final_results


# =========================================================
# PRETTY PRINT FUNCTION
# =========================================================
def display_results(results):
    if not results:
        print("❌ No relevant products found\n")
        return

    print("\n🎯 Recommended Products:\n")

    for i, item in enumerate(results, 1):
        print(f"{i}. {item['product_name']}")
        print(f"   Category: {item['category']}")
        print(f"   Description: {item['description']}")
        print(f"   Similarity Score: {item['score']}\n")


# =========================================================
# TEST CASES 
# =========================================================
if __name__ == "__main__":

    print("\n🔹 Test 1: Basic Search")
    results = search_products("modern wooden chair")
    display_results(results)

    print("\n🔹 Test 2: Category Filter")
    results = search_products("chair", category="chair")
    display_results(results)

    print("\n🔹 Test 3: Different Query")
    results = search_products("luxury sofa")
    display_results(results)