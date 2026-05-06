# 🧠 GenAI Product Recommendation System (Sprint 2 - Week 03)

## 📌 Project Overview

This project implements a **content-based recommendation system** using **Generative AI embeddings** and a **vector database (ChromaDB)**.

The system converts product attributes into natural language, generates embeddings, stores them in a vector database, and retrieves similar products based on semantic similarity.

---

## 🎯 Objectives

* Convert structured product data into meaningful text
* Generate high-quality embeddings using transformer models
* Store embeddings in a vector database (ChromaDB)
* Perform semantic search with filtering and ranking
* Evaluate and optimize embedding performance

---

## 🏗️ Project Structure

```
GenAI_Recommender_<TeamName>_Sprint2_Week3/
│
├── embeddings/
│   ├── product_embeddings.npy
│   └── embedding_metadata.json 
│
├── vector_db/
│   └── chroma_db/
│
├── src/
│   ├── embedder.py
│   ├── vector_store.py
│   └── content_recommender.py
│
├── notebooks/
│   └── embedding_analysis.ipynb
│
└── README.md
```

---

# ⚙️ Setup Instructions

## 1️⃣ Create Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate   # Windows
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Task Breakdown

---

## 🔹 Task 1: Product Description Generation

* Combines attributes:

  ```
  style + material + category + color + price
  ```

* Example:

  ```
  "modern wooden chair in brown color. price: 2500"
  ```

* Output:

  ```
  embeddings/embedding_metadata.json
  ```

---

## 🔹 Task 2: Embedding Generation

* Model used:

  * `all-MiniLM-L6-v2` (default)

* Converts descriptions → embeddings

* Output:

  ```
  embeddings/product_embeddings.npy
  ```

---

## 🔹 Task 3: Vector Database (ChromaDB)

* Stores embeddings + metadata

* Persistent storage

* Output:

  ```
  vector_db/chroma_db/
  ```

---

## 🔹 Task 4: Vector Search

* Query → Embedding → Search → Top-K results
* Features:

  * Semantic similarity
  * Category filtering
  * Similarity threshold

---

## 🔹 Task 5: Evaluation

* Cosine similarity testing
* PCA visualization
* Cluster analysis
* Observations of model performance

---

## 🔹 Task 6: Optimization

* PCA dimensionality reduction
* Model comparison (MiniLM, MPNet, BGE)
* Retrieval accuracy measurement
* Parameter tuning (Top-K, threshold)

---

# 🚀 How to Run

---

## ✅ Step 1: Generate Metadata + Embeddings

```bash
cd src
python embedder.py
```

✔ Creates:

* `embedding_metadata.json`
* `product_embeddings.npy`

---

## ✅ Step 2: Create Vector Database

```bash
python vector_store.py
```

✔ Stores data in ChromaDB

---

## ✅ Step 3: Run Recommender

```bash
python content_recommender.py
```

✔ Example Output:

```
🔍 Query: modern wooden chair

🎯 Recommended Products:

1. wooden chair
   Category: chair
   Description: modern wooden chair in green color. price: 340
   Score: 0.713
```

---

## 📈 Run Analysis (Task 5 & 6)

```bash
cd notebooks
jupyter notebook
```

Open:

```
embedding_analysis.ipynb
```

---

# 🧠 Technologies Used

* Python
* Pandas, NumPy
* Sentence Transformers
* ChromaDB
* Scikit-learn
* UMAP / PCA
* Matplotlib

---

# 📊 Key Features

✔ Semantic search using embeddings
✔ Vector database (ChromaDB)
✔ Metadata filtering
✔ Similarity thresholding
✔ Embedding visualization
✔ Model optimization

---

# ⚡ Improvements Made

* Added structured metadata (category, product_name)
* Implemented similarity threshold tuning
* Improved diversity in recommendations
* Applied PCA for optimization
* Compared multiple embedding models

---

# ⚠️ Limitations

* Price understanding is limited in embeddings
* Highly similar descriptions reduce diversity
* No user personalization (content-based only)

---

# 🚀 Future Work

* Hybrid recommendation (content + collaborative)
* User preference learning
* FastAPI backend deployment
* Streamlit UI for interaction
* Advanced ranking algorithms

---

# 👨‍💻 Author

Tushar Singh
Team Lambda A

---

# ✅ Conclusion

This project successfully demonstrates a **complete GenAI-powered recommendation pipeline**, from data preprocessing to optimized semantic retrieval using vector databases.

---
