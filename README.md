# 🛍️ Product Recommendation System

A full-stack, AI-powered product recommendation system built as a college project across two sprints. The backend implements multiple recommendation algorithms — from classical collaborative filtering to GenAI-based semantic search using vector embeddings — while the frontend provides a React + Vite UI to interact with the recommendations.

---

## 📁 Repository Structure

```
RECOMMENDATION_SYSTEM/
│
├── BACKEND/
│   ├── requirements.txt           # All Python dependencies
│   ├── SPRINT_01/
│   │   ├── WEEK_01/               # Data generation, cleaning & feature engineering
│   │   │   ├── data/              # Raw & processed datasets
│   │   │   ├── notebooks/         # EDA and ETL pipeline notebooks
│   │   │   ├── reports/           # EDA report (PDF)
│   │   │   └── src/               # data_generator.py, feature_engineer.py
│   │   └── WEEK_02/               # ML recommendation models
│   │       ├── data/              # User-item interaction matrices
│   │       ├── models/            # ALS, collaborative filter, matrix factorization, popularity
│   │       ├── pipelines/         # ETL pipeline
│   │       ├── evaluation/        # Evaluation scripts
│   │       └── tests/             # Model testing notebooks
│   └── SPRINT_02/
│       └── WEEK_03/               # GenAI content-based recommendation
│           ├── embeddings/        # Pre-generated product embeddings (.npy + metadata)
│           ├── vector_db/         # Persisted ChromaDB vector store
│           ├── src/               # embedder.py, vector_store.py, content_recommender.py
│           └── notebooks/         # Embedding analysis & evaluation notebook
│
└── FRONTEND/                      # React + Vite web application
    ├── src/
    │   ├── api/                   # API call definitions
    │   ├── components/            # Reusable UI components (Navbar, Footer, ProductCard, etc.)
    │   ├── context/               # React context for global state
    │   ├── hooks/                 # Custom hooks (e.g. useRecommendations)
    │   ├── pages/                 # Page components (Home, ProductPage)
    │   ├── services/              # Service layer for recommendation logic
    │   ├── styles/                # Global CSS
    │   └── utils/                 # Helper utilities
    ├── public/                    # Static assets
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## 🧠 Project Overview

### Sprint 01 — Classical Recommendation Models

> **Goal:** Build a solid data pipeline and implement traditional ML-based recommendation algorithms.

#### Week 01 — Data Engineering
- Synthetic e-commerce dataset generation (products + user behavior)
- Data cleaning and preprocessing
- Feature engineering (user features, item features)
- Full ETL pipeline with Jupyter notebooks

#### Week 02 — Recommendation Models
Four models were implemented and evaluated:

| Model | Description |
|---|---|
| **Popularity** | Baseline — recommends globally top-rated products |
| **Collaborative Filter** | User-based collaborative filtering using cosine similarity |
| **Matrix Factorization** | SVD-based latent factor model |
| **ALS** | Alternating Least Squares using the `implicit` library for implicit feedback |

All models include a **popularity fallback** for cold-start users.

---

### Sprint 02 — GenAI Content-Based Recommendation

> **Goal:** Replace/augment classical models with a semantic search engine powered by transformer embeddings and a vector database.

#### Week 03 — Embedding Pipeline + Vector Search

**How it works:**

```
Product Attributes → Natural Language Description → Transformer Embedding → ChromaDB → Semantic Search
```

1. **`embedder.py`** — Converts product attributes (style, material, category, color, price) into text descriptions and generates embeddings using `all-MiniLM-L6-v2` (Sentence Transformers)
2. **`vector_store.py`** — Stores embeddings + metadata in a persistent ChromaDB vector database
3. **`content_recommender.py`** — Accepts a natural language query, embeds it, and retrieves the most semantically similar products with optional category filtering and similarity thresholding

**Example:**
```
Query: "modern wooden chair"

🎯 Recommended Products:
1. wooden chair
   Category: chair
   Similarity Score: 0.713
```

---

## 🖥️ Frontend

A **React 19 + Vite** web application that serves as the UI layer for the recommendation system.

**Tech stack:**
- React 19
- Vite 8
- Vanilla CSS (custom design system)
- Custom hooks for data fetching
- Component-based architecture (ProductCard, RecommendationCard, Navbar, Footer)

**To run locally:**
```bash
cd FRONTEND
npm install
npm run dev
```
App will start at `http://localhost:5173`

---

## ⚙️ Backend Setup

### 1. Create & activate virtual environment
```bash
cd BACKEND
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Sprint 01 models
```bash
cd SPRINT_01/WEEK_02/models
python als.py
python collaborative_filter.py
python popularity.py
```

### 4. Run Sprint 02 GenAI recommender
```bash
cd SPRINT_02/WEEK_03/src

# Step 1: Generate embeddings
python embedder.py

# Step 2: Populate vector DB
python vector_store.py

# Step 3: Run semantic search
python content_recommender.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19, Vite 8, Vanilla CSS |
| **Classical ML** | NumPy, Pandas, SciPy, `implicit` (ALS) |
| **GenAI / Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Database** | ChromaDB |
| **Data Analysis** | Jupyter Notebooks, Matplotlib, Scikit-learn |
| **Language** | Python 3.10+, JavaScript (ESM) |

---

## 📊 Key Features

- ✅ End-to-end data generation → model training pipeline
- ✅ Four classical recommendation algorithms with popularity fallback
- ✅ GenAI semantic search using transformer embeddings
- ✅ Persistent vector database (ChromaDB)
- ✅ Category filtering + similarity thresholding in search
- ✅ React frontend with custom hooks, context and service layer
- ✅ Modular, sprint-based project structure

---

## ⚠️ Known Limitations

- Price semantics are partially captured by embeddings — not directly comparable
- No real-time user personalization (content-based only in Sprint 02)
- Frontend is not yet wired to a live backend API (in progress)

---

## 🚀 Future Work

- [ ] FastAPI backend exposing recommendation endpoints
- [ ] Connect React frontend to live API
- [ ] Hybrid recommendation (collaborative + content-based)
- [ ] User authentication and preference learning
- [ ] Deployment (Docker + cloud hosting)

---

## 👨‍💻 Author

**Tushar Singh Rajput** — Team Lambda A

---
