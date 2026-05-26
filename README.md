# 🛋️ FurniAI — AI-Powered Furniture Recommendation System

A full-stack product recommendation system that integrates classical machine learning (Collaborative Filtering) with modern Generative AI (Content-Based Semantic Search, Conversational RAG) to provide hyper-personalized furniture recommendations.

Built as a college project across multiple sprints.

## 🌟 Key Features

- **Four Recommendation Models**:
  - **Embedding**: Semantic search using Sentence Transformers and ChromaDB.
  - **Collaborative**: User-behavior pattern matching.
  - **Hybrid**: Blends CF and Embeddings, with optional XGBoost reranking.
  - **Conversational**: AI chat-based recommendations with product attachments.
- **Premium Frontend UI**: React 19 + Vite app featuring a dark-mode glassmorphism design system.
- **Performance Metrics**: Real-time display of response times and model comparisons on every query.
- **FastAPI Backend**: Modular architecture serving Python ML models.

---

## 📁 Repository Structure

```
RECOMMENDATION_SYSTEM/
│
├── BACKEND/
│   ├── SPRINT_01/ & SPRINT_02/    # Data generation & classical ML models
│   ├── SPRINT_03/WEEK_06/         # FastAPI Backend & React-connected API
│   │   ├── api/
│   │   │   ├── routes/            # collaborative.py, embedding.py, hybrid.py, conversational.py
│   │   │   ├── main.py            # FastAPI entry point
│   │   │   └── schemas.py         # Pydantic models
│   │   └── dashboard/app.py       # Streamlit testing dashboard
│   ├── myenv/                     # Python virtual environment
│   └── requirements.txt
│
└── FRONTEND/                      # React 19 + Vite web application
    ├── src/
    │   ├── api/                   # recommenderApi.js (FastAPI integration)
    │   ├── components/            # UI components (Navbar, ProductCard, SearchBar, etc.)
    │   ├── data/                  # embedding_metadata.json (Fallback product data)
    │   ├── hooks/                 # useRecommendations.js (API state management)
    │   ├── pages/                 # Home.jsx (Main application view)
    │   ├── App.jsx
    │   └── index.css              # Design system tokens and globals
    ├── index.html
    └── package.json
```

---

## 🚀 Running the Project

### 1. Start the FastAPI Backend

Open a terminal and navigate to the project directory:

```bash
cd BACKEND
# Activate virtual environment (Windows)
myenv\Scripts\activate
# Start the server
uvicorn SPRINT_03.WEEK_06.api.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`. You can view the Swagger UI documentation at `http://127.0.0.1:8000/docs`.

### 2. Start the React Frontend

Open a **second** terminal:

```bash
cd FRONTEND
npm install
npm run dev
```

The web app will start at `http://localhost:5173`.

---

## 🔌 API Endpoints Reference

All endpoints accept `POST` requests and return JSON.

### 1. Embedding (Content-Based)
`POST /recommend/embedding/`
```json
// Request
{ "query": "modern wooden sofa", "top_k": 10 }

// Response
{ "model": "Embedding Based Recommendation", "query": "...", "recommendations": [...] }
```

### 2. Collaborative Filtering
`POST /recommend/collaborative/`
```json
// Request
{ "user_id": 42, "query": "desk", "top_k": 5 }

// Response
{ "model": "Collaborative Filtering", "user_id": 42, "recommendations": [...] }
```

### 3. Hybrid
`POST /recommend/hybrid/`
```json
// Request
{ "user_id": 42, "query": "chair", "top_k": 10, "use_xgboost": true }

// Response
{ "model": "Hybrid Recommendation System", "query": "...", "recommendations": [...] }
```

### 4. Conversational RAG
`POST /recommend/conversational/`
```json
// Request
{ "user_id": 42, "query": "I need a comfortable reading chair" }

// Response
{ "model": "Conversational RAG Recommender", "response": "AI text here...", "products": [...] }
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 19, Vite, Vanilla CSS Modules |
| **Backend API**| FastAPI, Uvicorn, Pydantic |
| **Machine Learning**| Scikit-learn, Sentence Transformers (`all-MiniLM-L6-v2`), XGBoost |
| **Database**| ChromaDB (Vector Store) |

---

## 👨‍💻 Author
**Tushar Singh Rajput** — Team Lambda A
