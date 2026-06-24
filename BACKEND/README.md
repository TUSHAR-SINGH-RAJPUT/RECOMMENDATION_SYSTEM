# RoomSense Backend

This folder contains the Python recommendation system, sprint artifacts, FastAPI application, and optional Streamlit dashboard.

## Main Entry Point

```text
SPRINT_03/WEEK_06/api/main.py
```

Run it from inside the `BACKEND` folder:

```bash
python -m uvicorn SPRINT_03.WEEK_06.api.main:app --reload --host 127.0.0.1 --port 8000
```

## Folder Guide

| Folder | Purpose |
| --- | --- |
| `SPRINT_01/WEEK_01/` | Data generation, cleaning, feature engineering, notebooks, and raw/processed data |
| `SPRINT_01/WEEK_02/` | Collaborative filtering, ALS, matrix factorization, evaluation data |
| `SPRINT_02/WEEK_03/` | Product embeddings, ChromaDB vector store, content-based recommender |
| `SPRINT_02/WEEK_04/` | Hybrid recommender, XGBoost ranker, training, evaluation |
| `SPRINT_03/WEEK_05/` | RAG and LLM experiment files |
| `SPRINT_03/WEEK_06/api/` | Production-style FastAPI layer used by the React frontend |
| `SPRINT_03/WEEK_06/dashboard/` | Streamlit dashboard for manual testing |

## Setup

```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r requirement.txt
```

Git Bash activation:

```bash
source myenv/Scripts/activate
```

## API Routes

| Route | Description |
| --- | --- |
| `GET /` | Basic API running message |
| `GET /health` | Health check |
| `POST /recommend/embedding/` | Semantic product search |
| `POST /recommend/collaborative/` | User-behavior recommendation |
| `POST /recommend/hybrid/` | Combined semantic, collaborative, and ranking recommendation |
| `POST /recommend/conversational/` | Streaming chat response with optional product retrieval |

## Required Artifacts

The backend relies on stored data and model artifacts:

```text
SPRINT_01/WEEK_02/data/user_item_matrix.csv
SPRINT_02/WEEK_03/embeddings/embedding_metadata.json
SPRINT_02/WEEK_03/embeddings/product_embeddings.npy
SPRINT_02/WEEK_03/vector_db/chroma_db/
SPRINT_02/WEEK_04/Models/Model_checkpoints/xgb_ranker.pkl
```

If these are missing, some recommenders may return empty results or fail at startup.

## Conversational AI

The chat service uses Ollama by default:

```text
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=mistral
```

Prepare Ollama:

```bash
ollama pull mistral
ollama serve
```

## Streamlit Dashboard

```bash
streamlit run SPRINT_03/WEEK_06/dashboard/app.py
```
