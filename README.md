# RoomSense - AI Furniture Recommendation System

RoomSense is a full-stack furniture recommendation project built across multiple college sprints. It combines classical recommendation techniques, semantic search, hybrid ranking, and a conversational AI assistant into one React + FastAPI application.

The repository is documented so it can be pushed to GitHub, deleted locally, and restored later without losing the project context.

## What This Project Does

RoomSense helps users discover furniture products through four recommendation modes:

| Mode | Purpose | Main Technique |
| --- | --- | --- |
| Embedding recommendations | Finds products similar to a text query such as `modern wooden chair` | Sentence embeddings + ChromaDB |
| Collaborative filtering | Recommends products based on similar user behavior | User-user cosine similarity |
| Hybrid recommendations | Blends semantic matches, collaborative signals, user profile features, and optional XGBoost reranking | Content + CF + learning-to-rank style scoring |
| Conversational assistant | Lets users chat naturally and receive product-aware responses | RAG over product retrieval + local Ollama model |

## Tech Stack

| Layer | Tools |
| --- | --- |
| Frontend | React, Vite, CSS Modules, GSAP |
| Backend API | FastAPI, Uvicorn, Pydantic |
| ML and data | Pandas, NumPy, scikit-learn, sentence-transformers, ChromaDB, XGBoost |
| Conversational AI | Ollama with local `mistral` model |
| Optional dashboard | Streamlit |

## Repository Structure

```text
.
|-- BACKEND/
|   |-- requirement.txt
|   |-- SPRINT_01/
|   |   |-- WEEK_01/                 # Data generation, cleaning, feature engineering notebooks
|   |   `-- WEEK_02/                 # Collaborative filtering and matrix factorization work
|   |-- SPRINT_02/
|   |   |-- WEEK_03/                 # Embeddings, ChromaDB vector store, semantic search
|   |   `-- WEEK_04/                 # Hybrid recommender, XGBoost ranker, evaluation
|   `-- SPRINT_03/
|       |-- WEEK_05/                 # RAG/LLM experiments
|       `-- WEEK_06/
|           |-- api/                 # FastAPI app used by the frontend
|           `-- dashboard/           # Streamlit dashboard
|-- FRONTEND/
|   |-- package.json
|   |-- src/
|   |   |-- api/                     # API client functions
|   |   |-- components/              # Reusable UI components
|   |   |-- data/                    # Frontend product metadata fallback
|   |   |-- hooks/                   # Recommendation state management
|   |   `-- pages/                   # Main Home page
|   `-- public/
|-- docs/
|   `-- RESTORE_AND_MAINTENANCE.md   # Future setup, recovery, and maintenance notes
|-- requirement.txt                  # Minimal API-only dependency file
`-- run_all.sh                       # Starts backend and frontend together on Bash-compatible shells
```

## Important Runtime Files

These files are part of the trained/data-backed behavior. Keep them in GitHub unless they become too large for normal Git hosting.

| File or folder | Why it matters |
| --- | --- |
| `BACKEND/SPRINT_01/WEEK_02/data/user_item_matrix.csv` | Collaborative filtering input matrix |
| `BACKEND/SPRINT_02/WEEK_03/embeddings/embedding_metadata.json` | Product catalog metadata used by backend services |
| `BACKEND/SPRINT_02/WEEK_03/embeddings/product_embeddings.npy` | Precomputed product embeddings |
| `BACKEND/SPRINT_02/WEEK_03/vector_db/chroma_db/` | Persistent ChromaDB vector database |
| `BACKEND/SPRINT_02/WEEK_04/Models/Model_checkpoints/xgb_ranker.pkl` | Trained XGBoost ranking model |
| `FRONTEND/src/data/embedding_metadata.json` | Frontend fallback product metadata |

The local virtual environment `BACKEND/myenv/` is intentionally ignored and should not be pushed. Recreate it from `BACKEND/requirement.txt`.

## Prerequisites

Install these before running the project:

- Python 3.10 or newer
- Node.js and npm
- Git
- Bash-compatible terminal for `run_all.sh` on Windows, such as Git Bash
- Optional: Ollama for conversational chat

## Quick Start

From the repository root:

```bash
cd BACKEND
python -m venv myenv
source myenv/Scripts/activate   # Git Bash on Windows
pip install -r requirement.txt
cd ..

cd FRONTEND
npm install
cd ..

bash run_all.sh
```

Open these URLs:

- Frontend: `http://127.0.0.1:5173`
- API docs: `http://127.0.0.1:8000/docs`
- API health check: `http://127.0.0.1:8000/health`

## Manual Run Commands

Use these if you prefer separate terminals.

Terminal 1 - backend:

```bash
cd BACKEND
myenv\Scripts\activate
python -m uvicorn SPRINT_03.WEEK_06.api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 - frontend:

```bash
cd FRONTEND
npm install
npm run dev
```

If the backend is not running on `http://127.0.0.1:8000`, set the frontend API URL:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

On PowerShell:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

## Conversational Chat Setup

The chat endpoint streams responses from a local Ollama model. Product retrieval can work without Ollama, but generated chat text needs Ollama running.

Install and start Ollama, then pull Mistral:

```bash
ollama pull mistral
ollama serve
```

Optional environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/generate` | Ollama generate endpoint |
| `OLLAMA_MODEL` | `mistral` | Local model used for chat |
| `CHAT_MEMORY_TURNS` | `8` | Number of conversation turns kept per session |

## API Reference

Base URL:

```text
http://127.0.0.1:8000
```

### Health

```http
GET /
GET /health
```

### Embedding Recommendations

```http
POST /recommend/embedding/
Content-Type: application/json
```

Request:

```json
{
  "query": "modern wooden chair",
  "top_k": 10,
  "category": "chair"
}
```

Response shape:

```json
{
  "model": "Embedding Based Recommendation",
  "query": "modern wooden chair",
  "count": 10,
  "recommendations": [
    {
      "product_id": "P12",
      "product_name": "Modern Wooden Chair",
      "category": "chair",
      "description": "Product description",
      "score": 0.84,
      "metadata": {}
    }
  ]
}
```

### Collaborative Filtering

```http
POST /recommend/collaborative/
Content-Type: application/json
```

Request:

```json
{
  "user_id": 1,
  "query": "chair",
  "top_k": 10
}
```

The `query` field is accepted for a shared request shape, but collaborative filtering mainly uses `user_id`.

### Hybrid Recommendations

```http
POST /recommend/hybrid/
Content-Type: application/json
```

Request:

```json
{
  "user_id": 1,
  "query": "comfortable office chair",
  "top_k": 10,
  "category": "chair",
  "use_xgboost": true
}
```

If the XGBoost checkpoint cannot be loaded or `use_xgboost` is false, the hybrid recommender falls back to weighted scoring.

### Conversational Recommendations

```http
POST /recommend/conversational/
Content-Type: application/json
Accept: text/event-stream
```

Request:

```json
{
  "user_id": 1,
  "query": "I need a comfortable reading chair",
  "top_k": 5,
  "session_id": "optional-session-id"
}
```

This endpoint returns Server-Sent Events:

- `products`: retrieved products and session metadata
- `token`: streamed LLM tokens
- `done`: final response
- `error`: Ollama or streaming error details

## Main Backend Flow

```text
React UI
  -> FRONTEND/src/api/recommenderApi.js
  -> FastAPI route in BACKEND/SPRINT_03/WEEK_06/api/routes/
  -> service layer in api/services/
  -> sprint model modules and stored artifacts
  -> normalized product response
  -> React cards, metrics, and chatbot UI
```

Key backend entry point:

```text
BACKEND/SPRINT_03/WEEK_06/api/main.py
```

Key frontend entry point:

```text
FRONTEND/src/pages/Home.jsx
```

## Development Commands

Backend:

```bash
cd BACKEND
python -m uvicorn SPRINT_03.WEEK_06.api.main:app --reload
```

Frontend:

```bash
cd FRONTEND
npm run dev
npm run build
npm run lint
```

Streamlit dashboard:

```bash
cd BACKEND
streamlit run SPRINT_03/WEEK_06/dashboard/app.py
```

## Troubleshooting

### `ModuleNotFoundError` from backend

Run the API from inside the `BACKEND` folder:

```bash
cd BACKEND
python -m uvicorn SPRINT_03.WEEK_06.api.main:app --reload
```

### Chat says Ollama is not reachable

Start Ollama and confirm the model exists:

```bash
ollama pull mistral
ollama serve
```

### Frontend cannot reach backend

Check that the backend is running at `http://127.0.0.1:8000`, then set:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### Embedding endpoint fails

Confirm the ChromaDB files exist:

```text
BACKEND/SPRINT_02/WEEK_03/vector_db/chroma_db/
```

Also confirm product metadata exists:

```text
BACKEND/SPRINT_02/WEEK_03/embeddings/embedding_metadata.json
```

## Before Pushing to GitHub

1. Confirm the app runs locally.
2. Confirm `.gitignore` excludes virtual environments and `node_modules`.
3. Do not push `BACKEND/myenv/` or `FRONTEND/node_modules/`.
4. Commit the README, docs, source code, data artifacts needed for demo, and lock files.
5. After pushing, open GitHub and verify the README renders correctly.
6. Clone the repo into a temporary folder once to confirm the restore instructions work.

## Future Restore Summary

When you need this project again:

```bash
git clone <your-github-repo-url>
cd <repo-name>
cd BACKEND
python -m venv myenv
myenv\Scripts\activate
pip install -r requirement.txt
cd ..\FRONTEND
npm install
cd ..
bash run_all.sh
```

For a detailed checklist, see `docs/RESTORE_AND_MAINTENANCE.md`.

## Author

Tushar Singh Rajput

Team Lambda A
